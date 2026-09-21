"""A read-only HTTP face on a store.

`serve-export` writes adapters to a directory a server can load. That works when
the server shares a filesystem with the store and is awkward when it does not:
the fleet either mounts the store or somebody copies files around and the copies
drift.

This serves the same thing over HTTP. A serving runtime asks for a tenant's
adapters, gets back the list it should load with a stable id for each, and pulls
each one by commit id. Because a commit id names exactly one set of tensors, a
response is immutable and can be cached for ever, which is the property that
makes this cheap to put behind anything.

Reads and writes are separate scopes on separate tokens, because they are
separate jobs: a fleet of servers pulling adapters should not hold a credential
that can delete one. Tenant isolation is enforced on every route, and a token
that may not read a tenant gets a 404 rather than a 403, because a 403 confirms
the tenant exists. A token that may read but not write gets a 403, since it
already knows.
"""

from __future__ import annotations

import io
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ballast import peft as peft_io
from ballast import serving
from ballast import tensors as st
from ballast.store import BrokenView, NotGranted, Store

ADAPTER_MEDIA_TYPE = "application/octet-stream"


class Tokens:
    """What each bearer token may do, and to which tenants.

    A token maps to `{"read": [...], "write": [...]}`, and `"*"` in either list
    means every tenant. A bare list is shorthand for read-only, which is the
    common case and the safe default to mistype toward. Write implies read,
    because a token that can replace a delta can already learn it.

    An empty table means the server is open and read-only: right behind a
    gateway that has already decided who is calling, wrong anywhere else, and
    never enough to write. The CLI says so when it starts without one.
    """

    def __init__(self, table: Mapping[str, Mapping[str, list[str]] | list[str]] | None = None) -> None:
        self.table: dict[str, dict[str, set[str]]] = {}
        for token, grants in (table or {}).items():
            if isinstance(grants, list):
                self.table[token] = {"read": set(grants), "write": set()}
                continue
            unknown = set(grants) - {"read", "write"}
            if unknown:
                raise ValueError(f"token {token!r} has unknown scopes {sorted(unknown)}")
            write = set(grants.get("write", []))
            self.table[token] = {"read": set(grants.get("read", [])) | write, "write": write}

    @property
    def open(self) -> bool:
        return not self.table

    def _may(self, token: str | None, tenant: str, scope: str) -> bool:
        if token is None:
            return False
        allowed = self.table.get(token, {}).get(scope, set())
        return tenant in allowed or "*" in allowed

    def may_read(self, token: str | None, tenant: str) -> bool:
        # An open server is readable by anyone who can reach the port.
        return True if self.open else self._may(token, tenant, "read")

    def may_write(self, token: str | None, tenant: str) -> bool:
        # Never open: writing always needs a token that says so.
        return self._may(token, tenant, "write")

    @classmethod
    def from_file(cls, path: str) -> Tokens:
        """A JSON object of token -> scopes, or token -> list for read-only."""
        with open(path) as f:
            return cls(json.load(f))


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    return value or None if scheme.lower() == "bearer" else None


def create_app(store: Store, tokens: Tokens | None = None, lora_name: str | None = None) -> Starlette:
    """A Starlette app reading from `store`."""
    tokens = tokens or Tokens()

    def guard(request: Request) -> str:
        tenant = str(request.path_params["tenant"])
        if not tokens.may_read(_bearer(request), tenant):
            raise _NotFound(f"no tenant {tenant!r}")
        return tenant

    def guard_write(request: Request) -> str:
        """Read first, so a token that cannot see a tenant learns nothing more."""
        tenant = guard(request)
        if not tokens.may_write(_bearer(request), tenant):
            raise _Forbidden(f"this token may read {tenant!r} but not write to it")
        return tenant

    async def health(request: Request) -> Response:  # noqa: ARG001
        return JSONResponse(
            {
                "status": "ok",
                "blocks": store.chunks.backend.describe(),
                "metadata": store.db.describe(),
                "authenticated": not tokens.open,
            }
        )

    async def refs(request: Request) -> Response:
        tenant = guard(request)
        return JSONResponse({"tenant": tenant, "refs": store.refs(tenant)})

    async def commit(request: Request) -> Response:
        tenant = guard(request)
        found = _resolve(store, tenant, request.path_params["spec"])
        manifest = store.manifest(tenant, found.manifest_id)
        body: dict[str, Any] = {
            "tenant": tenant,
            "commit": found.id,
            "parent": found.parent_id,
            "manifest": manifest.id,
            "kind": manifest.kind,
            "base_model": manifest.base_model,
            "message": found.message,
            "metadata": found.metadata,
            "created_at": found.created_at,
            "int_id": serving.int_id(found.id),
            "tensors": {
                name: {"dtype": d, "shape": list(s)} for name, (d, s) in store.specs(tenant, found.id).items()
            }
            if manifest.kind == "leaf"
            else None,
        }
        if manifest.kind == "composite":
            body["recipe"] = {
                **manifest.config,
                "inputs": [
                    {"tenant": t, "manifest": m, "weight": w} for t, m, w in store.inputs(tenant, manifest.id)
                ],
            }
        return JSONResponse(body, headers=_immutable(found.id))

    async def adapter(request: Request) -> Response:
        """The delta itself, as one safetensors file."""
        tenant = guard(request)
        found = _resolve(store, tenant, request.path_params["spec"])
        try:
            specs = store.specs(tenant, found.id)
            tensors = dict(store.checkout_stream(tenant, found.id))
        except BrokenView as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        buffer = io.BytesIO()
        st.save_stream(buffer, specs, tensors.__getitem__, {"format": "pt"})
        return Response(
            buffer.getvalue(),
            media_type=ADAPTER_MEDIA_TYPE,
            headers={
                **_immutable(found.id),
                "content-disposition": f'attachment; filename="{peft_io.WEIGHTS}"',
                "x-ballast-commit": found.id,
                "x-ballast-int-id": str(serving.int_id(found.id)),
            },
        )

    async def adapter_config(request: Request) -> Response:
        tenant = guard(request)
        found = _resolve(store, tenant, request.path_params["spec"])
        manifest = store.manifest(tenant, found.manifest_id)
        config = manifest.config if manifest.kind == "leaf" else {}
        return JSONResponse(config, headers=_immutable(found.id))

    async def loras(request: Request) -> Response:
        """What a serving runtime should load for this tenant.

        Views that cannot resolve are listed under `skipped` with the reason
        rather than left out, so an operator can see why an adapter is missing
        instead of wondering.
        """
        tenant = guard(request)
        entries, skipped = [], {}
        for name, commit_id in sorted(store.refs(tenant).items()):
            try:
                found = store.resolve(tenant, commit_id)
                manifest = store.manifest(tenant, found.manifest_id)
                if manifest.kind == "composite":
                    store.specs(tenant, found.id)  # resolves, so a broken view is caught here
                entries.append(
                    {
                        "ref": name,
                        "name": lora_name.format(tenant=tenant, ref=name)
                        if lora_name
                        else f"{tenant}-{name}",
                        "int_id": serving.int_id(found.id),
                        "commit": found.id,
                        "base_model": manifest.base_model,
                        "url": request.url_for("adapter", tenant=tenant, spec=found.id).path,
                    }
                )
            except (BrokenView, LookupError) as exc:
                skipped[name] = str(exc)
        return JSONResponse({"tenant": tenant, "loras": entries, "skipped": skipped})

    async def create_commit(request: Request) -> Response:
        """Store an adapter posted as safetensors.

        The body is the file; everything else travels as query parameters, so a
        client needs no multipart encoder and the request streams.
        """
        tenant = guard_write(request)
        params = request.query_params
        message = params.get("message")
        if not message:
            raise _BadRequest("a commit needs a message")
        body = await request.body()
        if not body:
            raise _BadRequest("the body must be a safetensors file")
        path = store.root / "incoming" / f"{uuid.uuid4().hex}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_bytes(body)
            tensors, _ = st.load(path)
            if not tensors:
                raise _BadRequest("the file holds no tensors")
            commit = store.commit(
                tenant,
                dict(tensors),
                message=message,
                ref=params.get("ref", "main"),
                base_model=params.get("base_model"),
                config=_json_param(params.get("config"), "config"),
                metadata=_json_param(params.get("metadata"), "metadata"),
            )
        except st.MalformedSafetensors as exc:
            raise _BadRequest(str(exc)) from exc
        except ValueError as exc:
            raise _BadRequest(str(exc)) from exc
        finally:
            path.unlink(missing_ok=True)
        return JSONResponse(
            {"commit": commit.id, "ref": params.get("ref", "main"), "tensors": len(tensors)},
            status_code=201,
        )

    async def create_merge(request: Request) -> Response:
        """Record a view over commits already in the store."""
        tenant = guard_write(request)
        payload = await _json_body(request)
        inputs = payload.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise _BadRequest("inputs must be a non-empty list of {spec, weight}")
        try:
            commit = store.merge(
                tenant,
                str(payload.get("method", "linear")),
                [(str(i["spec"]), float(i.get("weight", 1.0))) for i in inputs],
                message=str(payload.get("message") or ""),
                ref=str(payload.get("ref", "main")),
                density=payload.get("density"),
                strict=bool(payload.get("strict", True)),
                seed=payload.get("seed"),
                normalize=payload.get("normalize"),
                t=payload.get("t"),
            )
        except NotGranted as exc:
            raise _Forbidden(str(exc)) from exc
        except (KeyError, TypeError) as exc:
            raise _BadRequest(f"malformed inputs: {exc}") from exc
        except LookupError as exc:
            raise _NotFound(str(exc)) from exc
        except ValueError as exc:
            raise _BadRequest(str(exc)) from exc
        return JSONResponse({"commit": commit.id, "ref": payload.get("ref", "main")}, status_code=201)

    async def delete_commit(request: Request) -> Response:
        """Forget one commit, returning the proof."""
        tenant = guard_write(request)
        reason = request.query_params.get("reason")
        if not reason:
            raise _BadRequest("deleting needs a reason; it goes into the record")
        found = _resolve(store, tenant, request.path_params["spec"])
        proof = store.forget_commit(tenant, found.id, reason)
        problems = store.verify(proof)
        return JSONResponse(
            {
                "attestation": proof.attestation,
                "signed": proof.signature is not None,
                "commits": list(proof.commits),
                "manifests": list(proof.manifests),
                "blocks": len(proof.chunks),
                "bytes_freed": proof.bytes_freed,
                "broken_composites": list(proof.broken_composites),
                "revoked_grants": list(proof.revoked_grants),
                "verified": not problems,
                "problems": problems,
            },
            status_code=200 if not problems else 500,
        )

    routes = [
        Route("/health", health),
        Route("/tenants/{tenant}/commits", create_commit, methods=["POST"]),
        Route("/tenants/{tenant}/merges", create_merge, methods=["POST"]),
        Route("/tenants/{tenant}/commits/{spec}", delete_commit, methods=["DELETE"]),
        Route("/tenants/{tenant}/refs", refs),
        Route("/tenants/{tenant}/loras", loras),
        Route("/tenants/{tenant}/commits/{spec}", commit),
        Route("/tenants/{tenant}/commits/{spec}/adapter", adapter, name="adapter"),
        Route("/tenants/{tenant}/commits/{spec}/config", adapter_config),
    ]
    app = Starlette(
        routes=routes,
        exception_handlers={
            _NotFound: _error(404),
            _Forbidden: _error(403),
            _BadRequest: _error(400),
        },
    )
    app.state.store = store
    return app


class _NotFound(Exception):
    pass


class _Forbidden(Exception):
    pass


class _BadRequest(Exception):
    pass


def _error(status: int) -> Callable[[Request, Exception], Awaitable[Response]]:
    async def handler(request: Request, exc: Exception) -> Response:  # noqa: ARG001
        return JSONResponse({"error": str(exc)}, status_code=status)

    return handler


def _json_param(raw: str | None, what: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _BadRequest(f"{what} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _BadRequest(f"{what} must be a JSON object")
    return value


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        payload = json.loads(await request.body())
    except json.JSONDecodeError as exc:
        raise _BadRequest(f"body is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _BadRequest("body must be a JSON object")
    return payload


def _resolve(store: Store, tenant: str, spec: str) -> Any:
    try:
        return store.resolve(tenant, spec)
    except (LookupError, ValueError) as exc:
        raise _NotFound(str(exc)) from exc


def _immutable(commit: str) -> dict[str, str]:
    """A commit names exactly one set of tensors, so its responses never change."""
    return {"cache-control": "public, max-age=31536000, immutable", "etag": f'"{commit}"'}


def run(
    store: Store,
    host: str = "127.0.0.1",
    port: int = 8080,
    tokens: Tokens | None = None,
) -> None:  # pragma: no cover - exercised by hand, not in tests
    import uvicorn  # noqa: PLC0415

    uvicorn.run(create_app(store, tokens), host=host, port=port, log_level="info")


Handler = Callable[[Request], Awaitable[Response]]
