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

Reads only. Writing is a control-plane job and wants authentication with more to
say than one token, so it is not here. Tenant isolation is enforced on every
route: a token is bound to the tenants it may read, and a request for another
tenant is a 404 rather than a 403, because a 403 confirms the tenant exists.
"""

from __future__ import annotations

import io
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from ballast import peft as peft_io
from ballast import serving
from ballast import tensors as st
from ballast.store import BrokenView, Store

ADAPTER_MEDIA_TYPE = "application/octet-stream"


class Tokens:
    """Which tenants a bearer token may read.

    An empty table means the server is open, which is right behind a gateway
    that has already decided who is calling and wrong anywhere else. The CLI
    says so when it starts without one.
    """

    def __init__(self, table: Mapping[str, list[str]] | None = None) -> None:
        self.table = {token: set(tenants) for token, tenants in (table or {}).items()}

    @property
    def open(self) -> bool:
        return not self.table

    def may_read(self, token: str | None, tenant: str) -> bool:
        if self.open:
            return True
        if token is None:
            return False
        allowed = self.table.get(token)
        return allowed is not None and (tenant in allowed or "*" in allowed)

    @classmethod
    def from_file(cls, path: str) -> Tokens:
        """A JSON object of token -> list of tenants, or ["*"] for all."""
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

    routes = [
        Route("/health", health),
        Route("/tenants/{tenant}/refs", refs),
        Route("/tenants/{tenant}/loras", loras),
        Route("/tenants/{tenant}/commits/{spec}", commit),
        Route("/tenants/{tenant}/commits/{spec}/adapter", adapter, name="adapter"),
        Route("/tenants/{tenant}/commits/{spec}/config", adapter_config),
    ]
    app = Starlette(routes=routes, exception_handlers={_NotFound: _not_found})
    app.state.store = store
    return app


class _NotFound(Exception):
    pass


async def _not_found(request: Request, exc: Exception) -> Response:  # noqa: ARG001
    return JSONResponse({"error": str(exc)}, status_code=404)


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
