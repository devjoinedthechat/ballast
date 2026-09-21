"""Hand a stored delta to a serving runtime.

vLLM loads a LoRA from a directory and identifies it by a name and a positive
integer. Both have to be stable: the same delta served from two processes, or
from the same process after a restart, must get the same id, or the server's
LoRA cache holds two copies of one adapter under different numbers.

A commit id is already a stable name for exactly one set of tensors, so the
integer is derived from it rather than handed out by a counter. Materialising is
idempotent for the same reason: the export directory is named after the commit,
so a delta that is already on disk is not written again.

What is tested here is the hand-off — the directory layout, the stable id, the
idempotence and the provenance. Constructing vLLM's own request object needs
vLLM installed, and it is not exercised in CI.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ballast import peft as peft_io
from ballast.store import Store

MAX_INT_ID = 2**31 - 1


@dataclass(frozen=True)
class LoRAExport:
    """A materialised delta and the identity a server should know it by."""

    name: str
    int_id: int
    path: Path
    tenant: str
    commit: str
    manifest: str
    base_model: str | None
    reused: bool
    """Whether the directory was already there from an earlier export."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "int_id": self.int_id,
            "path": str(self.path),
            "tenant": self.tenant,
            "commit": self.commit,
            "manifest": self.manifest,
            "base_model": self.base_model,
        }


def int_id(commit: str) -> int:
    """A stable positive integer for a commit, for servers that index by number."""
    return int(commit[:15], 16) % MAX_INT_ID + 1


def export(
    store: Store,
    tenant: str,
    spec: str,
    root: Path | str,
    *,
    name: str | None = None,
    refresh: bool = False,
) -> LoRAExport:
    """Materialise a commit into `root`, ready for a server to load.

    The directory is named after the commit, so calling this twice costs one
    directory listing the second time. `refresh` rewrites it anyway, which is
    only useful if something outside has edited it.
    """
    commit = store.resolve(tenant, spec)
    manifest = store.manifest(tenant, commit.manifest_id)
    # Checked even when the directory is already there. An export is a copy
    # outside the store, and nothing stops it outliving what it was copied from:
    # without this, a delta that was forgotten, or a grant that was revoked,
    # would go on being served from the last export of it. Resolving the graph
    # reads no tensors, so the check costs a few queries.
    store.specs(tenant, commit.id)
    path = Path(root) / tenant / commit.id
    reused = path.joinpath(peft_io.WEIGHTS).exists() and not refresh

    if not reused:
        provenance = {
            "tenant": tenant,
            "commit": commit.id,
            "manifest": manifest.id,
            "kind": manifest.kind,
            "base_model": manifest.base_model,
            "message": commit.message,
            "metadata": commit.metadata,
        }
        if manifest.kind == "composite":
            provenance["recipe"] = {
                **manifest.config,
                "inputs": [
                    {"tenant": t, "manifest": m, "weight": w} for t, m, w in store.inputs(tenant, manifest.id)
                ],
            }
        config = manifest.config if manifest.kind == "leaf" else {}
        peft_io.export(path, store.checkout(tenant, spec), config, provenance)

    return LoRAExport(
        name=name or f"{tenant}-{commit.id[:12]}",
        int_id=int_id(commit.id),
        path=path,
        tenant=tenant,
        commit=commit.id,
        manifest=manifest.id,
        base_model=manifest.base_model,
        reused=reused,
    )


class VLLMRuntime:
    """Tell a running vLLM server which adapters to hold.

    vLLM's OpenAI server can load and unload LoRAs while it is up, when it is
    started with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`. This drives those
    endpoints from a store, so a fleet converges on what the store says rather
    than on whatever it was started with.

    `sync` is a reconciliation, not a push: it loads what is missing, unloads
    what the store no longer has, and leaves the rest alone. Running it twice
    changes nothing the second time, which is what makes it safe to run on a
    timer.

    Tested against a stub transport rather than a live server. vLLM is
    CUDA-first and is not installed in CI, so what is checked is the requests
    this makes and how it reacts to the replies, not vLLM's behaviour.
    """

    def __init__(self, base_url: str, client: Any = None, timeout: float = 30.0) -> None:
        if client is None:
            import httpx  # noqa: PLC0415

            client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.client = client

    def loaded(self) -> set[str]:
        """The model names the server currently serves, adapters included."""
        response = self.client.get("/v1/models")
        response.raise_for_status()
        return {entry["id"] for entry in response.json().get("data", [])}

    def load(self, item: LoRAExport) -> bool:
        """Ask the server to hold this adapter. False if it already did."""
        response = self.client.post(
            "/v1/load_lora_adapter",
            json={"lora_name": item.name, "lora_path": str(item.path), "lora_int_id": item.int_id},
        )
        if response.status_code == 400 and "already" in response.text.lower():
            return False
        response.raise_for_status()
        return True

    def unload(self, name: str) -> None:
        response = self.client.post("/v1/unload_lora_adapter", json={"lora_name": name})
        if response.status_code == 404:
            return
        response.raise_for_status()

    def sync(
        self,
        store: Store,
        tenant: str,
        root: Path | str,
        specs: Sequence[str] | None = None,
        prune: bool = True,
    ) -> dict[str, list[str]]:
        """Converge the server on what the store holds for this tenant.

        Returns what changed, so a caller running this on a timer can log the
        turns where something did and stay quiet otherwise.
        """
        from ballast.store import BrokenView  # noqa: PLC0415

        wanted: dict[str, LoRAExport] = {}
        skipped: list[str] = []
        for spec in specs or sorted(store.refs(tenant)):
            try:
                # Named for the ref and the commit it points at: the ref makes
                # the name readable, the commit makes it change when the content
                # does, which is what lets a reconciliation notice.
                commit = store.resolve(tenant, spec)
                item = export(store, tenant, spec, root, name=f"{tenant}-{spec}-{commit.id[:8]}")
            except BrokenView:
                skipped.append(spec)
                continue
            wanted[item.name] = item

        present = self.loaded()
        # Only names this tenant owns are candidates for unloading: the server
        # may be serving other tenants, and its base model is in this list too.
        ours = {name for name in present if name.startswith(f"{tenant}-")}

        loaded = [name for name, item in sorted(wanted.items()) if name not in present and self.load(item)]
        unloaded: list[str] = []
        if prune:
            for name in sorted(ours - set(wanted)):
                self.unload(name)
                unloaded.append(name)
        return {
            "loaded": loaded,
            "unloaded": unloaded,
            "unchanged": sorted(set(wanted) & present),
            "skipped": skipped,
        }


def lora_request(item: LoRAExport) -> Any:
    """vLLM's `LoRARequest` for an export.

    Imports vLLM when called, so the rest of the module works without it.
    """
    from vllm.lora.request import LoRARequest  # noqa: PLC0415

    return LoRARequest(lora_name=item.name, lora_int_id=item.int_id, lora_path=str(item.path))


def manifest_json(exports: list[LoRAExport], path: Path | str) -> Path:
    """Write a list of exports for a server to load at start-up.

    A plain file rather than a live connection, because that is what most
    serving deployments actually take: a directory of adapters and a list.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([e.as_dict() for e in exports], indent=2, sort_keys=True))
    return path
