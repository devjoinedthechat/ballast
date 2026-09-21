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
