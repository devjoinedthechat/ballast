"""Content-addressed tensor storage on disk.

One file per tensor, named by its hash, under the tenant's directory. Reads are
memory-mapped so a checkout of a multi-gigabyte adapter costs no copies until
something touches the values.

This is the module a systems-language rewrite would replace. Nothing above it
touches the filesystem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ballast.tensors import DTYPES, dtype_name


class ChunkStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, tenant: str, digest: str) -> Path:
        return self.root / tenant / digest[:2] / digest

    def exists(self, tenant: str, digest: str) -> bool:
        return self.path(tenant, digest).exists()

    def put(self, tenant: str, digest: str, array: np.ndarray) -> bool:
        """Store a tensor. Returns False if it was already present."""
        target = self.path(tenant, digest)
        if target.exists():
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        with tmp.open("wb") as f:
            f.write(np.ascontiguousarray(array).tobytes())
        tmp.replace(target)
        return True

    def get(self, tenant: str, digest: str, dtype: str, shape: list[int]) -> np.ndarray:
        target = self.path(tenant, digest)
        if not target.exists():
            raise FileNotFoundError(f"chunk {digest[:12]} missing for tenant {tenant!r}")
        buffer = np.memmap(target, mode="r", dtype=DTYPES[dtype])
        return buffer.reshape(shape)

    def delete(self, tenant: str, digest: str) -> int:
        """Remove a chunk. Returns the bytes freed."""
        target = self.path(tenant, digest)
        if not target.exists():
            return 0
        size = target.stat().st_size
        target.unlink()
        parent = target.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        return size

    def delete_tenant(self, tenant: str) -> int:
        """Remove every chunk a tenant has. Returns the bytes freed."""
        root = self.root / tenant
        if not root.exists():
            return 0
        freed = 0
        for file in root.rglob("*"):
            if file.is_file():
                freed += file.stat().st_size
                file.unlink()
        for folder in sorted(root.rglob("*"), reverse=True):
            if folder.is_dir():
                folder.rmdir()
        root.rmdir()
        return freed


def describe(array: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": dtype_name(array.dtype),
        "shape": json.dumps(list(array.shape)),
        "nbytes": int(array.nbytes),
    }
