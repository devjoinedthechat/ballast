"""Tensors as content-addressed blocks.

A tensor's bytes are split into fixed-size blocks, each hashed and stored once.
Fixed-size is the right cut for weights: a changed value stays where it was, so
nothing shifts, and a one-value edit to a multi-gigabyte tensor re-stores one
block rather than the tensor. Content-defined chunking exists to handle
insertions, and weights have none.

A block's address is a hash over its raw bytes, independent of how it is
encoded on disk, so the same block deduplicates whether or not it was
compressed, and a store can change compression without losing history.

Blocks are compressed with zstd unless that would not help, in which case they
are kept raw. A single raw block read from a local backend is a memory map with
no copy; anything else is assembled into one buffer.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import blake3
import numpy as np
import zstandard

from ballast.backends import Backend, LocalBackend
from ballast.tensors import DTYPES, dtype_name

DEFAULT_BLOCK_SIZE = 8 * 1024 * 1024
ZSTD_LEVEL = 3


@dataclass(frozen=True)
class Block:
    digest: str
    nbytes: int
    stored_bytes: int
    encoding: str
    new: bool


@dataclass(frozen=True)
class TensorRecord:
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    blocks: tuple[Block, ...]

    @property
    def digests(self) -> tuple[str, ...]:
        return tuple(b.digest for b in self.blocks)

    def identity(self) -> dict[str, Any]:
        """What the manifest hash covers for this tensor."""
        return {"dtype": self.dtype, "shape": list(self.shape), "blocks": list(self.digests)}


def block_digest(data: bytes | memoryview) -> str:
    return blake3.blake3(
        bytes(data) if isinstance(data, memoryview) and not data.contiguous else data
    ).hexdigest()


class ChunkStore:
    def __init__(
        self, backend: Backend, block_size: int = DEFAULT_BLOCK_SIZE, compression: str = "zstd"
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if compression not in ("zstd", "raw"):
            raise ValueError(f"unknown compression {compression!r}; use 'zstd' or 'raw'")
        self.backend = backend
        self.block_size = block_size
        self.compression = compression
        self._compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
        self._decompressor = zstandard.ZstdDecompressor()

    # -- blocks -------------------------------------------------------------

    def put_block(self, tenant: str, data: memoryview, known: Callable[[str], bool] | None = None) -> Block:
        digest = block_digest(data)
        # The caller's index is the authority on what exists; asking the backend
        # would cost a round trip per block on a remote store.
        if known is not None and known(digest):
            return Block(digest, len(data), 0, "known", False)
        if known is None and self.backend.exists(tenant, digest):
            return Block(digest, len(data), 0, "known", False)
        encoding = "raw"
        payload: bytes | memoryview = data
        if self.compression == "zstd" and len(data) > 0:
            compressed = self._compressor.compress(bytes(data))
            if len(compressed) < len(data):
                payload, encoding = compressed, "zstd"
        self.backend.put(tenant, digest, payload)
        return Block(digest, len(data), len(payload), encoding, True)

    def get_block(self, tenant: str, digest: str, encoding: str, nbytes: int) -> bytes | memoryview:
        raw = self.backend.get(tenant, digest)
        if encoding == "zstd":
            return self._decompressor.decompress(bytes(raw), max_output_size=nbytes)
        return raw

    def verify_block(self, tenant: str, digest: str, encoding: str, nbytes: int) -> bool:
        try:
            data = self.get_block(tenant, digest, encoding, nbytes)
        except (FileNotFoundError, zstandard.ZstdError):
            return False
        return len(data) == nbytes and block_digest(data) == digest

    # -- tensors ------------------------------------------------------------

    def put_tensor(
        self, tenant: str, array: np.ndarray, known: Callable[[str], bool] | None = None
    ) -> TensorRecord:
        flat = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
        view = memoryview(flat)  # type: ignore[arg-type]
        blocks = tuple(
            self.put_block(tenant, view[start : start + self.block_size], known)
            for start in range(0, len(view), self.block_size)
        )
        return TensorRecord(
            dtype_name(array.dtype), tuple(int(d) for d in array.shape), int(flat.nbytes), blocks
        )

    def get_tensor(
        self,
        tenant: str,
        dtype: str,
        shape: Sequence[int],
        blocks: Sequence[tuple[str, str, int]],
    ) -> np.ndarray:
        """Reassemble a tensor from (digest, encoding, nbytes) blocks."""
        np_dtype = DTYPES[dtype]
        total = sum(n for _, _, n in blocks)
        if total == 0:
            return np.empty(tuple(shape), dtype=np_dtype)

        if len(blocks) == 1 and blocks[0][1] == "raw" and isinstance(self.backend, LocalBackend):
            digest, _, nbytes = blocks[0]
            mapped = self.backend.mmap(tenant, digest)
            if len(mapped) != nbytes:
                raise ValueError(f"block {digest[:12]} is {len(mapped)} bytes, expected {nbytes}")
            # A view keeps the memmap type, so the caller can see it is zero-copy.
            view: np.ndarray = mapped.view(np_dtype).reshape(tuple(shape))
            return view

        out = np.empty(total, dtype=np.uint8)
        offset = 0
        for digest, encoding, nbytes in blocks:
            data = self.get_block(tenant, digest, encoding, nbytes)
            if len(data) != nbytes:
                raise ValueError(f"block {digest[:12]} is {len(data)} bytes, expected {nbytes}")
            out[offset : offset + nbytes] = np.frombuffer(data, dtype=np.uint8)
            offset += nbytes
        return out.view(np_dtype).reshape(tuple(shape))

    # -- delegation ---------------------------------------------------------

    def exists(self, tenant: str, digest: str) -> bool:
        return self.backend.exists(tenant, digest)

    def delete(self, tenant: str, digest: str) -> int:
        return self.backend.delete(tenant, digest)

    def delete_tenant(self, tenant: str) -> int:
        return self.backend.delete_tenant(tenant)

    def path(self, tenant: str, digest: str) -> Any:
        if isinstance(self.backend, LocalBackend):
            return self.backend.path(tenant, digest)
        raise TypeError(f"{self.backend.describe()} has no filesystem paths")


def describe(array: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": dtype_name(array.dtype),
        "shape": json.dumps(list(array.shape)),
        "nbytes": int(array.nbytes),
    }
