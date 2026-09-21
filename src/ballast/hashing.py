"""Content addresses.

A chunk's address covers dtype and shape as well as bytes, so a float16 tensor
and a bfloat16 tensor with identical bit patterns do not collide, and neither do
two views of the same buffer with different shapes.
"""

from __future__ import annotations

import json
from typing import Any

import blake3
import numpy as np

from ballast.tensors import dtype_name


def tensor_hash(array: np.ndarray) -> str:
    hasher = blake3.blake3()
    hasher.update(dtype_name(array.dtype).encode())
    hasher.update(json.dumps(list(array.shape)).encode())
    # bfloat16 and float8 have no buffer-protocol code, so hash the bytes
    # through a uint8 view rather than the array's own buffer.
    hasher.update(np.ascontiguousarray(array).view(np.uint8).reshape(-1).data)
    return hasher.hexdigest()


def object_hash(payload: Any) -> str:
    """Address for metadata: canonical JSON, so equal content means equal id."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return blake3.blake3(encoded).hexdigest()
