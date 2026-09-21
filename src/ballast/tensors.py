"""Read and write the safetensors format directly.

The reference library's numpy path refuses bfloat16, and most adapters are
bfloat16. The format itself is small — an 8-byte header length, a JSON header, a
byte buffer — so reading it here costs twenty lines and buys a store that runs
without torch. `ml_dtypes` supplies the numpy bfloat16 and float8 types.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

DTYPES: dict[str, np.dtype[Any]] = {
    "F64": np.dtype(np.float64),
    "F32": np.dtype(np.float32),
    "F16": np.dtype(np.float16),
    "BF16": np.dtype(ml_dtypes.bfloat16),
    "F8_E4M3": np.dtype(ml_dtypes.float8_e4m3fn),
    "F8_E5M2": np.dtype(ml_dtypes.float8_e5m2),
    "I64": np.dtype(np.int64),
    "I32": np.dtype(np.int32),
    "I16": np.dtype(np.int16),
    "I8": np.dtype(np.int8),
    "U8": np.dtype(np.uint8),
    "BOOL": np.dtype(np.bool_),
}
NAMES: dict[np.dtype[Any], str] = {v: k for k, v in DTYPES.items()}


def dtype_name(dtype: np.dtype[Any]) -> str:
    try:
        return NAMES[np.dtype(dtype)]
    except KeyError:
        raise TypeError(f"{dtype} has no safetensors encoding") from None


def load(path: Path | str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Tensors and the file's metadata block."""
    path = Path(path)
    with path.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    data = np.memmap(path, mode="r", offset=8 + header_len)

    metadata = header.pop("__metadata__", {}) or {}
    tensors: dict[str, np.ndarray] = {}
    for name in sorted(header):
        entry = header[name]
        start, end = entry["data_offsets"]
        dtype = DTYPES[entry["dtype"]]
        buffer = data[start:end]
        tensors[name] = np.frombuffer(buffer, dtype=dtype).reshape(entry["shape"])
    return tensors, metadata


def save(path: Path | str, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
    """Write in the layout other loaders expect: header, then tensors in name order."""
    header: dict[str, Any] = {}
    offset = 0
    ordered = sorted(tensors)
    for name in ordered:
        array = np.ascontiguousarray(tensors[name])
        header[name] = {
            "dtype": dtype_name(array.dtype),
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes
    if metadata:
        header["__metadata__"] = metadata

    encoded = json.dumps(header, separators=(",", ":")).encode()
    # Pad the header to 8 bytes so the data buffer stays aligned.
    encoded += b" " * (-len(encoded) % 8)
    with Path(path).open("wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for name in ordered:
            f.write(np.ascontiguousarray(tensors[name]).tobytes())


def to_f32(array: np.ndarray) -> np.ndarray:
    return array.astype(np.float32)
