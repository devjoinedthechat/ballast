"""Read and write the safetensors format directly.

The reference library's numpy path refuses bfloat16, and most adapters are
bfloat16. The format itself is small — an 8-byte header length, a JSON header, a
byte buffer — so reading it here costs a page of code and buys a store that runs
without torch. `ml_dtypes` supplies the numpy bfloat16 and float8 types.

The reader takes files from users, so the header is validated before any byte
is interpreted: every offset in range, every span the size its dtype and shape
say, no two spans overlapping. A malformed file raises `MalformedSafetensors`
with the tensor and the reason, never a numpy error from deep inside.
"""

from __future__ import annotations

import itertools
import json
import math
import struct
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

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

MAX_HEADER_BYTES = 100 * 1024 * 1024


class MalformedSafetensors(ValueError):
    """The file does not describe its own bytes correctly."""


def dtype_name(dtype: np.dtype[Any]) -> str:
    try:
        return NAMES[np.dtype(dtype)]
    except KeyError:
        raise TypeError(f"{dtype} has no safetensors encoding") from None


def _validate(header: dict[str, Any], data_len: int, path: Path) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(header, dict):
        raise MalformedSafetensors(f"{path}: header is not an object")
    spans: list[tuple[int, int, str]] = []
    entries: list[tuple[str, dict[str, Any]]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise MalformedSafetensors(f"{path}: {name!r} is not an object")
        dtype = entry.get("dtype")
        if dtype not in DTYPES:
            raise MalformedSafetensors(f"{path}: {name!r} has unknown dtype {dtype!r}")
        shape = entry.get("shape")
        if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
            raise MalformedSafetensors(f"{path}: {name!r} has invalid shape {shape!r}")
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(o, int) for o in offsets):
            raise MalformedSafetensors(f"{path}: {name!r} has invalid data_offsets {offsets!r}")
        start, end = offsets
        if not 0 <= start <= end <= data_len:
            raise MalformedSafetensors(
                f"{path}: {name!r} offsets [{start}, {end}) fall outside the {data_len}-byte buffer"
            )
        expected = math.prod(shape) * DTYPES[dtype].itemsize
        if end - start != expected:
            raise MalformedSafetensors(
                f"{path}: {name!r} spans {end - start} bytes but {dtype} {shape} needs {expected}"
            )
        spans.append((start, end, name))
        entries.append((name, entry))
    spans.sort()
    for (_, e0, n0), (s1, _, n1) in itertools.pairwise(spans):
        if s1 < e0:
            raise MalformedSafetensors(f"{path}: {n0!r} and {n1!r} overlap")
    return entries


def read_header(path: Path | str) -> tuple[dict[str, Any], int]:
    """The parsed header and the byte offset where data begins."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise MalformedSafetensors(f"{path}: shorter than its own length prefix")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len <= 0 or header_len > MAX_HEADER_BYTES or 8 + header_len > size:
            raise MalformedSafetensors(
                f"{path}: header length {header_len} is impossible for a {size}-byte file"
            )
        try:
            header = json.loads(f.read(header_len))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MalformedSafetensors(f"{path}: header is not valid JSON: {exc}") from None
    return header, 8 + header_len


def load(path: Path | str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Tensors and the file's metadata block. Tensors are memory-mapped views."""
    path = Path(path)
    header, data_start = read_header(path)
    data_len = path.stat().st_size - data_start
    entries = _validate(header, data_len, path)
    metadata = header.get("__metadata__") or {}
    if not isinstance(metadata, dict):
        raise MalformedSafetensors(f"{path}: __metadata__ is not an object")

    data = np.memmap(path, mode="r", offset=data_start) if data_len else np.empty(0, dtype=np.uint8)
    tensors: dict[str, np.ndarray] = {}
    for name, entry in sorted(entries):
        start, end = entry["data_offsets"]
        tensors[name] = np.frombuffer(data[start:end], dtype=DTYPES[entry["dtype"]]).reshape(entry["shape"])
    return tensors, {str(k): str(v) for k, v in metadata.items()}


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
            np.ascontiguousarray(tensors[name]).view(np.uint8).reshape(-1).tofile(f)


def save_stream(
    target: Path | str | BinaryIO,
    specs: Mapping[str, tuple[str, Sequence[int]]],
    produce: Callable[[str], np.ndarray],
    metadata: dict[str, str] | None = None,
) -> None:
    """Write a file one tensor at a time.

    The format puts every offset in the header, so the shapes and dtypes have to
    be known before anything is written — but the values do not. `specs` gives
    the shape of the file and `produce` supplies each tensor as its turn comes,
    so peak memory is one tensor rather than the whole model. Writing a 70B
    model otherwise means holding a 70B model.

    Each tensor is checked against its declared spec as it arrives. A producer
    that returns the wrong shape would otherwise write a file whose header lies.

    `target` is a path or an open binary file, so the same code writes a shard
    to disk and an adapter to a response body.
    """
    header: dict[str, Any] = {}
    offset = 0
    ordered = sorted(specs)
    for name in ordered:
        dtype, shape = specs[name]
        if dtype not in DTYPES:
            raise ValueError(f"{name!r} has unknown dtype {dtype!r}")
        nbytes = math.prod(shape) * DTYPES[dtype].itemsize
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    if metadata:
        header["__metadata__"] = metadata

    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)

    def write(f: BinaryIO) -> None:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for name in ordered:
            dtype, shape = specs[name]
            array = np.ascontiguousarray(produce(name))
            if dtype_name(array.dtype) != dtype or tuple(array.shape) != tuple(shape):
                raise ValueError(
                    f"{name!r} was declared {dtype} {list(shape)} but produced "
                    f"{dtype_name(array.dtype)} {list(array.shape)}"
                )
            f.write(array.view(np.uint8).reshape(-1).tobytes())
            del array

    if isinstance(target, (str, Path)):
        with Path(target).open("wb") as f:
            write(f)
    else:
        write(target)


def specs_of(tensors: dict[str, np.ndarray]) -> dict[str, tuple[str, tuple[int, ...]]]:
    return {name: (dtype_name(a.dtype), tuple(a.shape)) for name, a in tensors.items()}


def load_dir(directory: Path | str) -> dict[str, np.ndarray]:
    """Every tensor from every safetensors shard in a model directory."""
    directory = Path(directory)
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files under {directory}")
    out: dict[str, np.ndarray] = {}
    for file in files:
        tensors, _ = load(file)
        for name, array in tensors.items():
            if name in out:
                raise MalformedSafetensors(f"{directory}: {name!r} appears in more than one shard")
            out[name] = array
    return out


def to_f32(array: np.ndarray) -> np.ndarray:
    return array.astype(np.float32)
