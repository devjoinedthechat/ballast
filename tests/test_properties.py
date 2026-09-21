"""Properties that must hold for every input, not just the ones in the fixtures."""

import json
import struct

import ml_dtypes
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as hs
from hypothesis.extra import numpy as hnp

from ballast import merge as merging
from ballast import tensors as st
from ballast.backends import LocalBackend
from ballast.chunks import ChunkStore
from ballast.hashing import tensor_hash
from ballast.tensors import MalformedSafetensors

DTYPES = [np.float32, np.float16, np.int8, np.uint8, ml_dtypes.bfloat16]


def arrays(max_side=12):
    return hs.sampled_from(DTYPES).flatmap(
        lambda dt: hnp.arrays(
            dtype=np.dtype(dt),
            shape=hnp.array_shapes(min_dims=1, max_dims=3, min_side=0, max_side=max_side),
            elements=(
                hs.integers(0, 100)
                if np.dtype(dt).kind == "u"
                else hs.integers(-100, 100)
                if np.dtype(dt).kind == "i"
                else hs.floats(-1e3, 1e3, width=16)
            ),
        )
    )


@settings(max_examples=60, deadline=None)
@given(array=arrays(), block_size=hs.integers(64, 3000))
def test_any_tensor_round_trips_through_any_block_size(tmp_path_factory, array, block_size):
    root = tmp_path_factory.mktemp("blocks")
    store = ChunkStore(LocalBackend(root), block_size=block_size, compression="zstd")
    record = store.put_tensor("t", array)
    blocks = [(b.digest, b.encoding if b.new else "raw", b.nbytes) for b in record.blocks]
    # Re-read encodings from what was written, as the store would from its index.
    blocks = [(d, "zstd" if len(LocalBackend(root).get("t", d)) < n else "raw", n) for d, _, n in blocks]
    out = store.get_tensor("t", record.dtype, record.shape, blocks)
    assert out.dtype == array.dtype
    assert out.shape == array.shape
    assert np.array_equal(out.view(np.uint8), np.ascontiguousarray(array).view(np.uint8))


@settings(max_examples=40, deadline=None)
@given(array=arrays())
def test_safetensors_round_trips_any_array(tmp_path_factory, array):
    path = tmp_path_factory.mktemp("st") / "t.safetensors"
    st.save(path, {"a": array})
    loaded, _ = st.load(path)
    assert loaded["a"].dtype == array.dtype
    assert loaded["a"].shape == array.shape
    assert np.array_equal(loaded["a"].view(np.uint8), np.ascontiguousarray(array).view(np.uint8))


@settings(max_examples=40, deadline=None)
@given(array=arrays())
def test_hash_is_deterministic_and_changes_with_content(array):
    h = tensor_hash(array)
    assert h == tensor_hash(array.copy())
    if array.size and array.dtype.kind in "iu":
        bumped = array.copy()
        bumped.flat[0] = bumped.flat[0] + 1 if bumped.flat[0] < 100 else bumped.flat[0] - 1
        assert tensor_hash(bumped) != h


@settings(max_examples=40, deadline=None)
@given(
    a=hnp.arrays(np.float32, (8,), elements=hs.floats(-10, 10, width=32)),
    b=hnp.arrays(np.float32, (8,), elements=hs.floats(-10, 10, width=32)),
    wa=hs.floats(-2, 2, width=32),
    wb=hs.floats(-2, 2, width=32),
)
def test_linear_is_exactly_the_weighted_sum(a, b, wa, wb):
    out = merging.linear([{"w": a}, {"w": b}], [wa, wb])["w"]
    assert np.allclose(out, wa * a + wb * b, atol=1e-4)


@settings(max_examples=30, deadline=None)
@given(a=hnp.arrays(np.float32, (16,), elements=hs.floats(-10, 10, width=32)))
def test_ties_of_one_input_at_full_density_is_the_input(a):
    out = merging.ties([{"w": a}], [1.0], density=1.0)["w"]
    assert np.allclose(out, a)


def _write(path, header: dict, data: bytes) -> None:
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


@pytest.mark.parametrize(
    ("header", "data", "reason"),
    [
        ({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 16]}}, b"\0" * 8, "outside"),
        ({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, b"\0" * 8, "needs 8"),
        ({"a": {"dtype": "F99", "shape": [2], "data_offsets": [0, 8]}}, b"\0" * 8, "unknown dtype"),
        ({"a": {"dtype": "F32", "shape": [-2], "data_offsets": [0, 8]}}, b"\0" * 8, "invalid shape"),
        ({"a": {"dtype": "F32", "shape": [2], "data_offsets": [4, 0]}}, b"\0" * 8, "outside"),
        (
            {
                "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},
            },
            b"\0" * 12,
            "overlap",
        ),
        ({"a": "not an object"}, b"", "not an object"),
    ],
)
def test_malformed_headers_are_refused_with_a_reason(tmp_path, header, data, reason):
    path = tmp_path / "bad.safetensors"
    _write(path, header, data)
    with pytest.raises(MalformedSafetensors, match=reason):
        st.load(path)


def test_an_absurd_header_length_is_refused(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    with pytest.raises(MalformedSafetensors, match="impossible"):
        st.load(path)


def test_a_truncated_file_is_refused(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"\x01\x02")
    with pytest.raises(MalformedSafetensors, match="shorter"):
        st.load(path)
