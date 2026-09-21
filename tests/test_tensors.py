import ml_dtypes
import numpy as np
import pytest

from ballast import tensors as st
from ballast.hashing import tensor_hash


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float16, ml_dtypes.bfloat16, ml_dtypes.float8_e4m3fn, np.int8]
)
def test_safetensors_round_trip_preserves_every_dtype(tmp_path, rng, dtype):
    arrays = {"a": rng.standard_normal((3, 5)).astype(dtype), "b": np.arange(7).astype(dtype)}
    st.save(tmp_path / "t.safetensors", arrays, {"format": "pt"})
    loaded, meta = st.load(tmp_path / "t.safetensors")
    assert meta == {"format": "pt"}
    for name, array in arrays.items():
        assert loaded[name].dtype == array.dtype
        assert loaded[name].shape == array.shape
        assert np.array_equal(loaded[name].view(np.uint8), array.view(np.uint8))


def test_header_is_padded_to_eight_bytes(tmp_path):
    st.save(tmp_path / "t.safetensors", {"x": np.zeros(1, dtype=np.float32)})
    raw = (tmp_path / "t.safetensors").read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    assert header_len % 8 == 0


def test_hash_covers_dtype_and_shape_not_just_bytes():
    bits = np.arange(8, dtype=np.uint8)
    as_u8 = bits.copy()
    as_i8 = bits.view(np.int8)
    reshaped = bits.reshape(2, 4)
    assert tensor_hash(as_u8) != tensor_hash(as_i8)
    assert tensor_hash(as_u8) != tensor_hash(reshaped)
    assert tensor_hash(as_u8) == tensor_hash(bits.copy())


def test_bfloat16_hashes_without_buffer_protocol(rng):
    a = rng.standard_normal((4, 4)).astype(ml_dtypes.bfloat16)
    assert len(tensor_hash(a)) == 64
