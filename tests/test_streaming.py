"""Reading and writing one tensor at a time.

A delta applied to a 70B model must not cost 70B of memory. Nothing here can
allocate that much to prove it, so the tests check the properties that make it
true: the file a stream writes is byte-identical to the one built in memory, the
producer is called once per tensor and its result is not retained, and a leaf
checkout hands back memory maps rather than copies.
"""

import gc
import weakref

import numpy as np
import pytest
from conftest import adapter, scaled

from ballast import Store
from ballast import tensors as st


def test_a_streamed_file_is_byte_identical_to_one_built_in_memory(tmp_path, rng):
    tensors = adapter(rng)
    st.save(tmp_path / "whole.safetensors", tensors, {"format": "pt"})
    st.save_stream(
        tmp_path / "streamed.safetensors", st.specs_of(tensors), tensors.__getitem__, {"format": "pt"}
    )
    assert (tmp_path / "whole.safetensors").read_bytes() == (tmp_path / "streamed.safetensors").read_bytes()


def test_the_producer_is_called_once_per_tensor_in_name_order(tmp_path, rng):
    tensors = adapter(rng)
    seen = []

    def produce(name):
        seen.append(name)
        return tensors[name]

    st.save_stream(tmp_path / "t.safetensors", st.specs_of(tensors), produce)
    assert seen == sorted(tensors)


def test_nothing_holds_a_reference_to_a_tensor_once_it_is_written(tmp_path):
    """The property that bounds memory: each tensor is released as it lands."""
    alive: list[weakref.ref] = []

    def produce(name):
        array = np.zeros((64, 64), dtype=np.float32)
        alive.append(weakref.ref(array))
        return array

    specs = {f"t{i}": ("F32", (64, 64)) for i in range(8)}
    st.save_stream(tmp_path / "t.safetensors", specs, produce)
    gc.collect()
    assert all(ref() is None for ref in alive)


def test_a_producer_that_breaks_its_promise_is_caught(tmp_path):
    """A wrong shape would write a file whose header lies about its contents."""
    specs = {"a": ("F32", (4, 4))}
    with pytest.raises(ValueError, match="was declared"):
        st.save_stream(tmp_path / "t.safetensors", specs, lambda _: np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="was declared"):
        st.save_stream(tmp_path / "t.safetensors", specs, lambda _: np.zeros((4, 4), dtype=np.float16))


def test_checkout_stream_yields_the_same_tensors_as_checkout(store, rng):
    tensors = adapter(rng)
    c = store.commit("t", tensors, message="v1")
    whole = store.checkout("t", c.id)
    streamed = dict(store.checkout_stream("t", c.id))
    assert sorted(streamed) == sorted(whole)
    for name, array in whole.items():
        assert np.array_equal(streamed[name].view(np.uint8), array.view(np.uint8))


def test_a_raw_leaf_streams_memory_maps_rather_than_copies(tmp_path, rng):
    raw = Store(tmp_path / "raw", compression="raw")
    raw.commit("t", {"w": rng.standard_normal((64, 64)).astype(np.float32)}, message="v1")
    (_, array) = next(iter(raw.checkout_stream("t", "main")))
    assert isinstance(array.base, np.memmap)


def test_specs_describe_a_leaf_without_reading_any_tensor(store, rng):
    tensors = adapter(rng)
    c = store.commit("t", tensors, message="v1")
    specs = store.specs("t", c.id)
    assert sorted(specs) == sorted(tensors)
    for name, array in tensors.items():
        assert specs[name] == (st.dtype_name(array.dtype), tuple(array.shape))


def test_a_view_streams_from_its_cache_once_it_has_one(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="a", base_model="b")
    c2 = store.commit("t", scaled(a, "layers.0.lora_A.weight", 2.0), message="b", base_model="b")
    m = store.merge("t", "linear", [(c1.id, 0.5), (c2.id, 0.5)], message="avg")

    first = dict(store.checkout_stream("t", m.id))
    assert store._cache_path("t", m.manifest_id).exists()
    second = dict(store.checkout_stream("t", m.id))
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert sorted(first) == sorted(a)


def test_specs_and_the_stream_agree_for_a_view(store, rng):
    c1 = store.commit("t", adapter(rng), message="a", base_model="b")
    c2 = store.commit("t", adapter(rng), message="b", base_model="b")
    m = store.merge("t", "linear", [(c1.id, 0.5), (c2.id, 0.5)], message="avg")
    specs = store.specs("t", m.id)
    for name, array in store.checkout_stream("t", m.id):
        assert specs[name] == (st.dtype_name(array.dtype), tuple(array.shape))
