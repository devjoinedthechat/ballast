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


def test_an_uncached_view_merges_one_tensor_at_a_time(store, rng):
    """The property that bounds a merge: inputs are read per tensor, not whole.

    Counting reads is how this is observable — a resolution that held its inputs
    would read each one once and then serve every tensor from memory.
    """
    a = adapter(rng)
    c1 = store.commit("t", a, message="a", base_model="b")
    c2 = store.commit("t", scaled(a, "layers.0.lora_A.weight", 2.0), message="b", base_model="b")
    m = store.merge("t", "linear", [(c1.id, 0.5), (c2.id, 0.5)], message="avg")

    reads: list[str] = []
    original = store._read_tensor

    def counted(tenant, manifest_id, name):
        reads.append(name)
        return original(tenant, manifest_id, name)

    store._read_tensor = counted
    try:
        out = dict(store.checkout_stream("t", m.id))
    finally:
        store._read_tensor = original

    # One read per tensor per input, not one read of each input.
    assert len(reads) == 2 * len(a)
    assert sorted(out) == sorted(a)


def test_streaming_a_view_fills_its_cache(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="a", base_model="b")
    c2 = store.commit("t", scaled(a, "layers.1.lora_B.weight", 3.0), message="b", base_model="b")
    m = store.merge("t", "linear", [(c1.id, 0.5), (c2.id, 0.5)], message="avg")

    streamed = dict(store.checkout_stream("t", m.id))
    assert store._cache_path("t", m.manifest_id).exists()

    cached = dict(store.checkout_stream("t", m.id))
    whole = store.checkout("t", m.id)
    for name, array in streamed.items():
        assert np.array_equal(cached[name], array)
        assert np.array_equal(whole[name], array)


def test_a_half_consumed_stream_leaves_no_cache_behind(store, rng):
    """A partial file would look like a resolved view and serve wrong tensors."""
    c1 = store.commit("t", adapter(rng), message="a", base_model="b")
    c2 = store.commit("t", adapter(rng), message="b", base_model="b")
    m = store.merge("t", "linear", [(c1.id, 0.5), (c2.id, 0.5)], message="avg")

    stream = store.checkout_stream("t", m.id)
    next(iter(stream))
    del stream
    gc.collect()

    assert not store._cache_path("t", m.manifest_id).exists()
    assert not list(store._cache_path("t", m.manifest_id).parent.glob("*.tmp"))


def test_a_stack_of_views_streams_through_every_level(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="a", base_model="b", ref="a")
    c2 = store.commit("t", scaled(a, "layers.0.lora_A.weight", 2.0), message="b", base_model="b", ref="b")
    store.merge("t", "linear", [("a", 0.5), ("b", 0.5)], message="inner", ref="inner")
    outer = store.merge("t", "linear", [("inner", 0.5), ("a", 0.5)], message="outer", ref="outer")

    streamed = dict(store.checkout_stream("t", outer.id))
    expected = store.checkout("t", outer.id)
    assert sorted(streamed) == sorted(expected)
    for name, array in expected.items():
        assert np.allclose(streamed[name].astype(np.float32), array.astype(np.float32), rtol=1e-2)
    assert c1.id != c2.id


def test_a_streamed_view_refuses_when_an_input_is_gone(store, rng):
    from ballast import BrokenView

    c1 = store.commit("t", adapter(rng), message="a", base_model="b")
    c2 = store.commit("t", adapter(rng), message="b", base_model="b")
    m = store.merge("t", "linear", [(c1.id, 0.5), (c2.id, 0.5)], message="avg")
    store.forget_commit("t", c1.id, "withdrawn")

    with pytest.raises(BrokenView, match="cannot resolve"):
        dict(store.checkout_stream("t", m.id))
    with pytest.raises(BrokenView, match="cannot resolve"):
        store.specs("t", m.id)


def test_a_union_view_streams_the_union(store, rng):
    a = adapter(rng, layers=2)
    b = {k: v for k, v in adapter(rng, layers=3).items() if ".2." in k}
    c1 = store.commit("t", a, message="a", base_model="b")
    c2 = store.commit("t", b, message="b", base_model="b")
    m = store.merge("t", "linear", [(c1.id, 1.0), (c2.id, 1.0)], message="u", strict=False)
    assert sorted(dict(store.checkout_stream("t", m.id))) == sorted(set(a) | set(b))
