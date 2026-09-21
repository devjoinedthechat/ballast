import numpy as np
import pytest
from conftest import adapter, scaled


def test_identical_content_is_stored_once(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="first")
    c2 = store.commit("t", a, message="again")
    assert c1.manifest_id == c2.manifest_id
    assert c1.id != c2.id
    s = store.stats("t")
    assert s.chunks == len(a)
    assert s.commits == 2


def test_a_commit_that_changes_one_tensor_stores_one_chunk(store, rng):
    a = adapter(rng)
    store.commit("t", a, message="v1")
    store.commit("t", scaled(a, "layers.2.lora_A.weight", 1.1), message="v2")
    s = store.stats("t")
    assert s.chunks == len(a) + 1
    assert s.dedup_ratio == pytest.approx(2 * sum(t.nbytes for t in a.values()) / s.physical_bytes)


def test_checkout_returns_the_bytes_that_went_in(store, rng):
    a = adapter(rng)
    c = store.commit("t", a, message="v1")
    out = store.checkout("t", c.id)
    assert set(out) == set(a)
    for name in a:
        assert out[name].dtype == a[name].dtype
        assert np.array_equal(out[name].view(np.uint8), a[name].view(np.uint8))


def test_refs_and_prefixes_resolve_and_ambiguity_is_an_error(store, rng):
    c = store.commit("t", adapter(rng), message="v1", ref="main")
    assert store.resolve("t", "main").id == c.id
    assert store.resolve("t", c.id[:8]).id == c.id
    with pytest.raises(LookupError, match="no ref or commit"):
        store.resolve("t", "nope")


def test_log_walks_the_chain_and_reset_rolls_back(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="v1")
    c2 = store.commit("t", scaled(a, "layers.0.lora_A.weight", 2.0), message="v2")
    c3 = store.commit("t", scaled(a, "layers.1.lora_A.weight", 2.0), message="v3")
    assert [c.id for c in store.log("t")] == [c3.id, c2.id, c1.id]

    store.reset("t", "main", c1.id)
    assert store.head("t").id == c1.id
    assert store.stats("t").commits == 3  # rollback deletes nothing


def test_tenants_cannot_see_each_other(store, rng):
    a = adapter(rng)
    c = store.commit("alpha", a, message="v1")
    store.commit("beta", a, message="v1")
    with pytest.raises(LookupError):
        store.resolve("beta", c.id)
    assert store.chunks.path("alpha", "ab" * 32).parts[-3] == "alpha"
    # Same bytes, separate chunks: the cost of clean deletion.
    assert store.stats("alpha").chunks == store.stats("beta").chunks == len(a)


def test_empty_commit_is_refused(store):
    with pytest.raises(ValueError, match="at least one tensor"):
        store.commit("t", {}, message="nothing")
