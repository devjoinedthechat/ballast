import ml_dtypes
import numpy as np
import pytest
from conftest import adapter

from ballast import BrokenView
from ballast import merge as merging


def test_linear_is_a_weighted_sum_in_float32_cast_back():
    a = {"w": np.array([1, 2, 3], dtype=ml_dtypes.bfloat16)}
    b = {"w": np.array([10, 20, 30], dtype=ml_dtypes.bfloat16)}
    out = merging.linear([a, b], [0.5, 0.5])
    assert out["w"].dtype == ml_dtypes.bfloat16
    assert np.allclose(out["w"].astype(np.float32), [5.5, 11, 16.5])


def test_ties_elects_the_sign_and_ignores_the_losers():
    # Two inputs agree on the first entry, disagree on the second with the
    # larger magnitude negative, and only one touches the third.
    a = {"w": np.array([1.0, 1.0, 0.0], dtype=np.float32)}
    b = {"w": np.array([1.0, -3.0, 2.0], dtype=np.float32)}
    out = merging.ties([a, b], [1.0, 1.0], density=1.0)
    assert np.allclose(out["w"], [1.0, -3.0, 2.0])


def test_ties_trims_to_density_before_merging():
    a = {"w": np.array([0.01, 5.0, 0.02, 4.0], dtype=np.float32)}
    out = merging.ties([a], [1.0], density=0.5)
    assert np.allclose(out["w"], [0.0, 5.0, 0.0, 4.0])


def test_mismatched_tensor_sets_are_refused():
    with pytest.raises(ValueError, match="do not share"):
        merging.linear([{"w": np.zeros(2)}, {"v": np.zeros(2)}], [1, 1])


def test_record_only_methods_refuse_to_resolve():
    with pytest.raises(NotImplementedError, match="recorded for provenance"):
        merging.resolve("slerp", [{"w": np.zeros(2)}], [1.0])


def test_a_merge_is_a_view_that_resolves_at_checkout(store, rng):
    a = adapter(rng)
    b = {k: (v.astype(np.float32) * 2).astype(v.dtype) for k, v in a.items()}
    c1 = store.commit("t", a, message="a", base_model="base")
    c2 = store.commit("t", b, message="b", base_model="base")
    before = store.stats("t").chunks
    m = store.merge("t", "linear", [(c1.id, 0.5), (c2.id, 0.5)], message="avg")
    assert store.stats("t").chunks == before  # nothing materialised
    out = store.checkout("t", m.id)
    expected = a["layers.0.lora_B.weight"].astype(np.float32) * 1.5
    assert np.allclose(out["layers.0.lora_B.weight"].astype(np.float32), expected, rtol=1e-2)


def test_merging_across_base_models_is_refused(store, rng):
    c1 = store.commit("t", adapter(rng), message="a", base_model="llama")
    c2 = store.commit("t", adapter(rng), message="b", base_model="mistral")
    with pytest.raises(ValueError, match="different base models"):
        store.merge("t", "linear", [(c1.id, 1.0), (c2.id, 1.0)], message="bad")


def test_deleting_an_input_breaks_the_view_and_the_proof_says_so(store, rng):
    c1 = store.commit("t", adapter(rng), message="a")
    c2 = store.commit("t", adapter(rng), message="b")
    m = store.merge("t", "linear", [(c1.id, 1.0), (c2.id, 1.0)], message="avg")
    proof = store.forget_commit("t", c1.id, "user request")
    assert m.manifest_id in proof.broken_composites
    with pytest.raises(BrokenView):
        store.checkout("t", m.id)
    assert store.checkout("t", c2.id)  # the surviving input is untouched
