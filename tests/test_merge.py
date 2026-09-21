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


@pytest.mark.parametrize("method", ["slerp", "breadcrumbs", "della", "passthrough"])
def test_record_only_methods_refuse_to_resolve(method):
    """A method that is not implemented must not quietly run as one that is."""
    with pytest.raises(NotImplementedError, match="recorded for provenance"):
        merging.resolve(method, [{"w": np.zeros(2)}], [1.0])


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
    assert f"t:{m.manifest_id}" in proof.broken_composites
    with pytest.raises(BrokenView):
        store.checkout("t", m.id)
    assert store.checkout("t", c2.id)  # the surviving input is untouched


# -- DARE ------------------------------------------------------------------


def test_dare_drops_entries_and_keeps_the_l1_norm():
    """Drop And REscale: a fraction survives, and the tensor's L1 norm is kept."""
    a = {"w": np.arange(1, 1001, dtype=np.float32)}
    out = merging.dare_linear([a], [1.0], density=0.3, seed=7)["w"]
    kept = int((out != 0).sum())
    assert 200 < kept < 400  # a draw around 300, not a fixed count
    assert float(np.abs(out).sum()) == pytest.approx(float(np.abs(a["w"]).sum()), rel=1e-5)


def test_the_same_seed_gives_the_same_mask_and_a_different_one_does_not():
    a = {"w": np.arange(1, 501, dtype=np.float32)}
    first = merging.dare_linear([a], [1.0], density=0.4, seed=11)["w"]
    again = merging.dare_linear([a], [1.0], density=0.4, seed=11)["w"]
    other = merging.dare_linear([a], [1.0], density=0.4, seed=12)["w"]
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_a_tensors_mask_does_not_depend_on_the_other_tensors():
    """Seeds are derived from the tensor's name, not drawn in sequence.

    A view resolved on its own has to give the same answer as the same view
    resolved inside a larger one, whatever else is being merged alongside it.
    """
    w = np.arange(1, 257, dtype=np.float32)
    alone = merging.dare_linear([{"w": w}], [1.0], density=0.5, seed=3)["w"]
    crowded = merging.dare_linear([{"a": w * 2, "w": w, "z": w * 3}], [1.0], density=0.5, seed=3)["w"]
    assert np.array_equal(alone, crowded)


def test_dare_ties_elects_signs_and_dare_linear_does_not():
    a = {"w": np.full(400, 1.0, dtype=np.float32)}
    b = {"w": np.full(400, -1.0, dtype=np.float32)}
    elected = merging.dare_ties([a, b], [0.6, 0.4], density=1.0, seed=1)["w"]
    summed = merging.dare_linear([a, b], [0.6, 0.4], density=1.0, seed=1)["w"]
    # At full density nothing is dropped: election keeps the majority sign only.
    assert np.allclose(elected, 0.6)
    assert np.allclose(summed, 0.2)


def test_resolving_a_seeded_method_without_a_seed_is_an_error():
    with pytest.raises(ValueError, match="needs a seed"):
        merging.resolve("dare_ties", [{"w": np.ones(4, dtype=np.float32)}], [1.0])


def test_a_dare_view_stores_its_seed_and_resolves_the_same_twice(store, rng):
    c1 = store.commit("t", adapter(rng), message="a", base_model="b")
    c2 = store.commit("t", adapter(rng), message="b", base_model="b")
    m = store.merge("t", "dare_ties", [(c1.id, 0.6), (c2.id, 0.4)], density=0.5, message="dare")
    config = store.manifest("t", m.manifest_id).config
    assert isinstance(config["seed"], int)

    first = store.checkout("t", m.id)
    store._cache_path("t", m.manifest_id).unlink()  # force a real re-resolve
    again = store.checkout("t", m.id)
    assert all(np.array_equal(first[k], again[k]) for k in first)


def test_two_dare_views_of_the_same_recipe_differ_unless_the_seed_is_given(store, rng):
    c1 = store.commit("t", adapter(rng), message="a", base_model="b")
    c2 = store.commit("t", adapter(rng), message="b", base_model="b")
    m1 = store.merge("t", "dare_ties", [(c1.id, 1.0), (c2.id, 1.0)], message="one", ref="a")
    m2 = store.merge("t", "dare_ties", [(c1.id, 1.0), (c2.id, 1.0)], message="two", ref="b")
    assert m1.manifest_id != m2.manifest_id  # different seeds, different views

    same = store.merge(
        "t",
        "dare_ties",
        [(c1.id, 1.0), (c2.id, 1.0)],
        message="pinned",
        ref="c",
        seed=store.manifest("t", m1.manifest_id).config["seed"],
    )
    assert same.manifest_id == m1.manifest_id  # identity is content, seed included


def test_a_method_that_draws_nothing_takes_no_seed(store, rng):
    c = store.commit("t", adapter(rng), message="a", base_model="b")
    with pytest.raises(ValueError, match="takes no seed"):
        store.merge("t", "linear", [(c.id, 1.0)], message="x", seed=5)
