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


@pytest.mark.parametrize("method", ["passthrough", "model_stock", "nuslerp", "sce"])
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


# -- SLERP -----------------------------------------------------------------


def test_slerp_travels_from_one_input_to_the_other():
    a = {"w": np.array([3.0, 4.0, 0.0], dtype=np.float32)}
    b = {"w": np.array([0.0, 4.0, 3.0], dtype=np.float32)}
    assert np.allclose(merging.slerp([a, b], 0.0)["w"], a["w"])
    assert np.allclose(merging.slerp([a, b], 1.0)["w"], b["w"])


def test_slerp_keeps_the_magnitude_a_straight_line_would_lose():
    """The whole reason to interpolate on the arc rather than the chord."""
    a = {"w": np.array([3.0, 4.0, 0.0], dtype=np.float32)}
    b = {"w": np.array([0.0, 4.0, 3.0], dtype=np.float32)}
    spherical = merging.slerp([a, b], 0.5)["w"]
    straight = 0.5 * a["w"] + 0.5 * b["w"]
    assert float(np.linalg.norm(spherical)) == pytest.approx(5.0, rel=1e-5)
    assert float(np.linalg.norm(straight)) < 5.0


def test_colinear_inputs_fall_back_to_a_straight_line():
    """The arc between two nearly parallel tensors is numerically undefined."""
    a = {"w": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    b = {"w": a["w"] * 2.0}
    got = merging.slerp([a, b], 0.25)["w"]
    assert np.allclose(got, 0.75 * a["w"] + 0.25 * b["w"])


def test_slerp_needs_exactly_two_inputs():
    one = {"w": np.ones(3, dtype=np.float32)}
    with pytest.raises(ValueError, match="exactly two"):
        merging.slerp([one, one, one], 0.5)


def test_a_slerp_view_must_carry_a_t(store, rng):
    c1 = store.commit("t", adapter(rng), message="a", base_model="b")
    c2 = store.commit("t", adapter(rng), message="b", base_model="b")
    with pytest.raises(ValueError, match="slerp needs a t"):
        store.merge("t", "slerp", [(c1.id, 1.0), (c2.id, 1.0)], message="x")
    m = store.merge("t", "slerp", [(c1.id, 1.0), (c2.id, 1.0)], message="x", t=0.3, ref="s")
    assert store.manifest("t", m.manifest_id).config["t"] == 0.3
    assert store.checkout("t", m.id)


# -- breadcrumbs -----------------------------------------------------------


def test_breadcrumbs_drops_the_largest_as_well_as_the_smallest():
    """TIES keeps the top. Breadcrumbs calls the very top outliers and drops it."""
    w = {"w": np.arange(1, 1001, dtype=np.float32)}
    out = merging.breadcrumbs([w], [1.0], density=0.5, gamma=0.1)["w"]
    kept = out[out != 0]
    assert len(kept) == 500
    assert kept.max() == 900.0  # the top 100 are gone
    assert kept.min() == 401.0  # and so is everything below the band


def test_breadcrumbs_ties_elects_a_sign_and_plain_breadcrumbs_does_not():
    a = {"w": np.full(200, 2.0, dtype=np.float32)}
    b = {"w": np.full(200, -1.0, dtype=np.float32)}
    elected = merging.breadcrumbs([a, b], [1.0, 1.0], density=1.0, elect=True)["w"]
    summed = merging.breadcrumbs([a, b], [1.0, 1.0], density=1.0, elect=False)["w"]
    assert np.allclose(elected, 2.0)
    assert np.allclose(summed, 1.0)


def test_the_outlier_cut_shrinks_when_the_density_leaves_no_room():
    """density 0.95 with gamma 0.1 cannot both hold; the cut gives way."""
    w = {"w": np.arange(1, 101, dtype=np.float32)}
    out = merging.breadcrumbs([w], [1.0], density=0.95, gamma=0.1)["w"]
    assert int((out != 0).sum()) == 95


# -- DELLA -----------------------------------------------------------------


def test_della_keeps_larger_entries_more_often_than_smaller_ones():
    """The point of DELLA: the keep probability rises with rank."""
    w = {"w": np.arange(1, 2001, dtype=np.float32)}
    out = merging.della([w], [1.0], density=0.5, seed=5, epsilon=0.3)["w"]
    kept = out != 0
    bottom = int(kept[:1000].sum())
    top = int(kept[1000:].sum())
    assert top > bottom * 1.3


def test_della_is_reproducible_from_its_seed():
    w = {"w": np.arange(1, 501, dtype=np.float32)}
    first = merging.della([w], [1.0], density=0.5, seed=9)["w"]
    assert np.array_equal(first, merging.della([w], [1.0], density=0.5, seed=9)["w"])
    assert not np.array_equal(first, merging.della([w], [1.0], density=0.5, seed=10)["w"])


def test_an_epsilon_that_pushes_the_probability_out_of_range_is_refused():
    w = {"w": np.arange(1, 101, dtype=np.float32)}
    with pytest.raises(ValueError, match="epsilon"):
        merging.della([w], [1.0], density=0.1, seed=1, epsilon=0.3)


def test_della_views_carry_a_seed_like_dare(store, rng):
    c1 = store.commit("t", adapter(rng), message="a", base_model="b")
    c2 = store.commit("t", adapter(rng), message="b", base_model="b")
    m = store.merge("t", "della", [(c1.id, 0.5), (c2.id, 0.5)], density=0.5, message="d")
    config = store.manifest("t", m.manifest_id).config
    assert isinstance(config["seed"], int)
    first = store.checkout("t", m.id)
    store._cache_path("t", m.manifest_id).unlink()
    assert all(np.array_equal(first[k], store.checkout("t", m.id)[k]) for k in first)


def test_della_linear_skips_the_sign_election(store, rng):
    c1 = store.commit("t", adapter(rng), message="a", base_model="b")
    c2 = store.commit("t", adapter(rng), message="b", base_model="b")
    m = store.merge("t", "della_linear", [(c1.id, 0.5), (c2.id, 0.5)], density=0.5, message="dl")
    assert store.checkout("t", m.id)


def test_a_method_is_either_resolvable_or_recorded_never_both():
    """The two lists decide whether a recipe runs or refuses.

    An overlap would make that depend on the order of checks inside `resolve`,
    which is exactly the kind of thing that silently runs the wrong merge.
    """
    assert not set(merging.RESOLVABLE) & set(merging.RECORD_ONLY)
    assert set(merging.SEEDED) <= set(merging.RESOLVABLE)
