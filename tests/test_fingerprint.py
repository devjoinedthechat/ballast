import pytest
from conftest import adapter, scaled

from ballast import FakeRunner, ProbeSet, fingerprint


def test_probes_that_see_the_change_report_it(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="v1")
    c2 = store.commit("t", scaled(a, "layers.2.lora_A.weight", 1.5), message="v2")
    ps = ProbeSet.of("What plan is Acme on?", "Refund policy?")
    fingerprint(store, "t", c1.id, ps, FakeRunner())
    fingerprint(store, "t", c2.id, ps, FakeRunner())
    diff = store.diff("t", c1.id, c2.id, probe_set=ps.id)
    assert diff.probes_changed == 2
    assert not diff.unobserved


def test_probes_blind_to_the_change_are_reported_as_unobserved(store, rng):
    """The dangerous case: weights moved, the probes said nothing."""
    a = adapter(rng)
    c1 = store.commit("t", a, message="v1")
    c2 = store.commit("t", scaled(a, "layers.2.lora_A.weight", 1.5), message="v2")
    ps = ProbeSet.of("hello")
    blind = FakeRunner(watch=["layers.0.lora_A.weight"])
    fingerprint(store, "t", c1.id, ps, blind)
    fingerprint(store, "t", c2.id, ps, blind)
    diff = store.diff("t", c1.id, c2.id, probe_set=ps.id)
    assert diff.relative_change > 0.01
    assert diff.probes_changed == 0
    assert diff.unobserved
    assert "UNOBSERVED" in str(diff)


def test_no_fingerprints_means_effect_unknown_not_unchanged(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="v1")
    c2 = store.commit("t", scaled(a, "layers.2.lora_A.weight", 1.5), message="v2")
    diff = store.diff("t", c1.id, c2.id)
    assert diff.probes_total is None
    assert not diff.unobserved
    assert "unknown" in str(diff)


def test_identical_commits_have_zero_change(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="v1")
    c2 = store.commit("t", a, message="v1 again")
    diff = store.diff("t", c1.id, c2.id)
    assert diff.changed == ()
    assert diff.relative_change == 0.0


def test_a_runner_returning_the_wrong_count_is_an_error(store, rng):
    c = store.commit("t", adapter(rng), message="v1")

    class Short:
        def run(self, tensors, config, base_model, probes):
            return ["only one"]

    with pytest.raises(ValueError, match="returned 1 outputs for 2"):
        fingerprint(store, "t", c.id, ProbeSet.of("a", "b"), Short())


def test_probe_set_id_is_content_derived():
    assert ProbeSet.of("a", "b").id == ProbeSet.of("a", "b").id
    assert ProbeSet.of("a", "b").id != ProbeSet.of("b", "a").id
