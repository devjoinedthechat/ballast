import numpy as np
import pytest
from conftest import adapter, scaled

from ballast import BrokenView, FakeRunner, ProbeSet, Store, fingerprint


def test_every_ref_move_is_logged_and_a_rollback_can_be_rolled_back(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="v1")
    c2 = store.commit("t", scaled(a, "layers.0.lora_A.weight", 2.0), message="v2")
    store.reset("t", "main", c1.id)
    log = store.reflog("t", "main")
    assert [(e.op, e.new_commit) for e in log] == [("reset", c1.id), ("commit", c2.id), ("commit", c1.id)]
    # The reflog knows where main was before the reset.
    store.reset("t", "main", log[0].old_commit)
    assert store.head("t").id == c2.id


def test_forgetting_the_head_is_a_logged_move(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="v1")
    c2 = store.commit("t", scaled(a, "layers.0.lora_A.weight", 2.0), message="v2")
    store.forget_commit("t", c2.id, "bad")
    entry = store.reflog("t", "main")[0]
    assert (entry.op, entry.old_commit, entry.new_commit) == ("forget", c2.id, c1.id)


def test_a_signed_proof_verifies_with_the_key_and_fails_without_it(tmp_path, rng):
    signed = Store(tmp_path / "s", signing_key=b"k1")
    signed.commit("t", adapter(rng), message="v1")
    proof = signed.forget("t", "erasure")
    assert proof.signature
    assert signed.verify(proof.attestation) == []
    signed.close()

    unkeyed = Store(tmp_path / "s")
    assert any("no signing key" in p for p in unkeyed.verify(proof.attestation))
    unkeyed.close()

    wrong = Store(tmp_path / "s", signing_key=b"k2")
    assert any("does not match the configured key" in p for p in wrong.verify(proof.attestation))


def test_an_unsigned_proof_is_reported_when_a_key_is_configured(tmp_path, rng):
    plain = Store(tmp_path / "s")
    plain.commit("t", adapter(rng), message="v1")
    proof = plain.forget("t", "erasure")
    assert proof.signature is None
    plain.close()
    keyed = Store(tmp_path / "s", signing_key=b"k")
    assert any("unsigned" in p for p in keyed.verify(proof.attestation))


def test_the_signing_key_comes_from_the_environment(tmp_path, rng, monkeypatch):
    monkeypatch.setenv("BALLAST_SIGNING_KEY", "from-env")
    store = Store(tmp_path / "s")
    store.commit("t", adapter(rng), message="v1")
    assert store.forget("t", "x").signature


def test_a_view_is_cached_and_the_cache_goes_when_an_input_does(store, rng):
    c1 = store.commit("t", adapter(rng), message="a", base_model="b")
    c2 = store.commit("t", adapter(rng), message="b", base_model="b")
    m = store.merge("t", "linear", [(c1.id, 0.5), (c2.id, 0.5)], message="avg")
    first = store.checkout("t", m.id)
    cached = store._cache_path("t", m.manifest_id)
    assert cached.exists()
    second = store.checkout("t", m.id)
    assert all(np.array_equal(first[k], second[k]) for k in first)

    store.forget_commit("t", c1.id, "gone")
    assert not cached.exists()
    with pytest.raises(BrokenView):
        store.checkout("t", m.id)


def test_revoking_a_grant_drops_the_grantees_cached_view(store, rng):
    store.commit("org", adapter(rng), message="org", base_model="b")
    store.commit("alice", adapter(rng), message="alice", base_model="b")
    store.grant("org", "main", "alice")
    m = store.merge("alice", "linear", [("org:main", 1.0), ("main", 1.0)], message="layered", ref="layered")
    store.checkout("alice", m.id)
    cached = store._cache_path("alice", m.manifest_id)
    assert cached.exists()
    store.revoke("org", "main", "alice")
    assert not cached.exists()


def test_union_merge_treats_absent_tensors_as_zero_and_strict_refuses(store, rng):
    a = adapter(rng, layers=2)
    b = {k: v for k, v in adapter(rng, layers=3).items() if ".2." in k}  # only layer 2
    c1 = store.commit("t", a, message="a", base_model="b")
    c2 = store.commit("t", b, message="b", base_model="b")
    with pytest.raises(ValueError, match="strict=False"):
        store.checkout("t", store.merge("t", "linear", [(c1.id, 1.0), (c2.id, 1.0)], message="strict").id)
    m = store.merge("t", "linear", [(c1.id, 1.0), (c2.id, 1.0)], message="union", strict=False, ref="u")
    out = store.checkout("t", m.id)
    assert set(out) == set(a) | set(b)
    name = "layers.0.lora_A.weight"
    assert np.allclose(out[name].astype(np.float32), a[name].astype(np.float32), rtol=1e-2)


def test_the_unobserved_threshold_is_configurable(store, rng):
    a = adapter(rng, layers=16)
    c1 = store.commit("t", a, message="v1")
    c2 = store.commit("t", scaled(a, "layers.3.lora_A.weight", 1.03), message="v2")
    ps = ProbeSet.of("hello")
    blind = FakeRunner(watch=["layers.0.lora_A.weight"])
    fingerprint(store, "t", c1.id, ps, blind)
    fingerprint(store, "t", c2.id, ps, blind)
    assert store.diff("t", c1.id, c2.id, probe_set=ps.id).unobserved
    assert not store.diff("t", c1.id, c2.id, probe_set=ps.id, threshold=0.5).unobserved


def test_probe_similarity_says_how_far_an_answer_moved(store, rng):
    store.commit("t", adapter(rng), message="v1")
    store.commit("t", adapter(rng), message="v2")
    c1, c2 = [c.id for c in store.log("t")][::-1]
    pid = store.record_fingerprint("t", c1, ["q"], ["The capital of France is Paris."])
    store.record_fingerprint("t", c2, ["q"], ["The capital of France is Paris!"])
    d = store.diff("t", c1, c2, probe_set=pid)
    assert d.probes_changed == 1
    assert d.probes_similarity is not None
    assert d.probes_similarity > 0.9
