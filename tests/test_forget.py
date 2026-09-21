from conftest import adapter, scaled


def test_forgetting_a_tenant_leaves_nothing_and_the_proof_verifies(store, rng):
    a = adapter(rng)
    store.commit("gone", a, message="v1")
    store.commit("gone", scaled(a, "layers.0.lora_A.weight", 2.0), message="v2")
    store.commit("stays", a, message="v1")

    proof = store.forget("gone", "gdpr erasure")
    assert len(proof.commits) == 2
    assert len(proof.chunks) == len(a) + 1
    assert proof.bytes_freed > 0
    assert store.verify(proof) == []

    s = store.stats("gone")
    assert (s.commits, s.manifests, s.chunks) == (0, 0, 0)
    assert not store.chunks.path("gone", "00" * 32).parent.parent.exists()
    assert store.stats("stays").commits == 1


def test_forgetting_a_commit_reparents_children_and_moves_refs(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="v1")
    c2 = store.commit("t", scaled(a, "layers.0.lora_A.weight", 2.0), message="v2")
    c3 = store.commit("t", scaled(a, "layers.1.lora_A.weight", 2.0), message="v3")

    store.forget_commit("t", c2.id, "bad training run")
    assert [c.id for c in store.log("t")] == [c3.id, c1.id]
    assert store.resolve("t", c3.id).parent_id == c1.id


def test_forgetting_the_head_moves_the_ref_to_its_parent(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="v1")
    c2 = store.commit("t", scaled(a, "layers.0.lora_A.weight", 2.0), message="v2")
    store.forget_commit("t", c2.id, "rollback for good")
    assert store.head("t").id == c1.id


def test_a_manifest_shared_by_another_commit_survives(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="v1")
    c2 = store.commit("t", a, message="same content again")
    proof = store.forget_commit("t", c1.id, "dup")
    assert proof.manifests == ()
    assert proof.chunks == ()
    assert store.checkout("t", c2.id)


def test_only_orphaned_chunks_are_removed(store, rng):
    a = adapter(rng)
    c1 = store.commit("t", a, message="v1")
    store.commit("t", scaled(a, "layers.0.lora_A.weight", 2.0), message="v2")
    proof = store.forget_commit("t", c1.id, "cleanup")
    # v1's only unique tensor was the unscaled layer 0 A; everything else is shared with v2.
    assert len(proof.chunks) == 1
    assert store.verify(proof) == []


def test_a_tampered_proof_fails_verification(store, rng):
    store.commit("t", adapter(rng), message="v1")
    proof = store.forget("t", "erasure")
    forged = proof.__class__(**{**proof.__dict__, "attestation": "0" * 64})
    assert any("no matching tombstone" in p for p in store.verify(forged))


def test_gc_frees_chunks_nothing_references(store, rng):
    a = adapter(rng)
    store.commit("t", a, message="v1")
    store.db.execute("DELETE FROM manifest_tensors WHERE tenant = 't' AND name = 'layers.0.lora_A.weight'")
    freed = store.gc("t")
    assert freed == a["layers.0.lora_A.weight"].nbytes
    assert store.stats("t").chunks == len(a) - 1
