import pytest
from conftest import adapter

from ballast import BrokenView
from ballast.store import NotGranted


def test_a_cross_tenant_reference_without_a_grant_is_refused(store, rng):
    store.commit("org", adapter(rng), message="org delta", base_model="base")
    store.commit("alice", adapter(rng), message="alice delta", base_model="base")
    with pytest.raises(NotGranted):
        store.merge("alice", "linear", [("org:main", 1.0), ("main", 1.0)], message="layered")


def test_a_grant_lets_the_grantee_build_a_view_and_the_owner_keeps_the_chunks(store, rng):
    store.commit("org", adapter(rng), message="org delta", base_model="base")
    store.commit("alice", adapter(rng), message="alice delta", base_model="base")
    store.grant("org", "main", "alice")

    m = store.merge("alice", "linear", [("org:main", 0.5), ("main", 0.5)], message="layered")
    assert store.checkout("alice", m.id)
    assert store.stats("alice").chunks == len(adapter(rng))  # only alice's own tensors
    assert [g.grantee for g in store.grants("org") if g.live] == ["alice"]


def test_revoking_a_grant_breaks_the_grantees_view_and_names_it(store, rng):
    store.commit("org", adapter(rng), message="org", base_model="base")
    store.commit("alice", adapter(rng), message="alice", base_model="base")
    store.grant("org", "main", "alice")
    m = store.merge("alice", "linear", [("org:main", 1.0), ("main", 1.0)], message="layered", ref="layered")

    broken = store.revoke("org", "main", "alice")
    assert broken == [f"alice:{m.manifest_id}"]
    with pytest.raises(BrokenView, match=r"grant .* missing or revoked"):
        store.checkout("alice", m.id)
    assert store.checkout("alice", "main")  # alice's own commit is unaffected


def test_the_owner_forgetting_breaks_the_grantee_and_the_proof_says_so(store, rng):
    store.commit("org", adapter(rng), message="org", base_model="base")
    store.commit("alice", adapter(rng), message="alice", base_model="base")
    store.grant("org", "main", "alice")
    m = store.merge("alice", "linear", [("org:main", 1.0), ("main", 1.0)], message="layered")

    proof = store.forget("org", "org left the platform")
    assert f"alice:{m.manifest_id}" in proof.broken_composites
    assert f"alice:{proof.manifests[0]}" in proof.revoked_grants
    assert store.verify(proof) == []
    with pytest.raises(BrokenView):
        store.checkout("alice", m.id)
    # Nothing of org's survives in alice's store.
    assert store.stats("org").chunks == 0


def test_forgetting_the_granted_commit_revokes_the_grant(store, rng):
    c = store.commit("org", adapter(rng), message="org", base_model="base")
    store.commit("alice", adapter(rng), message="alice", base_model="base")
    store.grant("org", "main", "alice")
    proof = store.forget_commit("org", c.id, "bad run")
    assert proof.revoked_grants == (f"alice:{c.manifest_id}",)
    assert all(not g.live for g in store.grants("org"))


def test_a_grant_covers_a_manifest_not_a_ref(store, rng):
    """Re-committing the ref does not silently extend the grant to new content."""
    a = adapter(rng)
    store.commit("org", a, message="v1", base_model="base")
    store.commit("alice", adapter(rng), message="alice", base_model="base")
    store.grant("org", "main", "alice")
    store.commit("org", adapter(rng), message="v2", base_model="base")  # main moves
    with pytest.raises(NotGranted):
        store.merge("alice", "linear", [("org:main", 1.0)], message="new content, no grant")


def test_a_tenant_cannot_grant_to_itself(store, rng):
    store.commit("org", adapter(rng), message="org")
    with pytest.raises(ValueError, match="own manifests"):
        store.grant("org", "main", "org")


def test_fsck_reports_a_view_whose_grant_was_revoked(store, rng):
    store.commit("org", adapter(rng), message="org", base_model="base")
    store.commit("alice", adapter(rng), message="alice", base_model="base")
    store.grant("org", "main", "alice")
    store.merge("alice", "linear", [("org:main", 1.0)], message="layered")
    assert store.fsck() == []
    store.revoke("org", "main", "alice")
    assert any("without a live grant" in p for p in store.fsck())


def test_a_view_over_a_view_resolves_across_three_tenants(store, rng):
    """Transitive delegation: alice needs a grant from org, not from bob.

    org builds on bob with bob's permission, and alice builds on org with
    org's permission. Alice never sees bob's manifest id and never needs a
    grant from bob; org vouches for what it passes on.
    """
    for tenant in ("bob", "org", "alice"):
        store.commit(tenant, adapter(rng), message=tenant, base_model="base")
    store.grant("bob", "main", "org")
    store.merge("org", "linear", [("bob:main", 0.5), ("main", 0.5)], message="org on bob", ref="layered")
    store.grant("org", "layered", "alice")
    stack = store.merge(
        "alice", "linear", [("org:layered", 0.5), ("main", 0.5)], message="stack", ref="stack"
    )

    assert store.checkout("alice", "stack")
    assert not store._granted("bob", store.resolve("bob", "main").manifest_id, "alice")
    assert store.fsck() == []
    assert stack.manifest_id


def test_revoking_the_first_hop_breaks_the_whole_stack(store, rng):
    """The leak a non-transitive cache would leave open.

    alice's view was resolved and cached while the chain was intact. When bob
    revokes org, alice's cached result is downstream of content she may no
    longer read, so it has to go with it.
    """
    for tenant in ("bob", "org", "alice"):
        store.commit(tenant, adapter(rng), message=tenant, base_model="base")
    store.grant("bob", "main", "org")
    org_view = store.merge("org", "linear", [("bob:main", 1.0), ("main", 1.0)], message="x", ref="layered")
    store.grant("org", "layered", "alice")
    alice_view = store.merge(
        "alice", "linear", [("org:layered", 1.0), ("main", 1.0)], message="y", ref="stack"
    )
    store.checkout("alice", "stack")
    assert store._cache_path("alice", alice_view.manifest_id).exists()

    broken = store.revoke("bob", "main", "org")
    assert f"alice:{alice_view.manifest_id}" in broken
    assert f"org:{org_view.manifest_id}" in broken
    assert not store._cache_path("alice", alice_view.manifest_id).exists()
    for tenant, ref in (("org", "layered"), ("alice", "stack")):
        with pytest.raises(BrokenView):
            store.checkout(tenant, ref)
    # Each tenant's own commit is untouched.
    assert store.checkout("alice", "main")


def test_an_owners_deletion_names_views_further_downstream(store, rng):
    for tenant in ("bob", "org", "alice"):
        store.commit(tenant, adapter(rng), message=tenant, base_model="base")
    store.grant("bob", "main", "org")
    store.merge("org", "linear", [("bob:main", 1.0), ("main", 1.0)], message="x", ref="layered")
    store.grant("org", "layered", "alice")
    alice_view = store.merge(
        "alice", "linear", [("org:layered", 1.0), ("main", 1.0)], message="y", ref="stack"
    )

    proof = store.forget("bob", "bob left")
    assert f"alice:{alice_view.manifest_id}" in proof.broken_composites
    assert store.verify(proof) == []
    with pytest.raises(BrokenView):
        store.checkout("alice", "stack")
