"""The Postgres metadata backend, against a real server.

Skipped unless one is reachable. Start one with:

    docker run -d --name ballast-pg -e POSTGRES_PASSWORD=ballast \\
      -e POSTGRES_USER=ballast -e POSTGRES_DB=ballast -p 55432:5432 postgres:16-alpine

and point `BALLAST_TEST_POSTGRES` at it to use a different server.
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import psycopg
import pytest
from conftest import adapter, scaled

from ballast import BrokenView, Store
from ballast.sql import PostgresDatabase, split

DEFAULT_DSN = "postgresql://ballast:ballast@localhost:55432/ballast"


def dsn() -> str:
    return os.environ.get("BALLAST_TEST_POSTGRES", DEFAULT_DSN)


def available() -> bool:
    try:
        PostgresDatabase(dsn()).close()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not available(), reason="no Postgres at BALLAST_TEST_POSTGRES")


@pytest.fixture
def pg_store(tmp_path):
    """A store whose metadata lives in its own Postgres schema."""
    schema = f"t{uuid.uuid4().hex[:12]}"
    with psycopg.connect(dsn(), autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}")
    store = Store(tmp_path / "blocks", metadata=f"{dsn()}?options=-csearch_path%3D{schema}")
    yield store
    store.close()
    with psycopg.connect(dsn(), autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")


def test_the_same_store_works_on_postgres(pg_store, rng):
    a = adapter(rng)
    c1 = pg_store.commit("t", a, message="v1", base_model="b", metadata={"run": 41})
    c2 = pg_store.commit("t", scaled(a, "layers.2.lora_A.weight", 1.5), message="v2", base_model="b")

    assert pg_store.db.dialect == "postgres"
    assert pg_store.resolve("t", c1.id[:8]).metadata == {"run": 41}
    assert pg_store.stats("t").commits == 2
    out = pg_store.checkout("t", c1.id)
    assert all(np.array_equal(out[k].view(np.uint8), a[k].view(np.uint8)) for k in a)
    assert pg_store.diff("t", c1.id, c2.id).changed == ("layers.2.lora_A.weight",)
    assert pg_store.fsck("t") == []


def test_one_changed_tensor_stores_one_block_on_postgres(pg_store, rng):
    a = adapter(rng)
    pg_store.commit("t", a, message="v1")
    before = pg_store.stats("t").chunks
    pg_store.commit("t", scaled(a, "layers.0.lora_A.weight", 2.0), message="v2")
    assert pg_store.stats("t").chunks == before + 1


def test_views_grants_and_revocation_on_postgres(pg_store, rng):
    for tenant in ("bob", "org", "alice"):
        pg_store.commit(tenant, adapter(rng), message=tenant, base_model="b")
    pg_store.grant("bob", "main", "org")
    pg_store.merge("org", "linear", [("bob:main", 0.5), ("main", 0.5)], message="x", ref="layered")
    pg_store.grant("org", "layered", "alice")
    stack = pg_store.merge("alice", "linear", [("org:layered", 0.5), ("main", 0.5)], message="y", ref="stack")
    assert pg_store.checkout("alice", "stack")

    # The recursive CTE that finds downstream views has to work on both engines.
    broken = pg_store.revoke("bob", "main", "org")
    assert f"alice:{stack.manifest_id}" in broken
    with pytest.raises(BrokenView):
        pg_store.checkout("alice", "stack")


def test_deletion_and_proofs_on_postgres(pg_store, rng):
    a = adapter(rng)
    pg_store.commit("gone", a, message="v1")
    pg_store.commit("gone", scaled(a, "layers.1.lora_B.weight", 3.0), message="v2")
    pg_store.commit("stays", adapter(rng), message="v1")

    proof = pg_store.forget("gone", "erasure")
    assert pg_store.verify(proof) == []
    assert pg_store.verify(proof.attestation) == []
    assert pg_store.stats("gone").commits == 0
    assert pg_store.stats("stays").commits == 1


def test_a_dare_view_keeps_its_seed_on_postgres(pg_store, rng):
    c1 = pg_store.commit("t", adapter(rng), message="a", base_model="b")
    c2 = pg_store.commit("t", adapter(rng), message="b", base_model="b")
    m = pg_store.merge("t", "dare_ties", [(c1.id, 1.0), (c2.id, 1.0)], density=0.5, message="dare")
    seed = pg_store.manifest("t", m.manifest_id).config["seed"]
    assert isinstance(seed, int)
    first = pg_store.checkout("t", m.id)
    pg_store._cache_path("t", m.manifest_id).unlink()
    assert all(np.array_equal(first[k], pg_store.checkout("t", m.id)[k]) for k in first)


def test_the_reflog_survives_a_reopen_on_postgres(pg_store, rng, tmp_path):
    a = adapter(rng)
    c1 = pg_store.commit("t", a, message="v1")
    pg_store.commit("t", scaled(a, "layers.0.lora_A.weight", 2.0), message="v2")
    pg_store.reset("t", "main", c1.id)
    assert [e.op for e in pg_store.reflog("t", "main")] == ["reset", "commit", "commit"]


# -- the adapter itself ----------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("SELECT ? FROM t", "SELECT %s FROM t"),
        ("SELECT * FROM t WHERE a = ? AND b = ?", "SELECT * FROM t WHERE a = %s AND b = %s"),
        # A question mark inside a literal is data, not a placeholder.
        ("SELECT * FROM t WHERE a = 'why?' AND b = ?", "SELECT * FROM t WHERE a = 'why?' AND b = %s"),
    ],
)
def test_placeholders_are_rewritten_outside_string_literals(query, expected):
    assert PostgresDatabase.adapt(query) == expected


def test_a_semicolon_in_a_comment_does_not_end_a_statement():
    script = (
        "CREATE TABLE a (x TEXT);\n"
        "-- stored thus; the address does not depend on it\n"
        "CREATE TABLE b (y TEXT);"
    )
    assert len(split(script)) == 2


def test_an_apostrophe_in_a_comment_does_not_open_a_string():
    script = "-- one tenant's blocks; not another's\nCREATE TABLE a (x TEXT);\nCREATE TABLE b (y TEXT);"
    statements = split(script)
    assert len(statements) == 2
    assert statements[0].startswith("CREATE TABLE a")
