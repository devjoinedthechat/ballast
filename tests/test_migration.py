"""A version-1 store opens under version 2 and keeps its data."""

import json
import sqlite3
import time

import blake3
import numpy as np
import pytest
from conftest import adapter

from ballast import Store
from ballast.chunks import block_digest
from ballast.db import SCHEMA_VERSION
from ballast.tensors import dtype_name

V1_SCHEMA = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE TABLE chunks (tenant TEXT NOT NULL, hash TEXT NOT NULL, dtype TEXT NOT NULL, shape TEXT NOT NULL,
    nbytes INTEGER NOT NULL, created_at REAL NOT NULL, PRIMARY KEY (tenant, hash));
CREATE TABLE manifests (tenant TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL, base_model TEXT,
    config TEXT NOT NULL, created_at REAL NOT NULL, PRIMARY KEY (tenant, id));
CREATE TABLE manifest_tensors (tenant TEXT NOT NULL, manifest_id TEXT NOT NULL, name TEXT NOT NULL,
    chunk_hash TEXT NOT NULL, PRIMARY KEY (tenant, manifest_id, name));
CREATE TABLE manifest_inputs (tenant TEXT NOT NULL, manifest_id TEXT NOT NULL, position INTEGER NOT NULL,
    input_tenant TEXT NOT NULL, input_id TEXT NOT NULL, weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (tenant, manifest_id, position));
CREATE TABLE grants (owner TEXT NOT NULL, manifest_id TEXT NOT NULL, grantee TEXT NOT NULL,
    created_at REAL NOT NULL, revoked_at REAL, PRIMARY KEY (owner, manifest_id, grantee));
CREATE TABLE commits (tenant TEXT NOT NULL, id TEXT NOT NULL, manifest_id TEXT NOT NULL, parent_id TEXT,
    message TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL,
    PRIMARY KEY (tenant, id));
CREATE TABLE refs (tenant TEXT NOT NULL, name TEXT NOT NULL, commit_id TEXT NOT NULL,
    PRIMARY KEY (tenant, name));
CREATE TABLE probe_sets (id TEXT NOT NULL PRIMARY KEY, probes TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE fingerprints (tenant TEXT NOT NULL, commit_id TEXT NOT NULL, probe_set TEXT NOT NULL,
    position INTEGER NOT NULL, output TEXT NOT NULL, PRIMARY KEY (tenant, commit_id, probe_set, position));
CREATE TABLE tombstones (tenant TEXT NOT NULL, kind TEXT NOT NULL, id TEXT NOT NULL, reason TEXT NOT NULL,
    deleted_at REAL NOT NULL, attestation TEXT NOT NULL, PRIMARY KEY (tenant, kind, id));
CREATE TABLE proofs (attestation TEXT NOT NULL PRIMARY KEY, tenant TEXT NOT NULL, body TEXT NOT NULL,
    created_at REAL NOT NULL);
INSERT INTO schema_version (version) VALUES (1);
"""


def v1_hash(array: np.ndarray) -> str:
    h = blake3.blake3()
    h.update(dtype_name(array.dtype).encode())
    h.update(json.dumps(list(array.shape)).encode())
    h.update(np.ascontiguousarray(array).view(np.uint8).reshape(-1).data)
    return h.hexdigest()


def build_v1_store(root, tensors):
    root.mkdir(parents=True)
    conn = sqlite3.connect(root / "ballast.db")
    conn.executescript(V1_SCHEMA)
    now = time.time()
    manifest_id = "m" * 64
    conn.execute("INSERT INTO manifests VALUES (?, ?, 'leaf', 'base', '{}', ?)", ("t", manifest_id, now))
    for name, array in tensors.items():
        digest = v1_hash(array)
        path = root / "chunks" / "t" / digest[:2] / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(np.ascontiguousarray(array).tobytes())
        conn.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
            ("t", digest, dtype_name(array.dtype), json.dumps(list(array.shape)), array.nbytes, now),
        )
        conn.execute("INSERT INTO manifest_tensors VALUES (?, ?, ?, ?)", ("t", manifest_id, name, digest))
    conn.execute(
        "INSERT INTO commits VALUES (?, ?, ?, NULL, 'v1 commit', '{}', ?)", ("t", "c" * 64, manifest_id, now)
    )
    conn.execute("INSERT INTO refs VALUES ('t', 'main', ?)", ("c" * 64,))
    conn.commit()
    conn.close()


def test_a_v1_store_migrates_in_place_and_reads_back(tmp_path, rng):
    tensors = adapter(rng)
    build_v1_store(tmp_path / "old", tensors)

    store = Store(tmp_path / "old")
    assert store.db.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION

    out = store.checkout("t", "main")
    assert set(out) == set(tensors)
    for name, array in tensors.items():
        assert np.array_equal(out[name].view(np.uint8), array.view(np.uint8))
        assert out[name].dtype == array.dtype

    # Blocks were renamed to their byte-only address, so new commits of the
    # same tensors deduplicate against the migrated ones.
    before = store.stats("t").chunks
    store.commit("t", tensors, message="same content after migration")
    assert store.stats("t").chunks == before
    assert store.fsck("t") == []
    assert "rehash_pending" not in store._settings()


def test_migrated_blocks_have_byte_addresses(tmp_path, rng):
    tensors = adapter(rng)
    build_v1_store(tmp_path / "old", tensors)
    store = Store(tmp_path / "old")
    digests = {r["hash"] for r in store.db.execute("SELECT hash FROM chunks")}
    expected = {
        block_digest(memoryview(np.ascontiguousarray(a).view(np.uint8).reshape(-1))) for a in tensors.values()
    }
    assert digests == expected


def test_a_newer_store_is_refused(tmp_path, rng):
    store = Store(tmp_path / "s")
    store.commit("t", adapter(rng), message="v1")
    store.db.execute("UPDATE schema_version SET version = 99")
    store.close()
    with pytest.raises(RuntimeError, match="Upgrade ballast"):
        Store(tmp_path / "s")
