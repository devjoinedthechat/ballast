"""The metadata graph, in SQLite.

Everything that is not tensor bytes lives here: manifests, what they reference,
commits, refs, fingerprints, tombstones. It is the part that survives the day the
chunk store is rewritten in a systems language, which is why it gets a real
schema rather than a directory of JSON.

Every table that holds tenant data carries the tenant in its key. Isolation is a
property of the schema, not a check in the application.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS chunks (
    tenant      TEXT NOT NULL,
    hash        TEXT NOT NULL,
    dtype       TEXT NOT NULL,
    shape       TEXT NOT NULL,
    nbytes      INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (tenant, hash)
);

CREATE TABLE IF NOT EXISTS manifests (
    tenant      TEXT NOT NULL,
    id          TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('leaf', 'composite')),
    base_model  TEXT,
    config      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (tenant, id)
);

-- A leaf manifest names its tensors.
CREATE TABLE IF NOT EXISTS manifest_tensors (
    tenant      TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    name        TEXT NOT NULL,
    chunk_hash  TEXT NOT NULL,
    PRIMARY KEY (tenant, manifest_id, name),
    FOREIGN KEY (tenant, manifest_id) REFERENCES manifests(tenant, id) ON DELETE CASCADE,
    FOREIGN KEY (tenant, chunk_hash)  REFERENCES chunks(tenant, hash)
);

-- A composite manifest names the manifests it is a view over. It never holds
-- tensors of its own, which is what makes deleting an input clean.
CREATE TABLE IF NOT EXISTS manifest_inputs (
    tenant      TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    position    INTEGER NOT NULL,
    input_id    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (tenant, manifest_id, position),
    FOREIGN KEY (tenant, manifest_id) REFERENCES manifests(tenant, id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS commits (
    tenant      TEXT NOT NULL,
    id          TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    parent_id   TEXT,
    message     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (tenant, id),
    FOREIGN KEY (tenant, manifest_id) REFERENCES manifests(tenant, id)
);

CREATE TABLE IF NOT EXISTS refs (
    tenant      TEXT NOT NULL,
    name        TEXT NOT NULL,
    commit_id   TEXT NOT NULL,
    PRIMARY KEY (tenant, name),
    FOREIGN KEY (tenant, commit_id) REFERENCES commits(tenant, id)
);

CREATE TABLE IF NOT EXISTS fingerprints (
    tenant      TEXT NOT NULL,
    commit_id   TEXT NOT NULL,
    probe_set   TEXT NOT NULL,
    position    INTEGER NOT NULL,
    output      TEXT NOT NULL,
    PRIMARY KEY (tenant, commit_id, probe_set, position),
    FOREIGN KEY (tenant, commit_id) REFERENCES commits(tenant, id) ON DELETE CASCADE
);

-- What was deleted, when, and under which attestation. Survives the deletion
-- so the record can be audited after the data is gone.
CREATE TABLE IF NOT EXISTS tombstones (
    tenant      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    id          TEXT NOT NULL,
    reason      TEXT NOT NULL,
    deleted_at  REAL NOT NULL,
    attestation TEXT NOT NULL,
    PRIMARY KEY (tenant, kind, id)
);

CREATE INDEX IF NOT EXISTS ix_commits_parent ON commits(tenant, parent_id);
CREATE INDEX IF NOT EXISTS ix_tensors_chunk ON manifest_tensors(tenant, chunk_hash);
CREATE INDEX IF NOT EXISTS ix_inputs_input ON manifest_inputs(tenant, input_id);
"""


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn
