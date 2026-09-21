"""The metadata graph, in SQLite.

Everything that is not tensor bytes lives here: manifests, what they reference,
commits, refs, grants, fingerprints, proofs, tombstones. It is the part that
survives the day the chunk store is rewritten in a systems language, which is why
it gets a real schema and a version number rather than a directory of JSON.

Every table that holds tenant data carries the tenant in its key. Isolation is a
property of the schema, not a check in the application. The one deliberate
crossing is `manifest_inputs.input_tenant`, which lets a composite reference
another tenant's manifest — and only resolves while a grant for it is live.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL
);

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

-- A composite manifest names the manifests it is a view over. It holds no
-- tensors, and input_id carries no foreign key on purpose: a view must be able
-- to outlive its input and be found broken, not be cascade-deleted with it.
CREATE TABLE IF NOT EXISTS manifest_inputs (
    tenant       TEXT NOT NULL,
    manifest_id  TEXT NOT NULL,
    position     INTEGER NOT NULL,
    input_tenant TEXT NOT NULL,
    input_id     TEXT NOT NULL,
    weight       REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (tenant, manifest_id, position),
    FOREIGN KEY (tenant, manifest_id) REFERENCES manifests(tenant, id) ON DELETE CASCADE
);

-- Permission for grantee to build views over one of owner's manifests. The
-- owner's chunks never move; the grantee reads through the grant, so revoking
-- it or deleting the manifest breaks the grantee's views and nothing lingers.
CREATE TABLE IF NOT EXISTS grants (
    owner       TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    grantee     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    revoked_at  REAL,
    PRIMARY KEY (owner, manifest_id, grantee)
);

CREATE TABLE IF NOT EXISTS commits (
    tenant      TEXT NOT NULL,
    id          TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    parent_id   TEXT,
    message     TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
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

-- The questions, kept so a fingerprint id can be explained later.
CREATE TABLE IF NOT EXISTS probe_sets (
    id          TEXT NOT NULL PRIMARY KEY,
    probes      TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS fingerprints (
    tenant      TEXT NOT NULL,
    commit_id   TEXT NOT NULL,
    probe_set   TEXT NOT NULL,
    position    INTEGER NOT NULL,
    output      TEXT NOT NULL,
    PRIMARY KEY (tenant, commit_id, probe_set, position),
    FOREIGN KEY (tenant, commit_id) REFERENCES commits(tenant, id) ON DELETE CASCADE,
    FOREIGN KEY (probe_set) REFERENCES probe_sets(id)
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

-- The proof itself, so it can be re-verified from another process, later.
CREATE TABLE IF NOT EXISTS proofs (
    attestation TEXT NOT NULL PRIMARY KEY,
    tenant      TEXT NOT NULL,
    body        TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_commits_parent ON commits(tenant, parent_id);
CREATE INDEX IF NOT EXISTS ix_tensors_chunk ON manifest_tensors(tenant, chunk_hash);
CREATE INDEX IF NOT EXISTS ix_inputs_input ON manifest_inputs(input_tenant, input_id);
CREATE INDEX IF NOT EXISTS ix_grants_grantee ON grants(grantee, owner, manifest_id);
"""


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        return
    current = int(row["version"])
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"store schema is version {current}; this build understands {SCHEMA_VERSION}. Upgrade ballast."
        )
    # Future migrations go here, one per version step, each ending in an UPDATE.
