"""The metadata graph, in SQLite.

Everything that is not tensor bytes lives here: manifests, what they reference,
commits, refs, the reflog, grants, fingerprints, proofs, tombstones. It is the
part that survives the day the chunk store is rewritten in a systems language,
which is why it gets a real schema and a version number rather than a directory
of JSON.

Every table that holds tenant data carries the tenant in its key. Isolation is a
property of the schema, not a check in the application. The one deliberate
crossing is `manifest_inputs.input_tenant`, which lets a composite reference
another tenant's manifest — and only resolves while a grant for it is live.

Schema changes are migrations. Version 1 stored one chunk per tensor; version 2
stores tensors as blocks. A version-1 store opens under this build, is migrated
in place inside one transaction, and its files on disk are untouched.
"""

from __future__ import annotations

from pathlib import Path

from ballast.sql import Database, SqliteDatabase
from ballast.sql import connect as connect_db

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT NOT NULL PRIMARY KEY,
    value       TEXT NOT NULL
);

-- A block of bytes, addressed by the hash of its raw content. `encoding` says
-- how it is stored; the address does not depend on it.
CREATE TABLE IF NOT EXISTS chunks (
    tenant       TEXT NOT NULL,
    hash         TEXT NOT NULL,
    nbytes       INTEGER NOT NULL,
    stored_bytes INTEGER NOT NULL,
    encoding     TEXT NOT NULL CHECK (encoding IN ('raw', 'zstd')),
    created_at   DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (tenant, hash)
);

CREATE TABLE IF NOT EXISTS manifests (
    tenant      TEXT NOT NULL,
    id          TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('leaf', 'composite')),
    base_model  TEXT,
    config      TEXT NOT NULL,
    created_at  DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (tenant, id)
);

-- A leaf manifest names its tensors; each tensor is an ordered run of blocks.
CREATE TABLE IF NOT EXISTS manifest_tensors (
    tenant      TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    name        TEXT NOT NULL,
    dtype       TEXT NOT NULL,
    shape       TEXT NOT NULL,
    nbytes      INTEGER NOT NULL,
    PRIMARY KEY (tenant, manifest_id, name),
    FOREIGN KEY (tenant, manifest_id) REFERENCES manifests(tenant, id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tensor_blocks (
    tenant      TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    name        TEXT NOT NULL,
    position    INTEGER NOT NULL,
    chunk_hash  TEXT NOT NULL,
    PRIMARY KEY (tenant, manifest_id, name, position),
    FOREIGN KEY (tenant, manifest_id, name) REFERENCES manifest_tensors(tenant, manifest_id, name)
        ON DELETE CASCADE,
    FOREIGN KEY (tenant, chunk_hash) REFERENCES chunks(tenant, hash)
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
    weight       DOUBLE PRECISION NOT NULL DEFAULT 1.0,
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
    created_at  DOUBLE PRECISION NOT NULL,
    revoked_at  DOUBLE PRECISION,
    PRIMARY KEY (owner, manifest_id, grantee)
);

CREATE TABLE IF NOT EXISTS commits (
    tenant      TEXT NOT NULL,
    id          TEXT NOT NULL,
    manifest_id TEXT NOT NULL,
    parent_id   TEXT,
    message     TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  DOUBLE PRECISION NOT NULL,
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

-- Every move of every ref, so a rollback can itself be rolled back.
CREATE TABLE IF NOT EXISTS reflog (
    tenant      TEXT NOT NULL,
    ref         TEXT NOT NULL,
    position    INTEGER NOT NULL,
    old_commit  TEXT,
    new_commit  TEXT,
    op          TEXT NOT NULL,
    at          DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (tenant, ref, position)
);

-- The questions, kept so a fingerprint id can be explained later.
CREATE TABLE IF NOT EXISTS probe_sets (
    id          TEXT NOT NULL PRIMARY KEY,
    probes      TEXT NOT NULL,
    created_at  DOUBLE PRECISION NOT NULL
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
    deleted_at  DOUBLE PRECISION NOT NULL,
    attestation TEXT NOT NULL,
    PRIMARY KEY (tenant, kind, id)
);

-- The proof itself, so it can be re-verified from another process, later. The
-- signature is an HMAC under a key held outside the store, when one is set.
CREATE TABLE IF NOT EXISTS proofs (
    attestation TEXT NOT NULL PRIMARY KEY,
    tenant      TEXT NOT NULL,
    body        TEXT NOT NULL,
    signature   TEXT,
    created_at  DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_commits_parent ON commits(tenant, parent_id);
CREATE INDEX IF NOT EXISTS ix_blocks_chunk ON tensor_blocks(tenant, chunk_hash);
CREATE INDEX IF NOT EXISTS ix_inputs_input ON manifest_inputs(input_tenant, input_id);
CREATE INDEX IF NOT EXISTS ix_grants_grantee ON grants(grantee, owner, manifest_id);
"""


def connect(target: Path | str) -> Database:
    """Open a metadata database, creating or migrating its schema."""
    db = connect_db(target)
    version = _current_version(db)
    if version is None:
        db.script(SCHEMA)
        db.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        return db
    if version > SCHEMA_VERSION:
        db.close()
        raise RuntimeError(
            f"store schema is version {version}; this build understands {SCHEMA_VERSION}. Upgrade ballast."
        )
    if version < SCHEMA_VERSION:
        _migrate(db, version)
    db.script(SCHEMA)  # idempotent: adds any table introduced after the migration
    return db


def _current_version(db: Database) -> int | None:
    if db.dialect == "postgres":
        exists = db.execute("SELECT to_regclass('schema_version') AS t").fetchone()
        if exists is None or exists["t"] is None:
            return None
    else:
        row = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
        ).fetchone()
        if row is None:
            return None
    version = db.execute("SELECT version FROM schema_version").fetchone()
    return int(version["version"]) if version else None


def _migrate(db: Database, version: int) -> None:
    """Apply every step from `version` up to the current one, each atomically.

    SQLite needs its foreign keys off while tables are reshaped; Postgres does
    not, because the steps drop and recreate in dependency order.
    """
    steps = {1: _MIGRATE_1_TO_2}
    if isinstance(db, SqliteDatabase):
        db.set_foreign_keys(False)
    while version < SCHEMA_VERSION:
        db.begin()
        try:
            db.script(steps[version])
            db.execute("UPDATE schema_version SET version = ?", (version + 1,))
            db.commit()
        except Exception:
            db.rollback()
            raise
        version += 1
    if isinstance(db, SqliteDatabase):
        db.set_foreign_keys(True)


# One chunk per tensor becomes one tensor with one block. Files are unchanged
# here; the store renames them to byte-only addresses when it next opens.
_MIGRATE_1_TO_2 = """
CREATE TABLE chunks_v2 (
            tenant TEXT NOT NULL, hash TEXT NOT NULL, nbytes INTEGER NOT NULL,
            stored_bytes INTEGER NOT NULL, encoding TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL,
            PRIMARY KEY (tenant, hash)
        );
        INSERT INTO chunks_v2 SELECT tenant, hash, nbytes, nbytes, 'raw', created_at FROM chunks;

        CREATE TABLE manifest_tensors_v2 (
            tenant TEXT NOT NULL, manifest_id TEXT NOT NULL, name TEXT NOT NULL,
            dtype TEXT NOT NULL, shape TEXT NOT NULL, nbytes BIGINT NOT NULL,
            PRIMARY KEY (tenant, manifest_id, name)
        );
        INSERT INTO manifest_tensors_v2
            SELECT t.tenant, t.manifest_id, t.name, c.dtype, c.shape, c.nbytes
            FROM manifest_tensors t JOIN chunks c ON c.tenant = t.tenant AND c.hash = t.chunk_hash;

        CREATE TABLE tensor_blocks (
            tenant TEXT NOT NULL, manifest_id TEXT NOT NULL, name TEXT NOT NULL,
            position INTEGER NOT NULL, chunk_hash TEXT NOT NULL,
            PRIMARY KEY (tenant, manifest_id, name, position)
        );
        INSERT INTO tensor_blocks SELECT tenant, manifest_id, name, 0, chunk_hash FROM manifest_tensors;

        DROP TABLE manifest_tensors;
        ALTER TABLE manifest_tensors_v2 RENAME TO manifest_tensors;
        DROP TABLE chunks;
        ALTER TABLE chunks_v2 RENAME TO chunks;

        ALTER TABLE proofs ADD COLUMN signature TEXT;
        CREATE TABLE IF NOT EXISTS settings (key TEXT NOT NULL PRIMARY KEY, value TEXT NOT NULL);
        INSERT OR REPLACE INTO settings (key, value) VALUES ('block_size', '8388608');
        INSERT OR REPLACE INTO settings (key, value) VALUES ('compression', 'zstd');
        INSERT OR REPLACE INTO settings (key, value) VALUES ('rehash_pending', '1');
"""
