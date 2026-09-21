"""The store.

A tenant is a namespace. Inside it: blocks, manifests, commits, refs. A manifest
is either a leaf (names tensors, each an ordered run of blocks) or a composite
(names other manifests and a merge method). Commits chain manifests into a
history; refs name commits; the reflog remembers every move a ref made.

Three decisions shape everything else.

Merges are views. A composite manifest holds no tensors; it resolves at
checkout, and the resolved tensors are cached until an input is deleted or a
grant revoked. Deleting an input breaks the view in a way the store detects and
reports, instead of leaving the input's contribution smeared through a
materialised matrix where nothing can find it.

Identity is content. Manifest and block ids are hashes of what they contain,
so committing the same adapter twice stores nothing new, a commit that changes
one value in a large tensor stores one block, and a composite can never form a
cycle because its id depends on inputs that already exist.

Deletion produces a record. Forgetting a tenant or a commit returns a proof
naming what was removed under an attestation hash, stores the proof, signs it
when a key is configured, and can re-verify it from any process afterwards. It
is an audit record, not a cryptographic guarantee against a hostile operator;
it answers "show me that it is gone" for an operator acting in good faith, and
with a key held outside the store, it also shows the record was not rewritten.

Tenants may share. A grant lets one tenant build views over another's manifest.
The owner's blocks never move — the grantee reads through the grant — so
revoking it, or the owner deleting the manifest, breaks the grantee's views and
leaves nothing behind. Without a grant, a cross-tenant reference is refused.
"""

from __future__ import annotations

import difflib
import hashlib
import hmac
import json
import os
import secrets
import shutil
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ballast import merge as merging
from ballast import names
from ballast import tensors as st
from ballast.backends import Backend, LocalBackend, from_url
from ballast.chunks import DEFAULT_BLOCK_SIZE, ChunkStore, block_digest
from ballast.db import connect
from ballast.hashing import object_hash
from ballast.sql import Database

DEFAULT_REF = "main"
SIGNING_KEY_ENV = "BALLAST_SIGNING_KEY"


@dataclass(frozen=True)
class Manifest:
    tenant: str
    id: str
    kind: str
    base_model: str | None
    config: dict[str, Any]
    created_at: float


@dataclass(frozen=True)
class Commit:
    tenant: str
    id: str
    manifest_id: str
    parent_id: str | None
    message: str
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def short(self) -> str:
        return self.id[:12]


@dataclass(frozen=True)
class Grant:
    owner: str
    manifest_id: str
    grantee: str
    created_at: float
    revoked_at: float | None

    @property
    def live(self) -> bool:
        return self.revoked_at is None


@dataclass(frozen=True)
class ReflogEntry:
    ref: str
    position: int
    old_commit: str | None
    new_commit: str | None
    op: str
    at: float


@dataclass(frozen=True)
class Stats:
    commits: int
    manifests: int
    chunks: int
    physical_bytes: int
    logical_bytes: int
    raw_bytes: int = 0
    """Bytes the blocks hold before compression."""

    @property
    def dedup_ratio(self) -> float:
        return 1.0 if not self.raw_bytes else self.logical_bytes / self.raw_bytes

    @property
    def compression_ratio(self) -> float:
        return 1.0 if not self.physical_bytes else self.raw_bytes / self.physical_bytes


@dataclass(frozen=True)
class Diff:
    a: str
    b: str
    changed: tuple[str, ...]
    only_in_a: tuple[str, ...]
    only_in_b: tuple[str, ...]
    shared: int
    relative_change: float
    """||b - a||_F / ||a||_F over shared tensors, in float32."""
    max_tensor_change: float = 0.0
    """The largest per-tensor relative change.

    The aggregate hides a targeted edit: one tensor moved 3% is 0.2% of the
    whole adapter. An audit cares about exactly that kind of change, so the
    per-tensor maximum is tracked and gates `unobserved` alongside the
    aggregate.
    """
    threshold: float = 0.01
    probe_set: str | None = None
    probes_changed: int | None = None
    probes_total: int | None = None
    probes_similarity: float | None = None
    """Mean character-level similarity of the answers that changed, 0..1.

    Exact mismatch says a probe moved; this says how far. Two answers that
    differ by one token score near 1; a rewritten answer scores low.
    """

    @property
    def unobserved(self) -> bool:
        """Weights moved and no probe noticed.

        The dangerous case. A diff that reported "no change" here would be
        trusted, and the change is real. It is surfaced as its own condition so
        silence is never mistaken for stability.
        """
        if self.probes_total is None:
            return False
        moved = max(self.relative_change, self.max_tensor_change) > self.threshold
        return moved and self.probes_changed == 0

    def __str__(self) -> str:
        lines = [
            f"{self.a[:12]} -> {self.b[:12]}",
            f"  tensors: {len(self.changed)} changed of {self.shared} shared, "
            f"{len(self.only_in_a)} removed, {len(self.only_in_b)} added",
            f"  relative change: {self.relative_change:.4f} overall, {self.max_tensor_change:.4f} in the "
            f"most changed tensor",
        ]
        if self.probes_total is not None:
            similarity = (
                "" if self.probes_similarity is None else f" (similarity {self.probes_similarity:.2f})"
            )
            lines.append(f"  probes: {self.probes_changed} of {self.probes_total} moved{similarity}")
        if self.unobserved:
            lines.append("  UNOBSERVED CHANGE: weights moved, no probe detected it")
        elif self.probes_total is None and self.changed:
            lines.append("  no fingerprints on both commits; behavioural effect unknown")
        return "\n".join(lines)


@dataclass(frozen=True)
class Proof:
    tenant: str
    reason: str
    deleted_at: float
    commits: tuple[str, ...]
    manifests: tuple[str, ...]
    chunks: tuple[str, ...]
    bytes_freed: int
    broken_composites: tuple[str, ...]
    """Views that referenced a deleted manifest, as `tenant:manifest_id`."""
    revoked_grants: tuple[str, ...]
    """Grants on deleted manifests, as `grantee:manifest_id`."""
    attestation: str
    signature: str | None = None
    """HMAC-SHA256 of the attestation under the store's signing key, if one was set."""

    def __str__(self) -> str:
        lines = [
            f"tenant {self.tenant!r}: {len(self.commits)} commits, {len(self.manifests)} manifests, "
            f"{len(self.chunks)} blocks, {self.bytes_freed:,} bytes",
            f"attestation {self.attestation}" + ("  (signed)" if self.signature else "  (unsigned)"),
        ]
        if self.broken_composites:
            lines.append(
                f"{len(self.broken_composites)} composite(s) now reference a deleted input "
                f"and will refuse to resolve: {', '.join(c[:20] for c in self.broken_composites)}"
            )
        if self.revoked_grants:
            lines.append(f"{len(self.revoked_grants)} grant(s) revoked")
        return "\n".join(lines)


class BrokenView(LookupError):
    """A view that cannot produce tensors.

    One condition with several causes — an input deleted, a grant revoked, a
    strict view whose inputs stopped agreeing — because to everything that
    consumes a view they are the same event: this one cannot be served, name it
    and carry on with the others.
    """


class NotGranted(PermissionError):
    """A cross-tenant reference without a live grant."""


class Store:
    def __init__(
        self,
        root: Path | str,
        *,
        backend: Backend | str | None = None,
        metadata: str | Path | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        compression: str = "zstd",
        signing_key: bytes | str | None = None,
        cache_views: bool = True,
        unobserved_threshold: float = 0.01,
    ) -> None:
        """Open or create a store.

        `root` holds the view cache and, unless told otherwise, the metadata
        database and the blocks. `backend` is a `Backend`, a path, or an
        `s3://bucket/prefix` URL; `metadata` is a path or a `postgresql://` URL.
        `block_size` and `compression` are fixed when a store is created and
        read back on every open after that.

        `signing_key` (or `$BALLAST_SIGNING_KEY`) signs deletion proofs. Keep it
        outside the store; that is what makes the signature mean something.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._metadata = metadata or (self.root / "ballast.db")

        settings = self._settings()
        if settings:
            block_size = int(settings["block_size"])
            compression = settings["compression"]
        else:
            self._write_settings({"block_size": str(block_size), "compression": compression})

        if backend is None:
            resolved: Backend = LocalBackend(self.root / "chunks")
        elif isinstance(backend, str):
            resolved = from_url(backend, self.root / "chunks")
        else:
            resolved = backend
        self.chunks = ChunkStore(resolved, block_size=block_size, compression=compression)
        self.cache_views = cache_views
        self.unobserved_threshold = unobserved_threshold

        key = signing_key if signing_key is not None else os.environ.get(SIGNING_KEY_ENV)
        self._signing_key = key.encode() if isinstance(key, str) else key

        if self._settings().get("rehash_pending") == "1":
            self._rehash_v1_blocks()

    # -- connections --------------------------------------------------------

    @property
    def db(self) -> Database:
        """One connection per thread. SQLite connections are not shareable."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self._metadata)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    @contextmanager
    def _tx(self) -> Iterator[Database]:
        db = self.db
        db.begin()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise

    def _settings(self) -> dict[str, str]:
        rows = self.db.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def _write_settings(self, values: dict[str, str]) -> None:
        with self._tx() as db:
            db.executemany(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                list(values.items()),
            )

    def _rehash_v1_blocks(self) -> None:
        """Version-1 blocks were addressed by a hash that covered dtype and shape.

        Version 2 addresses a block by its bytes alone, so the same bytes
        deduplicate whatever tensor they belong to. Finish the migration by
        renaming every block to its new address. This reads every block once.
        """
        rows = self.db.execute("SELECT tenant, hash, nbytes FROM chunks").fetchall()
        with self._tx() as db:
            for r in rows:
                tenant, old = r["tenant"], r["hash"]
                data = self.chunks.backend.get(tenant, old)
                new = block_digest(data)
                if new == old:
                    continue
                self.chunks.backend.put(tenant, new, data)
                self.chunks.backend.delete(tenant, old)
                db.execute(
                    "UPDATE tensor_blocks SET chunk_hash = ? WHERE tenant = ? AND chunk_hash = ?",
                    (new, tenant, old),
                )
                db.execute("UPDATE chunks SET hash = ? WHERE tenant = ? AND hash = ?", (new, tenant, old))
            db.execute("DELETE FROM settings WHERE key = 'rehash_pending'")

    # -- write --------------------------------------------------------------

    def commit(
        self,
        tenant: str,
        tensors: dict[str, np.ndarray],
        *,
        message: str,
        ref: str = DEFAULT_REF,
        base_model: str | None = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: str | None = None,
    ) -> Commit:
        """Store a delta as a leaf manifest and advance `ref` to it.

        `metadata` is free-form provenance — training run, dataset hash, who
        approved it — kept on the commit, not the manifest, because two commits
        of identical tensors can have different histories.
        """
        names.tenant(tenant)
        names.ref(ref)
        if not tensors:
            raise ValueError("a commit needs at least one tensor")
        config = config or {}
        now = time.time()

        with self._tx() as db:
            known_rows = {
                r["hash"] for r in db.execute("SELECT hash FROM chunks WHERE tenant = ?", (tenant,))
            }
            records = []
            for name in sorted(tensors):
                record = self.chunks.put_tensor(tenant, tensors[name], known=known_rows.__contains__)
                for block in record.blocks:
                    if block.new:
                        db.execute(
                            "INSERT INTO chunks (tenant, hash, nbytes, stored_bytes, encoding, created_at) "
                            "VALUES (?, ?, ?, ?, ?, ?)"
                            " ON CONFLICT (tenant, hash) DO NOTHING",
                            (tenant, block.digest, block.nbytes, block.stored_bytes, block.encoding, now),
                        )
                        known_rows.add(block.digest)
                records.append((name, record))

            manifest_id = object_hash(
                {
                    "kind": "leaf",
                    "base_model": base_model,
                    "config": config,
                    "tensors": [(name, rec.identity()) for name, rec in records],
                }
            )
            fresh = db.execute(
                "INSERT INTO manifests (tenant, id, kind, base_model, config, created_at) "
                "VALUES (?, ?, 'leaf', ?, ?, ?) ON CONFLICT (tenant, id) DO NOTHING",
                (tenant, manifest_id, base_model, json.dumps(config, sort_keys=True), now),
            ).rowcount
            if fresh:
                for name, rec in records:
                    db.execute(
                        "INSERT INTO manifest_tensors (tenant, manifest_id, name, dtype, shape, nbytes) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (tenant, manifest_id, name, rec.dtype, json.dumps(list(rec.shape)), rec.nbytes),
                    )
                    db.executemany(
                        "INSERT INTO tensor_blocks (tenant, manifest_id, name, position, chunk_hash) "
                        "VALUES (?, ?, ?, ?, ?)",
                        [(tenant, manifest_id, name, i, d) for i, d in enumerate(rec.digests)],
                    )
            return self._commit_manifest(db, tenant, manifest_id, message, ref, parent, metadata or {}, now)

    def merge(
        self,
        tenant: str,
        method: str,
        inputs: Sequence[tuple[str, float]],
        *,
        message: str,
        ref: str = DEFAULT_REF,
        density: float | None = None,
        strict: bool = False,
        seed: int | None = None,
        normalize: bool | None = None,
        lambda_: float = 1.0,
        gamma: float | None = None,
        epsilon: float | None = None,
        t: float | None = None,
        extra: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: str | None = None,
    ) -> Commit:
        """Record a merge as a view over existing commits.

        Nothing is computed here. The tensors exist when something checks the
        result out, and only for as long as every input still exists.

        An input is a ref or commit id in this tenant, or `other:ref` for a
        manifest another tenant has granted.

        Inputs are merged over the union of their tensors, treating one an input
        lacks as zero. That is the right reading for deltas, which is what this
        stores: two fine-tunes of the same base rarely touch the same weights,
        and a tensor one of them never changed is a change of zero. `strict=True`
        demands they carry exactly the same tensors, which is a useful check when
        they are meant to.

        A method that draws a random mask gets a seed, generated here if none is
        given, and stored in the view. Without it the same recipe would resolve
        to different weights on every checkout, and a view that cannot be
        resolved twice is not a version of anything.
        """
        names.tenant(tenant)
        names.ref(ref)
        if method not in merging.RESOLVABLE + merging.RECORD_ONLY and not method.endswith(":slices"):
            raise ValueError(f"unknown merge method {method!r}")
        if not inputs:
            raise ValueError("a merge needs at least one input")
        now = time.time()

        resolved: list[tuple[str, str, float]] = []
        bases: set[str | None] = set()
        for spec, weight in inputs:
            owner, commit = self._resolve_input(tenant, spec)
            manifest = self.manifest(owner, commit.manifest_id)
            if owner != tenant and not self._granted(owner, manifest.id, tenant):
                raise NotGranted(f"{tenant!r} has no grant on {owner}:{manifest.id[:12]}")
            bases.add(manifest.base_model)
            resolved.append((owner, manifest.id, float(weight)))
        if len(bases) > 1:
            raise ValueError(f"inputs have different base models: {sorted(str(b) for b in bases)}")
        base_model = next(iter(bases))

        config: dict[str, Any] = {"method": method}
        if density is not None:
            config["density"] = density
        # Written either way. A view resolved years later must not depend on
        # what the default happened to be when it was recorded.
        config["strict"] = bool(strict)
        if normalize is not None:
            config["normalize"] = bool(normalize)
        if lambda_ != 1.0:
            config["lambda"] = float(lambda_)
        if gamma is not None:
            config["gamma"] = float(gamma)
        if epsilon is not None:
            config["epsilon"] = float(epsilon)
        if t is not None:
            config["t"] = float(t)
        elif method == "slerp":
            raise ValueError("slerp needs a t: how far to travel from the first input to the second")
        if extra:
            # Validated here rather than at checkout, so a view that names a
            # parameter this resolver does not have is refused when it is made.
            merging.params_for(extra=extra)
            config.update(extra)
        if provenance:
            # Recorded, not interpreted: everything the recipe said that this
            # resolver does not act on, so the view still describes its origin.
            config["provenance"] = provenance
        if method in merging.SEEDED:
            config["seed"] = secrets.randbelow(2**63) if seed is None else int(seed)
        elif seed is not None:
            raise ValueError(f"{method!r} does not draw a random mask, so it takes no seed")
        manifest_id = object_hash(
            {"kind": "composite", "base_model": base_model, "config": config, "inputs": resolved}
        )

        with self._tx() as db:
            db.execute(
                "INSERT INTO manifests (tenant, id, kind, base_model, config, created_at) "
                "VALUES (?, ?, 'composite', ?, ?, ?) ON CONFLICT (tenant, id) DO NOTHING",
                (tenant, manifest_id, base_model, json.dumps(config, sort_keys=True), now),
            )
            db.executemany(
                "INSERT INTO manifest_inputs "
                "(tenant, manifest_id, position, input_tenant, input_id, weight) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (tenant, manifest_id, position) DO NOTHING",
                [(tenant, manifest_id, i, owner, mid, w) for i, (owner, mid, w) in enumerate(resolved)],
            )
            return self._commit_manifest(db, tenant, manifest_id, message, ref, parent, metadata or {}, now)

    def _resolve_input(self, tenant: str, spec: str) -> tuple[str, Commit]:
        owner, sep, rest = spec.partition(":")
        if sep and rest:
            names.tenant(owner)
            return owner, self.resolve(owner, rest)
        return tenant, self.resolve(tenant, spec)

    def _commit_manifest(
        self,
        db: Database,
        tenant: str,
        manifest_id: str,
        message: str,
        ref: str,
        parent: str | None,
        metadata: dict[str, Any],
        now: float,
    ) -> Commit:
        head = self.head(tenant, ref)
        if parent is None:
            parent = head.id if head else None
        commit_id = object_hash(
            {"tenant": tenant, "manifest": manifest_id, "parent": parent, "message": message, "at": now}
        )
        db.execute(
            "INSERT INTO commits (tenant, id, manifest_id, parent_id, message, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tenant, commit_id, manifest_id, parent, message, json.dumps(metadata, sort_keys=True), now),
        )
        self._move_ref(db, tenant, ref, head.id if head else None, commit_id, "commit", now)
        return Commit(tenant, commit_id, manifest_id, parent, message, now, metadata)

    def _move_ref(
        self,
        db: Database,
        tenant: str,
        ref: str,
        old: str | None,
        new: str | None,
        op: str,
        now: float,
    ) -> None:
        if new is None:
            db.execute("DELETE FROM refs WHERE tenant = ? AND name = ?", (tenant, ref))
        else:
            db.execute(
                "INSERT INTO refs (tenant, name, commit_id) VALUES (?, ?, ?) "
                "ON CONFLICT (tenant, name) DO UPDATE SET commit_id = excluded.commit_id",
                (tenant, ref, new),
            )
        position = db.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM reflog WHERE tenant = ? AND ref = ?", (tenant, ref)
        ).fetchone()[0]
        db.execute(
            "INSERT INTO reflog (tenant, ref, position, old_commit, new_commit, op, at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tenant, ref, position, old, new, op, now),
        )

    def reset(self, tenant: str, ref: str, target: str) -> Commit:
        """Point `ref` at an earlier commit. Nothing is deleted; this is rollback."""
        names.tenant(tenant)
        names.ref(ref)
        commit = self.resolve(tenant, target)
        head = self.head(tenant, ref)
        with self._tx() as db:
            self._move_ref(db, tenant, ref, head.id if head else None, commit.id, "reset", time.time())
        return commit

    def reflog(self, tenant: str, ref: str, limit: int = 100) -> list[ReflogEntry]:
        """Every move `ref` made, newest first."""
        names.tenant(tenant)
        names.ref(ref)
        rows = self.db.execute(
            "SELECT * FROM reflog WHERE tenant = ? AND ref = ? ORDER BY position DESC LIMIT ?",
            (tenant, ref, limit),
        ).fetchall()
        return [
            ReflogEntry(r["ref"], r["position"], r["old_commit"], r["new_commit"], r["op"], r["at"])
            for r in rows
        ]

    # -- grants -------------------------------------------------------------

    def grant(self, owner: str, spec: str, grantee: str) -> Grant:
        """Let `grantee` build views over the manifest behind `spec`."""
        names.tenant(owner)
        names.tenant(grantee)
        if owner == grantee:
            raise ValueError("a tenant does not need a grant on its own manifests")
        commit = self.resolve(owner, spec)
        now = time.time()
        with self._tx() as db:
            db.execute(
                "INSERT INTO grants (owner, manifest_id, grantee, created_at, revoked_at) "
                "VALUES (?, ?, ?, ?, NULL) "
                "ON CONFLICT (owner, manifest_id, grantee) DO UPDATE SET created_at = excluded.created_at, "
                "revoked_at = NULL",
                (owner, commit.manifest_id, grantee, now),
            )
        return Grant(owner, commit.manifest_id, grantee, now, None)

    def revoke(self, owner: str, spec: str, grantee: str) -> list[str]:
        """Withdraw a grant. Returns the grantee's views that now cannot resolve."""
        names.tenant(owner)
        names.tenant(grantee)
        commit = self.resolve(owner, spec)
        with self._tx() as db:
            db.execute(
                "UPDATE grants SET revoked_at = ? WHERE owner = ? AND manifest_id = ? AND grantee = ? "
                "AND revoked_at IS NULL",
                (time.time(), owner, commit.manifest_id, grantee),
            )
        # Everything downstream of the revoked manifest, not only the grantee's
        # own views: a third tenant building on the grantee's view loses access
        # at the same moment, and its cache has to go with it.
        broken = self._views_over(owner, commit.manifest_id)
        self._drop_cache(broken)
        return broken

    def grants(self, tenant: str) -> list[Grant]:
        """Grants this tenant has given or received, live ones first."""
        names.tenant(tenant)
        rows = self.db.execute(
            "SELECT * FROM grants WHERE owner = ? OR grantee = ? ORDER BY revoked_at IS NOT NULL, created_at",
            (tenant, tenant),
        ).fetchall()
        return [
            Grant(r["owner"], r["manifest_id"], r["grantee"], r["created_at"], r["revoked_at"]) for r in rows
        ]

    def _granted(self, owner: str, manifest_id: str, grantee: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM grants WHERE owner = ? AND manifest_id = ? AND grantee = ? AND revoked_at IS NULL",
            (owner, manifest_id, grantee),
        ).fetchone()
        return row is not None

    def _views_over(self, owner: str, manifest_id: str, only_tenant: str | None = None) -> list[str]:
        """Composites that reference this manifest, directly or through another view.

        Transitively, because a view over a view breaks too. A cached result
        served after its grandparent's grant was revoked would be exactly the
        leak the revocation was meant to stop, so this walks forward the whole
        way rather than one hop.
        """
        rows = self.db.execute(
            "WITH RECURSIVE deps(t, m) AS ("
            "  SELECT tenant, manifest_id FROM manifest_inputs WHERE input_tenant = ? AND input_id = ?"
            "  UNION"
            "  SELECT i.tenant, i.manifest_id FROM manifest_inputs i"
            "    JOIN deps d ON i.input_tenant = d.t AND i.input_id = d.m"
            ") SELECT t, m FROM deps",
            (owner, manifest_id),
        ).fetchall()
        views = sorted(f"{r['t']}:{r['m']}" for r in rows)
        if only_tenant is None:
            return views
        return [v for v in views if v.split(":", 1)[0] == only_tenant]

    # -- read ---------------------------------------------------------------

    def head(self, tenant: str, ref: str = DEFAULT_REF) -> Commit | None:
        row = self.db.execute(
            "SELECT commit_id FROM refs WHERE tenant = ? AND name = ?", (tenant, ref)
        ).fetchone()
        return self._commit(tenant, row["commit_id"]) if row else None

    def refs(self, tenant: str) -> dict[str, str]:
        names.tenant(tenant)
        rows = self.db.execute("SELECT name, commit_id FROM refs WHERE tenant = ?", (tenant,))
        return {r["name"]: r["commit_id"] for r in rows}

    def resolve(self, tenant: str, spec: str) -> Commit:
        """A ref name, a full commit id, or an unambiguous prefix of one."""
        names.tenant(tenant)
        head = self.head(tenant, spec)
        if head:
            return head
        if names.is_hex_prefix(spec):
            rows = self.db.execute(
                "SELECT id FROM commits WHERE tenant = ? AND substr(id, 1, ?) = ?",
                (tenant, len(spec), spec),
            ).fetchall()
            if len(rows) == 1:
                return self._commit(tenant, rows[0]["id"])
            if len(rows) > 1:
                raise LookupError(f"{spec!r} is ambiguous across {len(rows)} commits")
        raise LookupError(f"no ref or commit {spec!r} for tenant {tenant!r}")

    def _commit(self, tenant: str, commit_id: str) -> Commit:
        row = self.db.execute(
            "SELECT * FROM commits WHERE tenant = ? AND id = ?", (tenant, commit_id)
        ).fetchone()
        if row is None:
            raise LookupError(f"no commit {commit_id[:12]} for tenant {tenant!r}")
        return Commit(
            row["tenant"],
            row["id"],
            row["manifest_id"],
            row["parent_id"],
            row["message"],
            row["created_at"],
            json.loads(row["metadata"]),
        )

    def manifest(self, tenant: str, manifest_id: str) -> Manifest:
        row = self.db.execute(
            "SELECT * FROM manifests WHERE tenant = ? AND id = ?", (tenant, manifest_id)
        ).fetchone()
        if row is None:
            raise BrokenView(f"manifest {manifest_id[:12]} is missing for tenant {tenant!r}")
        return Manifest(
            row["tenant"],
            row["id"],
            row["kind"],
            row["base_model"],
            json.loads(row["config"]),
            row["created_at"],
        )

    def inputs(self, tenant: str, manifest_id: str) -> list[tuple[str, str, float]]:
        """A composite's inputs as (tenant, manifest_id, weight), in position order."""
        rows = self.db.execute(
            "SELECT input_tenant, input_id, weight FROM manifest_inputs WHERE tenant = ? AND manifest_id = ? "
            "ORDER BY position",
            (tenant, manifest_id),
        ).fetchall()
        return [(r["input_tenant"], r["input_id"], r["weight"]) for r in rows]

    def tensor_blocks(self, tenant: str, manifest_id: str) -> dict[str, tuple[str, ...]]:
        """Each tensor's block digests, for a leaf manifest."""
        rows = self.db.execute(
            "SELECT name, chunk_hash FROM tensor_blocks WHERE tenant = ? AND manifest_id = ? "
            "ORDER BY name, position",
            (tenant, manifest_id),
        ).fetchall()
        out: dict[str, list[str]] = {}
        for r in rows:
            out.setdefault(r["name"], []).append(r["chunk_hash"])
        return {k: tuple(v) for k, v in out.items()}

    def log(self, tenant: str, spec: str = DEFAULT_REF, limit: int = 100) -> list[Commit]:
        out: list[Commit] = []
        current: str | None = self.resolve(tenant, spec).id
        while current and len(out) < limit:
            commit = self._commit(tenant, current)
            out.append(commit)
            current = commit.parent_id
        return out

    def checkout(self, tenant: str, spec: str) -> dict[str, np.ndarray]:
        """Materialise a commit's tensors, resolving views as needed."""
        commit = self.resolve(tenant, spec)
        return self._materialise(tenant, commit.manifest_id, viewer=tenant)

    def _materialise(self, tenant: str, manifest_id: str, viewer: str) -> dict[str, np.ndarray]:
        if tenant != viewer and not self._granted(tenant, manifest_id, viewer):
            raise BrokenView(f"grant on {tenant}:{manifest_id[:12]} for {viewer!r} is missing or revoked")
        manifest = self.manifest(tenant, manifest_id)
        if manifest.kind == "leaf":
            return self._load_leaf(tenant, manifest_id)

        cached = self._cache_path(tenant, manifest_id)
        if self.cache_views and cached.exists():
            return st.load(cached)[0]

        parts = self.inputs(tenant, manifest_id)
        try:
            # A grantee's composite reads its inputs on behalf of the grantee, so
            # the grant check applies to each cross-tenant hop, not just the first.
            resolved = [self._materialise(owner, mid, viewer=tenant) for owner, mid, _ in parts]
        except BrokenView as exc:
            raise BrokenView(f"composite {manifest_id[:12]} cannot resolve: {exc}") from exc
        weights = [w for _, _, w in parts]
        try:
            out = merging.resolve(
                manifest.config["method"],
                resolved,
                weights,
                manifest.config.get("density"),
                manifest.config.get("strict", True),
                manifest.config.get("seed"),
                manifest.config.get("normalize"),
                manifest.config.get("lambda", 1.0),
                manifest.config.get("gamma", merging.DEFAULT_GAMMA),
                manifest.config.get("epsilon", merging.DEFAULT_EPSILON),
                manifest.config.get("t"),
                {k: v for k, v in manifest.config.items() if k in merging.EXTRA_KEYS},
            )
        except ValueError as exc:
            raise BrokenView(f"composite {manifest_id[:12]} cannot resolve: {exc}") from exc
        if self.cache_views:
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".tmp")
            st.save(tmp, out)
            tmp.replace(cached)
        return out

    def _load_leaf(self, tenant: str, manifest_id: str) -> dict[str, np.ndarray]:
        tensors = self.db.execute(
            "SELECT name, dtype, shape FROM manifest_tensors WHERE tenant = ? AND manifest_id = ? "
            "ORDER BY name",
            (tenant, manifest_id),
        ).fetchall()
        blocks = self.db.execute(
            "SELECT b.name, b.chunk_hash, c.encoding, c.nbytes FROM tensor_blocks b "
            "JOIN chunks c ON c.tenant = b.tenant AND c.hash = b.chunk_hash "
            "WHERE b.tenant = ? AND b.manifest_id = ? ORDER BY b.name, b.position",
            (tenant, manifest_id),
        ).fetchall()
        by_name: dict[str, list[tuple[str, str, int]]] = {}
        for r in blocks:
            by_name.setdefault(r["name"], []).append((r["chunk_hash"], r["encoding"], r["nbytes"]))
        return {
            r["name"]: self.chunks.get_tensor(
                tenant, r["dtype"], json.loads(r["shape"]), by_name.get(r["name"], [])
            )
            for r in tensors
        }

    def _cache_path(self, tenant: str, manifest_id: str) -> Path:
        return self.root / "cache" / tenant / f"{manifest_id}.safetensors"

    def _drop_cache(self, views: Sequence[str]) -> None:
        for item in views:
            tenant, _, manifest_id = item.partition(":")
            self._cache_path(tenant, manifest_id).unlink(missing_ok=True)

    def checkout_stream(self, tenant: str, spec: str) -> Iterator[tuple[str, np.ndarray]]:
        """A commit's tensors, one at a time, in name order.

        A leaf is read block by block. A cached view is read from its file. An
        uncached view is merged tensor by tensor: for each name, that one tensor
        is fetched from each input and merged, so what is held is one tensor per
        input rather than one model per input. A stack three deep over eight-
        gigabyte deltas costs megabytes.
        """
        commit = self.resolve(tenant, spec)
        manifest = self.manifest(tenant, commit.manifest_id)
        if manifest.kind == "leaf":
            yield from self._stream_leaf(tenant, manifest.id)
            return
        cached = self._cache_path(tenant, manifest.id)
        if self.cache_views and cached.exists():
            tensors, _ = st.load(cached)
            yield from sorted(tensors.items())
            return
        yield from self._stream_composite(tenant, manifest.id, viewer=tenant, cache=self.cache_views)

    def _stream_composite(
        self, tenant: str, manifest_id: str, viewer: str, cache: bool = False
    ) -> Iterator[tuple[str, np.ndarray]]:
        """Merge a view tensor by tensor, optionally filling its cache as it goes.

        The cache is written to a temporary file and renamed only once the last
        tensor is out, so a consumer that stops half way leaves no partial file
        behind pretending to be a resolved view.
        """
        manifest = self.manifest(tenant, manifest_id)
        merging.check(manifest.config["method"])
        parts = self.inputs(tenant, manifest_id)
        weights = [w for _, _, w in parts]
        params = merging.params_for(
            manifest.config.get("density"),
            manifest.config.get("normalize"),
            manifest.config.get("lambda", 1.0),
            manifest.config.get("gamma", merging.DEFAULT_GAMMA),
            manifest.config.get("epsilon", merging.DEFAULT_EPSILON),
            manifest.config.get("seed"),
            manifest.config.get("t"),
            {k: v for k, v in manifest.config.items() if k in merging.EXTRA_KEYS},
        )
        strict = manifest.config.get("strict", True)

        try:
            per_input = [self._specs_of(owner, mid, viewer) for owner, mid, _ in parts]
        except BrokenView as exc:
            raise BrokenView(f"composite {manifest_id[:12]} cannot resolve: {exc}") from exc
        try:
            names = self._merge_names(per_input, strict)
        except ValueError as exc:
            raise BrokenView(f"composite {manifest_id[:12]} cannot resolve: {exc}") from exc

        def produce(name: str) -> np.ndarray:
            arrays = [
                self._tensor_of(owner, mid, name, viewer) if name in specs else None
                for (owner, mid, _), specs in zip(parts, per_input, strict=True)
            ]
            try:
                return merging.merge_tensor(manifest.config["method"], name, arrays, weights, params)
            except (BrokenView, ValueError) as exc:
                raise BrokenView(f"composite {manifest_id[:12]} cannot resolve: {exc}") from exc

        if not cache:
            for name in names:
                yield name, produce(name)
            return

        target = self._cache_path(tenant, manifest_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        specs = self._merged_specs(per_input, names)
        try:
            with tmp.open("wb") as handle:
                yield from st.stream_writer(handle, specs, produce)
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _merged_specs(
        per_input: Sequence[dict[str, tuple[str, tuple[int, ...]]]], names: Sequence[str]
    ) -> dict[str, tuple[str, tuple[int, ...]]]:
        merged: dict[str, tuple[str, tuple[int, ...]]] = {}
        for name in names:
            for specs in per_input:
                if name in specs:
                    merged[name] = specs[name]
                    break
        return merged

    @staticmethod
    def _merge_names(per_input: Sequence[dict[str, Any]], strict: bool) -> list[str]:
        names = set(per_input[0])
        for specs in per_input[1:]:
            if strict and set(specs) != names:
                missing = sorted(names.symmetric_difference(specs))
                raise ValueError(
                    f"inputs do not share the same tensors; differ on {missing[:5]} "
                    f"(record the view with strict=False to union)"
                )
            names |= set(specs)
        return sorted(names)

    def _specs_of(self, tenant: str, manifest_id: str, viewer: str) -> dict[str, tuple[str, tuple[int, ...]]]:
        """A manifest's tensor names, dtypes and shapes, without reading any tensor."""
        self._check_grant(tenant, manifest_id, viewer)
        manifest = self.manifest(tenant, manifest_id)
        if manifest.kind == "leaf":
            rows = self.db.execute(
                "SELECT name, dtype, shape FROM manifest_tensors WHERE tenant = ? AND manifest_id = ? "
                "ORDER BY name",
                (tenant, manifest_id),
            ).fetchall()
            return {r["name"]: (r["dtype"], tuple(json.loads(r["shape"]))) for r in rows}
        parts = self.inputs(tenant, manifest_id)
        try:
            per_input = [self._specs_of(owner, mid, tenant) for owner, mid, _ in parts]
        except BrokenView as exc:
            raise BrokenView(f"composite {manifest_id[:12]} cannot resolve: {exc}") from exc
        try:
            names = self._merge_names(per_input, manifest.config.get("strict", True))
        except ValueError as exc:
            raise BrokenView(f"composite {manifest_id[:12]} cannot resolve: {exc}") from exc
        return self._merged_specs(per_input, names)

    def _tensor_of(self, tenant: str, manifest_id: str, name: str, viewer: str) -> np.ndarray:
        """One tensor from a manifest, resolving views for that tensor alone."""
        self._check_grant(tenant, manifest_id, viewer)
        manifest = self.manifest(tenant, manifest_id)
        if manifest.kind == "leaf":
            return self._read_tensor(tenant, manifest_id, name)
        for found, array in self._stream_composite(tenant, manifest_id, viewer=tenant):
            if found == name:
                return array
        raise KeyError(f"{name!r} is not in {tenant}:{manifest_id[:12]}")

    def _read_tensor(self, tenant: str, manifest_id: str, name: str) -> np.ndarray:
        row = self.db.execute(
            "SELECT dtype, shape FROM manifest_tensors WHERE tenant = ? AND manifest_id = ? AND name = ?",
            (tenant, manifest_id, name),
        ).fetchone()
        if row is None:
            raise KeyError(f"{name!r} is not in {tenant}:{manifest_id[:12]}")
        blocks = self.db.execute(
            "SELECT b.chunk_hash, c.encoding, c.nbytes FROM tensor_blocks b "
            "JOIN chunks c ON c.tenant = b.tenant AND c.hash = b.chunk_hash "
            "WHERE b.tenant = ? AND b.manifest_id = ? AND b.name = ? ORDER BY b.position",
            (tenant, manifest_id, name),
        ).fetchall()
        return self.chunks.get_tensor(
            tenant,
            row["dtype"],
            json.loads(row["shape"]),
            [(b["chunk_hash"], b["encoding"], b["nbytes"]) for b in blocks],
        )

    def _check_grant(self, tenant: str, manifest_id: str, viewer: str) -> None:
        if tenant != viewer and not self._granted(tenant, manifest_id, viewer):
            raise BrokenView(f"grant on {tenant}:{manifest_id[:12]} for {viewer!r} is missing or revoked")

    def _stream_leaf(self, tenant: str, manifest_id: str) -> Iterator[tuple[str, np.ndarray]]:
        rows = self.db.execute(
            "SELECT name, dtype, shape FROM manifest_tensors WHERE tenant = ? AND manifest_id = ? "
            "ORDER BY name",
            (tenant, manifest_id),
        ).fetchall()
        for row in rows:
            blocks = self.db.execute(
                "SELECT b.chunk_hash, c.encoding, c.nbytes FROM tensor_blocks b "
                "JOIN chunks c ON c.tenant = b.tenant AND c.hash = b.chunk_hash "
                "WHERE b.tenant = ? AND b.manifest_id = ? AND b.name = ? ORDER BY b.position",
                (tenant, manifest_id, row["name"]),
            ).fetchall()
            yield (
                row["name"],
                self.chunks.get_tensor(
                    tenant,
                    row["dtype"],
                    json.loads(row["shape"]),
                    [(b["chunk_hash"], b["encoding"], b["nbytes"]) for b in blocks],
                ),
            )

    def specs(self, tenant: str, spec: str) -> dict[str, tuple[str, tuple[int, ...]]]:
        """Each tensor's dtype and shape, without reading any of them.

        Enough to write a file's header, which is what streaming a checkout to
        disk needs before it can start.
        """
        commit = self.resolve(tenant, spec)
        manifest = self.manifest(tenant, commit.manifest_id)
        return self._specs_of(tenant, manifest.id, viewer=tenant)

    def stats(self, tenant: str) -> Stats:
        names.tenant(tenant)
        q = self.db.execute
        commits = q("SELECT COUNT(*) FROM commits WHERE tenant = ?", (tenant,)).fetchone()[0]
        manifests = q("SELECT COUNT(*) FROM manifests WHERE tenant = ?", (tenant,)).fetchone()[0]
        chunks, physical, raw = q(
            "SELECT COUNT(*), COALESCE(SUM(stored_bytes), 0), COALESCE(SUM(nbytes), 0) "
            "FROM chunks WHERE tenant = ?",
            (tenant,),
        ).fetchone()
        logical = q(
            "SELECT COALESCE(SUM(nbytes), 0) FROM manifest_tensors WHERE tenant = ?", (tenant,)
        ).fetchone()[0]
        return Stats(commits, manifests, chunks, physical, logical, raw)

    # -- analysis -----------------------------------------------------------

    def diff(
        self, tenant: str, a: str, b: str, probe_set: str | None = None, threshold: float | None = None
    ) -> Diff:
        ca, cb = self.resolve(tenant, a), self.resolve(tenant, b)
        threshold = self.unobserved_threshold if threshold is None else threshold

        # Identical block runs mean identical bytes; skip loading those tensors.
        ma, mb = self.manifest(tenant, ca.manifest_id), self.manifest(tenant, cb.manifest_id)
        same: set[str] = set()
        if ma.kind == "leaf" and mb.kind == "leaf":
            ba, bb = self.tensor_blocks(tenant, ma.id), self.tensor_blocks(tenant, mb.id)
            same = {n for n in ba if n in bb and ba[n] == bb[n]}

        ta = self._materialise(tenant, ca.manifest_id, viewer=tenant)
        tb = self._materialise(tenant, cb.manifest_id, viewer=tenant)

        shared = sorted(set(ta) & set(tb))
        changed = []
        num = 0.0
        den = 0.0
        worst = 0.0
        for name in shared:
            x = ta[name]
            if name in same:
                den += float(np.sum(x.astype(np.float32) ** 2))
                continue
            x = x.astype(np.float32)
            y = tb[name].astype(np.float32)
            if x.shape != y.shape or not np.array_equal(x, y):
                changed.append(name)
            if x.shape == y.shape:
                d = float(np.sum((y - x) ** 2))
                n = float(np.sum(x**2))
                num += d
                den += n
                worst = max(worst, float(np.sqrt(d) / np.sqrt(n)) if n > 0 else (1.0 if d > 0 else 0.0))
        relative = float(np.sqrt(num) / np.sqrt(den)) if den > 0 else (1.0 if num > 0 else 0.0)

        probes_changed = probes_total = None
        similarity: float | None = None
        if probe_set is not None:
            fa = self._fingerprint(tenant, ca.id, probe_set)
            fb = self._fingerprint(tenant, cb.id, probe_set)
            if fa is not None and fb is not None and len(fa) == len(fb):
                probes_total = len(fa)
                moved = [(x, y) for x, y in zip(fa, fb, strict=True) if x != y]
                probes_changed = len(moved)
                if moved:
                    similarity = sum(difflib.SequenceMatcher(None, x, y).ratio() for x, y in moved) / len(
                        moved
                    )

        return Diff(
            ca.id,
            cb.id,
            tuple(changed),
            tuple(sorted(set(ta) - set(tb))),
            tuple(sorted(set(tb) - set(ta))),
            len(shared),
            relative,
            worst,
            threshold,
            probe_set,
            probes_changed,
            probes_total,
            similarity,
        )

    # -- fingerprints -------------------------------------------------------

    def record_fingerprint(
        self, tenant: str, spec: str, probes: Sequence[str], outputs: Sequence[str]
    ) -> str:
        """Store a commit's answers to a probe set. Returns the probe set id."""
        names.tenant(tenant)
        if len(probes) != len(outputs):
            raise ValueError(f"{len(outputs)} outputs for {len(probes)} probes")
        commit = self.resolve(tenant, spec)
        probe_id = object_hash(list(probes))[:16]
        with self._tx() as db:
            db.execute(
                "INSERT INTO probe_sets (id, probes, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT (id) DO NOTHING",
                (probe_id, json.dumps(list(probes)), time.time()),
            )
            db.execute(
                "DELETE FROM fingerprints WHERE tenant = ? AND commit_id = ? AND probe_set = ?",
                (tenant, commit.id, probe_id),
            )
            db.executemany(
                "INSERT INTO fingerprints (tenant, commit_id, probe_set, position, output) "
                "VALUES (?, ?, ?, ?, ?)",
                [(tenant, commit.id, probe_id, i, out) for i, out in enumerate(outputs)],
            )
        return probe_id

    def probe_set(self, probe_id: str) -> list[str]:
        row = self.db.execute("SELECT probes FROM probe_sets WHERE id = ?", (probe_id,)).fetchone()
        if row is None:
            raise LookupError(f"no probe set {probe_id!r}")
        return list(json.loads(row["probes"]))

    def fingerprint_of(self, tenant: str, spec: str, probe_id: str) -> list[str] | None:
        return self._fingerprint(tenant, self.resolve(tenant, spec).id, probe_id)

    def _fingerprint(self, tenant: str, commit_id: str, probe_set: str) -> list[str] | None:
        rows = self.db.execute(
            "SELECT output FROM fingerprints WHERE tenant = ? AND commit_id = ? AND probe_set = ? "
            "ORDER BY position",
            (tenant, commit_id, probe_set),
        ).fetchall()
        return [r["output"] for r in rows] if rows else None

    # -- deletion -----------------------------------------------------------

    def forget(self, tenant: str, reason: str) -> Proof:
        """Remove everything a tenant has and return the record of it."""
        names.tenant(tenant)
        now = time.time()
        q = self.db.execute
        commits = [r["id"] for r in q("SELECT id FROM commits WHERE tenant = ?", (tenant,))]
        manifests = [r["id"] for r in q("SELECT id FROM manifests WHERE tenant = ?", (tenant,))]
        chunks = [r["hash"] for r in q("SELECT hash FROM chunks WHERE tenant = ?", (tenant,))]
        broken = sorted(
            {
                view
                for mid in manifests
                for view in self._views_over(tenant, mid)
                if not view.startswith(f"{tenant}:")
            }
        )
        revoked = [
            f"{r['grantee']}:{r['manifest_id']}"
            for r in q(
                "SELECT grantee, manifest_id FROM grants WHERE owner = ? AND revoked_at IS NULL", (tenant,)
            )
        ]
        attestation = _attest(tenant, commits, manifests, chunks, now)

        with self._tx() as db:
            for table in (
                "fingerprints",
                "refs",
                "reflog",
                "commits",
                "manifest_inputs",
                "tensor_blocks",
                "manifest_tensors",
                "manifests",
                "chunks",
            ):
                db.execute(f"DELETE FROM {table} WHERE tenant = ?", (tenant,))  # noqa: S608
            db.execute(
                "UPDATE grants SET revoked_at = ? WHERE owner = ? AND revoked_at IS NULL", (now, tenant)
            )
            self._tombstone(db, tenant, "commit", commits, reason, now, attestation)
            self._tombstone(db, tenant, "manifest", manifests, reason, now, attestation)
            self._tombstone(db, tenant, "chunk", chunks, reason, now, attestation)
            proof = self._proof(
                tenant,
                reason,
                now,
                tuple(commits),
                tuple(manifests),
                tuple(chunks),
                0,
                tuple(broken),
                tuple(revoked),
                attestation,
            )
            self._store_proof(db, proof)
        freed = self.chunks.delete_tenant(tenant)
        shutil.rmtree(self.root / "cache" / tenant, ignore_errors=True)
        self._drop_cache(broken)
        proof = self._proof(**{**asdict(proof), "bytes_freed": freed, "signature": None})
        with self._tx() as db:
            self._store_proof(db, proof)
        return proof

    def forget_commit(self, tenant: str, spec: str, reason: str) -> Proof:
        """Remove one commit from history.

        Its children are re-parented onto its parent, refs pointing at it move
        to its parent, and its manifest goes if no other commit uses it. Any
        composite — in this tenant or a grantee's — that used that manifest as
        an input is left in place, broken, and named in the proof: a view over
        a deleted input has nothing to show.
        """
        names.tenant(tenant)
        commit = self.resolve(tenant, spec)
        now = time.time()
        q = self.db.execute

        others = q(
            "SELECT COUNT(*) FROM commits WHERE tenant = ? AND manifest_id = ? AND id != ?",
            (tenant, commit.manifest_id, commit.id),
        ).fetchone()[0]
        drop_manifest = others == 0

        broken: list[str] = []
        orphaned: list[str] = []
        revoked: list[str] = []
        if drop_manifest:
            broken = self._views_over(tenant, commit.manifest_id)
            orphaned = [
                r["chunk_hash"]
                for r in q(
                    "SELECT DISTINCT chunk_hash FROM tensor_blocks WHERE tenant = ? AND manifest_id = ? "
                    "AND chunk_hash NOT IN (SELECT chunk_hash FROM tensor_blocks "
                    "WHERE tenant = ? AND manifest_id != ?)",
                    (tenant, commit.manifest_id, tenant, commit.manifest_id),
                )
            ]
            revoked = [
                f"{r['grantee']}:{r['manifest_id']}"
                for r in q(
                    "SELECT grantee, manifest_id FROM grants WHERE owner = ? AND manifest_id = ? "
                    "AND revoked_at IS NULL",
                    (tenant, commit.manifest_id),
                )
            ]

        manifests = [commit.manifest_id] if drop_manifest else []
        attestation = _attest(tenant, [commit.id], manifests, orphaned, now)
        moved_refs = [
            r["name"]
            for r in q("SELECT name FROM refs WHERE tenant = ? AND commit_id = ?", (tenant, commit.id))
        ]

        with self._tx() as db:
            db.execute(
                "UPDATE commits SET parent_id = ? WHERE tenant = ? AND parent_id = ?",
                (commit.parent_id, tenant, commit.id),
            )
            for ref in moved_refs:
                self._move_ref(db, tenant, ref, commit.id, commit.parent_id, "forget", now)
            db.execute("DELETE FROM commits WHERE tenant = ? AND id = ?", (tenant, commit.id))
            if drop_manifest:
                db.execute("DELETE FROM manifests WHERE tenant = ? AND id = ?", (tenant, commit.manifest_id))
                db.executemany(
                    "DELETE FROM chunks WHERE tenant = ? AND hash = ?", [(tenant, h) for h in orphaned]
                )
                db.execute(
                    "UPDATE grants SET revoked_at = ? WHERE owner = ? AND manifest_id = ? "
                    "AND revoked_at IS NULL",
                    (now, tenant, commit.manifest_id),
                )
            self._tombstone(db, tenant, "commit", [commit.id], reason, now, attestation)
            self._tombstone(db, tenant, "manifest", manifests, reason, now, attestation)
            self._tombstone(db, tenant, "chunk", orphaned, reason, now, attestation)
            freed = sum(self.chunks.delete(tenant, h) for h in orphaned)
            proof = self._proof(
                tenant,
                reason,
                now,
                (commit.id,),
                tuple(manifests),
                tuple(orphaned),
                freed,
                tuple(broken),
                tuple(revoked),
                attestation,
            )
            self._store_proof(db, proof)
        if drop_manifest:
            self._cache_path(tenant, commit.manifest_id).unlink(missing_ok=True)
        self._drop_cache(broken)
        return proof

    def _proof(self, *args: Any, **kwargs: Any) -> Proof:
        proof = Proof(*args, **kwargs)
        if self._signing_key and not proof.signature:
            proof = Proof(**{**asdict(proof), "signature": self._sign(proof.attestation)})
        return proof

    def _sign(self, attestation: str) -> str:
        if self._signing_key is None:
            raise RuntimeError("no signing key configured")
        return hmac.new(self._signing_key, attestation.encode(), hashlib.sha256).hexdigest()

    def proof(self, attestation: str) -> Proof:
        row = self.db.execute(
            "SELECT body, signature FROM proofs WHERE attestation = ?", (attestation,)
        ).fetchone()
        if row is None:
            raise LookupError(f"no proof with attestation {attestation[:12]}")
        body = json.loads(row["body"])
        return Proof(
            tenant=str(body["tenant"]),
            reason=str(body["reason"]),
            deleted_at=float(body["deleted_at"]),
            commits=tuple(body["commits"]),
            manifests=tuple(body["manifests"]),
            chunks=tuple(body["chunks"]),
            bytes_freed=int(body["bytes_freed"]),
            broken_composites=tuple(body["broken_composites"]),
            revoked_grants=tuple(body["revoked_grants"]),
            attestation=str(body["attestation"]),
            signature=row["signature"],
        )

    def verify(self, proof: Proof | str) -> list[str]:
        """Re-check a proof against the store. Empty list means it holds.

        Takes the proof or its attestation; the attestation alone is enough to
        verify from another process, which is the point of storing it. With a
        signing key configured, the signature is checked too, and an unsigned
        proof is reported.
        """
        if isinstance(proof, str):
            proof = self.proof(proof)
        q = self.db.execute
        problems: list[str] = []
        for cid in proof.commits:
            if q("SELECT 1 FROM commits WHERE tenant = ? AND id = ?", (proof.tenant, cid)).fetchone():
                problems.append(f"commit {cid[:12]} still present")
        for mid in proof.manifests:
            if q("SELECT 1 FROM manifests WHERE tenant = ? AND id = ?", (proof.tenant, mid)).fetchone():
                problems.append(f"manifest {mid[:12]} still present")
        for h in proof.chunks:
            if q("SELECT 1 FROM chunks WHERE tenant = ? AND hash = ?", (proof.tenant, h)).fetchone():
                problems.append(f"block {h[:12]} still indexed")
            if self.chunks.exists(proof.tenant, h):
                problems.append(f"block {h[:12]} still in the backend")
        for kind, ids in (("commit", proof.commits), ("manifest", proof.manifests), ("chunk", proof.chunks)):
            for item in ids:
                row = q(
                    "SELECT attestation FROM tombstones WHERE tenant = ? AND kind = ? AND id = ?",
                    (proof.tenant, kind, item),
                ).fetchone()
                if row is None or row["attestation"] != proof.attestation:
                    problems.append(f"{kind} {item[:12]} has no matching tombstone")
        expected = _attest(
            proof.tenant, list(proof.commits), list(proof.manifests), list(proof.chunks), proof.deleted_at
        )
        if expected != proof.attestation:
            problems.append("attestation does not match the proof's own contents")
        if self._signing_key:
            if not proof.signature:
                problems.append("proof is unsigned but a signing key is configured")
            elif not hmac.compare_digest(proof.signature, self._sign(proof.attestation)):
                problems.append("signature does not match the configured key")
        elif proof.signature:
            problems.append("proof is signed but no signing key is configured to check it")
        return problems

    def gc(self, tenant: str) -> int:
        """Drop blocks no tensor references and cache entries with no manifest. Returns bytes freed."""
        names.tenant(tenant)
        rows = self.db.execute(
            "SELECT hash FROM chunks WHERE tenant = ? AND hash NOT IN "
            "(SELECT chunk_hash FROM tensor_blocks WHERE tenant = ?)",
            (tenant, tenant),
        ).fetchall()
        freed = 0
        with self._tx() as db:
            for r in rows:
                db.execute("DELETE FROM chunks WHERE tenant = ? AND hash = ?", (tenant, r["hash"]))
                freed += self.chunks.delete(tenant, r["hash"])
        cache_dir = self.root / "cache" / tenant
        if cache_dir.exists():
            live = {r["id"] for r in self.db.execute("SELECT id FROM manifests WHERE tenant = ?", (tenant,))}
            for file in cache_dir.glob("*.safetensors"):
                if file.stem not in live:
                    freed += file.stat().st_size
                    file.unlink()
        return freed

    # -- integrity ----------------------------------------------------------

    def fsck(self, tenant: str | None = None, verify_bytes: bool = True) -> list[str]:
        """Check the store's invariants. Empty list means it is consistent.

        With `verify_bytes`, every block is re-read and re-hashed; that is the
        check that catches silent corruption. Without it, only the graph is
        checked.
        """
        q = self.db.execute
        scope = "WHERE tenant = ?" if tenant else ""
        params: tuple[Any, ...] = (tenant,) if tenant else ()
        problems: list[str] = []

        for r in q(f"SELECT tenant, hash, encoding, nbytes FROM chunks {scope}", params):  # noqa: S608
            if not self.chunks.exists(r["tenant"], r["hash"]):
                problems.append(f"{r['tenant']}: block {r['hash'][:12]} indexed but missing from the backend")
            elif verify_bytes and not self.chunks.verify_block(
                r["tenant"], r["hash"], r["encoding"], r["nbytes"]
            ):
                problems.append(f"{r['tenant']}: block {r['hash'][:12]} does not hash to its name")

        for r in q(
            f"SELECT b.tenant, b.manifest_id, b.name FROM tensor_blocks b "  # noqa: S608
            f"LEFT JOIN chunks c ON c.tenant = b.tenant AND c.hash = b.chunk_hash "
            f"{scope.replace('tenant', 'b.tenant')} {'AND' if scope else 'WHERE'} c.hash IS NULL",
            params,
        ):
            problems.append(
                f"{r['tenant']}: manifest {r['manifest_id'][:12]} tensor {r['name']!r} has a missing block"
            )

        for r in q(
            f"SELECT t.tenant, t.manifest_id, t.name, t.nbytes, "  # noqa: S608
            f"COALESCE((SELECT SUM(c.nbytes) FROM tensor_blocks b JOIN chunks c "
            f"ON c.tenant = b.tenant AND c.hash = b.chunk_hash "
            f"WHERE b.tenant = t.tenant AND b.manifest_id = t.manifest_id AND b.name = t.name), 0) AS have "
            f"FROM manifest_tensors t {scope.replace('tenant', 't.tenant')}",
            params,
        ):
            if r["have"] != r["nbytes"]:
                problems.append(
                    f"{r['tenant']}: manifest {r['manifest_id'][:12]} tensor {r['name']!r} "
                    f"has {r['have']} bytes of blocks for {r['nbytes']} declared"
                )

        for r in q(
            f"SELECT r.tenant, r.name FROM refs r "  # noqa: S608
            f"LEFT JOIN commits c ON c.tenant = r.tenant AND c.id = r.commit_id "
            f"{scope.replace('tenant', 'r.tenant')} {'AND' if scope else 'WHERE'} c.id IS NULL",
            params,
        ):
            problems.append(f"{r['tenant']}: ref {r['name']!r} points at a missing commit")

        for r in q(
            f"SELECT c.tenant, c.id FROM commits c "  # noqa: S608
            f"LEFT JOIN manifests m ON m.tenant = c.tenant AND m.id = c.manifest_id "
            f"{scope.replace('tenant', 'c.tenant')} {'AND' if scope else 'WHERE'} m.id IS NULL",
            params,
        ):
            problems.append(f"{r['tenant']}: commit {r['id'][:12]} has no manifest")

        for r in q(
            f"SELECT tenant, manifest_id, input_tenant, input_id FROM manifest_inputs {scope}",  # noqa: S608
            params,
        ):
            exists = q(
                "SELECT 1 FROM manifests WHERE tenant = ? AND id = ?", (r["input_tenant"], r["input_id"])
            ).fetchone()
            if not exists:
                problems.append(
                    f"{r['tenant']}: composite {r['manifest_id'][:12]} references deleted input "
                    f"{r['input_tenant']}:{r['input_id'][:12]} (broken view)"
                )
            elif r["input_tenant"] != r["tenant"] and not self._granted(
                r["input_tenant"], r["input_id"], r["tenant"]
            ):
                problems.append(
                    f"{r['tenant']}: composite {r['manifest_id'][:12]} references "
                    f"{r['input_tenant']}:{r['input_id'][:12]} without a live grant"
                )

        return problems

    def _store_proof(self, db: Database, proof: Proof) -> None:
        body = {k: v for k, v in asdict(proof).items() if k != "signature"}
        db.execute(
            "INSERT INTO proofs (attestation, tenant, body, signature, created_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (attestation) DO UPDATE SET body = excluded.body, signature = excluded.signature",
            (
                proof.attestation,
                proof.tenant,
                json.dumps(body, sort_keys=True),
                proof.signature,
                proof.deleted_at,
            ),
        )

    def _tombstone(
        self,
        db: Database,
        tenant: str,
        kind: str,
        ids: Sequence[str],
        reason: str,
        now: float,
        attestation: str,
    ) -> None:
        db.executemany(
            "INSERT INTO tombstones (tenant, kind, id, reason, deleted_at, attestation) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (tenant, kind, id) DO UPDATE SET "
            "reason = excluded.reason, deleted_at = excluded.deleted_at, attestation = excluded.attestation",
            [(tenant, kind, item, reason, now, attestation) for item in ids],
        )


def _attest(
    tenant: str, commits: Sequence[str], manifests: Sequence[str], chunks: Sequence[str], at: float
) -> str:
    return object_hash(
        {
            "tenant": tenant,
            "commits": sorted(commits),
            "manifests": sorted(manifests),
            "chunks": sorted(chunks),
            "at": at,
        }
    )
