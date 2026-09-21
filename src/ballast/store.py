"""The store.

A tenant is a namespace. Inside it: chunks, manifests, commits, refs. A manifest
is either a leaf (names tensors) or a composite (names other manifests and a
merge method). Commits chain manifests into a history; refs name commits.

Three decisions shape everything else.

Merges are views. A composite manifest holds no tensors; it resolves at
checkout. Deleting one of its inputs breaks it in a way the store can detect
and report, instead of leaving the input's contribution smeared through a
materialised matrix where nothing can find it.

Identity is content. Manifest and chunk ids are hashes of what they contain,
so committing the same adapter twice stores nothing new, a commit that changes
three tensors out of sixty stores three chunks, and a composite can never form a
cycle because its id depends on inputs that already exist.

Deletion produces a record. Forgetting a tenant or a commit returns a proof
naming what was removed under an attestation hash, stores the proof, and can
re-verify it from any process afterwards. It is an audit record, not a
cryptographic guarantee against a hostile operator; it answers "show me that it
is gone" for an operator acting in good faith.

Tenants may share. A grant lets one tenant build views over another's manifest.
The owner's chunks never move — the grantee reads through the grant — so
revoking it, or the owner deleting the manifest, breaks the grantee's views and
leaves nothing behind. Without a grant, a cross-tenant reference is refused.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ballast import merge as merging
from ballast import names
from ballast.chunks import ChunkStore, describe
from ballast.db import connect
from ballast.hashing import object_hash, tensor_hash

DEFAULT_REF = "main"


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
class Stats:
    commits: int
    manifests: int
    chunks: int
    physical_bytes: int
    logical_bytes: int

    @property
    def dedup_ratio(self) -> float:
        return 1.0 if not self.physical_bytes else self.logical_bytes / self.physical_bytes


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
    probe_set: str | None = None
    probes_changed: int | None = None
    probes_total: int | None = None

    @property
    def unobserved(self) -> bool:
        """Weights moved and no probe noticed.

        The dangerous case. A diff that reported "no change" here would be
        trusted, and the change is real. It is surfaced as its own condition so
        silence is never mistaken for stability.
        """
        if self.probes_total is None:
            return False
        moved = max(self.relative_change, self.max_tensor_change) > 0.01
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
            lines.append(f"  probes: {self.probes_changed} of {self.probes_total} moved")
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

    def __str__(self) -> str:
        lines = [
            f"tenant {self.tenant!r}: {len(self.commits)} commits, {len(self.manifests)} manifests, "
            f"{len(self.chunks)} chunks, {self.bytes_freed:,} bytes",
            f"attestation {self.attestation}",
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
    """A composite whose input has been deleted, or whose grant was revoked."""


class NotGranted(PermissionError):
    """A cross-tenant reference without a live grant."""


class Store:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = connect(self.root / "ballast.db")
        self.chunks = ChunkStore(self.root / "chunks")

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield self.db
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

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

        entries = []
        with self._tx() as db:
            for name in sorted(tensors):
                array = tensors[name]
                digest = tensor_hash(array)
                meta = describe(array)
                self.chunks.put(tenant, digest, array)
                db.execute(
                    "INSERT OR IGNORE INTO chunks (tenant, hash, dtype, shape, nbytes, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (tenant, digest, meta["dtype"], meta["shape"], meta["nbytes"], now),
                )
                entries.append((name, digest, meta["dtype"], meta["shape"]))

            manifest_id = object_hash(
                {"kind": "leaf", "base_model": base_model, "config": config, "tensors": entries}
            )
            db.execute(
                "INSERT OR IGNORE INTO manifests (tenant, id, kind, base_model, config, created_at) "
                "VALUES (?, ?, 'leaf', ?, ?, ?)",
                (tenant, manifest_id, base_model, json.dumps(config, sort_keys=True), now),
            )
            db.executemany(
                "INSERT OR IGNORE INTO manifest_tensors (tenant, manifest_id, name, chunk_hash) "
                "VALUES (?, ?, ?, ?)",
                [(tenant, manifest_id, name, digest) for name, digest, _, _ in entries],
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
        metadata: dict[str, Any] | None = None,
        parent: str | None = None,
    ) -> Commit:
        """Record a merge as a view over existing commits.

        Nothing is computed here. The tensors exist when something checks the
        result out, and only for as long as every input still exists.

        An input is a ref or commit id in this tenant, or `other:ref` for a
        manifest another tenant has granted.
        """
        names.tenant(tenant)
        names.ref(ref)
        if method not in merging.RESOLVABLE + merging.RECORD_ONLY:
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
        manifest_id = object_hash(
            {"kind": "composite", "base_model": base_model, "config": config, "inputs": resolved}
        )

        with self._tx() as db:
            db.execute(
                "INSERT OR IGNORE INTO manifests (tenant, id, kind, base_model, config, created_at) "
                "VALUES (?, ?, 'composite', ?, ?, ?)",
                (tenant, manifest_id, base_model, json.dumps(config, sort_keys=True), now),
            )
            db.executemany(
                "INSERT OR IGNORE INTO manifest_inputs "
                "(tenant, manifest_id, position, input_tenant, input_id, weight) VALUES (?, ?, ?, ?, ?, ?)",
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
        db: sqlite3.Connection,
        tenant: str,
        manifest_id: str,
        message: str,
        ref: str,
        parent: str | None,
        metadata: dict[str, Any],
        now: float,
    ) -> Commit:
        if parent is None:
            head = self.head(tenant, ref)
            parent = head.id if head else None
        commit_id = object_hash(
            {"tenant": tenant, "manifest": manifest_id, "parent": parent, "message": message, "at": now}
        )
        db.execute(
            "INSERT INTO commits (tenant, id, manifest_id, parent_id, message, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (tenant, commit_id, manifest_id, parent, message, json.dumps(metadata, sort_keys=True), now),
        )
        db.execute(
            "INSERT INTO refs (tenant, name, commit_id) VALUES (?, ?, ?) "
            "ON CONFLICT (tenant, name) DO UPDATE SET commit_id = excluded.commit_id",
            (tenant, ref, commit_id),
        )
        return Commit(tenant, commit_id, manifest_id, parent, message, now, metadata)

    def reset(self, tenant: str, ref: str, target: str) -> Commit:
        """Point `ref` at an earlier commit. Nothing is deleted; this is rollback."""
        names.tenant(tenant)
        names.ref(ref)
        commit = self.resolve(tenant, target)
        with self._tx() as db:
            db.execute(
                "INSERT INTO refs (tenant, name, commit_id) VALUES (?, ?, ?) "
                "ON CONFLICT (tenant, name) DO UPDATE SET commit_id = excluded.commit_id",
                (tenant, ref, commit.id),
            )
        return commit

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
        return self._views_over(owner, commit.manifest_id, only_tenant=grantee)

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
        """Composites anywhere that reference this manifest, as tenant:manifest_id."""
        query = (
            "SELECT DISTINCT tenant, manifest_id FROM manifest_inputs WHERE input_tenant = ? AND input_id = ?"
        )
        params: tuple[Any, ...] = (owner, manifest_id)
        if only_tenant is not None:
            query += " AND tenant = ?"
            params += (only_tenant,)
        return [f"{r['tenant']}:{r['manifest_id']}" for r in self.db.execute(query, params)]

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
            rows = self.db.execute(
                "SELECT t.name, t.chunk_hash, c.dtype, c.shape FROM manifest_tensors t "
                "JOIN chunks c ON c.tenant = t.tenant AND c.hash = t.chunk_hash "
                "WHERE t.tenant = ? AND t.manifest_id = ? ORDER BY t.name",
                (tenant, manifest_id),
            ).fetchall()
            return {
                r["name"]: self.chunks.get(tenant, r["chunk_hash"], r["dtype"], json.loads(r["shape"]))
                for r in rows
            }

        parts = self.inputs(tenant, manifest_id)
        try:
            # A grantee's composite reads its inputs on behalf of the grantee, so
            # the grant check applies to each cross-tenant hop, not just the first.
            resolved = [self._materialise(owner, mid, viewer=tenant) for owner, mid, _ in parts]
        except BrokenView as exc:
            raise BrokenView(f"composite {manifest_id[:12]} cannot resolve: {exc}") from exc
        weights = [w for _, _, w in parts]
        return merging.resolve(manifest.config["method"], resolved, weights, manifest.config.get("density"))

    def stats(self, tenant: str) -> Stats:
        names.tenant(tenant)
        q = self.db.execute
        commits = q("SELECT COUNT(*) FROM commits WHERE tenant = ?", (tenant,)).fetchone()[0]
        manifests = q("SELECT COUNT(*) FROM manifests WHERE tenant = ?", (tenant,)).fetchone()[0]
        chunks, physical = q(
            "SELECT COUNT(*), COALESCE(SUM(nbytes), 0) FROM chunks WHERE tenant = ?", (tenant,)
        ).fetchone()
        logical = q(
            "SELECT COALESCE(SUM(c.nbytes), 0) FROM manifest_tensors t "
            "JOIN chunks c ON c.tenant = t.tenant AND c.hash = t.chunk_hash WHERE t.tenant = ?",
            (tenant,),
        ).fetchone()[0]
        return Stats(commits, manifests, chunks, physical, logical)

    # -- analysis -----------------------------------------------------------

    def diff(self, tenant: str, a: str, b: str, probe_set: str | None = None) -> Diff:
        ca, cb = self.resolve(tenant, a), self.resolve(tenant, b)
        ta = self._materialise(tenant, ca.manifest_id, viewer=tenant)
        tb = self._materialise(tenant, cb.manifest_id, viewer=tenant)

        shared = sorted(set(ta) & set(tb))
        changed = []
        num = 0.0
        den = 0.0
        worst = 0.0
        for name in shared:
            x = ta[name].astype(np.float32)
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
        if probe_set is not None:
            fa = self._fingerprint(tenant, ca.id, probe_set)
            fb = self._fingerprint(tenant, cb.id, probe_set)
            if fa is not None and fb is not None and len(fa) == len(fb):
                probes_total = len(fa)
                probes_changed = sum(1 for x, y in zip(fa, fb, strict=True) if x != y)

        return Diff(
            ca.id,
            cb.id,
            tuple(changed),
            tuple(sorted(set(ta) - set(tb))),
            tuple(sorted(set(tb) - set(ta))),
            len(shared),
            relative,
            worst,
            probe_set,
            probes_changed,
            probes_total,
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
                "INSERT OR IGNORE INTO probe_sets (id, probes, created_at) VALUES (?, ?, ?)",
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
                f"{r['tenant']}:{r['manifest_id']}"
                for r in q(
                    "SELECT DISTINCT tenant, manifest_id FROM manifest_inputs "
                    "WHERE input_tenant = ? AND tenant != ?",
                    (tenant, tenant),
                )
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
                "commits",
                "manifest_inputs",
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
            proof = Proof(
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
        proof = Proof(**{**asdict(proof), "bytes_freed": freed})
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
                    "SELECT chunk_hash FROM manifest_tensors WHERE tenant = ? AND manifest_id = ? "
                    "AND chunk_hash NOT IN (SELECT chunk_hash FROM manifest_tensors "
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

        with self._tx() as db:
            db.execute(
                "UPDATE commits SET parent_id = ? WHERE tenant = ? AND parent_id = ?",
                (commit.parent_id, tenant, commit.id),
            )
            if commit.parent_id:
                db.execute(
                    "UPDATE refs SET commit_id = ? WHERE tenant = ? AND commit_id = ?",
                    (commit.parent_id, tenant, commit.id),
                )
            else:
                db.execute("DELETE FROM refs WHERE tenant = ? AND commit_id = ?", (tenant, commit.id))
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
            proof = Proof(
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
        return proof

    def proof(self, attestation: str) -> Proof:
        row = self.db.execute("SELECT body FROM proofs WHERE attestation = ?", (attestation,)).fetchone()
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
        )

    def verify(self, proof: Proof | str) -> list[str]:
        """Re-check a proof against the store. Empty list means it holds.

        Takes the proof or its attestation; the attestation alone is enough to
        verify from another process, which is the point of storing it.
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
                problems.append(f"chunk {h[:12]} still indexed")
            if self.chunks.exists(proof.tenant, h):
                problems.append(f"chunk {h[:12]} still on disk")
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
        return problems

    def gc(self, tenant: str) -> int:
        """Drop chunks no manifest references. Returns bytes freed."""
        names.tenant(tenant)
        rows = self.db.execute(
            "SELECT hash FROM chunks WHERE tenant = ? AND hash NOT IN "
            "(SELECT chunk_hash FROM manifest_tensors WHERE tenant = ?)",
            (tenant, tenant),
        ).fetchall()
        freed = 0
        with self._tx() as db:
            for r in rows:
                db.execute("DELETE FROM chunks WHERE tenant = ? AND hash = ?", (tenant, r["hash"]))
                freed += self.chunks.delete(tenant, r["hash"])
        return freed

    # -- integrity ----------------------------------------------------------

    def fsck(self, tenant: str | None = None, verify_bytes: bool = True) -> list[str]:
        """Check the store's invariants. Empty list means it is consistent.

        With `verify_bytes`, every chunk is re-hashed; that reads every byte and
        is the check that catches silent disk corruption. Without it, only the
        graph is checked.
        """
        q = self.db.execute
        scope = "WHERE tenant = ?" if tenant else ""
        params: tuple[Any, ...] = (tenant,) if tenant else ()
        problems: list[str] = []

        for r in q(f"SELECT tenant, hash, dtype, shape FROM chunks {scope}", params):  # noqa: S608
            if not self.chunks.exists(r["tenant"], r["hash"]):
                problems.append(f"{r['tenant']}: chunk {r['hash'][:12]} indexed but missing on disk")
            elif verify_bytes and not self.chunks.verify(
                r["tenant"], r["hash"], r["dtype"], json.loads(r["shape"])
            ):
                problems.append(f"{r['tenant']}: chunk {r['hash'][:12]} does not hash to its name")

        for r in q(
            f"SELECT t.tenant, t.manifest_id, t.name FROM manifest_tensors t "  # noqa: S608
            f"LEFT JOIN chunks c ON c.tenant = t.tenant AND c.hash = t.chunk_hash "
            f"{scope.replace('tenant', 't.tenant')} {'AND' if scope else 'WHERE'} c.hash IS NULL",
            params,
        ):
            problems.append(
                f"{r['tenant']}: manifest {r['manifest_id'][:12]} tensor {r['name']!r} has no chunk"
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

    def _store_proof(self, db: sqlite3.Connection, proof: Proof) -> None:
        db.execute(
            "INSERT INTO proofs (attestation, tenant, body, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (attestation) DO UPDATE SET body = excluded.body",
            (proof.attestation, proof.tenant, json.dumps(asdict(proof), sort_keys=True), proof.deleted_at),
        )

    def _tombstone(
        self,
        db: sqlite3.Connection,
        tenant: str,
        kind: str,
        ids: Sequence[str],
        reason: str,
        now: float,
        attestation: str,
    ) -> None:
        db.executemany(
            "INSERT OR REPLACE INTO tombstones (tenant, kind, id, reason, deleted_at, attestation) "
            "VALUES (?, ?, ?, ?, ?, ?)",
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
