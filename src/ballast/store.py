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
naming what was removed under an attestation hash, and the proof can be
re-verified against the store afterwards. It is an audit record, not a
cryptographic guarantee against a hostile operator; it answers "show me that it
is gone" for an operator acting in good faith.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ballast import merge as merging
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

    @property
    def short(self) -> str:
        return self.id[:12]


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
        return self.relative_change > 0.01 and self.probes_changed == 0

    def __str__(self) -> str:
        lines = [
            f"{self.a[:12]} -> {self.b[:12]}",
            f"  tensors: {len(self.changed)} changed of {self.shared} shared, "
            f"{len(self.only_in_a)} removed, {len(self.only_in_b)} added",
            f"  relative change: {self.relative_change:.4f}",
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
                f"and will refuse to resolve"
            )
        return "\n".join(lines)


class BrokenView(LookupError):
    """A composite whose input has been deleted."""


class Store:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = connect(self.root / "ballast.db")
        self.chunks = ChunkStore(self.root / "chunks")

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
        parent: str | None = None,
    ) -> Commit:
        """Store a delta as a leaf manifest and advance `ref` to it."""
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
            return self._commit_manifest(db, tenant, manifest_id, message, ref, parent, now)

    def merge(
        self,
        tenant: str,
        method: str,
        inputs: Sequence[tuple[str, float]],
        *,
        message: str,
        ref: str = DEFAULT_REF,
        density: float | None = None,
        parent: str | None = None,
    ) -> Commit:
        """Record a merge as a view over existing commits.

        Nothing is computed here. The tensors exist when something checks the
        result out, and only for as long as every input still exists.
        """
        if method not in merging.RESOLVABLE + merging.RECORD_ONLY:
            raise ValueError(f"unknown merge method {method!r}")
        if not inputs:
            raise ValueError("a merge needs at least one input")
        now = time.time()

        resolved = [(self.resolve(tenant, spec), weight) for spec, weight in inputs]
        bases = {self.manifest(tenant, c.manifest_id).base_model for c, _ in resolved}
        if len(bases) > 1:
            raise ValueError(f"inputs have different base models: {sorted(str(b) for b in bases)}")
        base_model = next(iter(bases))

        config: dict[str, Any] = {"method": method}
        if density is not None:
            config["density"] = density
        rows = [(c.manifest_id, float(weight)) for c, weight in resolved]
        manifest_id = object_hash(
            {"kind": "composite", "base_model": base_model, "config": config, "inputs": rows}
        )

        with self._tx() as db:
            db.execute(
                "INSERT OR IGNORE INTO manifests (tenant, id, kind, base_model, config, created_at) "
                "VALUES (?, ?, 'composite', ?, ?, ?)",
                (tenant, manifest_id, base_model, json.dumps(config, sort_keys=True), now),
            )
            db.executemany(
                "INSERT OR IGNORE INTO manifest_inputs (tenant, manifest_id, position, input_id, weight) "
                "VALUES (?, ?, ?, ?, ?)",
                [(tenant, manifest_id, i, mid, w) for i, (mid, w) in enumerate(rows)],
            )
            return self._commit_manifest(db, tenant, manifest_id, message, ref, parent, now)

    def _commit_manifest(
        self,
        db: sqlite3.Connection,
        tenant: str,
        manifest_id: str,
        message: str,
        ref: str,
        parent: str | None,
        now: float,
    ) -> Commit:
        if parent is None:
            head = self.head(tenant, ref)
            parent = head.id if head else None
        commit_id = object_hash(
            {"tenant": tenant, "manifest": manifest_id, "parent": parent, "message": message, "at": now}
        )
        db.execute(
            "INSERT INTO commits (tenant, id, manifest_id, parent_id, message, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (tenant, commit_id, manifest_id, parent, message, now),
        )
        db.execute(
            "INSERT INTO refs (tenant, name, commit_id) VALUES (?, ?, ?) "
            "ON CONFLICT (tenant, name) DO UPDATE SET commit_id = excluded.commit_id",
            (tenant, ref, commit_id),
        )
        return Commit(tenant, commit_id, manifest_id, parent, message, now)

    def reset(self, tenant: str, ref: str, target: str) -> Commit:
        """Point `ref` at an earlier commit. Nothing is deleted; this is rollback."""
        commit = self.resolve(tenant, target)
        with self._tx() as db:
            db.execute(
                "INSERT INTO refs (tenant, name, commit_id) VALUES (?, ?, ?) "
                "ON CONFLICT (tenant, name) DO UPDATE SET commit_id = excluded.commit_id",
                (tenant, ref, commit.id),
            )
        return commit

    # -- read ---------------------------------------------------------------

    def head(self, tenant: str, ref: str = DEFAULT_REF) -> Commit | None:
        row = self.db.execute(
            "SELECT commit_id FROM refs WHERE tenant = ? AND name = ?", (tenant, ref)
        ).fetchone()
        return self._commit(tenant, row["commit_id"]) if row else None

    def refs(self, tenant: str) -> dict[str, str]:
        rows = self.db.execute("SELECT name, commit_id FROM refs WHERE tenant = ?", (tenant,))
        return {r["name"]: r["commit_id"] for r in rows}

    def resolve(self, tenant: str, spec: str) -> Commit:
        """A ref name, a full commit id, or an unambiguous prefix of one."""
        head = self.head(tenant, spec)
        if head:
            return head
        rows = self.db.execute(
            "SELECT id FROM commits WHERE tenant = ? AND id LIKE ?", (tenant, spec + "%")
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
            row["tenant"], row["id"], row["manifest_id"], row["parent_id"], row["message"], row["created_at"]
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
        return self._materialise(tenant, commit.manifest_id)

    def _materialise(self, tenant: str, manifest_id: str) -> dict[str, np.ndarray]:
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

        rows = self.db.execute(
            "SELECT input_id, weight FROM manifest_inputs WHERE tenant = ? AND manifest_id = ? "
            "ORDER BY position",
            (tenant, manifest_id),
        ).fetchall()
        try:
            inputs = [self._materialise(tenant, r["input_id"]) for r in rows]
        except BrokenView as exc:
            raise BrokenView(f"composite {manifest_id[:12]} cannot resolve: {exc}") from exc
        weights = [r["weight"] for r in rows]
        return merging.resolve(manifest.config["method"], inputs, weights, manifest.config.get("density"))

    def stats(self, tenant: str) -> Stats:
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
        ta, tb = self._materialise(tenant, ca.manifest_id), self._materialise(tenant, cb.manifest_id)

        shared = sorted(set(ta) & set(tb))
        changed = []
        num = 0.0
        den = 0.0
        for name in shared:
            x = ta[name].astype(np.float32)
            y = tb[name].astype(np.float32)
            if x.shape != y.shape or not np.array_equal(x, y):
                changed.append(name)
            if x.shape == y.shape:
                num += float(np.sum((y - x) ** 2))
                den += float(np.sum(x**2))
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
            probe_set,
            probes_changed,
            probes_total,
        )

    # -- fingerprints -------------------------------------------------------

    def record_fingerprint(self, tenant: str, spec: str, probe_set: str, outputs: Sequence[str]) -> None:
        commit = self.resolve(tenant, spec)
        with self._tx() as db:
            db.execute(
                "DELETE FROM fingerprints WHERE tenant = ? AND commit_id = ? AND probe_set = ?",
                (tenant, commit.id, probe_set),
            )
            db.executemany(
                "INSERT INTO fingerprints (tenant, commit_id, probe_set, position, output) "
                "VALUES (?, ?, ?, ?, ?)",
                [(tenant, commit.id, probe_set, i, out) for i, out in enumerate(outputs)],
            )

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
        now = time.time()
        q = self.db.execute
        commits = [r["id"] for r in q("SELECT id FROM commits WHERE tenant = ?", (tenant,))]
        manifests = [r["id"] for r in q("SELECT id FROM manifests WHERE tenant = ?", (tenant,))]
        chunks = [r["hash"] for r in q("SELECT hash FROM chunks WHERE tenant = ?", (tenant,))]
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
            self._tombstone(db, tenant, "commit", commits, reason, now, attestation)
            self._tombstone(db, tenant, "manifest", manifests, reason, now, attestation)
            self._tombstone(db, tenant, "chunk", chunks, reason, now, attestation)
        freed = self.chunks.delete_tenant(tenant)

        return Proof(
            tenant, reason, now, tuple(commits), tuple(manifests), tuple(chunks), freed, (), attestation
        )

    def forget_commit(self, tenant: str, spec: str, reason: str) -> Proof:
        """Remove one commit from history.

        Its children are re-parented onto its parent, refs pointing at it move
        to its parent, and its manifest goes if no other commit uses it. Any
        composite that used that manifest as an input is left in place, broken,
        and named in the proof — a view over a deleted input has nothing to show.
        """
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
        if drop_manifest:
            broken = [
                r["manifest_id"]
                for r in q(
                    "SELECT DISTINCT manifest_id FROM manifest_inputs WHERE tenant = ? AND input_id = ?",
                    (tenant, commit.manifest_id),
                )
            ]
            orphaned = [
                r["chunk_hash"]
                for r in q(
                    "SELECT chunk_hash FROM manifest_tensors WHERE tenant = ? AND manifest_id = ? "
                    "AND chunk_hash NOT IN (SELECT chunk_hash FROM manifest_tensors "
                    "WHERE tenant = ? AND manifest_id != ?)",
                    (tenant, commit.manifest_id, tenant, commit.manifest_id),
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
            self._tombstone(db, tenant, "commit", [commit.id], reason, now, attestation)
            self._tombstone(db, tenant, "manifest", manifests, reason, now, attestation)
            self._tombstone(db, tenant, "chunk", orphaned, reason, now, attestation)

        freed = sum(self.chunks.delete(tenant, h) for h in orphaned)
        return Proof(
            tenant,
            reason,
            now,
            (commit.id,),
            tuple(manifests),
            tuple(orphaned),
            freed,
            tuple(broken),
            attestation,
        )

    def verify(self, proof: Proof) -> list[str]:
        """Re-check a proof against the store. Empty list means it holds."""
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
        return problems

    def gc(self, tenant: str) -> int:
        """Drop chunks no manifest references. Returns bytes freed."""
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
