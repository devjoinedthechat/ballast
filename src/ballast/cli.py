"""ballast — version control for what a model has learned."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ballast import mergekit
from ballast import peft as peft_io
from ballast.fingerprint import FakeRunner, ProbeSet, Runner, fingerprint
from ballast.store import BrokenView, NotGranted, Store


def _emit(a: argparse.Namespace, text: str, data: Any) -> None:
    if a.json:
        if is_dataclass(data) and not isinstance(data, type):
            data = asdict(data)
        print(json.dumps(data, indent=2, sort_keys=True, default=str))
    else:
        print(text)


def cmd_commit(store: Store, a: argparse.Namespace) -> int:
    arrays, config, base = peft_io.load(a.adapter)
    metadata = json.loads(a.metadata) if a.metadata else {}
    commit = store.commit(
        a.tenant,
        arrays,
        message=a.message,
        ref=a.ref,
        base_model=a.base_model or base,
        config=config,
        metadata=metadata,
    )
    _emit(a, f"{commit.short}  {len(arrays)} tensors  ref {a.ref}", commit)
    return 0


def cmd_log(store: Store, a: argparse.Namespace) -> int:
    commits = store.log(a.tenant, a.ref, limit=a.limit)
    lines = []
    for c in commits:
        kind = store.manifest(a.tenant, c.manifest_id).kind
        lines.append(f"{c.short}  {kind:<9}  {c.message}")
    _emit(a, "\n".join(lines), [asdict(c) for c in commits])
    return 0


def cmd_checkout(store: Store, a: argparse.Namespace) -> int:
    try:
        arrays = store.checkout(a.tenant, a.spec)
    except BrokenView as exc:
        print(f"cannot check out: {exc}", file=sys.stderr)
        return 2
    commit = store.resolve(a.tenant, a.spec)
    manifest = store.manifest(a.tenant, commit.manifest_id)
    provenance: dict[str, Any] = {
        "store": str(Path(a.root).resolve()),
        "tenant": a.tenant,
        "commit": commit.id,
        "manifest": manifest.id,
        "kind": manifest.kind,
        "base_model": manifest.base_model,
        "message": commit.message,
        "metadata": commit.metadata,
    }
    if manifest.kind == "composite":
        provenance["recipe"] = {
            **manifest.config,
            "inputs": [
                {"tenant": t, "manifest": m, "weight": w} for t, m, w in store.inputs(a.tenant, manifest.id)
            ],
        }
    peft_io.export(a.out, arrays, manifest.config if manifest.kind == "leaf" else {}, provenance)
    _emit(a, f"{len(arrays)} tensors -> {a.out}", {"tensors": len(arrays), "out": a.out, **provenance})
    return 0


def cmd_merge(store: Store, a: argparse.Namespace) -> int:
    inputs = []
    for item in a.inputs:
        spec, _, weight = item.rpartition("@")
        if not spec:
            spec, weight = item, ""
        inputs.append((spec, float(weight) if weight else 1.0))
    try:
        commit = store.merge(a.tenant, a.method, inputs, message=a.message, ref=a.ref, density=a.density)
    except NotGranted as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    _emit(a, f"{commit.short}  {a.method} of {len(inputs)} inputs  ref {a.ref}", commit)
    return 0


def cmd_diff(store: Store, a: argparse.Namespace) -> int:
    diff = store.diff(a.tenant, a.a, a.b, probe_set=a.probe_set)
    _emit(a, str(diff), {**asdict(diff), "unobserved": diff.unobserved})
    return 0


def cmd_reset(store: Store, a: argparse.Namespace) -> int:
    commit = store.reset(a.tenant, a.ref, a.spec)
    _emit(a, f"{a.ref} -> {commit.short}  {commit.message}", commit)
    return 0


def cmd_forget(store: Store, a: argparse.Namespace) -> int:
    proof = store.forget_commit(a.tenant, a.spec, a.reason) if a.spec else store.forget(a.tenant, a.reason)
    problems = store.verify(proof)
    verdict = "verified" if not problems else "\n".join(f"PROBLEM: {p}" for p in problems)
    _emit(a, f"{proof}\n{verdict}", {**asdict(proof), "problems": problems})
    return 0 if not problems else 1


def cmd_verify(store: Store, a: argparse.Namespace) -> int:
    problems = store.verify(a.attestation)
    proof = store.proof(a.attestation)
    text = f"{proof}\n" + ("verified" if not problems else "\n".join(f"PROBLEM: {p}" for p in problems))
    _emit(a, text, {**asdict(proof), "problems": problems})
    return 0 if not problems else 1


def cmd_grant(store: Store, a: argparse.Namespace) -> int:
    grant = store.grant(a.tenant, a.spec, a.to)
    _emit(a, f"{a.to} may build views over {a.tenant}:{grant.manifest_id[:12]}", grant)
    return 0


def cmd_revoke(store: Store, a: argparse.Namespace) -> int:
    broken = store.revoke(a.tenant, a.spec, a.to)
    text = f"revoked; {len(broken)} view(s) in {a.to!r} can no longer resolve"
    _emit(a, text, {"broken": broken})
    return 0


def cmd_grants(store: Store, a: argparse.Namespace) -> int:
    grants = store.grants(a.tenant)
    lines = [
        f"{'live' if g.live else 'revoked':<8} {g.owner}:{g.manifest_id[:12]} -> {g.grantee}" for g in grants
    ]
    _emit(a, "\n".join(lines) or "no grants", [asdict(g) for g in grants])
    return 0


def cmd_stats(store: Store, a: argparse.Namespace) -> int:
    s = store.stats(a.tenant)
    text = (
        f"{s.commits} commits, {s.manifests} manifests, {s.chunks} chunks\n"
        f"{s.physical_bytes:,} bytes on disk for {s.logical_bytes:,} logical ({s.dedup_ratio:.2f}x)"
    )
    _emit(a, text, {**asdict(s), "dedup_ratio": s.dedup_ratio})
    return 0


def cmd_gc(store: Store, a: argparse.Namespace) -> int:
    freed = store.gc(a.tenant)
    _emit(a, f"freed {freed:,} bytes", {"bytes_freed": freed})
    return 0


def cmd_fsck(store: Store, a: argparse.Namespace) -> int:
    problems = store.fsck(None if a.all else a.tenant, verify_bytes=not a.fast)
    _emit(a, "consistent" if not problems else "\n".join(problems), {"problems": problems})
    return 0 if not problems else 1


def cmd_import_mergekit(store: Store, a: argparse.Namespace) -> int:
    refs = dict(item.split("=", 1) for item in a.map or [])
    commit = mergekit.import_config(store, a.tenant, a.config, refs=refs, ref=a.ref)
    _emit(a, f"{commit.short}  {commit.message}", commit)
    return 0


def cmd_fingerprint(store: Store, a: argparse.Namespace) -> int:
    probes = ProbeSet(tuple(line for line in Path(a.probes).read_text().splitlines() if line.strip()))
    runner: Runner
    if a.runner == "fake":
        runner = FakeRunner()
    else:
        from ballast.runners import PeftRunner  # noqa: PLC0415

        base = (
            a.base_model or store.manifest(a.tenant, store.resolve(a.tenant, a.spec).manifest_id).base_model
        )
        if not base:
            print("commit has no base model; pass --base-model", file=sys.stderr)
            return 2
        runner = PeftRunner(base)
    outputs = fingerprint(store, a.tenant, a.spec, probes, runner)
    _emit(
        a,
        f"probe set {probes.id}: {len(outputs)} outputs recorded",
        {"probe_set": probes.id, "outputs": outputs},
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ballast", description=__doc__)
    p.add_argument("--root", default=".ballast", help="store directory")
    p.add_argument("--tenant", required=True)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("commit", help="store an adapter directory as a new commit")
    s.add_argument("adapter")
    s.add_argument("-m", "--message", required=True)
    s.add_argument("--ref", default="main")
    s.add_argument("--base-model")
    s.add_argument("--metadata", help="JSON object of provenance to keep on the commit")

    s = sub.add_parser("log")
    s.add_argument("ref", nargs="?", default="main")
    s.add_argument("--limit", type=int, default=50)

    s = sub.add_parser("checkout", help="materialise a commit as an adapter directory")
    s.add_argument("spec")
    s.add_argument("-o", "--out", required=True)

    s = sub.add_parser(
        "merge", help="record a merge as a view; inputs as ref, ref@weight, or tenant:ref@weight"
    )
    s.add_argument("inputs", nargs="+")
    s.add_argument("--method", default="linear")
    s.add_argument("--density", type=float)
    s.add_argument("-m", "--message", required=True)
    s.add_argument("--ref", default="main")

    s = sub.add_parser("diff")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--probe-set")

    s = sub.add_parser("reset", help="point a ref at an earlier commit")
    s.add_argument("ref")
    s.add_argument("spec")

    s = sub.add_parser("forget", help="delete a tenant, or one commit, with a proof")
    s.add_argument("spec", nargs="?")
    s.add_argument("--reason", required=True)

    s = sub.add_parser("verify", help="re-check a stored proof by its attestation")
    s.add_argument("attestation")

    s = sub.add_parser("grant", help="let another tenant build views over a commit's manifest")
    s.add_argument("spec")
    s.add_argument("--to", required=True)

    s = sub.add_parser("revoke")
    s.add_argument("spec")
    s.add_argument("--to", required=True)

    sub.add_parser("grants", help="grants given or received")
    sub.add_parser("stats")
    sub.add_parser("gc")

    s = sub.add_parser("fsck", help="check the store's invariants")
    s.add_argument("--all", action="store_true", help="every tenant, not just --tenant")
    s.add_argument("--fast", action="store_true", help="skip re-hashing chunk bytes")

    s = sub.add_parser("import-mergekit")
    s.add_argument("config")
    s.add_argument("--map", action="append", help="model=ref, repeatable")
    s.add_argument("--ref", default="main")

    s = sub.add_parser("fingerprint")
    s.add_argument("spec")
    s.add_argument("--probes", required=True, help="file with one probe per line")
    s.add_argument("--runner", choices=["fake", "peft"], default="peft")
    s.add_argument("--base-model")

    a = p.parse_args(argv)
    store = Store(a.root)
    handlers = {
        "commit": cmd_commit,
        "log": cmd_log,
        "checkout": cmd_checkout,
        "merge": cmd_merge,
        "diff": cmd_diff,
        "reset": cmd_reset,
        "forget": cmd_forget,
        "verify": cmd_verify,
        "grant": cmd_grant,
        "revoke": cmd_revoke,
        "grants": cmd_grants,
        "stats": cmd_stats,
        "gc": cmd_gc,
        "fsck": cmd_fsck,
        "import-mergekit": cmd_import_mergekit,
        "fingerprint": cmd_fingerprint,
    }
    try:
        return handlers[a.command](store, a)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
