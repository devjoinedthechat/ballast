"""ballast — version control for what a model has learned."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ballast import mergekit
from ballast import peft as peft_io
from ballast.fingerprint import FakeRunner, ProbeSet, Runner, fingerprint
from ballast.store import BrokenView, Store


def cmd_commit(store: Store, a: argparse.Namespace) -> int:
    arrays, config, base = peft_io.load(a.adapter)
    commit = store.commit(
        a.tenant, arrays, message=a.message, ref=a.ref, base_model=a.base_model or base, config=config
    )
    print(f"{commit.short}  {len(arrays)} tensors  ref {a.ref}")
    return 0


def cmd_log(store: Store, a: argparse.Namespace) -> int:
    for c in store.log(a.tenant, a.ref, limit=a.limit):
        kind = store.manifest(a.tenant, c.manifest_id).kind
        print(f"{c.short}  {kind:<9}  {c.message}")
    return 0


def cmd_checkout(store: Store, a: argparse.Namespace) -> int:
    try:
        arrays = store.checkout(a.tenant, a.spec)
    except BrokenView as exc:
        print(f"cannot check out: {exc}", file=sys.stderr)
        return 2
    manifest = store.manifest(a.tenant, store.resolve(a.tenant, a.spec).manifest_id)
    peft_io.export(a.out, arrays, manifest.config)
    print(f"{len(arrays)} tensors -> {a.out}")
    return 0


def cmd_merge(store: Store, a: argparse.Namespace) -> int:
    inputs = []
    for item in a.inputs:
        spec, _, weight = item.partition("@")
        inputs.append((spec, float(weight) if weight else 1.0))
    commit = store.merge(a.tenant, a.method, inputs, message=a.message, ref=a.ref, density=a.density)
    print(f"{commit.short}  {a.method} of {len(inputs)} inputs  ref {a.ref}")
    return 0


def cmd_diff(store: Store, a: argparse.Namespace) -> int:
    print(store.diff(a.tenant, a.a, a.b, probe_set=a.probe_set))
    return 0


def cmd_reset(store: Store, a: argparse.Namespace) -> int:
    commit = store.reset(a.tenant, a.ref, a.spec)
    print(f"{a.ref} -> {commit.short}  {commit.message}")
    return 0


def cmd_forget(store: Store, a: argparse.Namespace) -> int:
    proof = store.forget_commit(a.tenant, a.spec, a.reason) if a.spec else store.forget(a.tenant, a.reason)
    print(proof)
    problems = store.verify(proof)
    print("verified" if not problems else "\n".join(f"PROBLEM: {p}" for p in problems))
    return 0 if not problems else 1


def cmd_stats(store: Store, a: argparse.Namespace) -> int:
    s = store.stats(a.tenant)
    print(
        f"{s.commits} commits, {s.manifests} manifests, {s.chunks} chunks\n"
        f"{s.physical_bytes:,} bytes on disk for {s.logical_bytes:,} logical "
        f"({s.dedup_ratio:.2f}x)"
    )
    return 0


def cmd_gc(store: Store, a: argparse.Namespace) -> int:
    print(f"freed {store.gc(a.tenant):,} bytes")
    return 0


def cmd_import_mergekit(store: Store, a: argparse.Namespace) -> int:
    refs = dict(item.split("=", 1) for item in a.map or [])
    commit = mergekit.import_config(store, a.tenant, a.config, refs=refs, ref=a.ref)
    print(f"{commit.short}  {commit.message}")
    return 0


def cmd_fingerprint(store: Store, a: argparse.Namespace) -> int:
    probes = ProbeSet(tuple(Path(a.probes).read_text().splitlines()))
    runner: Runner
    if a.runner == "fake":
        runner = FakeRunner()
    else:
        from ballast.runners import PeftRunner  # noqa: PLC0415

        base = store.manifest(a.tenant, store.resolve(a.tenant, a.spec).manifest_id).base_model
        if not base:
            print("commit has no base model; pass one at commit time", file=sys.stderr)
            return 2
        runner = PeftRunner(base)
    outputs = fingerprint(store, a.tenant, a.spec, probes, runner)
    print(f"probe set {probes.id}: {len(outputs)} outputs recorded")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ballast", description=__doc__)
    p.add_argument("--root", default=".ballast", help="store directory")
    p.add_argument("--tenant", required=True)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("commit", help="store an adapter directory as a new commit")
    s.add_argument("adapter")
    s.add_argument("-m", "--message", required=True)
    s.add_argument("--ref", default="main")
    s.add_argument("--base-model")

    s = sub.add_parser("log")
    s.add_argument("ref", nargs="?", default="main")
    s.add_argument("--limit", type=int, default=50)

    s = sub.add_parser("checkout", help="materialise a commit as an adapter directory")
    s.add_argument("spec")
    s.add_argument("-o", "--out", required=True)

    s = sub.add_parser("merge", help="record a merge as a view; inputs as ref or ref@weight")
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

    sub.add_parser("stats")
    sub.add_parser("gc")

    s = sub.add_parser("import-mergekit")
    s.add_argument("config")
    s.add_argument("--map", action="append", help="model=ref, repeatable")
    s.add_argument("--ref", default="main")

    s = sub.add_parser("fingerprint")
    s.add_argument("spec")
    s.add_argument("--probes", required=True, help="file with one probe per line")
    s.add_argument("--runner", choices=["fake", "peft"], default="peft")

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
        "stats": cmd_stats,
        "gc": cmd_gc,
        "import-mergekit": cmd_import_mergekit,
        "fingerprint": cmd_fingerprint,
    }
    return handlers[a.command](store, a)


if __name__ == "__main__":
    sys.exit(main())
