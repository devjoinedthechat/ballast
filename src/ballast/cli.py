"""ballast — version control for what a model has learned."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ballast import mergekit, models, serving
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


def cmd_delta(store: Store, a: argparse.Namespace) -> int:
    metadata = json.loads(a.metadata) if a.metadata else {}
    commit, report = models.commit_delta(
        store, a.tenant, a.model, a.base, message=a.message, ref=a.ref, dtype=a.dtype, metadata=metadata
    )
    text = f"{commit.short}  {len(report.changed)} tensors changed, {len(report.unchanged)} unchanged  ref {a.ref}"
    if report.mismatched or report.only_in_model or report.only_in_base:
        text += f"\n  skipped: {report.summary()}"
    _emit(a, text, {**asdict(commit), "report": report.summary()})
    return 0


def cmd_apply(store: Store, a: argparse.Namespace) -> int:
    try:
        out = models.apply_commit(store, a.tenant, a.spec, a.base, a.out)
    except BrokenView as exc:
        print(f"cannot apply: {exc}", file=sys.stderr)
        return 2
    _emit(a, f"model written to {out}", {"out": str(out)})
    return 0


def cmd_log(store: Store, a: argparse.Namespace) -> int:
    commits = store.log(a.tenant, a.ref, limit=a.limit)
    lines = []
    for c in commits:
        kind = store.manifest(a.tenant, c.manifest_id).kind
        lines.append(f"{c.short}  {kind:<9}  {c.message}")
    _emit(a, "\n".join(lines), [asdict(c) for c in commits])
    return 0


def cmd_reflog(store: Store, a: argparse.Namespace) -> int:
    entries = store.reflog(a.tenant, a.ref, limit=a.limit)
    lines = [
        f"{e.position:>4}  {e.op:<7}  {(e.old_commit or '-')[:12]:<12} -> {(e.new_commit or '-')[:12]}"
        for e in entries
    ]
    _emit(a, "\n".join(lines) or "no history", [asdict(e) for e in entries])
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


def cmd_sync_vllm(store: Store, a: argparse.Namespace) -> int:
    runtime = serving.VLLMRuntime(a.url)
    changed = runtime.sync(store, a.tenant, a.out, specs=a.specs or None, prune=not a.no_prune)
    lines = [f"{action:<10} {name}" for action, names in changed.items() for name in names]
    _emit(a, "\n".join(lines) or "nothing to change", changed)
    return 0 if not changed["skipped"] else 4


def cmd_serve(store: Store, a: argparse.Namespace) -> int:
    from ballast.server import Tokens, run  # noqa: PLC0415

    tokens = Tokens.from_file(a.tokens) if a.tokens else Tokens()
    if tokens.open:
        print(
            "no --tokens given: every tenant is readable by anyone who can reach this port",
            file=sys.stderr,
        )
    print(f"serving {store.chunks.backend.describe()} on http://{a.host}:{a.port}", file=sys.stderr)
    run(store, host=a.host, port=a.port, tokens=tokens)
    return 0


def cmd_serve_export(store: Store, a: argparse.Namespace) -> int:
    """Materialise every named delta, and keep going past the ones that cannot.

    A fleet export that aborts because one view in it lost an input leaves the
    other adapters unserved for a reason that has nothing to do with them. The
    broken ones are named and the exit code says some failed.
    """
    specs = a.specs or sorted(store.refs(a.tenant))
    exports, failed = [], []
    for spec in specs:
        try:
            exports.append(serving.export(store, a.tenant, spec, a.out))
        except BrokenView as exc:
            failed.append((spec, str(exc)))
    if a.manifest:
        serving.manifest_json(exports, a.manifest)

    lines = [f"{e.int_id:<12} {e.name:<28} {e.path}" + ("  (reused)" if e.reused else "") for e in exports]
    lines += [f"{'-':<12} {spec:<28} skipped: {why}" for spec, why in failed]
    _emit(a, "\n".join(lines), {"exported": [e.as_dict() for e in exports], "skipped": dict(failed)})
    return 0 if not failed else 4


def cmd_merge(store: Store, a: argparse.Namespace) -> int:
    inputs = []
    for item in a.inputs:
        spec, _, weight = item.rpartition("@")
        if not spec:
            spec, weight = item, ""
        inputs.append((spec, float(weight) if weight else 1.0))
    try:
        commit = store.merge(
            a.tenant,
            a.method,
            inputs,
            message=a.message,
            ref=a.ref,
            density=a.density,
            strict=a.strict,
            seed=a.seed,
            t=a.t,
            normalize=a.normalize,
            extra={
                key: value
                for key, value in (
                    ("row_wise", a.row_wise),
                    ("flatten", a.flatten),
                    ("filter_wise", a.filter_wise),
                    ("select_topk", a.select_topk),
                )
                if value
            }
            or None,
            gamma=a.gamma,
            epsilon=a.epsilon,
        )
    except NotGranted as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    _emit(a, f"{commit.short}  {a.method} of {len(inputs)} inputs  ref {a.ref}", commit)
    return 0


def cmd_diff(store: Store, a: argparse.Namespace) -> int:
    diff = store.diff(a.tenant, a.a, a.b, probe_set=a.probe_set, threshold=a.threshold)
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
    _emit(a, f"revoked; {len(broken)} view(s) in {a.to!r} can no longer resolve", {"broken": broken})
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
        f"{s.commits} commits, {s.manifests} manifests, {s.chunks} blocks\n"
        f"{s.logical_bytes:,} logical bytes, {s.raw_bytes:,} unique ({s.dedup_ratio:.2f}x dedup), "
        f"{s.physical_bytes:,} on disk ({s.compression_ratio:.2f}x compression)"
    )
    _emit(a, text, {**asdict(s), "dedup_ratio": s.dedup_ratio, "compression_ratio": s.compression_ratio})
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
    p.add_argument(
        "--root", default=".ballast", help="store directory (metadata, cache, and blocks unless --backend)"
    )
    p.add_argument("--backend", help="where blocks live: a path or s3://bucket/prefix")
    p.add_argument("--db", help="where the graph lives: a path or postgresql://…")
    p.add_argument("--tenant", required=True)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("commit", help="store a PEFT adapter directory as a new commit")
    s.add_argument("adapter")
    s.add_argument("-m", "--message", required=True)
    s.add_argument("--ref", default="main")
    s.add_argument("--base-model")
    s.add_argument("--metadata", help="JSON object of provenance to keep on the commit")

    s = sub.add_parser("delta", help="commit model - base from two model directories")
    s.add_argument("model")
    s.add_argument("--base", required=True)
    s.add_argument("-m", "--message", required=True)
    s.add_argument("--ref", default="main")
    s.add_argument("--dtype", help="safetensors dtype for the delta, e.g. F32; default is the model's")
    s.add_argument("--metadata")

    s = sub.add_parser("apply", help="write a commit applied to a base as a model directory")
    s.add_argument("spec")
    s.add_argument("--base", required=True)
    s.add_argument("-o", "--out", required=True)

    s = sub.add_parser("log")
    s.add_argument("ref", nargs="?", default="main")
    s.add_argument("--limit", type=int, default=50)

    s = sub.add_parser("reflog", help="every move a ref made")
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
    s.add_argument("--density", type=float, help="the share of entries a sparsifying method keeps")
    s.add_argument("--t", type=float, help="slerp: how far to travel from the first input to the second")
    s.add_argument("--seed", type=int, help="pin the draw a random method makes; one is chosen otherwise")
    s.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        help="divide by the weights that agreed; on for ties, off for dare",
    )
    s.add_argument("--gamma", type=float, help="breadcrumbs: the share of largest entries dropped")
    s.add_argument("--epsilon", type=float, help="della: how far the keep probability swings by rank")
    s.add_argument("--select-topk", type=float, help="sce: the share of entries kept, by disagreement")
    s.add_argument("--row-wise", action="store_true", help="nuslerp: interpolate rows, not the tensor")
    s.add_argument("--flatten", action="store_true", help="nuslerp: treat the tensor as one vector")
    s.add_argument("--filter-wise", action="store_true", help="model_stock: one angle per row")
    s.add_argument(
        "--strict",
        action="store_true",
        help="require every input to carry the same tensors; by default an absent one counts as zero",
    )
    s.add_argument("-m", "--message", required=True)
    s.add_argument("--ref", default="main")

    s = sub.add_parser("serve", help="read-only HTTP API a serving runtime can pull from")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--tokens", help='JSON file of bearer token -> list of tenants, or ["*"]')

    s = sub.add_parser("sync-vllm", help="converge a running vLLM on what the store holds")
    s.add_argument("specs", nargs="*", help="refs or commits; every ref by default")
    s.add_argument("--url", required=True, help="base URL of the vLLM OpenAI server")
    s.add_argument("-o", "--out", required=True, help="directory the server reads adapters from")
    s.add_argument("--no-prune", action="store_true", help="load, but do not unload what is gone")

    s = sub.add_parser("serve-export", help="materialise deltas for a serving runtime to load")
    s.add_argument("specs", nargs="*", help="refs or commits; every ref by default")
    s.add_argument("-o", "--out", required=True, help="directory of adapters")
    s.add_argument("--manifest", help="also write a JSON list of what was exported")

    s = sub.add_parser("diff")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--probe-set")
    s.add_argument("--threshold", type=float, help="relative change above which an unnoticed move is flagged")

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
    s.add_argument("--fast", action="store_true", help="skip re-hashing block bytes")

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
    store = Store(a.root, backend=a.backend, metadata=a.db)
    handlers = {
        "commit": cmd_commit,
        "delta": cmd_delta,
        "apply": cmd_apply,
        "log": cmd_log,
        "reflog": cmd_reflog,
        "checkout": cmd_checkout,
        "serve": cmd_serve,
        "sync-vllm": cmd_sync_vllm,
        "serve-export": cmd_serve_export,
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
    except LookupError as exc:
        # An unknown ref or commit is a mistake at the keyboard, not a bug.
        print(f"{exc}", file=sys.stderr)
        return 2
    except BrokenView as exc:
        print(f"cannot resolve: {exc}", file=sys.stderr)
        return 2
    except NotGranted as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
