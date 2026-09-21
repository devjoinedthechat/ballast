"""Measure the claims the README makes, at sizes worth measuring.

Four of them:

- a commit that changes a few tensors stores a few blocks, not the adapter
- zstd earns its place on real deltas
- applying a delta costs one tensor of memory, not a model
- resolving a view streams, so a merge holds one tensor per input

Every scenario runs in its own process, and the fixtures it reads are built by a
different process again. Both matter. Peak resident memory is a high-water mark
that freeing does not lower, so a process that generated a four-gigabyte model
before measuring reports four gigabytes whatever the measured code went on to
do — the first version of this script did exactly that and reported the apply
holding the whole model.

    python scripts/benchmark.py            # ~4 GB of disk, a few minutes
    python scripts/benchmark.py --size s   # seconds, small enough for CI
    python scripts/benchmark.py --size l   # a 4 GB model, the real memory test
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

BF16 = np.dtype(ml_dtypes.bfloat16)

# Shapes from real adapters: LoRA over q, k, v and o projections, and a base
# model to apply a delta onto.
SIZES: dict[str, dict[str, int]] = {
    "s": {"layers": 8, "hidden": 1024, "rank": 16, "model_layers": 8, "model_hidden": 1024},
    "m": {"layers": 32, "hidden": 4096, "rank": 16, "model_layers": 16, "model_hidden": 2048},
    "l": {"layers": 32, "hidden": 4096, "rank": 64, "model_layers": 32, "model_hidden": 4096},
}


def lora(layers: int, hidden: int, rank: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for layer in range(layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            stem = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
            out[f"{stem}.lora_A.weight"] = rng.standard_normal((rank, hidden)).astype(BF16)
            out[f"{stem}.lora_B.weight"] = rng.standard_normal((hidden, rank)).astype(BF16)
    return out


def model_tensor_names(layers: int) -> list[str]:
    return [
        f"model.layers.{layer}.self_attn.{proj}.weight"
        for layer in range(layers)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]


def nbytes(tensors: dict[str, np.ndarray]) -> int:
    return sum(int(t.nbytes) for t in tensors.values())


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (peak if sys.platform == "darwin" else peak * 1024) / 1e6


# -- fixtures, built where nothing is measured -----------------------------


def prepare_apply(root: Path, spec: dict[str, int]) -> dict[str, Any]:
    from ballast import Store
    from ballast import tensors as st

    base = root / "base"
    base.mkdir(parents=True)
    hidden = spec["model_hidden"]
    names = model_tensor_names(spec["model_layers"])
    rng = np.random.default_rng(0)
    # A tensor at a time, so building the fixture does not need the memory the
    # fixture exists to show is unnecessary.
    st.save_stream(
        base / "model.safetensors",
        dict.fromkeys(names, ("BF16", (hidden, hidden))),
        lambda _: rng.standard_normal((hidden, hidden)).astype(BF16),
        {"format": "pt"},
    )
    (base / "config.json").write_text(json.dumps({"_name_or_path": "bench/base"}))

    delta = {name: rng.standard_normal((hidden, hidden)).astype(BF16) for name in names[:4]}
    Store(root / "store").commit("bench", delta, message="delta", base_model="bench/base")
    return {
        "model_mb": (base / "model.safetensors").stat().st_size / 1e6,
        "delta_mb": nbytes(delta) / 1e6,
        "model_tensors": len(names),
    }


def prepare_merge(root: Path, spec: dict[str, int]) -> dict[str, Any]:
    from ballast import Store

    store = Store(root / "store")
    for ref, seed in (("a", 1), ("b", 2)):
        store.commit(
            "bench",
            lora(spec["layers"], spec["hidden"], spec["rank"], seed),
            message=ref,
            ref=ref,
            base_model="bench/base",
        )
    store.merge("bench", "ties", [("a", 0.6), ("b", 0.4)], density=0.5, message="blend", ref="v")
    return {"inputs_mb": store.stats("bench").logical_bytes / 1e6}


# -- scenarios -------------------------------------------------------------


def scenario_commit(root: Path, spec: dict[str, int]) -> dict[str, Any]:
    """Storing an adapter, and storing a retrain of it.

    Peak memory here legitimately includes the adapter: `commit` takes it as a
    dictionary, so it is in memory by definition.
    """
    from ballast import Store

    adapter = lora(spec["layers"], spec["hidden"], spec["rank"])
    logical = nbytes(adapter)
    store = Store(root / "store")

    start = time.perf_counter()
    store.commit("bench", adapter, message="v1", base_model="bench/base")
    first = time.perf_counter() - start
    after_first = store.stats("bench")

    changed = sorted(adapter)[:2]
    retrained = dict(adapter)
    for name in changed:
        retrained[name] = (adapter[name].astype(np.float32) * 1.1).astype(BF16)
    touched = nbytes({name: adapter[name] for name in changed})

    start = time.perf_counter()
    store.commit("bench", retrained, message="v2", base_model="bench/base")
    retrain = time.perf_counter() - start
    after = store.stats("bench")

    start = time.perf_counter()
    store.checkout("bench", "main")
    read = time.perf_counter() - start

    return {
        "tensors": len(adapter),
        "adapter_mb": logical / 1e6,
        "commit_s": first,
        "commit_mbs": logical / 1e6 / first,
        "retrain_s": retrain,
        "checkout_s": read,
        "checkout_mbs": logical / 1e6 / read,
        "blocks_first": after_first.chunks,
        "blocks_added": after.chunks - after_first.chunks,
        "bytes_added_mb": (after.raw_bytes - after_first.raw_bytes) / 1e6,
        "touched_mb": touched / 1e6,
        "on_disk_mb": after.physical_bytes / 1e6,
        "logical_mb": after.logical_bytes / 1e6,
        "compression": after.compression_ratio,
        "dedup": after.dedup_ratio,
    }


def scenario_apply(root: Path, _spec: dict[str, int]) -> dict[str, Any]:
    """Applying a delta, streamed, as `apply_commit` does it."""
    from ballast import Store, models

    facts = json.loads((root / "facts.json").read_text())
    store = Store(root / "store")
    before = peak_rss_mb()
    start = time.perf_counter()
    models.apply_commit(store, "bench", "main", root / "base", root / "out")
    elapsed = time.perf_counter() - start
    return {
        **facts,
        "mode": "streamed",
        "apply_s": elapsed,
        "apply_mbs": facts["model_mb"] / elapsed,
        "rss_before_mb": before,
        "peak_rss_mb": peak_rss_mb(),
    }


def scenario_apply_whole(root: Path, _spec: dict[str, int]) -> dict[str, Any]:
    """The same result built in memory first, which is what streaming avoids.

    Here to be compared against. Reading a mapped file makes its pages resident,
    so peak RSS cannot fall below the model once every tensor has been touched —
    the question is whether a second copy is allocated on top, and that is what
    the difference between these two answers.
    """
    from ballast import Store
    from ballast import tensors as st

    facts = json.loads((root / "facts.json").read_text())
    store = Store(root / "store")
    before = peak_rss_mb()
    start = time.perf_counter()
    base = st.load_dir(root / "base")
    deltas = store.checkout("bench", "main")
    merged = {
        name: (value.astype(np.float32) + deltas[name].astype(np.float32)).astype(value.dtype)
        if name in deltas
        else np.array(value)
        for name, value in base.items()
    }
    out = root / "out-whole"
    out.mkdir(exist_ok=True)
    st.save(out / "model.safetensors", merged, {"format": "pt"})
    elapsed = time.perf_counter() - start
    return {
        **facts,
        "mode": "whole",
        "apply_s": elapsed,
        "apply_mbs": facts["model_mb"] / elapsed,
        "rss_before_mb": before,
        "peak_rss_mb": peak_rss_mb(),
    }


def scenario_merge_streamed(root: Path, _spec: dict[str, int]) -> dict[str, Any]:
    return _merge(root, stream=True)


def scenario_merge_whole(root: Path, _spec: dict[str, int]) -> dict[str, Any]:
    return _merge(root, stream=False)


def _merge(root: Path, *, stream: bool) -> dict[str, Any]:
    from ballast import Store

    facts = json.loads((root / "facts.json").read_text())
    store = Store(root / "store", cache_views=False)
    before = peak_rss_mb()
    start = time.perf_counter()
    count = (
        sum(1 for _ in store.checkout_stream("bench", "v")) if stream else len(store.checkout("bench", "v"))
    )
    elapsed = time.perf_counter() - start
    return {
        **facts,
        "mode": "streamed" if stream else "whole",
        "tensors": count,
        "resolve_s": elapsed,
        "rss_before_mb": before,
        "peak_rss_mb": peak_rss_mb(),
    }


SCENARIOS = {
    "commit": scenario_commit,
    "apply": scenario_apply,
    "apply-whole": scenario_apply_whole,
    "merge-streamed": scenario_merge_streamed,
    "merge-whole": scenario_merge_whole,
}
BUILDERS = {"prepare-apply": prepare_apply, "prepare-merge": prepare_merge}


# -- driver ----------------------------------------------------------------


def child(name: str, size: str, root: Path) -> dict[str, Any]:
    done = subprocess.run(
        [sys.executable, __file__, "--child", name, "--size", size, "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        print(done.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"{name} failed")
    return dict(json.loads(done.stdout.strip().splitlines()[-1]))


def group(names: list[str], builder: str, size: str) -> list[dict[str, Any]]:
    """Build one set of fixtures, then measure against it in fresh processes."""
    root = Path(tempfile.mkdtemp(prefix="ballast-bench-"))
    try:
        (root / "facts.json").write_text(json.dumps(child(builder, size, root)))
        out = []
        for name in names:
            out.append(child(name, size, root))
            shutil.rmtree(root / "store" / "cache", ignore_errors=True)
        return out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="m", choices=sorted(SIZES))
    parser.add_argument("--child", choices=[*sorted(SCENARIOS), *sorted(BUILDERS)])
    parser.add_argument("--root")
    args = parser.parse_args()
    spec = SIZES[args.size]

    if args.child:
        root = Path(args.root)
        work = BUILDERS[args.child] if args.child in BUILDERS else SCENARIOS[args.child]
        print(json.dumps(work(root, spec)))
        return 0

    started = time.perf_counter()
    print(f"ballast benchmark — size {args.size}, {spec}\n")

    root = Path(tempfile.mkdtemp(prefix="ballast-bench-"))
    try:
        c = child("commit", args.size, root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("Storing an adapter")
    print(f"  {c['tensors']} tensors, {c['adapter_mb']:.0f} MB")
    print(f"  commit          {c['commit_s']:.2f} s   {c['commit_mbs']:.0f} MB/s")
    print(f"  checkout        {c['checkout_s']:.2f} s   {c['checkout_mbs']:.0f} MB/s")
    print(f"  on disk         {c['on_disk_mb']:.0f} MB for {c['logical_mb']:.0f} MB logical")
    print(f"  dedup {c['dedup']:.2f}x, compression {c['compression']:.2f}x")
    print()
    print("A retrain that changes 2 of the adapter's tensors")
    print(f"  blocks added    {c['blocks_added']} of {c['blocks_first']}")
    print(f"  bytes added     {c['bytes_added_mb']:.2f} MB")
    print(f"  those tensors   {c['touched_mb']:.2f} MB")
    print(f"  whole adapter   {c['adapter_mb']:.0f} MB")
    print(f"  {c['adapter_mb'] / max(c['bytes_added_mb'], 1e-9):.0f}x less than storing the adapter again")
    print()

    whole_apply, streamed_apply = group(["apply-whole", "apply"], "prepare-apply", args.size)
    print("Applying a delta to a model")
    print(
        f"  model           {streamed_apply['model_mb']:.0f} MB in {streamed_apply['model_tensors']} tensors"
    )
    print(f"  delta           {streamed_apply['delta_mb']:.0f} MB")
    for r in (whole_apply, streamed_apply):
        held = r["peak_rss_mb"] - r["rss_before_mb"]
        print(
            f"  {r['mode']:<14}  {r['apply_s']:.2f} s   {r['apply_mbs']:.0f} MB/s   "
            f"peak RSS {r['peak_rss_mb']:.0f} MB ({held / r['model_mb'] * 100:.0f}% of the model)"
        )
    saved = whole_apply["peak_rss_mb"] - streamed_apply["peak_rss_mb"]
    print(f"  streaming holds {saved:.0f} MB less")
    print("  (reading a mapped file makes its pages resident, so neither can sit far")
    print("   below the model; what streaming avoids is the second copy on top)")
    print()

    whole, streamed = group(["merge-whole", "merge-streamed"], "prepare-merge", args.size)
    print("Resolving a view over two adapters")
    print(f"  inputs          {streamed['inputs_mb']:.0f} MB, {streamed['tensors']} tensors")
    for r in (whole, streamed):
        held = r["peak_rss_mb"] - r["rss_before_mb"]
        print(
            f"  {r['mode']:<14}  {r['resolve_s']:.2f} s   peak RSS {r['peak_rss_mb']:.0f} MB ({held:+.0f} MB)"
        )
    delta = (whole["peak_rss_mb"] - whole["rss_before_mb"]) - (
        streamed["peak_rss_mb"] - streamed["rss_before_mb"]
    )
    print(f"  streaming holds {delta:.0f} MB less")
    print(f"\nran in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
