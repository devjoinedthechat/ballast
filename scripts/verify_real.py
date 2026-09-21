"""Verify ballast against a live model and against mergekit's own output.

Not part of the test suite: it downloads HuggingFaceTB/SmolLM2-135M (~270 MB),
runs generation on CPU, and shells out to mergekit-yaml. Run it by hand:

    uv pip install -e ".[peft]" mergekit
    python scripts/verify_real.py

It checks three things and prints a verdict for each.

1. PeftRunner fingerprints a real adapter, and a retrained adapter moves probes.
2. A tiny perturbation that greedy decoding cannot see is reported as
   UNOBSERVED CHANGE rather than as no change.
3. A linear and a TIES view resolved by ballast match, tensor for tensor, what
   mergekit writes to disk for the same recipe.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ballast import Store, mergekit
from ballast import peft as peft_io
from ballast import tensors as st
from ballast.fingerprint import ProbeSet, fingerprint
from ballast.runners import PeftRunner

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROBES = ProbeSet.of(
    "The capital of France is",
    "def fibonacci(n):",
    "The customer asked for a refund because",
    "Once upon a time",
)


def banner(text: str) -> None:
    print(f"\n=== {text} ===", flush=True)


def make_adapters(work: Path) -> tuple[Path, Path, Path]:
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
    model = get_peft_model(base, cfg)
    # lora_B is zero at init, which makes the adapter a no-op. Give it a signal.
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.normal_(0, 0.05)
    v1 = work / "v1"
    model.save_pretrained(v1)

    # v2: the same adapter with two layers pushed hard. Behaviour should move.
    for name, param in model.named_parameters():
        if "lora_B" in name and (".layers.3." in name or ".layers.9." in name):
            param.data.mul_(40.0)
    v2 = work / "v2"
    model.save_pretrained(v2)

    # v3: v1 with one tensor nudged by an amount greedy decoding will not see.
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32), cfg)
    sd = dict(st.load(v1 / "adapter_model.safetensors")[0])
    target = next(k for k in sd if ".layers.20." in k and "lora_A" in k)
    sd[target] = (sd[target].astype(np.float32) * 1.03).astype(sd[target].dtype)
    v3 = work / "v3"
    v3.mkdir()
    st.save(v3 / "adapter_model.safetensors", sd, {"format": "pt"})
    shutil.copy(v1 / "adapter_config.json", v3 / "adapter_config.json")
    return v1, v2, v3


def check_fingerprints(store: Store, work: Path) -> bool:
    banner("1+2: PeftRunner on a live model")
    v1, v2, v3 = make_adapters(work)
    runner = PeftRunner(MODEL, max_new_tokens=12)  # in-process: no export between store and model
    ok = True
    ids = {}
    for tag, path in (("v1", v1), ("v2", v2), ("v3", v3)):
        arrays, config, base = peft_io.load(path)
        c = store.commit("verify", arrays, message=tag, base_model=base, config=config, ref=tag)
        outputs = fingerprint(store, "verify", tag, PROBES, runner)
        ids[tag] = c.id
        print(f"{tag} {c.short}")
        for probe, out in zip(PROBES.probes, outputs, strict=True):
            print(f"   {probe!r:<45} -> {out[len(probe) :].strip()!r}")

    d12 = store.diff("verify", "v1", "v2", probe_set=PROBES.id)
    print(f"\nv1 -> v2\n{d12}")
    if d12.probes_changed and d12.probes_changed > 0 and not d12.unobserved:
        print("PASS: retrained layers moved the probes")
    else:
        print("FAIL: expected probes to move")
        ok = False

    d13 = store.diff("verify", "v1", "v3", probe_set=PROBES.id)
    print(f"\nv1 -> v3\n{d13}")
    if d13.unobserved:
        print("PASS: a change greedy decoding cannot see is reported as UNOBSERVED, not as nothing")
    elif d13.probes_changed:
        print(
            "INCONCLUSIVE: the nudge was visible to the probes on this model; "
            "the unobserved path is covered by unit tests"
        )
    else:
        print("FAIL: unobserved change not flagged")
        ok = False
    return ok


def make_variants(work: Path) -> tuple[Path, Path, Path]:
    torch.manual_seed(1)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base_dir = work / "base"
    AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).save_pretrained(base_dir)
    tok.save_pretrained(base_dir)
    out = []
    for tag, scale in (("a", 0.02), ("b", 0.03)):
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "q_proj" in name or "v_proj" in name:
                    param.add_(torch.randn_like(param) * scale)
        d = work / tag
        model.save_pretrained(d)
        tok.save_pretrained(d)
        out.append(d)
    return base_dir, out[0], out[1]


def run_mergekit(config: dict, out: Path) -> dict[str, np.ndarray]:
    """Run mergekit in-process and return what it wrote.

    mergekit 0.1.4 with transformers 5 and pydantic 2.10 fails to instantiate its
    architecture models unless they are rebuilt with torch in scope. That is a
    bug in the combination, not in the merge, so it is worked around here rather
    than pinned around.
    """
    import mergekit.architecture as arch
    import mergekit.architecture.base as arch_base
    from mergekit.config import MergeConfiguration
    from mergekit.merge import run_merge
    from mergekit.options import MergeOptions
    from pydantic import BaseModel

    for mod in (arch_base, arch):
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel:
                obj.model_rebuild(_types_namespace={"torch": torch}, force=True)

    (out.parent / f"{out.name}.yaml").write_text(json.dumps(config))
    run_merge(
        MergeConfiguration.model_validate(config),
        str(out),
        MergeOptions(copy_tokenizer=False, lazy_unpickle=False, low_cpu_memory=False),
    )
    return load_model_tensors(out)


def load_model_tensors(path: Path) -> dict[str, np.ndarray]:
    merged: dict[str, np.ndarray] = {}
    for file in sorted(path.glob("*.safetensors")):
        merged.update(st.load(file)[0])
    return merged


def check_mergekit(store: Store, work: Path) -> bool:
    banner("3: views match mergekit output")
    base_dir, a_dir, b_dir = make_variants(work)
    base = load_model_tensors(base_dir)
    a = load_model_tensors(a_dir)
    b = load_model_tensors(b_dir)
    names = sorted(k for k in a if "q_proj" in k or "v_proj" in k)
    ok = True

    def f32(t: np.ndarray) -> np.ndarray:
        return t.astype(np.float32)

    # linear: mergekit averages full weights; ballast averages whatever it is given.
    store.commit("mk", {k: a[k] for k in names}, message="a", ref="a", base_model="base")
    store.commit("mk", {k: b[k] for k in names}, message="b", ref="b", base_model="base")
    linear_cfg = {
        "merge_method": "linear",
        "models": [
            {"model": str(a_dir), "parameters": {"weight": 0.7}},
            {"model": str(b_dir), "parameters": {"weight": 0.3}},
        ],
        "parameters": {"normalize": False},
        "dtype": "float32",
    }
    ref_out = run_mergekit(linear_cfg, work / "mk-linear")
    cfg_path = work / "linear.yaml"
    cfg_path.write_text(json.dumps(linear_cfg))
    c = mergekit.import_config(store, "mk", cfg_path, refs={str(a_dir): "a", str(b_dir): "b"}, ref="linear")
    ours = store.checkout("mk", c.id)
    worst = max(float(np.max(np.abs(f32(ours[k]) - f32(ref_out[k])))) for k in names)
    print(f"linear: {len(names)} tensors, max abs difference vs mergekit {worst:.2e}")
    if worst < 1e-5:
        print("PASS: linear view matches mergekit")
    else:
        print("FAIL: linear view differs from mergekit")
        ok = False

    # ties: mergekit works on deltas from base_model; ballast is given the deltas.
    deltas_a = {k: f32(a[k]) - f32(base[k]) for k in names}
    deltas_b = {k: f32(b[k]) - f32(base[k]) for k in names}
    store.commit("mk", deltas_a, message="delta a", ref="da", base_model="base")
    store.commit("mk", deltas_b, message="delta b", ref="db", base_model="base")
    density = 0.4
    ties_cfg = {
        "merge_method": "ties",
        "base_model": str(base_dir),
        "models": [
            {"model": str(a_dir), "parameters": {"weight": 0.6, "density": density}},
            {"model": str(b_dir), "parameters": {"weight": 0.4, "density": density}},
        ],
        "parameters": {"normalize": True, "int8_mask": False},
        "dtype": "float32",
    }
    ref_out = run_mergekit(ties_cfg, work / "mk-ties")
    cfg_path = work / "ties.yaml"
    cfg_path.write_text(json.dumps(ties_cfg))
    c = mergekit.import_config(store, "mk", cfg_path, refs={str(a_dir): "da", str(b_dir): "db"}, ref="ties")
    ours = store.checkout("mk", c.id)

    # Entries tied exactly at the density boundary are resolved by an unstable
    # sort inside mergekit and a stable one here, so they are the one legitimate
    # source of difference. Count them per tensor and require every differing
    # entry to be accounted for by one.
    differing_total = 0
    unexplained = 0
    worst = 0.0
    for k in names:
        real = f32(ref_out[k]) - f32(base[k])
        mine = f32(ours[k])
        differing = int((~np.isclose(mine, real, atol=1e-6, rtol=0)).sum())
        differing_total += differing
        worst = max(worst, float(np.abs(mine - real).max()))
        tied = 0
        for delta in (deltas_a[k], deltas_b[k]):
            flat = np.abs(delta).ravel()
            kk = int(density * flat.size)
            threshold = np.partition(flat, -kk)[-kk]
            tied += max(0, int((flat == threshold).sum()) - 1)
        # A tie can move one entry in one input, which changes at most two entries
        # of the merged tensor (the one dropped and the one kept).
        if differing > 2 * tied:
            unexplained += differing - 2 * tied
    print(
        f"ties: {len(names)} tensors, {differing_total} entries differ, max abs {worst:.2e}, "
        f"{unexplained} not explained by a boundary tie"
    )
    if unexplained == 0:
        print("PASS: ties view matches mergekit up to entries tied at the density boundary")
    else:
        print("FAIL: ties view differs from mergekit beyond boundary ties")
        ok = False

    # The full round trip a mergekit user takes: extract deltas from model
    # directories, merge as a view, apply back onto the base, load the result.
    banner("4: delta -> view -> apply reproduces mergekit's output model")
    from ballast import models

    models.commit_delta(store, "rt", a_dir, base_dir, message="delta a", ref="a")
    models.commit_delta(store, "rt", b_dir, base_dir, message="delta b", ref="b")
    view = store.merge("rt", "linear", [("a", 0.7), ("b", 0.3)], message="linear view", ref="lin")
    rebuilt_dir = models.apply_commit(store, "rt", view.id, base_dir, work / "rebuilt")
    rebuilt = load_model_tensors(rebuilt_dir)
    mk_linear = load_model_tensors(work / "mk-linear")
    worst = max(float(np.max(np.abs(f32(rebuilt[k]) - f32(mk_linear[k])))) for k in mk_linear)
    untouched = [k for k in mk_linear if k not in names]
    exact = all(np.array_equal(rebuilt[k].view(np.uint8), mk_linear[k].view(np.uint8)) for k in untouched)
    print(
        f"apply: {len(mk_linear)} tensors, max abs difference vs mergekit's model {worst:.2e}; "
        f"{len(untouched)} untouched tensors byte-identical: {exact}"
    )
    from transformers import AutoModelForCausalLM

    AutoModelForCausalLM.from_pretrained(rebuilt_dir, dtype=torch.float32)
    print("apply: transformers loads the rebuilt directory")
    if worst < 1e-5 and exact:
        print("PASS: the model-directory round trip reproduces mergekit's output")
    else:
        print("FAIL: the rebuilt model differs from mergekit's output")
        ok = False
    return ok


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="ballast-verify-"))
    print(f"working in {work}")
    store = Store(work / "store")
    results = [check_fingerprints(store, work), check_mergekit(store, work)]
    banner("verdict")
    print("ALL PASS" if all(results) else "FAILURES ABOVE")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
