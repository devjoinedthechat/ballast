"""Full models in, full models out.

People who merge models have model directories, not deltas. `delta` turns a
fine-tuned model and its base into the tensors that differ, which is what the
store versions; `apply` turns a stored delta and a base back into a model
directory that transformers loads. The round trip is exact for float32 and
within rounding for narrower dtypes, and it is checked against mergekit's own
output by scripts/verify_real.py.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ballast import tensors as st
from ballast.store import Commit, Store

MODEL_FILE = "model.safetensors"
EXTRA_SKIP = {"model.safetensors.index.json"}


@dataclass
class DeltaReport:
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    """Identical in both; stored as absent, which `apply` reads as unchanged."""
    mismatched: list[str] = field(default_factory=list)
    """Different shapes; skipped, because a delta between them has no meaning."""
    only_in_model: list[str] = field(default_factory=list)
    only_in_base: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "changed": len(self.changed),
            "unchanged": len(self.unchanged),
            "mismatched": self.mismatched,
            "only_in_model": self.only_in_model,
            "only_in_base": self.only_in_base,
        }


def delta(
    model_dir: Path | str, base_dir: Path | str, dtype: str | None = None
) -> tuple[dict[str, np.ndarray], DeltaReport]:
    """`model - base` for every tensor both hold, computed in float32.

    The result is cast to the model's dtype, or to `dtype` (a safetensors name
    such as `F32`) when given. Tensors identical in both are omitted and listed
    in the report; `apply` treats an absent tensor as unchanged.
    """
    model = st.load_dir(model_dir)
    base = st.load_dir(base_dir)
    report = DeltaReport()
    out: dict[str, np.ndarray] = {}
    for name in sorted(set(model) | set(base)):
        if name not in base:
            report.only_in_model.append(name)
            continue
        if name not in model:
            report.only_in_base.append(name)
            continue
        m, b = model[name], base[name]
        if m.shape != b.shape:
            report.mismatched.append(name)
            continue
        if np.array_equal(m.view(np.uint8), b.view(np.uint8)):
            report.unchanged.append(name)
            continue
        d = m.astype(np.float32) - b.astype(np.float32)
        target = st.DTYPES[dtype] if dtype else m.dtype
        out[name] = d.astype(target)
        report.changed.append(name)
    return out, report


def apply(
    deltas: Mapping[str, np.ndarray], base_dir: Path | str, out_dir: Path | str, copy_extras: bool = True
) -> Path:
    """Write `base + delta` as a model directory.

    Tensors the delta lacks are copied from the base unchanged. Every file in
    the base directory that is not a weights shard is copied alongside, so the
    result carries its config and tokenizer and loads as the base did.

    Written one tensor at a time. The base is memory-mapped and each sum is
    released as soon as it is on disk, so applying a delta to a 70B model costs
    one tensor of memory rather than the model.
    """
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    base = st.load_dir(base_dir)
    unknown = sorted(set(deltas) - set(base))
    if unknown:
        raise KeyError(f"delta names tensors the base does not have: {unknown[:5]}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def produce(name: str) -> np.ndarray:
        original = base[name]
        if name not in deltas:
            return original
        return (original.astype(np.float32) + deltas[name].astype(np.float32)).astype(original.dtype)

    st.save_stream(out_dir / MODEL_FILE, st.specs_of(base), produce, {"format": "pt"})
    if copy_extras:
        for file in base_dir.iterdir():
            if file.is_file() and not file.name.endswith(".safetensors") and file.name not in EXTRA_SKIP:
                shutil.copy2(file, out_dir / file.name)
    return out_dir


def base_name(base_dir: Path | str) -> str:
    """What to call the base: its config's own name if it has one, else the directory."""
    base_dir = Path(base_dir)
    config = base_dir / "config.json"
    if config.exists():
        try:
            name = json.loads(config.read_text()).get("_name_or_path")
            if isinstance(name, str) and name:
                return name
        except json.JSONDecodeError:
            pass
    return base_dir.name


def commit_delta(
    store: Store,
    tenant: str,
    model_dir: Path | str,
    base_dir: Path | str,
    *,
    message: str,
    ref: str = "main",
    dtype: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[Commit, DeltaReport]:
    """Extract and commit `model - base` in one step."""
    tensors, report = delta(model_dir, base_dir, dtype)
    if not tensors:
        raise ValueError("model and base are identical; nothing to commit")
    config = {"delta_of": base_name(base_dir), **report.summary()}
    commit = store.commit(
        tenant,
        tensors,
        message=message,
        ref=ref,
        base_model=base_name(base_dir),
        config=config,
        metadata=metadata,
    )
    return commit, report


def apply_commit(store: Store, tenant: str, spec: str, base_dir: Path | str, out_dir: Path | str) -> Path:
    """Write a commit applied to a base as a model directory, streaming both sides.

    The delta is read from the store one tensor at a time and the base is
    memory-mapped, so the peak is one tensor of each rather than two models.
    """
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    base = st.load_dir(base_dir)
    deltas = dict(store.checkout_stream(tenant, spec))
    unknown = sorted(set(deltas) - set(base))
    if unknown:
        raise KeyError(f"delta names tensors the base does not have: {unknown[:5]}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def produce(name: str) -> np.ndarray:
        original = base[name]
        if name not in deltas:
            return original
        return (original.astype(np.float32) + deltas[name].astype(np.float32)).astype(original.dtype)

    st.save_stream(out_dir / MODEL_FILE, st.specs_of(base), produce, {"format": "pt"})
    for file in base_dir.iterdir():
        if file.is_file() and not file.name.endswith(".safetensors") and file.name not in EXTRA_SKIP:
            shutil.copy2(file, out_dir / file.name)
    return out_dir
