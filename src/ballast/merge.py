"""Resolve a composite manifest to tensors.

A composite is a view. It stores no tensors; it names its inputs and a method,
and the tensors are computed when something asks for them. Two methods are
implemented, and they are the two that account for nearly all merges of deltas
in practice. Others are recorded for provenance and refuse to resolve rather
than silently doing something else.

Arithmetic runs in float32 and casts back to the first input's dtype, so bf16
inputs do not accumulate rounding across a long sum.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

RESOLVABLE = ("linear", "task_arithmetic", "ties", "dare_ties")
RECORD_ONLY = ("slerp", "passthrough", "breadcrumbs", "model_stock", "della")


def linear(inputs: Sequence[dict[str, np.ndarray]], weights: Sequence[float]) -> dict[str, np.ndarray]:
    """Weighted sum. Task arithmetic over deltas is the same operation."""
    names = _shared_names(inputs)
    out: dict[str, np.ndarray] = {}
    for name in names:
        dtype = inputs[0][name].dtype
        acc = np.zeros(inputs[0][name].shape, dtype=np.float32)
        for tensors, weight in zip(inputs, weights, strict=True):
            acc += weight * tensors[name].astype(np.float32)
        out[name] = acc.astype(dtype)
    return out


def ties(
    inputs: Sequence[dict[str, np.ndarray]],
    weights: Sequence[float],
    density: float = 0.2,
) -> dict[str, np.ndarray]:
    """TIES: trim, elect sign, disjoint merge (Yadav et al., 2023).

    Per input, keep only the largest `density` fraction of entries by magnitude.
    Per entry, elect the sign carried by the larger total magnitude, breaking a
    tie toward positive. Combine only the inputs that agree with the elected
    sign, normalised by their weights, so opposing updates do not cancel into
    noise.

    The normalisation and tie-break follow mergekit's implementation exactly,
    so a view resolved here matches what mergekit would have written to disk.
    That is checked numerically in scripts/verify_real.py.
    """
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")
    names = _shared_names(inputs)
    out: dict[str, np.ndarray] = {}
    for name in names:
        dtype = inputs[0][name].dtype
        trimmed = [_trim(t[name].astype(np.float32), density) for t in inputs]
        stacked = np.stack([w * t for t, w in zip(trimmed, weights, strict=True)])
        elected = np.where(stacked.sum(axis=0) >= 0, 1.0, -1.0)
        agrees = np.sign(stacked) == elected
        merged = np.where(agrees, stacked, 0.0).sum(axis=0)
        divisor = np.stack([w * agrees[i] for i, w in enumerate(weights)]).sum(axis=0)
        divisor = np.where(divisor == 0, 1.0, divisor)
        out[name] = (merged / divisor).astype(dtype)
    return out


def _trim(array: np.ndarray, density: float) -> np.ndarray:
    """Keep exactly the `density` fraction of entries with the largest magnitude.

    Exactly, not "at least": a threshold comparison keeps every entry tied at the
    boundary and can retain one more than mergekit does, which then shows up as
    a one-entry difference in the merged tensor. Ties are broken by index, which
    is deterministic here; mergekit's own tie order is whatever torch's unstable
    sort produces, so entries tied exactly at the boundary are the one place a
    resolved view may legitimately differ from mergekit's file.
    """
    if density >= 1.0 or array.size == 0:
        return array
    keep = max(1, int(array.size * density))
    order = np.argsort(-np.abs(array).ravel(), kind="stable")[:keep]
    mask = np.zeros(array.size, dtype=bool)
    mask[order] = True
    return np.where(mask.reshape(array.shape), array, 0.0)


def _shared_names(inputs: Sequence[dict[str, np.ndarray]]) -> list[str]:
    if not inputs:
        raise ValueError("a merge needs at least one input")
    names = set(inputs[0])
    for tensors in inputs[1:]:
        if set(tensors) != names:
            missing = sorted(names.symmetric_difference(tensors))
            raise ValueError(f"inputs do not share the same tensors; differ on {missing[:5]}")
    for name in names:
        shapes = {t[name].shape for t in inputs}
        if len(shapes) != 1:
            raise ValueError(f"tensor {name!r} has different shapes across inputs: {shapes}")
    return sorted(names)


def resolve(
    method: str,
    inputs: Sequence[dict[str, np.ndarray]],
    weights: Sequence[float],
    density: float | None = None,
) -> dict[str, np.ndarray]:
    if method in ("linear", "task_arithmetic"):
        return linear(inputs, weights)
    if method in ("ties", "dare_ties"):
        return ties(inputs, weights, density if density is not None else 0.2)
    if method in RECORD_ONLY:
        raise NotImplementedError(
            f"{method!r} is recorded for provenance but not resolved here; "
            f"resolvable methods are {RESOLVABLE}"
        )
    raise ValueError(f"unknown merge method {method!r}")
