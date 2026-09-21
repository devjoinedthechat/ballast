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
    Per entry, elect the sign carried by the larger total magnitude. Average only
    the inputs that agree with the elected sign, so opposing updates do not
    cancel into noise.
    """
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")
    names = _shared_names(inputs)
    out: dict[str, np.ndarray] = {}
    for name in names:
        dtype = inputs[0][name].dtype
        stacked = np.stack(
            [w * _trim(t[name].astype(np.float32), density) for t, w in zip(inputs, weights, strict=True)]
        )
        elected = np.sign(stacked.sum(axis=0))
        agrees = (np.sign(stacked) == elected) & (stacked != 0)
        count = agrees.sum(axis=0)
        merged = np.where(agrees, stacked, 0.0).sum(axis=0)
        merged = np.divide(merged, count, out=np.zeros_like(merged), where=count > 0)
        out[name] = merged.astype(dtype)
    return out


def _trim(array: np.ndarray, density: float) -> np.ndarray:
    if density >= 1.0 or array.size == 0:
        return array
    keep = max(1, round(array.size * density))
    flat = np.abs(array).ravel()
    threshold = np.partition(flat, -keep)[-keep]
    return np.where(np.abs(array) >= threshold, array, 0.0)


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
