"""Resolve a composite manifest to tensors.

A composite is a view. It stores no tensors; it names its inputs and a method,
and the tensors are computed when something asks for them. Four methods resolve
and the rest are recorded for provenance and refuse, rather than silently
running as something they are not.

Arithmetic runs in float32 and casts back to the first input's dtype, so bf16
inputs do not accumulate rounding across a long sum.

The sparsification and consensus steps follow mergekit's implementation, and
`linear` and `ties` are checked against its output numerically by
scripts/verify_real.py. The DARE variants cannot be: mergekit draws their masks
from the global torch RNG, so two runs of the same recipe there produce
different weights. Here the seed is part of the view, drawn once when the merge
is recorded and stored with it, and every tensor's mask is derived from that
seed and the tensor's own name — so the result is reproducible, independent of
how many tensors there are or the order they are resolved in, and by
construction not bit-identical to any particular mergekit run.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

import blake3
import numpy as np

RESOLVABLE = ("linear", "task_arithmetic", "ties", "dare_ties", "dare_linear")
RECORD_ONLY = (
    "slerp",
    "passthrough",
    "breadcrumbs",
    "breadcrumbs_ties",
    "model_stock",
    "della",
    "della_linear",
)
SEEDED = ("dare_ties", "dare_linear")
"""Methods whose result depends on a random draw, so a view must store its seed."""

DEFAULT_DENSITY = 0.2
EPS = 1e-7


def linear(
    inputs: Sequence[dict[str, np.ndarray]], weights: Sequence[float], strict: bool = True
) -> dict[str, np.ndarray]:
    """Weighted sum. Task arithmetic over deltas is the same operation."""
    names, template = _names(inputs, strict)
    out: dict[str, np.ndarray] = {}
    for name in names:
        ref = template[name]
        acc = np.zeros(ref.shape, dtype=np.float32)
        for tensors, weight in zip(inputs, weights, strict=True):
            if name in tensors:
                acc += weight * tensors[name].astype(np.float32)
        out[name] = acc.astype(ref.dtype)
    return out


def ties(
    inputs: Sequence[dict[str, np.ndarray]],
    weights: Sequence[float],
    density: float = DEFAULT_DENSITY,
    strict: bool = True,
    normalize: bool = True,
    lambda_: float = 1.0,
) -> dict[str, np.ndarray]:
    """TIES: trim, elect sign, disjoint merge (Yadav et al., 2023).

    Per input, keep only the largest `density` fraction of entries by magnitude.
    Per entry, elect the sign carried by the larger total magnitude, breaking a
    tie toward positive. Combine only the inputs that agree with the elected
    sign, normalised by their weights, so opposing updates do not cancel into
    noise.
    """
    return _generalized(
        inputs,
        weights,
        density,
        strict,
        sparsify="magnitude",
        elect=True,
        normalize=normalize,
        lambda_=lambda_,
    )


def dare_ties(
    inputs: Sequence[dict[str, np.ndarray]],
    weights: Sequence[float],
    density: float = DEFAULT_DENSITY,
    strict: bool = True,
    seed: int = 0,
    normalize: bool = False,
    lambda_: float = 1.0,
) -> dict[str, np.ndarray]:
    """DARE with sign election (Yu et al., 2023).

    Drop each entry independently with probability `1 - density`, rescale the
    survivors so the tensor keeps its L1 norm, then elect a sign as TIES does.
    Not normalised by the agreeing weights, which is mergekit's default for this
    method and the reason it and TIES differ by more than the sparsifier.
    """
    return _generalized(
        inputs,
        weights,
        density,
        strict,
        sparsify="random",
        elect=True,
        normalize=normalize,
        seed=seed,
        lambda_=lambda_,
    )


def dare_linear(
    inputs: Sequence[dict[str, np.ndarray]],
    weights: Sequence[float],
    density: float = DEFAULT_DENSITY,
    strict: bool = True,
    seed: int = 0,
    lambda_: float = 1.0,
) -> dict[str, np.ndarray]:
    """DARE without sign election: drop, rescale, weighted sum."""
    return _generalized(
        inputs,
        weights,
        density,
        strict,
        sparsify="random",
        elect=False,
        normalize=False,
        seed=seed,
        lambda_=lambda_,
    )


def _generalized(
    inputs: Sequence[dict[str, np.ndarray]],
    weights: Sequence[float],
    density: float,
    strict: bool,
    *,
    sparsify: str,
    elect: bool,
    normalize: bool,
    seed: int = 0,
    lambda_: float = 1.0,
) -> dict[str, np.ndarray]:
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")
    names, template = _names(inputs, strict)
    out: dict[str, np.ndarray] = {}
    for name in names:
        ref = template[name]
        prepared = []
        for index, tensors in enumerate(inputs):
            if name not in tensors:
                prepared.append(np.zeros(ref.shape, dtype=np.float32))
                continue
            value = tensors[name].astype(np.float32)
            if sparsify == "magnitude":
                prepared.append(_trim(value, density))
            else:
                prepared.append(_drop_and_rescale(value, density, _seed_for(seed, index, name)))

        stacked = np.stack([w * t for t, w in zip(prepared, weights, strict=True)])
        if not elect:
            merged = stacked.sum(axis=0)
        else:
            elected = np.where(stacked.sum(axis=0) >= 0, 1.0, -1.0)
            agrees = np.sign(stacked) == elected
            merged = np.where(agrees, stacked, 0.0).sum(axis=0)
            if normalize:
                divisor = np.stack([w * agrees[i] for i, w in enumerate(weights)]).sum(axis=0)
                merged = merged / np.where(divisor == 0, 1.0, divisor)
        if lambda_ != 1.0:
            merged = merged * lambda_
        out[name] = merged.astype(ref.dtype)
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


def _drop_and_rescale(array: np.ndarray, density: float, seed: int) -> np.ndarray:
    """Keep each entry with probability `density`, then restore the L1 norm.

    The classic DARE rescale is 1/density, which is that in expectation;
    matching the norm of the actual draw is what mergekit does and is stabler on
    a small tensor, where the realised keep rate strays from the nominal one.
    """
    if density >= 1.0 or array.size == 0:
        return array
    rng = np.random.default_rng(seed)
    mask = rng.random(array.shape) < density
    masked = np.where(mask, array, 0.0)
    before = float(np.abs(array).sum())
    after = float(np.abs(masked).sum())
    if before < EPS or after < EPS:
        return masked
    return masked * (before / after)


def _seed_for(seed: int, index: int, name: str) -> int:
    """A tensor's own seed, from the view's seed, the input's position and the name.

    Derived rather than sequential so the draw for one tensor does not depend on
    how many came before it. A view resolved on its own gives the same answer as
    the same view resolved inside a larger one.
    """
    digest = blake3.blake3(struct.pack("<qq", seed, index) + name.encode()).digest(8)
    return int.from_bytes(digest, "little")


def _names(inputs: Sequence[dict[str, np.ndarray]], strict: bool) -> tuple[list[str], dict[str, np.ndarray]]:
    """The tensors to merge, and one representative of each for shape and dtype.

    Strict requires every input to carry the same tensors. Non-strict takes the
    union and treats a tensor an input lacks as zero, which is what two adapters
    over the same base with different target modules need.
    """
    if not inputs:
        raise ValueError("a merge needs at least one input")
    names = set(inputs[0])
    for tensors in inputs[1:]:
        if strict and set(tensors) != names:
            missing = sorted(names.symmetric_difference(tensors))
            raise ValueError(
                f"inputs do not share the same tensors; differ on {missing[:5]} (use strict=False to union)"
            )
        names |= set(tensors)
    template: dict[str, np.ndarray] = {}
    for name in names:
        present = [t[name] for t in inputs if name in t]
        shapes = {t.shape for t in present}
        if len(shapes) != 1:
            raise ValueError(f"tensor {name!r} has different shapes across inputs: {shapes}")
        template[name] = present[0]
    return sorted(names), template


def resolve(
    method: str,
    inputs: Sequence[dict[str, np.ndarray]],
    weights: Sequence[float],
    density: float | None = None,
    strict: bool = True,
    seed: int | None = None,
    normalize: bool | None = None,
    lambda_: float = 1.0,
) -> dict[str, np.ndarray]:
    """Resolve a view.

    `normalize` and `lambda_` are mergekit's own parameters, and a real config
    sets them: TIES defaults to normalising and DARE does not, and either can be
    overridden. Ignoring them would mean a view that reads like a recipe and
    resolves to something else.
    """
    if method in ("linear", "task_arithmetic"):
        return linear(inputs, weights, strict)
    density = DEFAULT_DENSITY if density is None else density
    if method == "ties":
        return ties(inputs, weights, density, strict, True if normalize is None else normalize, lambda_)
    if method in SEEDED:
        if seed is None:
            raise ValueError(f"{method!r} needs a seed to be reproducible; the view should carry one")
        if method == "dare_linear":
            return dare_linear(inputs, weights, density, strict, seed, lambda_)
        return dare_ties(
            inputs, weights, density, strict, seed, False if normalize is None else normalize, lambda_
        )
    if method.endswith(":slices"):
        raise NotImplementedError(
            f"{method[:-7]!r} was imported from a slice configuration, which composes layer "
            f"ranges rather than whole deltas; the recipe is recorded and cannot be resolved here"
        )
    if method in RECORD_ONLY:
        raise NotImplementedError(
            f"{method!r} is recorded for provenance but not resolved here; "
            f"resolvable methods are {RESOLVABLE}"
        )
    raise ValueError(f"unknown merge method {method!r}")
