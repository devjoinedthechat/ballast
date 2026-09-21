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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import blake3
import numpy as np

RESOLVABLE = (
    "linear",
    "task_arithmetic",
    "ties",
    "dare_ties",
    "dare_linear",
    "slerp",
    "breadcrumbs",
    "breadcrumbs_ties",
    "della",
    "della_linear",
    "passthrough",
    "nuslerp",
    "multislerp",
    "model_stock",
    "sce",
)
RECORD_ONLY = ("arcee_fusion", "karcher", "nearswap")
EXTRA_KEYS = frozenset({"row_wise", "flatten", "filter_wise", "select_topk"})
"""Parameters only one or two methods use, stored alongside a view's config."""

SEEDED = ("dare_ties", "dare_linear", "della", "della_linear")
"""Methods whose result depends on a random draw, so a view must store its seed."""

DEFAULT_DENSITY = 0.2
DEFAULT_GAMMA = 0.01
"""Breadcrumbs: the share of largest-magnitude entries dropped as outliers."""
DEFAULT_EPSILON = 0.15
"""DELLA: how far the keep probability swings either side of the density."""
EPS = 1e-7
DOT_THRESHOLD = 0.9995
"""Above this cosine the two tensors are colinear and interpolation is linear."""


def linear(
    inputs: Sequence[dict[str, np.ndarray]], weights: Sequence[float], strict: bool = True
) -> dict[str, np.ndarray]:
    """Weighted sum. Task arithmetic over deltas is the same operation."""
    return merge_all("linear", inputs, weights, Params(), strict)


def slerp(inputs: Sequence[dict[str, np.ndarray]], t: float, strict: bool = True) -> dict[str, np.ndarray]:
    """Spherical interpolation between exactly two inputs.

    Interpolates along the arc between the two tensors rather than the chord,
    which keeps the magnitude of the result closer to the magnitude of its
    inputs. `t` is how far to travel, 0 giving the first input and 1 the second.

    Two tensors that already point almost the same way have no meaningful arc —
    the angle between them is numerically zero and the divisions below blow up —
    so a cosine above `DOT_THRESHOLD` falls back to a straight line.
    """
    if len(inputs) != 2:
        raise ValueError(f"slerp interpolates between exactly two inputs, got {len(inputs)}")
    names, template = _names(inputs, strict)
    out: dict[str, np.ndarray] = {}
    for name in names:
        ref = template[name]
        v0 = inputs[0].get(name, np.zeros(ref.shape, dtype=ref.dtype)).astype(np.float32)
        v1 = inputs[1].get(name, np.zeros(ref.shape, dtype=ref.dtype)).astype(np.float32)
        out[name] = _slerp_one(t, v0, v1).astype(ref.dtype)
    return out


def _slerp_one(t: float, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    u0, u1 = _unit(v0), _unit(v1)
    dot = float(np.sum(u0 * u1))
    if abs(dot) > DOT_THRESHOLD:
        return (1 - t) * v0 + t * v1
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    theta_t = theta * t
    s0 = float(np.sin(theta - theta_t) / sin_theta)
    s1 = float(np.sin(theta_t) / sin_theta)
    return s0 * v0 + s1 * v1


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-8 else v


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
    return merge_all(
        "ties", inputs, weights, Params(density=density, normalize=normalize, lambda_=lambda_), strict
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
    return merge_all(
        "dare_ties",
        inputs,
        weights,
        Params(density=density, normalize=normalize, seed=seed, lambda_=lambda_),
        strict,
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
    return merge_all(
        "dare_linear", inputs, weights, Params(density=density, seed=seed, lambda_=lambda_), strict
    )


def breadcrumbs(
    inputs: Sequence[dict[str, np.ndarray]],
    weights: Sequence[float],
    density: float = DEFAULT_DENSITY,
    strict: bool = True,
    gamma: float = DEFAULT_GAMMA,
    elect: bool = False,
    lambda_: float = 1.0,
) -> dict[str, np.ndarray]:
    """Model Breadcrumbs (Davari & Belilovsky, 2024): drop the tails, keep the middle.

    TIES keeps the largest entries. Breadcrumbs argues the very largest are
    outliers that carry noise rather than skill, so it removes the top `gamma`
    fraction as well as the bottom, and merges what is left. `breadcrumbs_ties`
    is the same sparsifier with TIES' sign election on top.
    """
    method = "breadcrumbs_ties" if elect else "breadcrumbs"
    return merge_all(method, inputs, weights, Params(density=density, gamma=gamma, lambda_=lambda_), strict)


def della(
    inputs: Sequence[dict[str, np.ndarray]],
    weights: Sequence[float],
    density: float = DEFAULT_DENSITY,
    strict: bool = True,
    seed: int = 0,
    epsilon: float = DEFAULT_EPSILON,
    elect: bool = True,
    normalize: bool = True,
    lambda_: float = 1.0,
) -> dict[str, np.ndarray]:
    """DELLA (Deep et al., 2024): drop by rank rather than uniformly.

    DARE keeps every entry with the same probability. DELLA ranks entries by
    magnitude within each row and makes the largest likelier to survive, sliding
    the keep probability from `density - epsilon` at the smallest to
    `density + epsilon` at the largest. Like DARE it draws a mask, so a view
    carries its seed.
    """
    method = "della" if elect else "della_linear"
    return merge_all(
        method,
        inputs,
        weights,
        Params(density=density, normalize=normalize, seed=seed, epsilon=epsilon, lambda_=lambda_),
        strict,
    )


@dataclass(frozen=True)
class Params:
    """Everything a method needs beyond its inputs and their weights."""

    density: float = DEFAULT_DENSITY
    normalize: bool | None = None
    lambda_: float = 1.0
    gamma: float = DEFAULT_GAMMA
    epsilon: float = DEFAULT_EPSILON
    seed: int | None = None
    t: float | None = None
    row_wise: bool = False
    """nuSLERP: interpolate along rows rather than the last axis."""
    flatten: bool = False
    """nuSLERP: treat the whole tensor as one vector."""
    filter_wise: bool = False
    """Model Stock: one angle per row rather than one for the tensor."""
    select_topk: float = 1.0
    """SCE: the share of entries kept, chosen by how much the inputs disagree."""


def merge_tensor(
    method: str,
    name: str,
    arrays: Sequence[np.ndarray | None],
    weights: Sequence[float],
    params: Params,
) -> np.ndarray:
    """Merge one named tensor across the inputs.

    The unit every method is built from, and the reason a view can be resolved
    without holding its inputs: a caller that can fetch one tensor from each
    input can produce the whole result one tensor at a time.

    `None` means an input does not carry this tensor, which only happens in a
    union merge and counts as zero. `name` is not decoration — the random
    methods derive their mask from it, so that a tensor's draw does not depend
    on what else is being merged alongside it.
    """
    present = [a for a in arrays if a is not None]
    if not present:
        raise ValueError(f"no input carries {name!r}")
    shapes = {a.shape for a in present}
    if len(shapes) != 1:
        raise ValueError(f"tensor {name!r} has different shapes across inputs: {shapes}")
    ref = present[0]
    values = [np.zeros(ref.shape, dtype=np.float32) if a is None else a.astype(np.float32) for a in arrays]

    if method in ("linear", "task_arithmetic"):
        out = sum(w * v for v, w in zip(values, weights, strict=True))
        return np.asarray(out).astype(ref.dtype)

    if method == "slerp":
        if params.t is None:
            raise ValueError("slerp needs a t; the view should carry one")
        if len(values) != 2:
            raise ValueError(f"slerp interpolates between exactly two inputs, got {len(values)}")
        return _slerp_one(params.t, values[0], values[1]).astype(ref.dtype)

    if method == "passthrough":
        if len(values) != 1:
            raise ValueError(f"passthrough takes exactly one input, got {len(values)}")
        return (values[0] * weights[0]).astype(ref.dtype)

    if method == "nuslerp":
        if len(values) != 2:
            raise ValueError(f"nuslerp interpolates between exactly two inputs, got {len(values)}")
        total = weights[0] + weights[1]
        # Weights that cancel leave no direction to travel in; halfway is the
        # least surprising answer, and mergekit's.
        share = 0.5 if abs(total) < 1e-6 else weights[1] / total
        return _nuslerp(share, values[0], values[1], params.row_wise, params.flatten).astype(ref.dtype)

    if method == "multislerp":
        return _multislerp(values, weights, params.normalize is not False).astype(ref.dtype)

    if method == "model_stock":
        return _model_stock(values, params.filter_wise).astype(ref.dtype)

    if method == "sce":
        return _sce(values, params.select_topk).astype(ref.dtype)

    spec = _SPECS.get(method)
    if spec is None:
        raise ValueError(f"unknown merge method {method!r}")
    sparsify, elect, default_normalize = spec
    if not 0.0 < params.density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {params.density}")

    prepared = []
    for index, value in enumerate(values):
        if sparsify == "magnitude":
            prepared.append(_trim(value, params.density))
        elif sparsify == "outliers":
            prepared.append(_trim_outliers(value, params.density, params.gamma))
        elif sparsify == "rank":
            prepared.append(
                _rank_drop(value, params.density, params.epsilon, _seed_for(params.seed, index, name))
            )
        elif sparsify == "random":
            prepared.append(_drop_and_rescale(value, params.density, _seed_for(params.seed, index, name)))
        else:
            prepared.append(value)

    stacked = np.stack([w * t for t, w in zip(prepared, weights, strict=True)])
    if not elect:
        merged = stacked.sum(axis=0)
    else:
        elected = np.where(stacked.sum(axis=0) >= 0, 1.0, -1.0)
        agrees = np.sign(stacked) == elected
        merged = np.where(agrees, stacked, 0.0).sum(axis=0)
        normalize = default_normalize if params.normalize is None else params.normalize
        if normalize:
            divisor = np.stack([w * agrees[i] for i, w in enumerate(weights)]).sum(axis=0)
            merged = merged / np.where(divisor == 0, 1.0, divisor)
    if params.lambda_ != 1.0:
        merged = merged * params.lambda_
    return np.asarray(merged).astype(ref.dtype)


# method -> (sparsifier, elect a sign, normalise by default)
_SPECS: dict[str, tuple[str, bool, bool]] = {
    "ties": ("magnitude", True, True),
    "dare_ties": ("random", True, False),
    "dare_linear": ("random", False, False),
    "breadcrumbs": ("outliers", False, False),
    "breadcrumbs_ties": ("outliers", True, False),
    "della": ("rank", True, True),
    "della_linear": ("rank", False, False),
}


def merge_all(
    method: str,
    inputs: Sequence[dict[str, np.ndarray]],
    weights: Sequence[float],
    params: Params,
    strict: bool = True,
) -> dict[str, np.ndarray]:
    """Every tensor at once, for callers that already hold the inputs."""
    names = _union(inputs, strict)
    return {
        name: merge_tensor(method, name, [t.get(name) for t in inputs], weights, params) for name in names
    }


def _union(inputs: Sequence[dict[str, np.ndarray]], strict: bool) -> list[str]:
    """The tensors to merge.

    Strict requires every input to carry the same ones. Non-strict takes the
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
    return sorted(names)


def _nuslerp(t: float, v0: np.ndarray, v1: np.ndarray, row_wise: bool, flatten: bool) -> np.ndarray:
    """Slerp that interpolates each row separately rather than the whole tensor.

    Plain SLERP treats a weight matrix as one long vector, so a single angle
    describes the whole thing. nuSLERP works along the last axis, which for a
    weight matrix means one angle per row, and usually preserves more structure.
    """
    shape = v0.shape
    if flatten:
        a, b = v0.reshape(1, -1), v1.reshape(1, -1)
    elif row_wise and v0.ndim >= 2:
        a, b = np.swapaxes(v0, 0, -1), np.swapaxes(v1, 0, -1)
    else:
        a, b = (v0[None, :], v1[None, :]) if v0.ndim == 1 else (v0, v1)

    ua = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-7)
    ub = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-7)
    cos_theta = np.clip(np.sum(ua * ub, axis=-1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / sin_theta
    colinear = np.abs(sin_theta) < 1e-8
    out = np.where(colinear, (1 - t) * a + t * b, out)

    if flatten:
        return out.reshape(shape)
    if row_wise and v0.ndim >= 2:
        return np.swapaxes(out, 0, -1)
    return out.reshape(shape)


def _multislerp(values: Sequence[np.ndarray], weights: Sequence[float], normalize: bool) -> np.ndarray:
    """Barycentric interpolation on a hypersphere, for more than two inputs.

    SLERP walks an arc between two points. With three or more there is no single
    arc, so each input is projected onto the unit sphere, averaged in the tangent
    space at their weighted mean, and mapped back. The magnitude comes from the
    weighted average of the inputs' own norms.
    """
    if len(values) == 1:
        return values[0]
    shape = values[0].shape
    stacked = np.stack([v.reshape(-1) for v in values])
    w = np.asarray(weights, dtype=np.float32)
    if normalize:
        total = w.sum()
        w = w / total if abs(total) > 1e-8 else np.full_like(w, 1.0 / len(w))

    norms = np.linalg.norm(stacked, axis=-1, keepdims=True)
    unit = stacked / (norms + 1e-8)
    mean = (unit * w[:, None]).sum(0)
    mean_norm = float(np.linalg.norm(mean))
    if mean_norm < 1e-8:
        if len(values) == 2:
            # Antipodal pair: there is no mean direction, so fall back to a line.
            return np.asarray(stacked[0] * w[0] + stacked[1] * w[1]).reshape(shape)
        raise ValueError(
            "the weighted inputs cancel out, so they have no mean direction to "
            "interpolate around; change the weights or use a different method"
        )
    mean = mean / mean_norm

    tangent = unit - (unit * mean).sum(-1, keepdims=True) * mean
    combined = (tangent * w[:, None]).sum(0)
    tangent_norm = float(np.linalg.norm(combined)) + 1e-8
    result = mean * np.cos(tangent_norm) + combined * (np.sin(tangent_norm) / tangent_norm)
    return np.asarray(result * float((norms.squeeze(-1) * w).sum())).reshape(shape)


def _model_stock(values: Sequence[np.ndarray], filter_wise: bool) -> np.ndarray:
    """Model Stock (Jang et al., 2024): interpolate toward the mean by angle.

    How far to move from the base toward the average of the fine-tunes is set by
    how much they agree: the closer the angle between their offsets, the further
    it is safe to go. Over deltas the base is the origin, so the offsets are the
    deltas themselves.

    The paper gives an exact angle for two models and does not say what to do
    with more; mergekit averages the pairwise angles, and so does this.
    """
    if len(values) < 2:
        raise ValueError(f"model_stock needs at least two inputs, got {len(values)}")
    shape = values[0].shape
    offsets = (
        [v[None, :] if v.ndim == 1 else v for v in values] if filter_wise else [v.reshape(-1) for v in values]
    )

    cosines = []
    for i, a in enumerate(offsets):
        for b in offsets[i + 1 :]:
            product = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
            cosines.append(np.clip((a * b).sum(axis=-1) / np.maximum(product, 1e-6), -1.0, 1.0))
    cos_theta = np.stack(cosines).mean(axis=0)[..., None]

    n = len(offsets)
    t = (n * cos_theta) / (1 + (n - 1) * cos_theta)
    average = sum(offsets) / n
    return np.asarray(t * average).reshape(shape)


def _sce(values: Sequence[np.ndarray], select_topk: float) -> np.ndarray:
    """SCE (Yang et al., 2024): select by variance, weight by magnitude, erase by sign.

    Entries where the inputs disagree most carry the most information, so those
    are selected. Each input is then weighted by its mean square, and entries
    that disagree with the elected sign are erased before the weighted average.
    """
    if not values:
        raise ValueError("sce needs at least one input")
    stacked = np.stack(values)
    if select_topk < 1.0:
        stacked = stacked * _variance_mask(stacked, select_topk)

    elected = np.where(stacked.sum(axis=0) >= 0, 1.0, -1.0)
    erase = (np.sign(stacked) == elected).astype(np.float32)

    magnitudes = np.mean(stacked**2, axis=tuple(range(1, stacked.ndim)))
    total = float(magnitudes.sum())
    share = np.full_like(magnitudes, 1.0 / len(magnitudes)) if abs(total) < 1e-6 else magnitudes / total
    weights = share.reshape((-1,) + (1,) * (stacked.ndim - 1)) * erase
    merged = (stacked * weights).sum(axis=0)
    return np.asarray(merged / np.maximum(weights.sum(axis=0), 1e-6))


def _variance_mask(stacked: np.ndarray, density: float) -> np.ndarray:
    """Keep the entries the inputs disagree about most."""
    if density <= 0:
        return np.zeros(stacked.shape[1:], dtype=np.float32)
    variance = stacked.var(axis=0)
    nonzero = int(np.count_nonzero(variance))
    keep = int(nonzero * density)
    if keep == 0:
        return np.zeros_like(variance, dtype=np.float32)
    flat = np.abs(variance).ravel()
    order = np.argsort(-flat, kind="stable")[:keep]
    mask = np.zeros(flat.size, dtype=np.float32)
    mask[order] = 1.0
    return mask.reshape(variance.shape)


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


def _trim_outliers(array: np.ndarray, density: float, gamma: float) -> np.ndarray:
    """Keep the middle band: drop the largest `gamma` share and enough of the smallest.

    When the density leaves no room for the outlier cut, the cut shrinks rather
    than the target density, which is mergekit's own resolution.
    """
    if density >= 1.0 or array.size == 0:
        return array
    total = array.size
    target = int(density * total)
    top = int(gamma * total)
    bottom = total - target - top
    if bottom < 0:
        top += bottom
        bottom = 0
    order = np.argsort(np.abs(array).ravel(), kind="stable")
    keep = order[bottom : total - top] if top > 0 else order[bottom:]
    mask = np.zeros(total, dtype=bool)
    mask[keep] = True
    return np.where(mask.reshape(array.shape), array, 0.0)


def _rank_drop(array: np.ndarray, density: float, epsilon: float, seed: int) -> np.ndarray:
    """Keep each entry with a probability that rises with its rank in its row.

    Ranks run within axis 1, as mergekit's does, so a 1-D tensor is treated as a
    single row. The L1 norm is restored afterwards, as in DARE.
    """
    if not (density - epsilon > 0.0 and density + epsilon < 1.0):
        raise ValueError(
            f"epsilon must keep density +/- epsilon inside (0, 1); "
            f"density {density} with epsilon {epsilon} does not"
        )
    if density >= 1.0 or array.size == 0:
        return array
    work = array.reshape(1, -1) if array.ndim < 2 else array
    magnitudes = np.abs(work)
    ranks = np.argsort(np.argsort(magnitudes, axis=1, kind="stable"), axis=1, kind="stable") + 1.0
    low = ranks.min(axis=1, keepdims=True)
    high = ranks.max(axis=1, keepdims=True)
    span = np.where(high == low, 1.0, high - low)
    rank_norm = np.clip((ranks - low) / span, 0.0, 1.0)
    probs = (density - epsilon) + rank_norm * 2 * epsilon

    rng = np.random.default_rng(seed)
    mask = rng.random(work.shape) < probs
    masked = np.where(mask, work, 0.0)
    before = float(np.abs(work).sum())
    after = float(np.abs(masked).sum())
    if before >= EPS and after >= EPS:
        masked = masked * (before / after)
    return masked.reshape(array.shape)


def _seed_for(seed: int | None, index: int, name: str) -> int:
    """A tensor's own seed, from the view's seed, the input's position and the name.

    Derived rather than sequential so the draw for one tensor does not depend on
    how many came before it. A view resolved on its own gives the same answer as
    the same view resolved inside a larger one.
    """
    if seed is None:
        raise ValueError("this method draws a random mask and needs a seed; the view should carry one")
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
    gamma: float = DEFAULT_GAMMA,
    epsilon: float = DEFAULT_EPSILON,
    t: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Resolve a view, holding every input.

    `normalize` and `lambda_` are mergekit's own parameters, and a real config
    sets them: TIES defaults to normalising and DARE does not, and either can be
    overridden. Ignoring them would mean a view that reads like a recipe and
    resolves to something else.
    """
    check(method)
    params = params_for(density, normalize, lambda_, gamma, epsilon, seed, t, extra)
    return merge_all(method, inputs, weights, params, strict)


def params_for(
    density: float | None = None,
    normalize: bool | None = None,
    lambda_: float = 1.0,
    gamma: float = DEFAULT_GAMMA,
    epsilon: float = DEFAULT_EPSILON,
    seed: int | None = None,
    t: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Params:
    """Gather a view's stored configuration into the shape the core takes.

    `extra` carries the parameters only one or two methods use, so the common
    call does not have to name them and an unknown one is refused rather than
    ignored.
    """
    unknown = set(extra or {}) - EXTRA_KEYS
    if unknown:
        raise ValueError(f"unknown merge parameters {sorted(unknown)}; known extras are {sorted(EXTRA_KEYS)}")
    return Params(
        density=DEFAULT_DENSITY if density is None else density,
        normalize=normalize,
        lambda_=lambda_,
        gamma=gamma,
        epsilon=epsilon,
        seed=seed,
        t=t,
        row_wise=bool((extra or {}).get("row_wise", False)),
        flatten=bool((extra or {}).get("flatten", False)),
        filter_wise=bool((extra or {}).get("filter_wise", False)),
        select_topk=float((extra or {}).get("select_topk", 1.0)),
    )


def check(method: str) -> None:
    """Refuse a method this resolver does not implement, before any work."""
    if method in RESOLVABLE:
        return
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
