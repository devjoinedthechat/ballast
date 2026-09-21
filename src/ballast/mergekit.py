"""Import a mergekit configuration as a view.

mergekit is where merge provenance is lost today: a YAML file describes the
recipe, the output is a directory of tensors, and the link between them is a
commit message if anyone wrote one. Importing the recipe records it as a view
over the inputs, so the result is reproducible from the store and the inputs can
be traced from the output.

Real configurations in the wild carry more than a method and a list of models,
and the reader takes the lot: per-model `density` and `weight`, top-level
`normalize` and `lambda`, `base_model`, `dtype`, `tokenizer_source`, and the
`slices` form that takes different layer ranges from different models.

What it does not do is guess. A parameter this resolver acts on is stored where
the resolver reads it; everything else is recorded under `provenance`, where it
describes the recipe without changing the result. A `slices` merge is recorded
and refuses to resolve, because ballast composes whole deltas and a slice merge
is not one.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ballast import merge as merging
from ballast.store import Commit, Store

# Parameters this resolver acts on. Anything else is recorded, not interpreted.
ACTED_ON = frozenset({"weight", "density", "normalize", "lambda", "gamma", "epsilon", "t"})


class SliceMerge(ValueError):
    """A configuration that takes layer ranges rather than whole models."""


def parse(path: Path | str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "merge_method" not in raw:
        raise ValueError(f"{path} does not look like a mergekit config")
    return raw


def _entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    """The `models` list, normalised to dicts.

    A `slices` configuration has none, and is refused with its shape named
    rather than silently flattened into something that merges differently.
    """
    models = config.get("models")
    if not models:
        if config.get("slices"):
            raise SliceMerge(
                "this config merges layer ranges (`slices`), which composes parts of models "
                "rather than whole deltas; import it with allow_slices=True to record it unresolved"
            )
        raise ValueError("mergekit config lists no models")
    return [m if isinstance(m, dict) else {"model": str(m)} for m in models]


def _slice_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Every distinct model a `slices` config draws from, in first-seen order."""
    seen: dict[str, dict[str, Any]] = {}
    for section in config.get("slices") or []:
        for source in section.get("sources") or []:
            name = source["model"] if isinstance(source, dict) else str(source)
            seen.setdefault(name, source if isinstance(source, dict) else {"model": name})
    return list(seen.values())


def import_config(
    store: Store,
    tenant: str,
    path: Path | str,
    *,
    refs: Mapping[str, str] | None = None,
    ref: str = "main",
    message: str | None = None,
    seed: int | None = None,
    allow_slices: bool = False,
) -> Commit:
    """Record a mergekit configuration as a view over commits already in the store.

    Each `model` entry is mapped to a ref through `refs`, or taken as a ref name
    when it is not listed.
    """
    config = parse(path)
    method = config["merge_method"]
    defaults = config.get("parameters") or {}

    sliced = False
    try:
        entries = _entries(config)
    except SliceMerge:
        if not allow_slices:
            raise
        entries, sliced = _slice_models(config), True

    inputs: list[tuple[str, float]] = []
    densities: list[float] = []
    for entry in entries:
        params = entry.get("parameters") or {}
        weight = _scalar(params.get("weight", defaults.get("weight", 1.0)))
        if "density" in params or "density" in defaults:
            densities.append(_scalar(params.get("density", defaults.get("density"))))
        name = entry["model"]
        inputs.append(((refs or {}).get(name, name), weight))
    if not inputs:
        raise ValueError("mergekit config lists no models")

    if len(set(densities)) > 1:
        raise ValueError(
            f"per-model densities differ ({sorted(set(densities))}); a view carries one density, "
            f"so this recipe cannot be recorded faithfully"
        )
    density = densities[0] if densities else None

    # SLERP interpolates from the base model to the other one, so the base is the
    # first input whether or not the config lists it among `models`.
    if method == "slerp" and config.get("base_model"):
        base_ref = (refs or {}).get(config["base_model"], config["base_model"])
        if base_ref not in [spec for spec, _ in inputs]:
            inputs.insert(0, (base_ref, 1.0))

    resolvable = method in merging.RESOLVABLE and not sliced
    provenance: dict[str, Any] = {"source": Path(path).name}
    for key in ("base_model", "dtype", "tokenizer_source", "out_dtype", "chat_template"):
        if config.get(key) is not None:
            provenance[key] = config[key]
    # A parameter is left out of provenance only when the resolver genuinely
    # acted on all of it. Nothing is acted on in a recipe that cannot resolve,
    # and a gradient is only acted on at its first point, so the list is kept.
    acted = ACTED_ON if resolvable else frozenset()
    extra = {key: value for key, value in defaults.items() if key not in acted or isinstance(value, list)}
    if extra:
        provenance["parameters"] = extra
    if sliced:
        provenance["slices"] = config["slices"]
        provenance["unresolvable"] = "slice merges compose layer ranges, not whole deltas"

    return store.merge(
        tenant,
        method if resolvable else f"{method}:slices" if sliced else method,
        inputs,
        message=message or f"mergekit {method} of {len(inputs)} inputs from {Path(path).name}",
        ref=ref,
        density=density,
        seed=seed if method in merging.SEEDED and resolvable else None,
        normalize=_maybe_bool(defaults.get("normalize")),
        lambda_=_scalar(defaults.get("lambda", 1.0)),
        gamma=_scalar(defaults["gamma"]) if "gamma" in defaults else None,
        epsilon=_scalar(defaults["epsilon"]) if "epsilon" in defaults else None,
        t=_scalar(defaults["t"]) if "t" in defaults and resolvable else None,
        provenance=provenance,
    )


def _scalar(value: Any) -> float:
    """Take a parameter that may be a number or mergekit's gradient form.

    A gradient varies the value by layer. A view carries one number, so the
    first point is taken and the whole list is kept in provenance by the caller.
    """
    if isinstance(value, list):
        if not value:
            raise ValueError("empty parameter list")
        first = value[0]
        return float(first["value"] if isinstance(first, dict) else first)
    return float(value)


def _maybe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)
