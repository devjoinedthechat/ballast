"""Import a mergekit configuration as a composite manifest.

mergekit is where merge provenance is lost today: a YAML file describes the
recipe, the output is a directory of tensors, and the link between them is a
commit message if anyone wrote one. Importing the recipe records it as a view
over the inputs, so the result is reproducible from the store and the inputs
can be traced from the output.

Each `model` entry has to already be committed. By default its name is taken as
a ref; pass `refs` to map names to refs explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ballast.store import Commit, Store


def parse(path: Path | str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "merge_method" not in raw:
        raise ValueError(f"{path} does not look like a mergekit config")
    return raw


def import_config(
    store: Store,
    tenant: str,
    path: Path | str,
    *,
    refs: Mapping[str, str] | None = None,
    ref: str = "main",
    message: str | None = None,
) -> Commit:
    config = parse(path)
    method = config["merge_method"]
    defaults = config.get("parameters") or {}
    density = defaults.get("density")

    inputs: list[tuple[str, float]] = []
    for entry in config.get("models") or []:
        name = entry["model"] if isinstance(entry, dict) else str(entry)
        params = (entry.get("parameters") or {}) if isinstance(entry, dict) else {}
        weight = float(params.get("weight", defaults.get("weight", 1.0)))
        if density is None and "density" in params:
            density = params["density"]
        spec = (refs or {}).get(name, name)
        inputs.append((spec, weight))
    if not inputs:
        raise ValueError("mergekit config lists no models")

    return store.merge(
        tenant,
        method,
        inputs,
        message=message or f"mergekit {method} of {len(inputs)} inputs from {Path(path).name}",
        ref=ref,
        density=float(density) if density is not None else None,
    )
