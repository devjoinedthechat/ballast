"""Move adapters between the store and the directory layout PEFT and vLLM load.

An adapter directory is `adapter_model.safetensors` plus `adapter_config.json`.
Tensor names are kept exactly as found; the store has no opinion about them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ballast import tensors as st

WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"
PROVENANCE = "ballast.json"
"""Written next to the adapter on checkout: which store, tenant, commit and
manifest it came from, and for a view, the recipe. Provenance travels with the
artifact instead of living in whoever's memory did the export."""


def load(directory: Path | str) -> tuple[dict[str, np.ndarray], dict[str, Any], str | None]:
    """Tensors, adapter config, and the base model the config names."""
    directory = Path(directory)
    weights = directory / WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found")
    arrays, _ = st.load(weights)
    config: dict[str, Any] = {}
    config_path = directory / CONFIG
    if config_path.exists():
        config = json.loads(config_path.read_text())
    base = config.get("base_model_name_or_path")
    return arrays, config, base


def export(
    directory: Path | str,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    st.save(directory / WEIGHTS, arrays, {"format": "pt"})
    (directory / CONFIG).write_text(json.dumps(config or {}, indent=2, sort_keys=True))
    if provenance is not None:
        (directory / PROVENANCE).write_text(json.dumps(provenance, indent=2, sort_keys=True))
    return directory
