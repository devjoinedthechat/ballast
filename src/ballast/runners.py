"""A runner backed by transformers and PEFT.

Optional. Imports its dependencies when constructed, not when the module loads,
so the store works on a machine that has never seen torch.

The adapter is applied in process: the PEFT wrapper is built once from the
adapter config, and each run loads tensors straight into it with
`set_peft_model_state_dict`. Nothing is written to disk between the store and
the model. A session-scoped delta that changes every few minutes should not
pay for an export and a reload each time.

Verified against a live model on CPU by scripts/verify_real.py. CI only imports
it; there is no GPU there.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np


class PeftRunner:
    def __init__(self, base_model: str, device: str = "cpu", max_new_tokens: int = 32) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32).to(device)  # type: ignore[arg-type]
        self.base.eval()
        self.model: Any = None
        self._config_key: str | None = None

    def _ensure_wrapped(self, config: dict[str, Any]) -> None:
        """Build the PEFT wrapper for this adapter config, rebuilding if it changed."""
        from peft import get_peft_model  # noqa: PLC0415
        from peft.config import PeftConfig  # noqa: PLC0415

        key = json.dumps(config, sort_keys=True)
        if self.model is not None and key == self._config_key:
            return
        if self.model is not None:
            # A different adapter shape: strip the old wrapper before adding a new one.
            self.base = self.model.unload()
            self.model = None
        peft_config = PeftConfig.from_peft_type(  # type: ignore[no-untyped-call]
            **{**config, "inference_mode": True}
        )
        self.model = get_peft_model(self.base, peft_config)
        self.model.eval()
        self._config_key = key

    def run(
        self,
        tensors: dict[str, np.ndarray],
        config: dict[str, Any],
        base_model: str | None,
        probes: Sequence[str],
    ) -> list[str]:
        from peft import set_peft_model_state_dict  # noqa: PLC0415

        self._ensure_wrapped(config)
        state = {
            name: self._torch.from_numpy(np.ascontiguousarray(array.astype(np.float32)))
            for name, array in tensors.items()
        }
        result = set_peft_model_state_dict(self.model, state)
        unexpected = getattr(result, "unexpected_keys", None)
        if unexpected:
            raise ValueError(
                f"{len(unexpected)} tensors did not map onto the adapter: {sorted(unexpected)[:3]}"
            )

        outputs: list[str] = []
        with self._torch.no_grad():
            for probe in probes:
                ids = self.tokenizer(probe, return_tensors="pt").to(self.device)
                generated = self.model.generate(  # type: ignore[no-untyped-call]
                    **ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
                text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                outputs.append(text if isinstance(text, str) else "".join(text))
        return outputs
