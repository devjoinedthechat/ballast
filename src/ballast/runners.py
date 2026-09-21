"""A runner backed by transformers and PEFT.

Optional. Imports its dependencies when constructed, not when the module loads,
so the store works on a machine that has never seen torch.

Attaching an adapter with `PeftModel.from_pretrained` modifies the base model in
place, so a second call would stack adapters. Every run unloads the adapter
afterwards and keeps the restored base for the next one.

Verified against a live model on CPU by scripts/verify_real.py. CI only imports
it; there is no GPU there.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ballast import peft as peft_io


class PeftRunner:
    def __init__(self, base_model: str, device: str = "cpu", max_new_tokens: int = 32) -> None:
        import torch  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        self._torch = torch
        self._PeftModel = PeftModel
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32).to(device)  # type: ignore[arg-type]
        self.base.eval()

    def run(
        self,
        tensors: dict[str, np.ndarray],
        config: dict[str, Any],
        base_model: str | None,
        probes: Sequence[str],
    ) -> list[str]:
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            peft_io.export(Path(tmp), tensors, config)
            model = self._PeftModel.from_pretrained(self.base, tmp).to(self.device)
            model.eval()
            outputs: list[str] = []
            try:
                with self._torch.no_grad():
                    for probe in probes:
                        ids = self.tokenizer(probe, return_tensors="pt").to(self.device)
                        generated = model.generate(  # type: ignore[no-untyped-call]
                            **ids,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        )
                        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                        outputs.append(text if isinstance(text, str) else "".join(text))
            finally:
                # Strip the adapter layers and keep the bare base for the next run.
                self.base = model.unload()
            return outputs
