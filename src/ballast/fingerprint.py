"""Behavioural fingerprints.

A diff of two low-rank matrices is noise. The only way to say what changed
between two versions of a delta is to ask both the same questions and compare
the answers. A probe set is the questions; a fingerprint is one version's
answers; the store keeps them per commit so a diff can report how many moved.

The store does not know how to run a model. A `Runner` does, and the two that
ship are a deterministic fake for tests and a PEFT-backed one for real use. The
protocol is small on purpose: anything that can turn tensors and prompts into
strings can fingerprint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ballast.hashing import object_hash, tensor_hash


@dataclass(frozen=True)
class ProbeSet:
    probes: tuple[str, ...]

    @property
    def id(self) -> str:
        return object_hash(list(self.probes))[:16]

    @classmethod
    def of(cls, *probes: str) -> ProbeSet:
        return cls(tuple(probes))


class Runner(Protocol):
    def run(
        self,
        tensors: dict[str, np.ndarray],
        config: dict[str, Any],
        base_model: str | None,
        probes: Sequence[str],
    ) -> list[str]:
        """One output string per probe."""


class FakeRunner:
    """Deterministic outputs that change when the tensors change.

    Sensitive to the tensors named in `watch`; blind to the rest. That makes it
    possible to test the unobserved-change path, where weights move and no probe
    notices, without a model in the loop.
    """

    def __init__(self, watch: Sequence[str] | None = None) -> None:
        self.watch = tuple(watch) if watch is not None else None

    def run(
        self,
        tensors: dict[str, np.ndarray],
        config: dict[str, Any],
        base_model: str | None,
        probes: Sequence[str],
    ) -> list[str]:
        names = sorted(tensors) if self.watch is None else [n for n in sorted(tensors) if n in self.watch]
        state = object_hash([tensor_hash(tensors[n]) for n in names])
        return [object_hash([state, probe])[:12] for probe in probes]


def fingerprint(store: Any, tenant: str, spec: str, probe_set: ProbeSet, runner: Runner) -> list[str]:
    """Run the probes against a commit and record the answers."""
    commit = store.resolve(tenant, spec)
    manifest = store.manifest(tenant, commit.manifest_id)
    tensors = store.checkout(tenant, spec)
    outputs = runner.run(tensors, manifest.config, manifest.base_model, probe_set.probes)
    if len(outputs) != len(probe_set.probes):
        raise ValueError(f"runner returned {len(outputs)} outputs for {len(probe_set.probes)} probes")
    store.record_fingerprint(tenant, spec, probe_set.probes, outputs)
    return outputs
