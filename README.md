# ballast

Version control for what a model has learned.

A fine-tuned adapter, a merged model, a per-user delta: each is a set of tensors on
top of a base model, and today each is a file in a bucket with a name someone
chose. `ballast` stores them content-addressed, chains them into a history per
tenant, records merges as views over their inputs, diffs versions by behaviour
rather than by bytes, and deletes with a proof that can be re-verified afterwards.

It runs without torch. The core is numpy, SQLite and blake3.

## A session

Three adapters over the same base, two of them successive runs of the same one.

```
$ ballast commit support-v1 -m "support adapter, run 41" --ref support
e650c06b351e  32 tensors  ref support
$ ballast commit support-v2 -m "run 42: retrained layers 3 and 9" --ref support
e4457af6f34d  32 tensors  ref support
$ ballast commit finance-v1 -m "finance adapter" --ref finance
94d436abdce3  32 tensors  ref finance

$ ballast stats
3 commits, 3 manifests, 66 chunks
8,650,752 bytes on disk for 12,582,912 logical (1.45x)

$ ballast log support
e4457af6f34d  leaf       run 42: retrained layers 3 and 9
e650c06b351e  leaf       support adapter, run 41

$ ballast diff e650c06b351e e4457af6f34d
e650c06b351e -> e4457af6f34d
  tensors: 2 changed of 32 shared, 0 removed, 0 added
  relative change: 0.0499
  no fingerprints on both commits; behavioural effect unknown

$ ballast merge support@0.6 finance@0.4 --method ties --density 0.5 -m "support/finance blend" --ref blend
16970e12af20  ties of 2 inputs  ref blend

$ ballast checkout blend -o ./blend-adapter
32 tensors -> ./blend-adapter

$ ballast forget e650c06b351e --reason "run 41 trained on withdrawn data"
tenant 'acme': 1 commits, 1 manifests, 2 chunks, 262,144 bytes
attestation 5171369ada4afbc62486eaa2ee4d1f0429d4de7b7da56c1ffa21c36f447467eb
verified

$ ballast checkout blend -o ./again
32 tensors -> ./again

$ ballast forget e4457af6f34d --reason "run 42 also trained on withdrawn data"
tenant 'acme': 1 commits, 1 manifests, 32 chunks, 4,194,304 bytes
attestation 8ba11df0b380b1466ffa03efa0a92955806b6d444320abc9b953d4b64362efbd
1 composite(s) now reference a deleted input and will refuse to resolve
verified

$ ballast checkout blend -o ./again
cannot check out: composite 83b97f22a066 cannot resolve: manifest 36ce0e22bf6f is missing for tenant 'acme'
exit=2

$ ballast log blend
16970e12af20  composite  support/finance blend
```

Run 42 changed two of thirty-two tensors, so it stored two chunks. The merge stored
nothing: it is a view, resolved when checked out. Forgetting run 41 removed exactly
the two chunks nothing else referenced. Forgetting run 42 — an input to the blend —
removed its chunks, named the composite it broke, and left the blend in history as
a record that refuses to resolve rather than a matrix with a deleted contribution
smeared through it.

## What it stores

**Chunks** are tensors, one file each, named by a hash over dtype, shape and bytes,
under the tenant's directory. Reads are memory-mapped. Two versions of an adapter
that differ in three tensors share every other chunk.

**Manifests** are either a leaf, which names tensors, or a composite, which names
other manifests and a merge method. A manifest's id is a hash of its content, so
committing the same adapter twice stores nothing new, and a composite can never
form a cycle because its id depends on inputs that already exist.

**Commits** chain manifests into a history. **Refs** name commits. `reset` moves a
ref; nothing is deleted.

Everything is keyed by tenant, in the schema. Chunks are not deduplicated across
tenants — different fine-tunes share essentially no tensors, and a global object
store would make one tenant's deletion proof depend on another tenant's data.

## Merges are views

```python
from ballast import Store

store = Store("./store")
store.merge("acme", "ties", [("support", 0.6), ("finance", 0.4)], density=0.5, message="blend")
```

Nothing is computed until `checkout`. `linear`, `task_arithmetic`, `ties` and
`dare_ties` resolve; `slerp`, `passthrough` and the others are recorded for
provenance and raise when asked to resolve, rather than doing something else
quietly. Arithmetic runs in float32 and casts back to the inputs' dtype.

A mergekit configuration imports as a composite, with each `model` entry mapped to
a ref, so the recipe is stored as the thing it describes:

```
ballast --tenant acme import-mergekit merge.yaml --map org/finance-lora=finance --map org/support-lora=support
```

## Diffs report behaviour, and say when they cannot

A diff of two low-rank matrices is noise. `diff` reports the relative Frobenius
change and, if both commits carry fingerprints for the same probe set, how many
probes produced a different answer.

```python
from ballast import FakeRunner, ProbeSet, fingerprint

probes = ProbeSet.of("What plan is this account on?", "Summarise the refund policy.")
fingerprint(store, "acme", "support", probes, runner)
```

A runner is anything with `run(tensors, config, base_model, probes) -> list[str]`.
`FakeRunner` is deterministic and used in tests. `ballast.runners.PeftRunner`
loads the adapter onto a base model through PEFT and generates; it is exercised
only by import in CI, since there is no GPU there, and has not been run against a
live model.

Three outcomes, and the difference between the last two is the point:

```
probes: 2 of 2 moved                                   the change was seen
no fingerprints on both commits; behavioural effect unknown
UNOBSERVED CHANGE: weights moved, no probe detected it
```

The last is weights that moved more than 1% with no probe noticing. It is reported
as its own condition so silence is never mistaken for stability.

## Deletion produces a record

`forget` removes a tenant entirely. `forget <commit>` removes one commit: children
are re-parented onto its parent, refs pointing at it move to its parent, its
manifest goes if no other commit uses it, and chunks nothing else references go
with it. Both return a `Proof` naming what was removed under an attestation hash
and writing tombstones the store keeps after the data is gone. `verify(proof)`
re-checks it against the store.

It is an audit record for an operator acting in good faith. It does not defend
against an operator who copied the chunk directory first.

## Loading into a model

`checkout` writes a PEFT adapter directory — `adapter_model.safetensors` and
`adapter_config.json` — which PEFT and vLLM load as they would any other.

## Scope

Anthropic-style hosted models expose no weights and are out of scope by
construction. This is for open-weight models and the deltas people train on them.

Cross-tenant composition — an org delta over a team delta over a user delta — is
not supported. Tenants are strictly separate, and layering across them needs a
share grant with its own deletion semantics.

The safetensors reader and writer are this package's own, so bfloat16 and float8
work without torch. They read and write the standard layout.

## Install

```
uv pip install -e ".[dev]"
pytest -q
```

`.[peft]` adds torch, transformers and peft for the PEFT runner.

## License

Apache-2.0
