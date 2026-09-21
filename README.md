# ballast

Version control for what a model has learned.

A fine-tuned adapter, a merged model, a per-user delta: each is a set of tensors on
top of a base model, and today each is a file in a bucket with a name someone
chose. `ballast` stores them content-addressed, chains them into a history per
tenant, records merges as views over their inputs, diffs versions by behaviour
rather than by bytes, shares them across tenants through revocable grants, and
deletes with a proof that can be re-verified from another process afterwards.

The core is numpy, SQLite and blake3. It runs without torch.

## A session

Two tenants. `acme` has three adapters over the same base, two of them successive
runs of the same one. `partner` builds on one of them through a grant.

```
$ ballast --tenant acme commit support-v1 -m "support adapter, run 41" --ref support --metadata '{"run": 41}'
a3d509ff9689  32 tensors  ref support

$ ballast --tenant acme commit support-v2 -m "run 42: retrained layers 3 and 9" --ref support
c02850e20a1c  32 tensors  ref support

$ ballast --tenant acme commit finance-v1 -m "finance adapter" --ref finance
2b30d87d4370  32 tensors  ref finance

$ ballast --tenant acme stats
3 commits, 3 manifests, 66 chunks
8,650,752 bytes on disk for 12,582,912 logical (1.45x)

$ ballast --tenant acme log support
c02850e20a1c  leaf       run 42: retrained layers 3 and 9
a3d509ff9689  leaf       support adapter, run 41

$ ballast --tenant acme diff a3d509ff9689 c02850e20a1c
a3d509ff9689 -> c02850e20a1c
  tensors: 2 changed of 32 shared, 0 removed, 0 added
  relative change: 0.0499 overall, 0.2000 in the most changed tensor
  no fingerprints on both commits; behavioural effect unknown

$ ballast --tenant acme merge support@0.6 finance@0.4 --method ties --density 0.5 -m "blend" --ref blend
d8ab4508172a  ties of 2 inputs  ref blend

$ ballast --tenant acme checkout blend -o ./blend-adapter
32 tensors -> ./blend-adapter

$ cat blend-adapter/ballast.json
{
  "base_model": "meta-llama/Llama-3.1-8B",
  "commit": "d8ab4508172a286e45e36b50ed4a989308a94186fa3e5e2c8b71cf5503bc091b",
  "kind": "composite",
  "manifest": "2d16ace07c1b447e32db3fff42c56822bc364e083f8473861886d8e7174a1237",
  "message": "support/finance blend",
  "metadata": {},
  "recipe": {
    "density": 0.5,
    "inputs": [
      {
        "manifest": "36ce0e22bf6fba802340f82db73800da462ccc7fa7e362ee596ae4dda7f80100",
        "tenant": "acme",
        "weight": 0.6
      },
      {
        "manifest": "175b5a05edc3262b192fe6eac2edee90c95b25fd59eafbde7628f76470ade73e",
        "tenant": "acme",
        "weight": 0.4
      }
    ],
    "method": "ties"
  },
  "store": "…",
  "tenant": "acme"
}
$ ballast --tenant acme grant finance --to partner
partner may build views over acme:175b5a05edc3

$ ballast --tenant partner commit finance-v1 -m "partner adapter"
25887c16f76e  32 tensors  ref main

$ ballast --tenant partner merge acme:finance@0.5 main@0.5 -m "layered on acme" --ref layered
bfb82ad84dde  linear of 2 inputs  ref layered

$ ballast --tenant partner checkout layered -o ./layered
32 tensors -> ./layered

$ ballast --tenant acme revoke finance --to partner
revoked; 1 view(s) in 'partner' can no longer resolve

$ ballast --tenant partner checkout layered -o ./layered
cannot check out: composite 15be099ddfa8 cannot resolve: grant on acme:175b5a05edc3 for 'partner' is missing or revoked
exit=2

$ ballast --tenant acme forget c02850e20a1c --reason "run 42 trained on withdrawn data"
tenant 'acme': 1 commits, 1 manifests, 2 chunks, 262,144 bytes
attestation 1e2e80f4987360ffa9aec4668e79d0e359bd31b28d9a441ebb978c12b1b4f321
1 composite(s) now reference a deleted input and will refuse to resolve: acme:2d16ace07c1b447
verified

$ ballast --tenant acme checkout blend -o ./again
cannot check out: composite 2d16ace07c1b cannot resolve: manifest 36ce0e22bf6f is missing for tenant 'acme'
exit=2

$ ballast --tenant acme verify 1e2e80f4987360ffa9aec4668e79d0e359bd31b28d9a441ebb978c12b1b4f321
tenant 'acme': 1 commits, 1 manifests, 2 chunks, 262,144 bytes
attestation 1e2e80f4987360ffa9aec4668e79d0e359bd31b28d9a441ebb978c12b1b4f321
1 composite(s) now reference a deleted input and will refuse to resolve: acme:2d16ace07c1b447
verified

$ ballast --tenant acme fsck
acme: composite 2d16ace07c1b references deleted input acme:36ce0e22bf6f (broken view)
exit=1
```

Run 42 changed two of thirty-two tensors, so it stored two chunks. The blend stored
nothing: it is a view, resolved when checked out, and its recipe travels with the
exported adapter in `ballast.json`. The partner's layered view resolved through the
grant and stopped resolving the moment the grant was revoked — nothing was copied,
so nothing lingered. Forgetting run 42 removed exactly the two chunks nothing else
referenced, named the composite it broke, wrote a proof, and that proof verified
again by its attestation alone. `fsck` reports the broken view, which is the
correct state: a record that a merge happened, refusing to produce a matrix with a
deleted contribution smeared through it.

## What it stores

**Chunks** are tensors, one file each, named by a hash over dtype, shape and bytes,
under the tenant's directory. Reads are memory-mapped. Two versions of an adapter
that differ in three tensors share every other chunk.

**Manifests** are either a leaf, which names tensors, or a composite, which names
other manifests and a merge method. A manifest's id is a hash of its content, so
committing the same adapter twice stores nothing new, and a composite can never
form a cycle because its id depends on inputs that already exist.

**Commits** chain manifests into a history, each carrying a message and free-form
metadata — training run, dataset hash, who approved it. **Refs** name commits.
`reset` moves a ref; nothing is deleted.

Everything is keyed by tenant, in the schema. Chunks are not deduplicated across
tenants: different fine-tunes share essentially no tensors, and a global object
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

`linear` resolves to exactly what mergekit writes to disk. `ties` matches it
entry for entry except where two inputs tie exactly at the density boundary, where
mergekit's unstable sort and ballast's stable one may keep a different one of the
tied pair. Both are checked numerically by `scripts/verify_real.py`.

## Diffs report behaviour, and say when they cannot

A diff of two low-rank matrices is noise. `diff` reports the relative Frobenius
change overall and in the most changed tensor — the aggregate hides a targeted
edit, and a targeted edit is exactly what an audit cares about — and, if both
commits carry fingerprints for the same probe set, how many probes produced a
different answer.

```python
from ballast import ProbeSet, fingerprint
from ballast.runners import PeftRunner

probes = ProbeSet.of("What plan is this account on?", "Summarise the refund policy.")
fingerprint(store, "acme", "support", probes, PeftRunner("meta-llama/Llama-3.1-8B"))
```

A runner is anything with `run(tensors, config, base_model, probes) -> list[str]`.
`PeftRunner` loads the adapter onto the base through PEFT and generates greedily;
`FakeRunner` is deterministic and used in tests. Probe sets are stored, so a
fingerprint id can be explained later.

Three outcomes, and the difference between the last two is the point:

```
probes: 2 of 2 moved                                   the change was seen
no fingerprints on both commits; behavioural effect unknown
UNOBSERVED CHANGE: weights moved, no probe detected it
```

The last is weights that moved more than 1% — overall or in any one tensor — with
no probe noticing. It is reported as its own condition so silence is never
mistaken for stability.

## Sharing across tenants

```
ballast --tenant acme grant finance --to partner
ballast --tenant partner merge acme:finance@0.5 main@0.5 -m "layered" --ref layered
ballast --tenant acme revoke finance --to partner
```

A grant covers one manifest, not a ref. Granting `main` and then committing new
content to `main` does not extend the grant. The owner's chunks never move; the
grantee resolves through the grant at every hop of a view, so revoking it, or the
owner forgetting the manifest, breaks the grantee's views and leaves no copy
behind. Without a grant, a cross-tenant reference is refused at merge time.

## Deletion produces a record

`forget` removes a tenant entirely. `forget <commit>` removes one commit: children
are re-parented onto its parent, refs pointing at it move to its parent, its
manifest goes if no other commit uses it, chunks nothing else references go with
it, and grants on it are revoked. Both return a `Proof` naming what was removed
under an attestation hash, write tombstones the store keeps after the data is
gone, and store the proof itself. `verify <attestation>` re-checks it from any
process, later.

It is an audit record for an operator acting in good faith. It does not defend
against an operator who copied the chunk directory first. See
[SECURITY.md](SECURITY.md).

## Integrity

`fsck` re-hashes every chunk, checks every reference in the graph — tensors
without chunks, refs without commits, commits without manifests — and reports
views whose input was deleted or whose grant was revoked. `--fast` skips the
re-hash. `gc` drops chunks no manifest references.

## Loading into a model

`checkout` writes a PEFT adapter directory — `adapter_model.safetensors` and
`adapter_config.json` — which PEFT and vLLM load as they would any other, plus
`ballast.json` with the commit, manifest, base model, metadata, and for a view the
full recipe.

## Python API

```python
store = Store("./store")

c = store.commit("acme", tensors, message="run 41", ref="support",
                 base_model="meta-llama/Llama-3.1-8B", config=adapter_config,
                 metadata={"run": 41})
store.merge("acme", "ties", [("support", 0.6), ("finance", 0.4)], density=0.5, message="blend")
store.reset("acme", "support", older_commit_id)

store.head("acme", "support");  store.log("acme", "support");  store.resolve("acme", "a3d5")
tensors = store.checkout("acme", "blend")
store.diff("acme", "a3d5", "c028", probe_set=probes.id)

store.grant("acme", "finance", "partner");  store.revoke("acme", "finance", "partner")

proof = store.forget_commit("acme", "c028", reason="withdrawn data")
store.verify(proof.attestation)
store.fsck("acme");  store.gc("acme");  store.stats("acme")
```

Every method that takes a tenant or ref validates the name first. Anything that
can resolve a view raises `BrokenView` when it cannot; a cross-tenant reference
without a grant raises `NotGranted`.

## CLI

| command | what it does |
| --- | --- |
| `commit <dir> -m … [--ref] [--metadata JSON]` | store a PEFT adapter directory |
| `log [ref]` | history, newest first |
| `checkout <spec> -o <dir>` | materialise, with `ballast.json` provenance |
| `merge <in>… --method … [--density]` | record a view; inputs as `ref`, `ref@w`, `tenant:ref@w` |
| `diff <a> <b> [--probe-set]` | weight change and, if fingerprinted, probe change |
| `fingerprint <spec> --probes <file> [--runner peft or fake]` | run and record a probe set |
| `reset <ref> <spec>` | roll a ref back |
| `grant <spec> --to <tenant>` / `revoke` / `grants` | cross-tenant sharing |
| `forget [<spec>] --reason …` | delete a tenant or a commit, with a proof |
| `verify <attestation>` | re-check a stored proof |
| `fsck [--all] [--fast]` / `gc` / `stats` | integrity and housekeeping |
| `import-mergekit <yaml> --map model=ref…` | store a mergekit recipe as a view |

`--json` on any command prints the result as JSON. `--root` selects the store.

## What is verified

The test suite runs offline in about two seconds and covers the invariants:
content addressing, per-version chunk sharing, tenant isolation, views breaking on
input deletion and grant revocation, proofs verifying and tampered proofs failing,
`fsck` catching corrupted and missing chunks, concurrent commits from threads,
and name validation against path traversal.

`scripts/verify_real.py` runs by hand and checks against real systems:

- `PeftRunner` on `HuggingFaceTB/SmolLM2-135M` on CPU. An adapter with two layers
  retrained moves every probe. A 3% nudge to one tensor that greedy decoding
  cannot see is reported as UNOBSERVED CHANGE.
- `linear` views match mergekit's output on 60 tensors with zero difference.
- `ties` views match mergekit's output except at entries tied exactly at the
  density boundary, which the script counts and requires every difference to be
  explained by.

## Scope

Hosted models expose no weights and are out of scope by construction. This is for
open-weight models and the deltas people train on them.

The chunk store is one Python module with a hard interface; nothing above it
touches the filesystem. When this runs as a daemon serving many tenants at
gigabyte scale, that module is the one to rewrite in a systems language, and the
SQLite graph above it does not change.

The safetensors reader and writer are this package's own, so bfloat16 and float8
work without torch. They read and write the standard layout.

## Install

```
uv pip install -e ".[dev]"      # store, tests, lint
uv pip install -e ".[peft]"     # PeftRunner
uv pip install -e ".[verify]"   # scripts/verify_real.py
pytest -q
```

## License

Apache-2.0
