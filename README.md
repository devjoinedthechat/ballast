<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo-light.svg" alt="" width="112" height="112">
  </picture>
</p>

<h1 align="center">ballast</h1>

<p align="center">
  <b>Version control for what a model has learned.</b><br>
  Content-addressed weight deltas with history, views, behavioural diffs, grants,
  and deletion with proof.
</p>

<p align="center">
  <a href="https://github.com/devjoinedthechat/ballast/actions/workflows/ci.yml"><img src="https://github.com/devjoinedthechat/ballast/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.14-blue" alt="Python 3.10 | 3.12 | 3.14">
  <img src="https://img.shields.io/badge/tests-117-brightgreen" alt="117 tests">
  <img src="https://img.shields.io/badge/verified%20against-mergekit%20%C2%B7%20PEFT-2e7d32" alt="Verified against mergekit and PEFT">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0">
  <img src="https://img.shields.io/badge/status-pre--alpha-orange" alt="Status: pre-alpha">
</p>

<p align="center">
  <a href="#why">Why</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#merges-are-views">Views</a> ·
  <a href="#diffs-report-behaviour">Diffs</a> ·
  <a href="#sharing-across-tenants">Sharing</a> ·
  <a href="#deletion-produces-a-record">Deletion</a> ·
  <a href="#cli">CLI</a> ·
  <a href="#what-is-verified">Verification</a> ·
  <a href="docs/walkthrough.md">Walkthrough</a>
</p>

---

A fine-tuned adapter, a merged model, a per-user delta: each is a set of tensors on
top of a base model, and today each is a file in a bucket with a name someone chose.
`ballast` stores them as content-addressed blocks, chains them into a history per
tenant, records merges as views over their inputs, diffs versions by what the model
*does* rather than by bytes, shares them across tenants through revocable grants, and
deletes with a signed proof that can be re-verified from another process afterwards.

The core is numpy, SQLite, blake3 and zstd. It runs without torch.

## Why

Weights are the one artifact in an ML system that nobody versions. Data has lineage,
code has git, prompts have a registry, and the adapter that actually changes what the
model says is `adapter_v3_final2.safetensors` in a bucket. Nobody can say what went
into a merge, whether the retrain that shipped last Tuesday changed the answers that
matter, or — when a customer asks — whether their data is out of the weights.

That gets worse, not better. As memory moves from the context window into weights,
every user and session carries a delta of its own, and the questions above are asked
per person: what did this model learn about me, roll it back, prove you deleted it.

`ballast` answers them with a data structure rather than a policy. Identity is
content, so history is cheap. Merges are views, so deletion is clean. Diffs are
behavioural, so a change that matters is visible and a change nothing noticed is
reported as exactly that.

## Features

- **Content-addressed blocks.** Tensors are split into fixed-size blocks, each stored
  once. A one-value edit to a multi-gigabyte tensor stores one block. Blocks are
  zstd-compressed unless that would not help, and a single raw block is read by
  memory map with no copy.
- **History that rolls back.** Commits chain manifests per tenant with a message and
  free-form provenance. Refs name commits, `reset` moves one, and the reflog
  remembers every move so a rollback can be rolled back.
- **Merges are views.** A merge stores no tensors; it names its inputs and a method
  and resolves at checkout. `linear` and `ties` match mergekit's output numerically.
  Deleting an input breaks the view detectably instead of leaving its contribution
  smeared through a matrix.
- **Behavioural diffs.** Fingerprint a commit by running probes through the model,
  and `diff` reports how many answers moved and how far. Weights that moved with no
  probe noticing are reported as an unobserved change, never as silence.
- **Grants, not copies.** One tenant can build views over another's manifest through
  a grant scoped to that manifest. The owner's blocks never move; revoking the grant
  or deleting the manifest breaks the grantee's views and leaves nothing behind.
- **Deletion with proof.** Forget a tenant or a single commit and get a record naming
  everything removed under an attestation, signed when a key is configured, stored,
  and re-verifiable later by attestation alone.
- **Full models in, full models out.** `delta` extracts what a fine-tune changed from
  two model directories; `apply` writes a stored delta or view back onto a base as a
  directory transformers loads.
- **Local or S3.** Blocks live on disk or in an object store behind one interface.
  `fsck` re-hashes every block and checks every reference either way.

## Quickstart

```
uv pip install ballast-store        # or: uv pip install -e ".[dev]" from a checkout
```

Two fine-tunes of the same base, merged and applied back onto it:

```
$ ballast --tenant acme delta ./support-ft --base ./llama-8b -m "support, run 41" --ref support
a3d509ff9689  120 tensors changed, 171 unchanged  ref support

$ ballast --tenant acme delta ./finance-ft --base ./llama-8b -m "finance" --ref finance
2b30d87d4370  120 tensors changed, 171 unchanged  ref finance

$ ballast --tenant acme merge support@0.6 finance@0.4 --method ties --density 0.5 -m "blend" --ref blend
d8ab4508172a  ties of 2 inputs  ref blend

$ ballast --tenant acme apply blend --base ./llama-8b -o ./blend-model
model written to blend-model
```

Adapters work the same way with `commit` and `checkout`, which read and write PEFT
adapter directories. The [walkthrough](docs/walkthrough.md) runs the whole session:
two tenants, a grant, a revocation, a deletion, and a proof.

## How it works

```
 tenant
 ├── blocks        content-addressed bytes, zstd or raw, local or S3
 ├── manifests     leaf: tensors as runs of blocks · composite: inputs + method
 ├── commits       manifest + parent + message + metadata
 ├── refs          name → commit            reflog: every move
 ├── grants        manifest → grantee       revocable
 ├── fingerprints  commit × probe set → answers
 └── proofs        what was deleted, attested and signed
```

**Identity is content.** A block is named by the hash of its bytes; a manifest by the
hash of its tensors' dtypes, shapes and block runs; a composite by its inputs and
method. Committing the same adapter twice stores nothing new, and a composite cannot
form a cycle because its id depends on inputs that already exist.

**Tenants are namespaces in the schema.** Every table holding tenant data carries the
tenant in its primary key, and blocks are keyed by tenant too. They are deliberately
not deduplicated across tenants: different fine-tunes share almost no bytes, and a
shared object store would make one tenant's deletion proof depend on another's data.

**Blocks are fixed-size on purpose.** Content-defined chunking exists to survive
insertions, and weights have none: a changed value stays where it was. Fixed blocks
give the same dedup at a fraction of the complexity.

## Merges are views

```python
from ballast import Store

store = Store("./store")
store.merge("acme", "ties", [("support", 0.6), ("finance", 0.4)], density=0.5, message="blend")
```

Nothing is computed until `checkout`, and the result is cached until an input is
deleted or a grant revoked. `linear`, `task_arithmetic` and `ties` resolve. `dare_ties`,
`slerp`, `passthrough` and the rest are recorded for provenance and refuse to resolve,
rather than quietly running as something else.

A mergekit configuration imports as a view, each `model` entry mapped to a ref, so the
recipe is stored as the thing it describes:

```
ballast --tenant acme import-mergekit merge.yaml --map org/finance-lora=finance --map org/support-lora=support
```

`--union` merges adapters with different target modules, treating a tensor an input
lacks as zero.

## Diffs report behaviour

A diff of two low-rank matrices is noise. `diff` reports the relative change overall
and in the most changed tensor — the aggregate hides a targeted edit, and a targeted
edit is what an audit cares about — and, when both commits carry fingerprints for the
same probe set, how many answers moved and how similar the moved ones still are.

```python
from ballast import ProbeSet, fingerprint
from ballast.runners import PeftRunner

probes = ProbeSet.of("What plan is this account on?", "Summarise the refund policy.")
fingerprint(store, "acme", "support", probes, PeftRunner("meta-llama/Llama-3.1-8B"))
```

`PeftRunner` applies the adapter in process — tensors go from the store into the live
model, nothing is written between them — and generates greedily. A runner is anything
with `run(tensors, config, base_model, probes) -> list[str]`.

Three outcomes, and the difference between the last two is the point:

```
probes: 2 of 2 moved (similarity 0.41)              the change was seen
no fingerprints on both commits; behavioural effect unknown
UNOBSERVED CHANGE: weights moved, no probe detected it
```

## Sharing across tenants

```
ballast --tenant acme grant finance --to partner
ballast --tenant partner merge acme:finance@0.5 main@0.5 -m "layered" --ref layered
ballast --tenant acme revoke finance --to partner
```

A grant covers one manifest, not a ref: granting `main` and then committing new
content to `main` does not extend it. The grantee resolves through the grant at every
hop of a view, so revocation or the owner's deletion breaks the grantee's views and
leaves no copy behind. Without a grant, a cross-tenant reference is refused.

## Deletion produces a record

`forget` removes a tenant. `forget <commit>` removes one commit: children are
re-parented, refs move to the parent, its manifest goes if nothing else uses it,
blocks nothing else references go with it, grants on it are revoked, and the reflog
records the move. Both return a `Proof`:

```
tenant 'acme': 1 commits, 1 manifests, 2 blocks, 200,893 bytes
attestation f1c9c512e77c…  (signed)
1 composite(s) now reference a deleted input and will refuse to resolve: acme:b9ad3cb2…
verified
```

The proof is stored, and `verify <attestation>` re-checks it from any process: the
data is gone from the index and the backend, the tombstones match, the attestation
matches the proof's own contents, and the signature matches the key. Set
`BALLAST_SIGNING_KEY` outside the store and the record is tamper-evident against
anyone who can edit the database. See [SECURITY.md](SECURITY.md) for what that does
and does not defend against.

## Storage

```python
Store("./store")                                   # blocks under ./store/chunks
Store("./meta", backend="s3://bucket/prefix")      # blocks in S3, metadata local
Store("./store", block_size=4 << 20, compression="raw")
```

Block size and compression are fixed when a store is created and read back on every
open. Writes to the local backend are atomic and durable: a per-writer temporary
name, fsync of the file, rename, fsync of the directory. `fsck` re-reads and
re-hashes every block; `--fast` checks only the graph.

## CLI

| command | what it does |
| --- | --- |
| `delta <model> --base <base> -m …` | commit `model − base` from two model directories |
| `apply <spec> --base <base> -o <dir>` | write a commit or view onto a base as a model directory |
| `commit <adapter-dir> -m …` | store a PEFT adapter directory |
| `checkout <spec> -o <dir>` | materialise as a PEFT adapter directory, with `ballast.json` provenance |
| `merge <in>… --method … [--density] [--union]` | record a view; inputs as `ref`, `ref@w`, `tenant:ref@w` |
| `diff <a> <b> [--probe-set] [--threshold]` | weight change and, if fingerprinted, probe change |
| `fingerprint <spec> --probes <file>` | run a probe set through the model and record the answers |
| `log` / `reflog` / `reset <ref> <spec>` | history, every ref move, rollback |
| `grant <spec> --to <t>` / `revoke` / `grants` | cross-tenant sharing |
| `forget [<spec>] --reason …` | delete a tenant or a commit, with a proof |
| `verify <attestation>` | re-check a stored proof |
| `fsck [--all] [--fast]` / `gc` / `stats` | integrity and housekeeping |
| `import-mergekit <yaml> --map model=ref…` | store a mergekit recipe as a view |

`--json` on any command prints the result as JSON. `--root` selects the store,
`--backend` where its blocks live.

## What is verified

The test suite runs offline and covers the invariants: content addressing, one block
per changed value, tenant isolation, views breaking on deletion and revocation, proofs
verifying and forged ones failing, signatures, `fsck` catching corrupted and missing
blocks, the version-1 migration, the S3 backend against a mock, thread safety, name
validation against path traversal, and property-based round trips of arbitrary
tensors through arbitrary block sizes.

`scripts/verify_real.py` runs by hand against real systems, on CPU, and every check
in it passes:

- **`PeftRunner` on `HuggingFaceTB/SmolLM2-135M`.** An adapter with two layers
  retrained moves every probe. A 3% nudge to one tensor that greedy decoding cannot
  see is reported as an unobserved change.
- **`linear` views match mergekit's output** on 60 tensors with zero difference.
- **`ties` views match mergekit's output** except at entries tied exactly at the
  density boundary, where mergekit's unstable sort and ballast's stable one may keep a
  different member of the pair. The script counts those ties and requires every
  difference to be explained by one.
- **`delta` → view → `apply` reproduces mergekit's output model** directory, and
  transformers loads it.

## Project status

Pre-alpha. The schema is versioned and migrates; the API is not yet stable.

What is not there: cross-tenant composition beyond one hop of grants, a metadata
backend other than SQLite (so one node), a live hand-off into a running vLLM server
rather than an in-process PEFT model, and a `dare_ties` that resolves. The chunk
store is one module behind a narrow interface; when this runs as a daemon serving many
tenants at gigabyte scale, that module is the one to rewrite in a systems language.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for the layout and what a change needs, and
[SECURITY.md](SECURITY.md) for the trust boundary.

## License

Apache-2.0
