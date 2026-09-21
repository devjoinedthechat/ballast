# Changelog

Notable changes, newest first. Dates are the day the work landed on `main`.

This project is pre-alpha: the schema is versioned and migrates, and the Python
API is not yet stable.

## Unreleased

### Added

- **A benchmark,** `scripts/benchmark.py`, measuring the storage and memory
  claims at three sizes, the largest a four-gigabyte model. Every scenario runs
  in its own process and its fixtures are built in another, because peak
  resident memory is a high-water mark that freeing does not lower. Results and
  method in [docs/benchmark.md](docs/benchmark.md); the small size runs in CI.

- **Fifteen merge methods resolve**, up from five: `slerp`, `nuslerp`,
  `multislerp`, `breadcrumbs`, `breadcrumbs_ties`, `della`, `della_linear`,
  `model_stock`, `sce` and `passthrough` joined the four that already did. Each
  is compared against mergekit's own function numerically — the deterministic
  sparsifiers to zero, the geometric and consensus methods to float32 epsilon,
  and DELLA's keep probabilities to 1e-8.
- **Every method is one per-tensor operation.** `merge_tensor` merges one named
  tensor across the inputs, and the dictionary-level functions map over names.
  That is what lets a view resolve without holding its inputs, and it makes a new
  method one function rather than a new path through the store.
- **Views resolve incrementally.** An uncached view is merged tensor by tensor —
  one tensor per input rather than one model per input — and its cache is written
  as it streams, to a temporary file renamed only once the last tensor is out. A
  consumer that stops half way leaves nothing behind pretending to be a view.
- **Writes over HTTP,** on a separate scope from reads: a fleet pulling adapters
  should not hold a credential that can delete one. `POST .../commits` takes a
  safetensors body, `POST .../merges` records a view, `DELETE .../commits/{spec}`
  forgets one and returns its proof.
- **`sync-vllm`,** which converges a running vLLM on what the store holds: loads
  what is missing, unloads what is gone, leaves other tenants alone, and changes
  nothing on a second run. Names carry the ref and the commit, so a ref moving is
  visible as a different name.

- **Five more merge methods that resolve**, taking the total to ten: `slerp`,
  `breadcrumbs`, `breadcrumbs_ties`, `della` and `della_linear`. Each is compared
  against mergekit's own function numerically — the two deterministic sparsifiers
  match to zero, SLERP to float32 epsilon, and DELLA's keep probabilities to 1e-8.
- **`ballast serve`,** a read-only HTTP API a serving runtime can pull from: the
  list of adapters to load with a stable id for each, the delta itself as
  safetensors, and its adapter config. A commit id names one set of tensors, so
  every response is immutable and carries an ETag. Bearer tokens are scoped to
  tenants, and a refused tenant is a 404 rather than a 403.
- **Bounded memory.** `save_stream` writes a file one tensor at a time,
  `checkout_stream` reads one at a time, and `specs` describes a commit without
  reading any of it. Applying a delta to a 70B model costs one tensor, not a model.

- **Sub-tensor blocks.** Tensors are stored as fixed-size content-addressed
  blocks rather than whole, so a one-value edit to a large tensor stores one
  block. Blocks are compressed with zstd unless that does not help, and are
  addressed by their raw bytes so the same block deduplicates whichever way it
  was encoded.
- **Storage backends.** Blocks live on a local filesystem or in S3 behind one
  interface. Local writes fsync the file and its directory around the rename.
- **A Postgres metadata backend.** `Store(..., metadata="postgresql://…")` puts
  the graph in Postgres instead of SQLite. One dialect of SQL serves both.
- **Full model directories.** `delta` extracts what a fine-tune changed from a
  model and its base; `apply` writes a delta or a view back onto a base as a
  directory transformers loads.
- **`dare_ties` and `dare_linear`,** with the seed drawn once when the view is
  recorded and stored with it, so a DARE view resolves the same way twice.
- **A reflog.** Every move of every ref is recorded, so a rollback can itself be
  rolled back.
- **Signed deletion proofs.** With `BALLAST_SIGNING_KEY` set outside the store,
  a proof carries an HMAC and `verify` checks it.
- **`serve-export`.** Materialises deltas into the layout a serving runtime
  loads, with a stable positive integer id per commit, skipping views that
  cannot resolve rather than failing the whole fleet.
- **Union merges** (`--union`) for adapters with different target modules.
- **`fsck`** re-hashes every block and checks every reference in the graph.
- **Real-world mergekit configurations**: per-model density and weight,
  `normalize`, `lambda`, `base_model`, `dtype`, `tokenizer_source`, gradient
  parameters, and the `slices` form, which is recorded and refuses to resolve.
- Commit metadata, probe sets stored by id, a `ballast.json` provenance file
  beside every checkout, `--json` on every command, and a configurable
  unobserved-change threshold with a similarity score for probes that moved.

### Fixed

- **Writing a tensor copied it first.** Every write went through `tobytes()`,
  which builds a second copy of the tensor before the write: a few megabytes per
  tensor, gigabytes over a model, and enough to make the streaming apply slower
  than the version that holds everything in memory. Tensors are written straight
  from the array now, and the copy is kept only for destinations with no file
  descriptor, such as an HTTP response body. Found by the benchmark.

- **Merging refused the ordinary case.** Two fine-tunes of the same base rarely
  touch the same tensors, so `delta` produces deltas with different tensor sets
  and a strict merge refused fourteen of the fifteen methods on them. The union
  is the default now — a weight an input never changed is a change of zero — and
  `--strict` asks for the old behaviour. Every view records which rule it was
  made under, so one resolved later does not depend on what the default was then.
- **A forgotten delta went on being served.** `serve-export` skipped the work
  when the export directory already existed, so a commit that had been deleted,
  or a grant that had been revoked, kept being exported from the last copy. It
  re-checks that the commit resolves on every export, which reads no tensors.
- **`slerp` was unreachable from the command line.** It resolved, it was tested,
  and `merge` had no `--t`. Every per-method parameter is a flag now, and a test
  walks every resolvable method through the CLI.
- **An unknown ref printed a traceback** instead of one line and an exit code.
- **A view that could not resolve reported two different ways** depending on why.
  Deleted input, revoked grant and a strict mismatch are one condition to
  everything downstream — this one cannot be served — and all raise `BrokenView`,
  so `serve-export` and `sync-vllm` skip it and name it rather than failing.

- **`passthrough` was implemented but listed as record-only,** so a recipe that
  ballast could resolve refused instead.

- **`RESOLVABLE` and `RECORD_ONLY` overlapped,** so whether a recipe ran or
  refused depended on the order of checks inside `resolve`. A test now asserts
  the two are disjoint.
- **A parameter that could not be acted on was dropped from provenance** rather
  than recorded, and a gradient reduced to its first value lost the rest of the
  list. Both are kept now.

- **A revoked grant kept serving cached results.** Cache invalidation followed
  only direct dependents, so a view built on a view kept resolving after the
  grant underneath it was withdrawn. It is transitive now, and the proof for a
  deletion names every view downstream rather than the first hop.
- **`dare_ties` resolved as TIES.** DARE drops entries at random and rescales;
  TIES trims by magnitude. Aliasing one onto the other produced different
  weights than the recipe described, silently. It is implemented properly now.
- **`normalize` and `lambda` were ignored** when importing a mergekit
  configuration, so a recipe that said not to normalise was normalised anyway.
- **The safetensors reader trusted its header.** Offsets, spans and overlaps are
  validated before any byte is interpreted, with the tensor and the reason named.
- **Block writes were not durable.** A crash between the rename and the flush
  could leave an indexed block with no bytes behind it.
- **`_trim` kept one entry too many** when two values tied at the density
  boundary, which showed up as a numeric difference against mergekit's output.
- **The unobserved-change gate used only the aggregate**, which hid a targeted
  edit: one tensor moved 3% is 0.2% of an adapter. It gates on the per-tensor
  maximum as well.
- Tenant and ref names are validated before they reach a path or a query, and
  commit prefix lookup uses `substr` rather than `LIKE`, so `%` is not a wildcard.

### Verified

Against real systems, by `scripts/verify_real.py`:

- `PeftRunner` on `HuggingFaceTB/SmolLM2-135M`, in process.
- `linear` views match mergekit's output on 60 tensors with zero difference.
- `ties` views match except at entries tied exactly at the density boundary.
- `delta` → view → `apply` reproduces mergekit's output model, which
  transformers then loads.

And against a real Postgres in CI.
