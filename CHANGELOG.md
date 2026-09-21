# Changelog

Notable changes, newest first. Dates are the day the work landed on `main`.

This project is pre-alpha: the schema is versioned and migrates, and the Python
API is not yet stable.

## Unreleased

### Added

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
