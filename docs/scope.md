# Scope

What ballast does not do, and the reasoning. Everything here is a deliberate
edge rather than an oversight; where something is untested rather than absent,
it says so.

- **A live vLLM in the tests.** `sync-vllm` drives vLLM's runtime LoRA endpoints
  and is tested against a stub transport, which checks the conversation — the
  requests made and how each reply is handled — but not vLLM's own behaviour.
  vLLM is CUDA-first and is not installed in CI.
- **Three merge methods.** `arcee_fusion`, `karcher` and `nearswap` are read and
  recorded, and refuse rather than approximate. So does the `slices` form, which
  composes layer ranges rather than whole deltas.
- **Authentication beyond bearer tokens.** Scoped static tokens are enough for a
  fleet behind a gateway and are not an identity system.
- **A systems-language core.** `chunks.py` is one module with a narrow interface
  and nothing above it touches a backend directly, so it is the piece to rewrite
  when a single node stops being enough. There is no evidence it is the
  bottleneck yet, so it has not been.

## Where the boundaries are

**The store is the trust boundary.** A deletion proof shows that the store no
longer holds the data and that tombstones and an attestation were written at the
time. Signed, it also shows the record was not rewritten afterwards. It cannot
show that nobody copied the chunk directory first. See
[SECURITY.md](../SECURITY.md).

**An export is a copy that leaves the store.** `serve-export` and `sync-vllm`
write adapters to a directory a server reads. Ballast re-checks on every export
that the commit still resolves, so a forgotten delta or a revoked grant stops
being exported — but a copy someone else made is beyond its reach.

**A view is a recipe, not a result.** Deleting an input breaks every view over
it, and ballast reports that rather than repairing it. There is no way to
"re-point" a view at a replacement, because the replacement is a different thing
and pretending otherwise would lose exactly the history the store exists to keep.

**Deltas, not models.** The unit is what a fine-tune changed. Full model
directories go in and come out through `delta` and `apply`, but what is versioned
in between is the difference. A merge that composes layer ranges rather than
whole deltas — mergekit's `slices` form — is recorded and refuses to resolve,
because it is not a thing this store can represent.
