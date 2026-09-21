# Security

## What the store defends against

- **Tenant isolation.** Every table holding tenant data carries the tenant in its
  primary key, and chunks are stored under a per-tenant directory. A tenant cannot
  resolve, read, or reference another tenant's data without a grant, and a grant
  is scoped to one manifest, not to a ref that could later move onto new content.
- **Path traversal.** Tenant and ref names are validated against a fixed alphabet
  at every public entry point before they reach a path or a query.
- **Query injection.** Every query is parameterised. Commit prefix lookup uses
  `substr` rather than `LIKE`, so `%` and `_` in a spec are not wildcards.
- **Silent corruption.** `fsck` re-hashes every chunk and checks every reference
  in the graph.

## What it does not defend against

- **A hostile operator.** A deletion proof is an audit record for an operator
  acting in good faith. It shows that the store no longer holds the data and
  that tombstones and an attestation were written at the time. It cannot show
  that nobody copied the chunk directory first. Treat the store's host as the
  trust boundary.
- **Side channels through grants.** A grantee who can resolve a view over an
  owner's manifest can read the resolved tensors. That is what a grant is for.
  Do not grant on a manifest whose contents the grantee should not see.
- **The model itself.** Nothing here inspects what a delta does. A fingerprint
  reports how outputs changed on the probes you chose, and says so when a change
  in the weights produced no change in any probe.

## Reporting

Open a private security advisory on the repository, or email the address on the
maintainer's GitHub profile. Include the store's schema version (`SELECT version
FROM schema_version`) and, if it is safe to share, the output of `ballast fsck
--all --json`.
