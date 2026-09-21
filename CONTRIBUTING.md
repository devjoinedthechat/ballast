# Contributing

```
uv venv && uv pip install -e ".[dev]"
pytest -q
ruff check . && mypy src
```

The suite runs offline in about a second. `scripts/verify_real.py` is the check
against a live model and against mergekit's own output; it downloads ~270 MB and
takes a few minutes on CPU, and it is run by hand before a release rather than in
CI.

## Where things go

- `tensors.py` — the safetensors reader and writer. Add a dtype here. The reader
  takes untrusted files: validate the header before interpreting a byte.
- `backends.py` — where blobs live. A new backend is one class and no changes
  above it. Local writes must stay atomic and durable.
- `chunks.py` — tensors as blocks. This is the module a systems-language rewrite
  replaces, and nothing above it touches a backend directly.
- `models.py` — full model directories in and out, for people who merge models
  rather than adapters.
- `db.py` — the schema. Bump `SCHEMA_VERSION` and add a migration step for any
  change to a table; never edit a `CREATE TABLE` in place.
- `store.py` — the API. Every public method validates tenant and ref names first.
- `merge.py` — how a view resolves. A new method is either resolvable, with a
  numeric test against mergekit's output, or record-only and refuses. Aliasing an
  unimplemented method onto a similar one is the bug this rule exists to prevent.

## What a change needs

A test that fails without it. For anything touching deletion, a test that
`verify` still passes afterwards. For anything touching merges, a numeric
comparison rather than a shape check. For anything touching storage, a property
test over arbitrary tensors rather than the fixture.

Run `scripts/verify_real.py` before a release. It is the only check that runs a
real model and compares against mergekit's own output, and it has caught two bugs
the unit tests could not: an aggregate change metric that hid a targeted edit, and
a trim that kept one entry too many at a density tie.

Keep the README to what is true of the code as it is. Fix history goes in the
commit message.
