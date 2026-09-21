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

- `tensors.py` — the safetensors reader and writer. Add a dtype here.
- `chunks.py` — bytes on disk. This is the module a systems-language rewrite
  replaces, and nothing above it touches the filesystem.
- `db.py` — the schema. Bump `SCHEMA_VERSION` and add a migration step for any
  change to a table; never edit a `CREATE TABLE` in place.
- `store.py` — the API. Every public method validates tenant and ref names first.
- `merge.py` — how a view resolves. A new method is either resolvable, with a
  test against mergekit's output, or record-only and refuses.

## What a change needs

A test that fails without it. For anything touching deletion, a test that
`verify` still passes afterwards. For anything touching merges, a numeric
comparison rather than a shape check.

Keep the README to what is true of the code as it is. Fix history goes in the
commit message.
