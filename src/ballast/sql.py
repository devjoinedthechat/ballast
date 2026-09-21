"""One SQL dialect, two engines.

The metadata graph is ordinary relational SQL, and there is no reason it has to
live in SQLite — SQLite is the right default for one machine and the wrong answer
for several. Rather than keep two copies of every query, the store writes one
dialect and this module adapts it.

The dialect is the portable subset: `?` placeholders, `ON CONFLICT` upserts with
an explicit target, `WITH RECURSIVE`, `substr`, `COALESCE`. Column types are
chosen so both engines accept them literally — SQLite reads `DOUBLE PRECISION`
as REAL affinity and `BIGINT` as INTEGER, so no translation is needed there.

What differs is genuinely small: the placeholder mark, how a transaction begins,
and the pragmas that only SQLite has. Everything else is the same statement.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

_PLACEHOLDER = re.compile(r"\?(?=(?:[^']*'[^']*')*[^']*$)")


class Row(Protocol):
    """A row readable by column name or by position."""

    def __getitem__(self, key: Any) -> Any: ...
    def __iter__(self) -> Iterator[Any]: ...


class Cursor(Protocol):
    @property
    def rowcount(self) -> int: ...

    def fetchone(self) -> Any:
        """The next row, or None.

        Typed loosely on purpose. Callers that ask for one row of an aggregate
        know it is always there — `SELECT COUNT(*)` cannot return nothing — and
        making every one of those guard against None would be noise around a
        case that cannot happen.
        """

    def fetchall(self) -> list[Row]: ...
    def __iter__(self) -> Iterator[Row]: ...


class Database(Protocol):
    """What the store needs from a metadata engine."""

    dialect: str

    def execute(self, query: str, params: Sequence[Any] = ()) -> Cursor: ...
    def executemany(self, query: str, params: Sequence[Sequence[Any]]) -> None: ...
    def script(self, statements: str) -> None:
        """Run several statements, inside whatever transaction is open."""

    def begin(self) -> None: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...
    def describe(self) -> str: ...


class SqliteDatabase:
    dialect = "sqlite"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = sqlite3.connect(path, isolation_level=None, timeout=30.0, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 30000")
        self.conn.execute("PRAGMA foreign_keys = ON")

    def execute(self, query: str, params: Sequence[Any] = ()) -> Cursor:
        return self.conn.execute(query, tuple(params))

    def executemany(self, query: str, params: Sequence[Sequence[Any]]) -> None:
        self.conn.executemany(query, [tuple(p) for p in params])

    def script(self, statements: str) -> None:
        # Statement by statement rather than `executescript`, which commits any
        # open transaction before it runs and would break a migration mid-step.
        for statement in split(statements):
            self.conn.execute(statement)

    def begin(self) -> None:
        self.conn.execute("BEGIN IMMEDIATE")

    def commit(self) -> None:
        self.conn.execute("COMMIT")

    def rollback(self) -> None:
        self.conn.execute("ROLLBACK")

    def close(self) -> None:
        self.conn.close()

    def set_foreign_keys(self, on: bool) -> None:
        self.conn.execute(f"PRAGMA foreign_keys = {'ON' if on else 'OFF'}")

    def describe(self) -> str:
        return f"sqlite:{self.path}"


class HybridRow:
    """A row readable by column name or by position, as `sqlite3.Row` is.

    psycopg offers one or the other. Supporting both is what lets the store
    write a single set of queries instead of two, and unpacking works because
    iteration yields the values in column order.
    """

    __slots__ = ("_index", "_values")

    def __init__(self, fields: dict[str, int], values: Sequence[Any]) -> None:
        self._index = fields
        self._values = tuple(values)

    def __getitem__(self, key: Any) -> Any:
        return self._values[self._index[key]] if isinstance(key, str) else self._values[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"HybridRow({dict(zip(self._index, self._values, strict=False))})"

    def keys(self) -> list[str]:
        return list(self._index)


def _hybrid_rows(cursor: Any) -> Any:
    fields = {c.name: i for i, c in enumerate(cursor.description or [])}

    def make(values: Sequence[Any]) -> HybridRow:
        return HybridRow(fields, values)

    return make


class PostgresDatabase:
    """Postgres over psycopg 3.

    Placeholders are rewritten from `?` to `%s`, skipping any inside a quoted
    string. Everything else the store writes is already valid Postgres.
    """

    dialect = "postgres"

    def __init__(self, dsn: str) -> None:
        import psycopg  # noqa: PLC0415

        self.dsn = dsn
        self.conn = psycopg.connect(dsn, autocommit=True, row_factory=_hybrid_rows)
        self._in_tx = False

    @staticmethod
    def adapt(query: str) -> str:
        return _PLACEHOLDER.sub("%s", query)

    def execute(self, query: str, params: Sequence[Any] = ()) -> Cursor:
        cur = self.conn.cursor()
        cur.execute(self.adapt(query), tuple(params))
        return cur

    def executemany(self, query: str, params: Sequence[Sequence[Any]]) -> None:
        rows = [tuple(p) for p in params]
        if not rows:
            return
        with self.conn.cursor() as cur:
            cur.executemany(self.adapt(query), rows)

    def script(self, statements: str) -> None:
        with self.conn.cursor() as cur:
            for statement in split(statements):
                cur.execute(self.adapt(statement))

    def begin(self) -> None:
        self.conn.execute("BEGIN")
        self._in_tx = True

    def commit(self) -> None:
        self.conn.execute("COMMIT")
        self._in_tx = False

    def rollback(self) -> None:
        self.conn.execute("ROLLBACK")
        self._in_tx = False

    def close(self) -> None:
        self.conn.close()

    def set_foreign_keys(self, on: bool) -> None:
        """Postgres has no session switch for this.

        The migration path turns constraints off to reshape tables. Here the
        equivalent is that the migration drops and recreates in dependency
        order inside one transaction, which is what it already does.
        """

    def describe(self) -> str:
        redacted = re.sub(r"(password=)[^ ]+", r"\1***", self.dsn)
        return f"postgres:{redacted}"


def split(statements: str) -> list[str]:
    """Split a script into statements.

    Semicolons inside string literals and inside `--` comments do not end a
    statement, and comments are dropped rather than carried along — a comment
    containing an apostrophe would otherwise look like an open string literal
    to everything after it.
    """
    out: list[str] = []
    current: list[str] = []
    in_string = False
    in_comment = False
    previous = ""
    for char in statements:
        if in_comment:
            if char == "\n":
                in_comment = False
                current.append(char)
            continue
        if char == "'":
            in_string = not in_string
        elif char == "-" and previous == "-" and not in_string:
            in_comment = True
            current.pop()  # the first dash of the marker
            previous = ""
            continue
        if char == ";" and not in_string:
            statement = "".join(current).strip()
            if statement:
                out.append(statement)
            current = []
        else:
            current.append(char)
        previous = char
    tail = "".join(current).strip()
    if tail:
        out.append(tail)
    return out


def connect(target: Path | str) -> Database:
    """A path opens SQLite; a `postgres://` or `postgresql://` URL opens Postgres."""
    if isinstance(target, str) and target.startswith(("postgres://", "postgresql://")):
        return PostgresDatabase(target)
    return SqliteDatabase(Path(target))
