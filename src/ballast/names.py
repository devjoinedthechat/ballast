"""Identifiers that end up in paths and queries.

Tenant names become directory names under the chunk root, so an unvalidated one
is a path traversal. Ref names go into queries and into the CLI. Both are held to
a small safe alphabet, and the check runs at every public entry point rather than
being trusted to callers.
"""

from __future__ import annotations

import re

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_HEX = re.compile(r"^[0-9a-f]{4,64}$")


def tenant(name: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name) or ".." in name:
        raise ValueError(
            f"invalid tenant {name!r}: 1-64 characters from [A-Za-z0-9._-], not starting with '.' or '-'"
        )
    return name


def ref(name: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"invalid ref {name!r}: 1-64 characters from [A-Za-z0-9._-]")
    return name


def is_hex_prefix(spec: str) -> bool:
    return bool(_HEX.match(spec))
