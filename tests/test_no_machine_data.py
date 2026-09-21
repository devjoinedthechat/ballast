"""Nothing committed identifies the machine it was written on.

Documentation quotes real output, and real output is where a home directory or
a temporary path leaks in. This walks everything git tracks and fails on the
shapes that carry them, so a pasted result cannot bring one along unnoticed.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Each pattern is a shape that only appears by accident: an absolute path from
# somebody's machine, a per-user temporary directory, a hostname.
LEAKS = {
    "a home directory": re.compile(r"/(?:Users|home)/(?!<|\.\.\.)[A-Za-z0-9_.-]+"),
    "a macOS temporary directory": re.compile(r"/var/folders/[a-z0-9]{2}/[a-z0-9]{10,}"),
    "a private temporary directory": re.compile(r"/private/(?:var|tmp)/[A-Za-z0-9_.-]+"),
    "a local checkout path": re.compile(r"/(?:Desktop|Documents|Downloads)/[A-Za-z0-9_.-]+"),
}

# Text files only: a tracked binary is not going to hold a path by accident,
# and decoding one as UTF-8 would fail noisily for no purpose.
BINARY = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".safetensors", ".db", ".zst"}


def tracked_text_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [ROOT / name for name in listed if Path(name).suffix.lower() not in BINARY]


@pytest.mark.parametrize("path", tracked_text_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_machine_data_in(path: Path):
    try:
        content = path.read_text()
    except UnicodeDecodeError:
        return
    for what, pattern in LEAKS.items():
        found = pattern.search(content)
        assert found is None, f"{path.relative_to(ROOT)} contains {what}: {found.group(0)!r}"


def test_the_patterns_catch_what_they_are_for():
    """A guard that never fires is not a guard.

    The samples are assembled rather than written out, because this file is
    tracked and the guard reads it: a literal example would be caught as the
    thing it is an example of.
    """
    samples = {
        "a home directory": "/" + "Users/someone/ballast",
        "a macOS temporary directory": "/var/" + "folders/d6/0a1b2c3d4e5f/T/x",
        "a private temporary directory": "/private/" + "tmp/ballast-verify-abc",
        "a local checkout path": "/" + "Desktop/ballast",
    }
    for what, sample in samples.items():
        assert LEAKS[what].search(sample), what


def test_a_documentation_placeholder_is_not_a_leak():
    """An angle-bracket placeholder in an example is deliberate and stays allowed."""
    assert LEAKS["a home directory"].search("/" + "home/<user>/models") is None


def test_the_guard_reads_what_git_tracks():
    """A file not yet added is not scanned, so run this after staging.

    The guard's own first version passed locally and failed in CI for exactly
    that reason.
    """
    assert Path(__file__) in tracked_text_files()
