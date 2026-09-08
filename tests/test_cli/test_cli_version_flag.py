"""``agentos --version`` prints the installed version and exits 0.

Before this existed the only way to read the running version was
``uv tool list`` (issue #1364).
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from agentos import __version__
from agentos.cli.main import app

runner = CliRunner()


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_version_and_exits_zero(flag: str) -> None:
    result = runner.invoke(app, [flag])

    assert result.exit_code == 0
    assert __version__ in result.stdout.splitlines()


def test_version_flag_is_listed_in_help() -> None:
    result = runner.invoke(app, ["--help"], terminal_width=100)

    assert result.exit_code == 0
    assert "--version" in result.stdout


def test_version_flag_wins_over_a_subcommand() -> None:
    """Eager, so it answers before ``doctor`` (or anything else) can run."""

    result = runner.invoke(app, ["--version", "doctor"])

    assert result.exit_code == 0
    assert __version__ in result.stdout.splitlines()


def test_bare_invocation_still_shows_help() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "Usage:" in result.stdout
