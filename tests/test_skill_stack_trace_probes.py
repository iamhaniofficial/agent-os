"""Issue #3662: the stack-trace probes named a parent skill that doesn't exist.

The five ``stack-trace-*-probe`` skills began their description with "Internal
helper for meta-stack-trace-investigator", but no skill by that name has ever
shipped. They are hidden, fixed-shape step helpers for meta-skill DAGs (like
``http-fetch`` and ``text-file-read``), so the descriptions now say that and
stop pointing at a missing skill.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agentos.skills.loader import SkillLoader
from agentos.skills.types import SkillSpec

BUNDLED_DIR = Path(__file__).resolve().parent.parent / "src" / "agentos" / "skills" / "bundled"

PROBES = (
    "stack-trace-generic-probe",
    "stack-trace-go-probe",
    "stack-trace-js-probe",
    "stack-trace-python-probe",
    "stack-trace-rust-probe",
)

# The phrasing the probes used; any skill that still claims a parent this way
# has to name one that is actually bundled.
_INTERNAL_HELPER_RE = re.compile(r"Internal helper for ([a-z0-9][a-z0-9-]*)")


@pytest.fixture(scope="module")
def bundled() -> dict[str, SkillSpec]:
    return {s.name: s for s in SkillLoader(bundled_dir=BUNDLED_DIR).load_all()}


@pytest.mark.parametrize("name", PROBES)
def test_probe_is_a_hidden_dag_step_helper_without_a_missing_parent(
    bundled: dict[str, SkillSpec], name: str
) -> None:
    probe = bundled[name]
    assert probe.disable_model_invocation is True
    assert probe.user_invocable is False

    description = probe.description
    assert "meta-stack-trace-investigator" not in description
    assert "meta-skill DAGs" in description


def test_every_named_parent_skill_is_bundled(bundled: dict[str, SkillSpec]) -> None:
    assert _INTERNAL_HELPER_RE.findall("Internal helper for meta-stack-trace-investigator.") == [
        "meta-stack-trace-investigator"
    ]

    claimed = {
        parent
        for skill in bundled.values()
        for parent in _INTERNAL_HELPER_RE.findall(skill.description)
    }
    missing = claimed - set(bundled)
    assert not missing, f"helpers name skills that aren't bundled: {missing}"
