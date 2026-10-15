"""The ACP docs, bundled skills and config example stay in step with the code."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agentos.acp.registry import list_harnesses
from agentos.cli.acp_cmd import app as acp_app
from agentos.gateway.config import AcpAgentOverride, AcpConfig

ROOT = Path(__file__).resolve().parents[2]
ACP_DOC = ROOT / "docs" / "acp-agents.md"
CLI_DOC = ROOT / "docs" / "cli.md"
CONFIG_DOC = ROOT / "docs" / "configuration.md"
TOML_EXAMPLE = ROOT / "agentos.toml.example"
AGENTOS_SKILL = ROOT / "src" / "agentos" / "skills" / "bundled" / "agentos" / "SKILL.md"
SUB_AGENT_SKILL = ROOT / "src" / "agentos" / "skills" / "bundled" / "sub-agent" / "SKILL.md"

# Letters only Vietnamese uses: a/d/o/u with breve/stroke/horn, i/u with tilde,
# and the Latin Extended Additional block of tone-marked vowels.
_VIETNAMESE = re.compile(
    "[\u0102\u0103\u0110\u0111\u01a0\u01a1\u01af\u01b0\u0128\u0129\u0168\u0169\u1ea0-\u1ef9]"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _commands() -> list[str]:
    return sorted(c.name or "" for c in acp_app.registered_commands)


def test_cli_commands_are_the_documented_set() -> None:
    assert _commands() == ["cancel", "close", "doctor", "list", "run", "sessions"]


@pytest.mark.parametrize("doc", [CLI_DOC, AGENTOS_SKILL, ACP_DOC], ids=lambda p: p.name)
def test_every_acp_command_is_documented(doc: Path) -> None:
    text = _text(doc)
    for name in _commands():
        assert f"agentos acp {name}" in text or f"`{name}" in text, (doc, name)


def test_every_harness_id_is_in_the_doc_table_and_router() -> None:
    doc = _text(ACP_DOC)
    skill = _text(SUB_AGENT_SKILL)
    for harness in list_harnesses():
        assert f"| `{harness.agent_id}` |" in doc, harness.agent_id
        assert harness.agent_id in skill, harness.agent_id


@pytest.mark.parametrize(
    "doc", [CONFIG_DOC, TOML_EXAMPLE, AGENTOS_SKILL, ACP_DOC], ids=lambda p: p.name
)
def test_every_acp_config_key_is_documented(doc: Path) -> None:
    text = _text(doc)
    assert "[acp" in text or "`[acp]`" in text
    for key in AcpConfig.model_fields:
        if key == "agents":
            assert "acp.agents" in text, doc
            continue
        assert key in text, (doc.name, key)
    for key in AcpAgentOverride.model_fields:
        assert key in text, (doc.name, key)


@pytest.mark.parametrize("doc", [SUB_AGENT_SKILL, ACP_DOC], ids=lambda p: p.name)
def test_acp_prompt_docs_are_english_only(doc: Path) -> None:
    hits = _VIETNAMESE.findall(_text(doc))
    assert not hits, f"{doc.name} contains non-English letters: {sorted(set(hits))}"


def test_sub_agent_skill_has_no_stale_cli_advice() -> None:
    text = _text(SUB_AGENT_SKILL)
    for stale in ("--yolo", "--full-auto", "gpt-5.2-codex", "pty", "PTY"):
        assert stale not in text, stale
    assert 'runtime="acp"' in text
    assert "sessions_yield" in text


def test_docs_index_links_the_acp_page() -> None:
    for rel in ("docs/README.md", "docs/features.md"):
        assert "acp-agents.md" in _text(ROOT / rel), rel
