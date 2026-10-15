"""Built-in ACP harness registry.

Derived from OpenClaw's ``acpx`` agent registry (MIT; see
``THIRD_PARTY_NOTICES.md``). ``argv`` is the zero-install launch (``npx``
/ ``uvx`` where an adapter package exists); ``installed_argv`` is preferred
when the binary is already on ``PATH``. ``required_commands`` must resolve
on ``PATH`` for the harness to be usable at all (the adapter only wraps it).
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class HarnessDefinition:
    agent_id: str
    name: str
    argv: tuple[str, ...]
    installed_argv: tuple[str, ...] = ()
    required_commands: tuple[str, ...] = ()
    package_name: str | None = None
    package_range: str | None = None
    preferred_bin: str | None = None
    notes: str = ""


_CLAUDE_ACP_RANGE = "^0.91.0"
_CODEX_ACP_RANGE = "^2.2.2"
_PI_ACP_RANGE = "^0.0.33"
_MUX_RANGE = "^0.28.0"


def _antigravity_argv() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("agy_acp_server.exe",)
    if sys.platform.startswith("linux"):
        return ("agy_acp_server.par", "--uid=")
    return ("agy_acp_server.par",)


BUILTIN_HARNESSES: tuple[HarnessDefinition, ...] = (
    HarnessDefinition(
        "claude",
        "Claude Code",
        ("npx", "-y", f"@agentclientprotocol/claude-agent-acp@{_CLAUDE_ACP_RANGE}"),
        installed_argv=("claude-agent-acp",),
        required_commands=("claude",),
        package_name="@agentclientprotocol/claude-agent-acp",
        package_range=_CLAUDE_ACP_RANGE,
        preferred_bin="claude-agent-acp",
        notes="Wraps the Claude Agent SDK; reports USD cost in usage_update.",
    ),
    HarnessDefinition(
        "codex",
        "Codex",
        ("npx", "-y", f"@agentclientprotocol/codex-acp@{_CODEX_ACP_RANGE}"),
        installed_argv=("codex-acp",),
        required_commands=("codex",),
        package_name="@agentclientprotocol/codex-acp",
        package_range=_CODEX_ACP_RANGE,
        preferred_bin="codex-acp",
        notes="Built on codex app-server; modes read-only / workspace-write / full-access.",
    ),
    HarnessDefinition("gemini", "Gemini CLI", ("gemini", "--acp"), required_commands=("gemini",)),
    HarnessDefinition(
        "cursor", "Cursor", ("cursor-agent", "acp"), required_commands=("cursor-agent",)
    ),
    HarnessDefinition(
        "copilot",
        "GitHub Copilot",
        ("copilot", "--acp", "--stdio"),
        required_commands=("copilot",),
    ),
    HarnessDefinition(
        "opencode",
        "OpenCode",
        ("npx", "-y", "opencode-ai", "acp"),
        installed_argv=("opencode", "acp"),
    ),
    HarnessDefinition(
        "pi",
        "Pi",
        ("npx", f"pi-acp@{_PI_ACP_RANGE}"),
        installed_argv=("pi-acp",),
        required_commands=("pi",),
        package_name="pi-acp",
        package_range=_PI_ACP_RANGE,
        preferred_bin="pi-acp",
    ),
    HarnessDefinition(
        "droid",
        "Factory Droid",
        ("droid", "exec", "--output-format", "acp"),
        required_commands=("droid",),
    ),
    HarnessDefinition(
        "grok-build", "Grok Build", ("grok", "agent", "stdio"), required_commands=("grok",)
    ),
    HarnessDefinition(
        "iflow", "iFlow", ("iflow", "--experimental-acp"), required_commands=("iflow",)
    ),
    HarnessDefinition("junie", "Junie", ("junie", "--acp=true"), required_commands=("junie",)),
    HarnessDefinition(
        "kilocode",
        "Kilo Code",
        ("npx", "-y", "@kilocode/cli", "acp"),
        installed_argv=("kilo", "acp"),
    ),
    HarnessDefinition("kimi", "Kimi Code", ("kimi", "acp"), required_commands=("kimi",)),
    HarnessDefinition(
        "kiro", "Kiro", ("kiro-cli-chat", "acp"), required_commands=("kiro-cli-chat",)
    ),
    HarnessDefinition("mcode", "MiniMax Code", ("mcode", "acp"), required_commands=("mcode",)),
    HarnessDefinition("mux", "Mux", ("npx", "-y", f"mux@{_MUX_RANGE}", "acp")),
    HarnessDefinition("pool", "Poolside", ("pool", "acp"), required_commands=("pool",)),
    HarnessDefinition("qoder", "Qoder", ("qodercli", "--acp"), required_commands=("qodercli",)),
    HarnessDefinition("qwen", "Qwen Code", ("qwen", "--acp"), required_commands=("qwen",)),
    HarnessDefinition("trae", "Trae", ("traecli", "acp", "serve"), required_commands=("traecli",)),
    HarnessDefinition("zeroclaw", "ZeroClaw", ("zeroclaw", "acp"), required_commands=("zeroclaw",)),
    HarnessDefinition("devin", "Devin", ("devin", "acp"), required_commands=("devin",)),
    HarnessDefinition("fx", "fx", ("fx", "acp"), required_commands=("fx",)),
    HarnessDefinition(
        "fast-agent", "Fast Agent", ("uvx", "fast-agent-mcp", "acp"), required_commands=("uvx",)
    ),
    HarnessDefinition(
        "antigravity",
        "Google Antigravity",
        _antigravity_argv(),
        required_commands=(_antigravity_argv()[0],),
    ),
    HarnessDefinition(
        "openclaw", "OpenClaw ACP bridge", ("openclaw", "acp"), required_commands=("openclaw",)
    ),
)

ALIASES: dict[str, str] = {
    "claude-code": "claude",
    "claudecode": "claude",
    "factorydroid": "droid",
    "factory-droid": "droid",
    "gemini-cli": "gemini",
    "github-copilot": "copilot",
    "cursor-agent": "cursor",
    "kilo": "kilocode",
    "kimi-cli": "kimi",
    "kiro-cli": "kiro",
    "qwen-code": "qwen",
    "grok": "grok-build",
}

_BY_ID: dict[str, HarnessDefinition] = {h.agent_id: h for h in BUILTIN_HARNESSES}


def normalize_agent_id(value: str) -> str:
    key = (value or "").strip().lower()
    return ALIASES.get(key, key)


def list_harnesses() -> list[HarnessDefinition]:
    return list(BUILTIN_HARNESSES)


def get_harness(agent_id: str) -> HarnessDefinition | None:
    return _BY_ID.get(normalize_agent_id(agent_id))


@dataclass(slots=True)
class ResolvedLaunch:
    agent_id: str
    argv: list[str]
    source: str  # installed | package-exec | override
    missing_commands: list[str] = field(default_factory=list)


def resolve_launch(
    agent_id: str,
    *,
    overrides: dict[str, list[str]] | None = None,
    which=shutil.which,
) -> ResolvedLaunch:
    """Pick argv for a harness: config override > installed binary > npx form.

    ``missing_commands`` lists required CLIs that are not on PATH so callers
    (``doctor``, spawn) can fail with a precise message instead of a stuck
    ``npx`` download.
    """
    aid = normalize_agent_id(agent_id)
    if overrides and aid in overrides and overrides[aid]:
        return ResolvedLaunch(aid, list(overrides[aid]), "override")
    h = _BY_ID.get(aid)
    if h is None:
        raise KeyError(aid)
    missing = [c for c in h.required_commands if which(c) is None]
    if h.installed_argv and which(h.installed_argv[0]) is not None:
        return ResolvedLaunch(aid, list(h.installed_argv), "installed", missing)
    return ResolvedLaunch(aid, list(h.argv), "package-exec", missing)
