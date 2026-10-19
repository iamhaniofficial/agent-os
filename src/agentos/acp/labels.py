"""Short human labels for harness steps ("Run git status", "Read ~/app/x.py").

One builder serves the session status (``activity``) and the tool rows in the
child transcript, so the desktop bar and the transcript say the same thing.
A label is one line, at most :data:`MAX_LABEL_CHARS` characters, with the
user's home directory shown as ``~``. It is built from the tool call's
``rawInput`` first (the command, path, query or URL), then from its title,
then from its kind; a harness-internal tool name such as ``ToolSearch`` is
humanized rather than shown raw. Dependency-free (stdlib only).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

#: Longest label (one line, ``$HOME`` shortened to ``~``).
MAX_LABEL_CHARS = 80

#: The ACP tool kinds a label (and ``activity_kind``) may carry.
TOOL_KINDS = frozenset(
    {"read", "edit", "delete", "move", "search", "execute", "think", "fetch", "other"}
)

KIND_LABELS: dict[str, str] = {
    "read": "Reading a file",
    "edit": "Editing a file",
    "delete": "Deleting a file",
    "move": "Moving a file",
    "search": "Searching",
    "execute": "Running a command",
    "think": "Thinking",
    "fetch": "Fetching",
    "other": "Working",
}

#: Harness-internal tools (Claude Code names) and what they do for the user.
INTERNAL_TOOL_LABELS: dict[str, str] = {
    "toolsearch": "Looking up tools",
    "todowrite": "Updating the plan",
    "todoread": "Reading the plan",
    "taskcreate": "Updating the plan",
    "taskupdate": "Updating the plan",
    "tasklist": "Reading the plan",
    "taskget": "Reading the plan",
    "skill": "Loading a skill",
    "exitplanmode": "Finishing the plan",
    "enterplanmode": "Planning",
    "askuserquestion": "Asking a question",
    "listmcpresourcestool": "Listing MCP resources",
    "readmcpresourcetool": "Reading an MCP resource",
    "bashoutput": "Reading command output",
    "killshell": "Stopping a command",
    "killbash": "Stopping a command",
    "monitor": "Watching a task",
    "sendmessage": "Messaging a helper",
    "taskstop": "Stopping a helper",
    "listagents": "Listing helpers",
}

#: Titles a harness sends before the tool's input is known.
PLACEHOLDER_TITLES = frozenset(
    {"terminal", "preparing file…", "preparing file...", "edit", "read", "unknown tool", "other"}
)

_PATH_KEYS = ("file_path", "path", "abs_path", "filePath", "notebook_path", "filename", "file")
_QUERY_KEYS = ("pattern", "query", "regex", "q", "search")
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def shorten_home(text: str) -> str:
    """Show the user's home directory as ``~``."""
    home = os.path.expanduser("~")
    if not home or home in ("/", "~"):
        return text
    home = home.rstrip("/\\")
    text = text.replace(home + "/", "~/").replace(home + "\\", "~\\")
    if text == home or text.endswith(" " + home):
        text = text[: -len(home)] + "~"
    return text


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _first_line(value: Any) -> str:
    for line in str(value or "").splitlines():
        if line.strip():
            return _one_line(line)
    return ""


def _cap(text: str) -> str:
    text = shorten_home(_one_line(text))
    if len(text) > MAX_LABEL_CHARS:
        text = text[: MAX_LABEL_CHARS - 1].rstrip() + "…"
    return text


def _input_value(raw_input: Mapping[str, Any] | None, keys: tuple[str, ...]) -> str:
    if not isinstance(raw_input, Mapping):
        return ""
    for key in keys:
        value = raw_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _command(raw_input: Mapping[str, Any] | None) -> str:
    if not isinstance(raw_input, Mapping):
        return ""
    value = raw_input.get("command") or raw_input.get("cmd")
    if isinstance(value, list) and value:
        parts = [str(p) for p in value]
        # ``["bash", "-lc", "git status"]`` (Codex): the script is the command.
        if len(parts) >= 3 and parts[1] in ("-c", "-lc", "-ic") and parts[0].endswith("sh"):
            value = parts[2]
        else:
            value = " ".join(parts)
    return _first_line(value) if isinstance(value, str) else ""


def humanize_tool_name(name: str) -> str:
    """``ToolSearch`` -> "Looking up tools"; ``mcp__github__create_issue`` ->
    "github: create issue"; ``read_logs`` -> "Read logs"."""
    raw = (name or "").strip()
    if not raw:
        return ""
    known = INTERNAL_TOOL_LABELS.get(raw.replace("_", "").replace("-", "").lower())
    if known:
        return known
    if raw.startswith("mcp__"):
        parts = [p for p in raw.split("__")[1:] if p]
        if len(parts) >= 2:
            return f"{parts[0]}: {humanize_tool_name(parts[-1]).lower()}"
    words = _CAMEL.sub(" ", raw).replace("_", " ").replace("-", " ").split()
    if not words:
        return raw
    text = " ".join(words).lower()
    return text[:1].upper() + text[1:]


def harness_tool_name(raw: Mapping[str, Any] | None) -> str:
    """The harness's own tool name (``_meta.claudeCode.toolName`` or ``name``)."""
    if not isinstance(raw, Mapping):
        return ""
    meta = raw.get("_meta")
    if isinstance(meta, Mapping):
        claude = meta.get("claudeCode")
        if isinstance(claude, Mapping) and isinstance(claude.get("toolName"), str):
            return str(claude["toolName"])
    name = raw.get("name")
    return name if isinstance(name, str) else ""


def _is_placeholder(title: str) -> bool:
    return title.strip().lower() in PLACEHOLDER_TITLES


def _strip_ticks(text: str) -> str:
    return text.replace("`", "").strip()


def tool_label(
    title: str | None,
    kind: str | None,
    raw_input: Mapping[str, Any] | None = None,
    *,
    tool_name: str | None = None,
) -> str:
    """The label of one harness tool call.

    ``execute`` -> "Run <command>", ``read`` -> "Read <path>", ``edit`` ->
    "Edit <path>", ``search`` -> "Search <query>", ``fetch`` -> "Fetch <host>"
    (a web search: "Search <query>"), ``think`` -> "Thinking"; anything else
    keeps a descriptive title or humanizes the tool's name.
    """
    kind = kind if kind in TOOL_KINDS else "other"
    text = _first_line(title)
    title_text = "" if _is_placeholder(text) else _strip_ticks(text)
    if kind == "execute":
        command = _command(raw_input) or title_text
        if command.lower().startswith("run "):
            return _cap(command)
        return _cap(f"Run {command}") if command else KIND_LABELS["execute"]
    if kind in ("read", "edit", "delete", "move"):
        path = _input_value(raw_input, _PATH_KEYS)
        verb = {"read": "Read", "edit": "Edit", "delete": "Delete", "move": "Move"}[kind]
        if path:
            first = title_text.split(" ", 1)[0] if title_text else ""
            if kind == "edit" and first in ("Write", "Create"):
                verb = first
            return _cap(f"{verb} {path}")
        return _cap(title_text) if title_text else KIND_LABELS[kind]
    if kind == "search":
        query = _input_value(raw_input, _QUERY_KEYS)
        if query:
            return _cap(f"Search {query}")
        return _cap(title_text) if title_text else KIND_LABELS["search"]
    if kind == "fetch":
        url = _input_value(raw_input, ("url", "uri"))
        if url:
            host = urlparse(url).netloc or url
            return _cap(f"Fetch {host}")
        query = _input_value(raw_input, ("query", "q"))
        if query:
            return _cap(f"Search {query}")
        return _cap(title_text) if title_text else KIND_LABELS["fetch"]
    if kind == "think":
        return KIND_LABELS["think"]
    # other: a descriptive title wins; a bare identifier is a tool name.
    if title_text and not _IDENTIFIER.match(title_text):
        return _cap(title_text)
    name = title_text or (tool_name or "")
    humanized = humanize_tool_name(name)
    return _cap(humanized) if humanized else KIND_LABELS["other"]


def tool_label_for_event(event: Any) -> str:
    """:func:`tool_label` for an :class:`~agentos.acp.types.AcpEvent`."""
    return tool_label(
        getattr(event, "title", None),
        getattr(event, "tool_kind", None),
        getattr(event, "raw_input", None),
        tool_name=harness_tool_name(getattr(event, "raw", None)),
    )
