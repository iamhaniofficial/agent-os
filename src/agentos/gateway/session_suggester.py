"""Suggest the user's next prompt once a turn has finished.

The chat input shows the suggestion as ghost text; Tab takes it, Enter sends
it. Clients ask for it through ``chat.suggestNext`` after they see the turn
end, so this module never sits on the turn path: a slow or failing model only
means no ghost text.

Rules that keep it cheap and quiet:

* Only a finished turn gets a suggestion: the latest entry (tool results
  aside) must be an assistant reply with text. An errored turn ends on a
  ``system`` "Error: …" entry and an aborted one before any reply, so neither
  costs a model call.
* The turn runner yields ``DoneEvent`` before it persists the reply, so a
  client asking the moment it sees ``done`` can arrive first. While the
  latest entry is still the user's message the suggester waits briefly
  (``settle_seconds``) for the reply to land.
* One model call per finished turn, however many clients ask. Results — a
  ``None`` included, so a failing model is not hammered — are cached by the
  reply's ``message_id``, and concurrent askers share the in-flight call.
* No heuristic fallback: no suggestion beats a bad one.
"""

from __future__ import annotations

import asyncio
import re
from collections import OrderedDict
from typing import Any

import structlog

from agentos.provider.auxiliary import AuxiliaryError, get_auxiliary_client
from agentos.provider.types import ChatConfig, Message

from .session_titler import ModelHint

log = structlog.get_logger()

TASK = "suggest_next"
MAX_SUGGESTION_CHARS = 160
# Each side of the exchange the model sees. The reply keeps its tail, which is
# where an agent says what is left to do.
MAX_USER_CHARS = 1_500
MAX_REPLY_CHARS = 1_500
# Enough to step over the tool results a turn leaves behind its last reply.
RECENT_ENTRIES = 12
CACHE_SIZE = 256
SETTLE_SECONDS = 3.0
SETTLE_POLL_SECONDS = 0.1
# Reasoning models think before the visible answer; see TITLE_MAX_TOKENS.
SUGGEST_MAX_TOKENS = 512

SYSTEM_PROMPT = (
    "You predict the user's next message in a chat with an AI agent. Given the "
    "user's last message and the agent's reply, reply with ONLY the single most "
    "likely next message the user would type, written as the user, in the same "
    "language as the user. One line, at most 15 words. No quotes, no "
    "explanation, no prefix. If the conversation is finished or no follow-up is "
    "natural, reply with exactly NONE."
)

_QUOTES = "\"'“”‘’«»`"
_PREFIX = re.compile(
    r"^\s*(suggestion|suggested|next( message| prompt)?|user|gợi ý)\s*[:：]\s*",
    re.IGNORECASE,
)

CacheKey = tuple[str, str]


def clean_suggestion(raw: str | None) -> str | None:
    """Turn model output into one line of ghost text, or ``None``."""
    lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    if not lines:
        return None
    text = _PREFIX.sub("", lines[0]).strip().strip(_QUOTES).strip()
    if not text or text.rstrip(".!").upper() == "NONE":
        return None
    if len(text) > MAX_SUGGESTION_CHARS:
        cut = text[: MAX_SUGGESTION_CHARS + 1].rsplit(" ", 1)[0].rstrip()
        text = cut or text[:MAX_SUGGESTION_CHARS]
    return text


def _last_exchange(entries: list[Any]) -> tuple[str, str, str] | None:
    """(reply message_id, user text, reply text) of a finished turn, or None."""
    reply = None
    index = len(entries) - 1
    while index >= 0 and getattr(entries[index], "role", "") == "tool":
        index -= 1
    if index >= 0:
        reply = entries[index]
    if reply is None or reply.role != "assistant" or not (reply.content or "").strip():
        return None
    user_text = ""
    for entry in reversed(entries[:index]):
        if entry.role == "user" and (entry.content or "").strip():
            user_text = entry.content
            break
    return str(reply.message_id), user_text, reply.content


def _awaiting_reply(entries: list[Any]) -> bool:
    """True while the latest entry (tool results aside) is the user's message."""
    for entry in reversed(entries):
        role = getattr(entry, "role", "")
        if role != "tool":
            return bool(role == "user")
    return False


def _prompt(user_text: str, reply_text: str) -> str:
    user = user_text.strip()[:MAX_USER_CHARS]
    reply = reply_text.strip()
    if len(reply) > MAX_REPLY_CHARS:
        reply = "…" + reply[-MAX_REPLY_CHARS:]
    return f"User's last message:\n{user or '(none)'}\n\nAgent's reply:\n{reply}"


class SessionSuggester:
    """Answers ``chat.suggestNext``: one cached model call per finished turn."""

    def __init__(
        self,
        session_manager: Any,
        *,
        timeout: float = 15.0,
        hint: ModelHint = ("", ""),
        settle_seconds: float = SETTLE_SECONDS,
    ) -> None:
        self._mgr = session_manager
        self._timeout = timeout
        self._hint = hint
        self._settle_seconds = settle_seconds
        self._cache: OrderedDict[CacheKey, str | None] = OrderedDict()
        self._inflight: dict[CacheKey, asyncio.Future[str | None]] = {}

    async def suggest(self, session_key: str) -> str | None:
        entries = await self._settled_transcript(session_key)
        if entries is None:
            return None
        exchange = _last_exchange(entries)
        if exchange is None:
            return None
        message_id, user_text, reply_text = exchange
        key = (session_key, message_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        pending = self._inflight.get(key)
        if pending is not None:
            return await asyncio.shield(pending)

        future: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            text = await self._generate(session_key, _prompt(user_text, reply_text))
        except BaseException:
            # A cancelled asker must not strand the others on a future that
            # never resolves; they get no suggestion, and nothing is cached.
            future.set_result(None)
            raise
        finally:
            self._inflight.pop(key, None)
        future.set_result(text)
        self._remember(key, text)
        return text

    async def _settled_transcript(self, session_key: str) -> list[Any] | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._settle_seconds
        while True:
            try:
                entries = await self._mgr.get_recent_transcript(session_key, RECENT_ENTRIES)
            except KeyError:
                return None
            if not _awaiting_reply(entries) or loop.time() >= deadline:
                return list(entries)
            await asyncio.sleep(SETTLE_POLL_SECONDS)

    async def _generate(self, session_key: str, prompt: str) -> str | None:
        try:
            result = await get_auxiliary_client().complete(
                task=TASK,
                messages=[Message(role="user", content=prompt)],
                preferred_provider=self._hint[0],
                preferred_model=self._hint[1],
                chat_config=ChatConfig(
                    max_tokens=SUGGEST_MAX_TOKENS, temperature=0.3, system=SYSTEM_PROMPT
                ),
                timeout=self._timeout,
                session_key=session_key,
            )
            return clean_suggestion(result.text)
        except AuxiliaryError as exc:
            log.info("suggest_next.unavailable", session_key=session_key, error=str(exc))
        except Exception:
            log.exception("suggest_next.failed", session_key=session_key)
        return None

    def _remember(self, key: CacheKey, text: str | None) -> None:
        self._cache[key] = text
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)


_suggesters: dict[int, SessionSuggester] = {}


def suggester_for(
    session_manager: Any,
    *,
    timeout: float = 15.0,
    hint: ModelHint = ("", ""),
) -> SessionSuggester:
    """One suggester per session manager, so its cache spans every client.

    ``timeout`` and ``hint`` are refreshed on each call: both come from config
    an operator may change while the gateway runs.
    """
    key = id(session_manager)
    suggester = _suggesters.get(key)
    if suggester is None:
        suggester = SessionSuggester(session_manager, timeout=timeout, hint=hint)
        _suggesters[key] = suggester
    else:
        suggester._timeout = timeout
        suggester._hint = hint
    return suggester


def reset_suggesters() -> None:
    """Forget every suggester (tests)."""
    _suggesters.clear()
