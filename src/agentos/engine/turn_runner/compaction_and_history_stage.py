"""Stage object for the before-turn transcript history load.

Owns the per-turn slice between the agent-bootstrap stage boundary (Agent
construction) and the stream-consumer stage boundary (the
``async for event in agent.run_turn(...)`` loop). The harness invokes
``CompactionAndHistoryStage.run`` once per turn, AFTER
AgentBootstrapStage and BEFORE the attachment-message build +
stream-consumer loop.

Compaction no longer runs here. The before-turn paths this stage used to
own (T3-upgrade compaction, DB preflight compaction and the emergency
request-scoped trim) were folded into the agent's single
:class:`agentos.engine.compaction_service.CompactionService`, which checks
the live request before every provider call -- including the first one of
the turn -- and persists through the in-turn ``CompactionEvent`` path
(``persist_compaction_result`` + the durable checkpoint receipt). The class
name is kept so the runtime wiring and imports stay stable.

Side-effect contract:

- ``history_loader.load`` mutates ``agent._history`` via
  ``agent.set_history`` and returns a (possibly ``None``) durable
  compaction-summary context string.
- ``request_context_prepender.prepend`` is a pure string function whose
  return value the harness applies as
  ``agent.config.request_context_prompt = ...``.

NEVER terminates. Always returns ``StageOutcome.success(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentos.engine.agent import Agent
    from agentos.engine.turn_runner.outcome import StageOutcome

# ---------------------------------------------------------------------------
# Ports -- two narrow Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class HistoryLoaderPort(Protocol):
    """Wraps ``TurnRunner._load_history``.

    Reads the persisted transcript, reconstructs per-entry messages,
    and mutates ``agent._history`` via ``agent.set_history(history)``
    when any history was reconstructed.

    Returns ``str | None`` — the formatted compaction-summary context
    if durable summaries existed, else ``None``. The stage body feeds
    the return value into the prepender port.

    NOT a compaction port; does NOT fire ``CompactionHook``. History
    load is a read of already-persisted (possibly already-compacted)
    state.
    """

    async def load(
        self,
        *,
        agent: Agent,
        session_key: str,
        trim_last_user: bool,
    ) -> str | None: ...


@runtime_checkable
class RequestContextPrependPort(Protocol):
    """Wraps ``_prepend_request_context_prompt`` (module-level pure helper).

    Pure string function. Returns the prepended context (or the existing
    value unchanged when there is nothing to prepend). The harness
    applies the result with ``agent.config.request_context_prompt =
    ...`` after the stage returns; the stage itself does NOT mutate the
    agent.

    Reason for promoting a small pure function to a port: testability.
    A recording fake lets us assert "prepender was called with exactly
    these two arguments" without parsing log lines.
    """

    def prepend(
        self,
        *,
        existing: str | None,
        prepended: str | None,
    ) -> str | None: ...


# ---------------------------------------------------------------------------
# Stage I/O dataclasses (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactionAndHistoryStageInput:
    """Inputs the stage needs at its boundary.

    ``agent`` comes from AgentBootstrapStage; ``session_key``, ``agent_id``
    and ``history_has_persisted_user`` come from the ``_run_turn`` call site.
    """

    agent: Agent
    session_key: str
    agent_id: str
    history_has_persisted_user: bool


@dataclass(frozen=True)
class CompactionAndHistoryStageOutput:
    """The pieces of state subsequent stages and the harness consume.

    - ``compaction_summary_context``: the string passed to the
      prepender. ``None`` when no durable summaries exist.
    - ``final_request_context_prompt``: the result of the prepend.
      The harness applies this to
      ``agent.config.request_context_prompt`` after the stage returns.
      Surfaced (not applied inside the stage) so the harness remains
      the only place that mutates ``agent.config``.
    """

    compaction_summary_context: str | None
    final_request_context_prompt: str | None


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class CompactionAndHistoryStage:
    """Load history, prepend durable compaction context.

    Stable boundary: runs ONCE per turn, after ``AgentBootstrapStage``
    and before the attachment-build + stream-consumer steps:

    1. ``history_loader.load`` (always called).
    2. ``request_context_prepender.prepend`` (always called; pure).

    Exception model: history/prepender exceptions propagate to the outer
    terminal handler in ``_run_turn``.
    """

    name = "compaction_and_history_stage"

    def __init__(
        self,
        *,
        history_loader: HistoryLoaderPort,
        request_context_prepender: RequestContextPrependPort,
    ) -> None:
        self._history_loader = history_loader
        self._request_context_prepender = request_context_prepender

    async def run(
        self,
        inp: CompactionAndHistoryStageInput,
    ) -> StageOutcome[CompactionAndHistoryStageOutput]:
        # Local import keeps the module import-cycle-free.
        from agentos.engine.turn_runner.outcome import StageOutcome

        # 1. Load history (transcript + structured/legacy messages + durable summary).
        compaction_summary_context = await self._history_loader.load(
            agent=inp.agent,
            session_key=inp.session_key,
            trim_last_user=inp.history_has_persisted_user,
        )

        # 2. Prepend compaction summary context to request_context_prompt (pure).
        final_request_context_prompt = self._request_context_prepender.prepend(
            existing=inp.agent.config.request_context_prompt,
            prepended=compaction_summary_context,
        )

        return StageOutcome.success(
            CompactionAndHistoryStageOutput(
                compaction_summary_context=compaction_summary_context,
                final_request_context_prompt=final_request_context_prompt,
            )
        )
