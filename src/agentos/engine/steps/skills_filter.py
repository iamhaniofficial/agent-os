"""Step 3: Gate skills deterministically, optionally filter by relevance, inject."""

from __future__ import annotations

import threading
from typing import Any, cast

import structlog

from agentos.engine.pipeline import TurnContext
from agentos.skills.availability import gate_skills, plan_injection, retrieval_availability
from agentos.skills.eligibility import EligibilityContext
from agentos.skills.injector import DEFAULT_MAX_SKILLS_PROMPT_CHARS, RENDER_MODE_FULL
from agentos.skills.retrieval import HybridRetriever, Strategy

log = structlog.get_logger(__name__)

_retriever: HybridRetriever | None = None
_retriever_lock = threading.Lock()


def _eligibility_context() -> EligibilityContext:
    """Build a fresh eligibility context for this turn.

    A process-wide context cached negative `shutil.which()` and env lookups
    forever: installing a missing binary or setting a credential never took
    effect until restart, while every RPC surface builds its own context per
    call and so reported the skill as ready. ~15 `which()` calls per turn is
    cheap next to that divergence."""
    return EligibilityContext.auto()


def _get_retriever(skills_cfg: Any) -> HybridRetriever:
    """Return a process-wide HybridRetriever sized to current config.
    Recreate when retrieval-shaping fields change so that tuning takes
    effect on next turn."""
    global _retriever
    rrf_k = getattr(skills_cfg, "filter_rrf_k", 60)
    lex_top_n = getattr(skills_cfg, "filter_lexical_top_n", 20)
    sem_top_n = getattr(skills_cfg, "filter_semantic_top_n", 20)
    model_name = getattr(skills_cfg, "filter_embedding_model", None)
    strategy = cast(Strategy, getattr(skills_cfg, "filter_strategy", "lexical"))
    config_key = (rrf_k, lex_top_n, sem_top_n, model_name, strategy)
    with _retriever_lock:
        if _retriever is None or getattr(_retriever, "_config_key", None) != config_key:
            from agentos.skills.retrieval.embedder import get_embedder

            # strategy="lexical" never needs an embedder; skip the lookup
            # so a missing skill-filter extra is not even attempted.
            embedder = None
            if strategy != "lexical" and model_name:
                try:
                    embedder = get_embedder(model_name)
                except ImportError:
                    embedder = None
            r = HybridRetriever(
                embedder=embedder,
                rrf_k=rrf_k,
                lexical_top_n=lex_top_n,
                semantic_top_n=sem_top_n,
                strategy=strategy,
            )
            setattr(r, "_config_key", config_key)
            _retriever = r
        return _retriever


async def filter_skills(ctx: TurnContext) -> TurnContext:
    """Gate, optionally filter, and inject skills into the system prompt."""
    skill_loader = ctx.metadata.get("skill_loader")
    if skill_loader is None:
        return ctx

    tools_cfg = getattr(ctx.config, "tools", None) if ctx.config else None
    if getattr(tools_cfg, "profile", None) == "memory_only":
        ctx.metadata["filtered_skill_ids"] = []
        ctx.metadata["skill_count"] = 0
        ctx.metadata["skills_prompt_chars"] = 0
        ctx.metadata["skill_availability"] = {}
        log.debug("skills_filter.skipped", reason="memory_only")
        return ctx

    all_skills = skill_loader.load_all()
    if not all_skills:
        return ctx

    # ── deterministic gate (no LLM, pure Python) ──
    # gate_skills / plan_injection are shared with the RPC surface that answers
    # "why is this skill not offered?" — the engine must not compute the gate
    # its own way, or the two answers drift.
    available_tools = {t.name for t in ctx.tool_defs} if ctx.tool_defs else set()
    # A deferred tool is one tool_search (or skill_view) away, so a skill that
    # requires it is still offered.
    from agentos.tools.deferred import deferred_tool_names

    available_tools.update(deferred_tool_names(ctx.session_key))
    gated, availability = gate_skills(all_skills, available_tools, _eligibility_context())

    # ── always skills bypass filter, guaranteed visibility ──
    pinned = [s for s in gated if s.always]
    filterable = [s for s in gated if not s.always]

    skills_cfg = getattr(ctx.config, "skills", None) if ctx.config else None
    filter_enabled = getattr(skills_cfg, "filter_enabled", False) if skills_cfg else False
    max_chars = getattr(skills_cfg, "max_skills_prompt_chars", DEFAULT_MAX_SKILLS_PROMPT_CHARS)
    injection_mode = getattr(skills_cfg, "injection_mode", "system")
    semantic_message = getattr(ctx, "semantic_message", None)
    if semantic_message is None:
        semantic_message = getattr(ctx, "raw_message", None)
    if semantic_message is None:
        semantic_message = ctx.message

    if filter_enabled:
        top_k = getattr(skills_cfg, "filter_top_k", 5)
        # Strategy ("lexical" / "semantic" / "hybrid") is handled inside
        # HybridRetriever — see agentos.skills.retrieval.HybridRetriever.
        retriever = _get_retriever(skills_cfg)
        filtered = retriever.retrieve(filterable, semantic_message, top_k=top_k)
        retrieved = {s.name for s in filtered}
        availability.update(
            retrieval_availability([s for s in filterable if s.name not in retrieved], top_k)
        )
    else:
        filtered = filterable

    final = pinned + filtered

    # tuple[1] is the uncached suffix slot: may already carry the
    # per-turn recalled-memory block produced upstream by
    # TurnRunner._assemble_prompt. Append instead of overwriting so that
    # both recall and skills survive the pipeline together.
    if isinstance(ctx.system_prompt, str):
        base, suffix = ctx.system_prompt, ""
    else:
        base, suffix = ctx.system_prompt

    plan = plan_injection(
        final,
        max_chars,
        injection_mode,
        # Which route back to the descriptions the block advertises has to match
        # the live tool surface: cron agents run under an allowlist, so pointing
        # them at a tool they do not have is worse than saying nothing.
        skill_list_tool="skill_list" in available_tools,
    )
    skills_prompt, dropped = plan.prompt, plan.dropped
    availability.update(plan.availability)

    # Everything below describes what actually reached the prompt, not the
    # pre-truncation list: a metadata count that includes skills the budget
    # threw away is how this stayed invisible.
    dropped_names = set(dropped)
    injected = plan.offered

    # Publish the injected skill-ID list so the pipeline wrapper can
    # surface it in the decision log's PipelineStepRecord. Non-mutating
    # additive read for callers that don't consume the metadata.
    try:
        ctx.metadata["filtered_skill_ids"] = [
            getattr(s, "id", None) or getattr(s, "name", None)
            for s in filtered
            if (getattr(s, "id", None) or getattr(s, "name", None)) and s.name not in dropped_names
        ]
    except Exception:  # pragma: no cover — metadata is best-effort
        ctx.metadata["filtered_skill_ids"] = []

    ctx.metadata["skill_count"] = len(injected)
    ctx.metadata["skills_prompt_chars"] = len(skills_prompt)
    ctx.metadata["skills_injection_mode"] = injection_mode
    ctx.metadata["skills_dropped_for_budget"] = dropped
    # How the block was rendered, not just how big it came out. A budget that
    # silently threw away every description reported an empty `dropped` list and
    # a smaller char count — indistinguishable from the operator uninstalling
    # skills, which is how it went unnoticed across every turn of an install.
    ctx.metadata["skills_render_mode"] = plan.mode
    ctx.metadata["skills_description_max_chars"] = plan.description_max_chars
    # One entry per loaded skill: offered, or the reason it is not. Same map the
    # RPC surface builds from the same two functions.
    ctx.metadata["skill_availability"] = availability

    if dropped:
        log.warning(
            "skills_filter.budget_truncated",
            dropped=dropped,
            budget=max_chars,
            kept=len(injected),
        )
    elif plan.mode != RENDER_MODE_FULL and injection_mode != "user_message":
        # "user_message" renders compact by configuration, not under budget
        # pressure, so the same warning there would be noise on every turn and
        # its advice — raise the budget — would not change anything.
        log.warning(
            "skills_filter.budget_degraded",
            mode=plan.mode,
            description_max_chars=plan.description_max_chars,
            budget=max_chars,
            skills=len(injected),
            hint="raise skills.max_skills_prompt_chars or enable skills.filter_enabled",
        )

    # Surface the actual skill IDs the retriever picked. Without this,
    # operators can see a query passed through (total → filtered count)
    # but cannot tell which skills were chosen vs missed — the diagnostic
    # signal needed to debug recall quality (e.g. "why did 'commit my
    # changes to git' not surface `git`?").
    pinned_ids = [
        getattr(s, "id", None) or getattr(s, "name", None)
        for s in pinned
        if getattr(s, "id", None) or getattr(s, "name", None)
    ]
    filtered_ids = ctx.metadata.get("filtered_skill_ids") or []
    log.debug(
        "skills_filter.applied",
        total=len(all_skills),
        gated=len(gated),
        pinned=len(pinned),
        filtered=len(injected),
        mode=injection_mode,
        strategy=getattr(skills_cfg, "filter_strategy", "lexical") if filter_enabled else "off",
        query_preview=(
            semantic_message[:60] + "..."
            if isinstance(semantic_message, str) and len(semantic_message) > 60
            else semantic_message
        ),
        pinned_skills=pinned_ids,
        filtered_skills=filtered_ids,
    )

    if skills_prompt and injection_mode == "user_context":
        ctx.metadata["skills_context_prompt"] = skills_prompt
    elif skills_prompt and not filter_enabled:
        # The block belongs in the cacheable base, not the dynamic suffix.
        # Downstream, an enabled prompt cache splits a (base, dynamic) prompt by
        # sending base as the system message and the dynamic half as a *user*
        # message headed "not a user request ... use it only when it is
        # relevant" — which is the weakest position in the request and directly
        # contradicts the block's own "read this before answering". Skills are
        # instructions, so they have to stay system-role.
        #
        # Putting them in the cached half is also the cheaper choice: with
        # relevance filtering off the list is the same on every turn, so paying
        # to re-send it uncached bought nothing. It only busts the cached prefix
        # when the offered set genuinely changes — an install, a removal, or a
        # skill becoming eligible.
        ctx.system_prompt = (f"{base}{skills_prompt}", suffix) if suffix else base + skills_prompt
    elif skills_prompt:
        # Relevance filtering re-picks the list from the message, so it varies
        # per turn and would bust the cached prefix every time. Keep it in the
        # dynamic slot.
        combined_suffix = f"{suffix}\n\n{skills_prompt}" if suffix else skills_prompt
        ctx.system_prompt = (base, combined_suffix)
    # else: leave ctx.system_prompt unchanged — preserves any upstream tuple
    # carrying the recall block, or stays str when neither is present.

    return ctx
