"""Show what the fixed part of every provider request costs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.table import Table

from agentos.cli.output import print_json
from agentos.cli.ui import ACCENT_HEADER, console

if TYPE_CHECKING:
    from agentos.engine.context_breakdown import ContextBreakdown

app = typer.Typer(help="Inspect the fixed per-request context cost.")

_PROFILE_ORDER = ("full", "coding", "messaging", "memory_only", "minimal")


def _defer_loading_configured() -> bool:
    """``tools.defer_loading`` as the gateway would resolve it (file + env)."""

    try:
        from agentos.gateway.config import GatewayConfig
        from agentos.tools.deferred import defer_loading_enabled

        return defer_loading_enabled(GatewayConfig.load())
    except Exception:  # noqa: BLE001 - an unreadable config keeps the default
        return True


def _measure(defer_loading: bool = True) -> ContextBreakdown:
    from agentos.engine.context_breakdown import build_breakdown, tokens_from_chars
    from agentos.tools.policy_config import profile_allowlist
    from agentos.tools.registry import get_default_registry
    from agentos.tools.types import ToolContext

    registry = get_default_registry()
    available = frozenset(registry.list_names())

    # What a request carries, so deferred tools (``tools.defer_loading``) are
    # priced as the tool_search catalog line they cost, not their full schema.
    def cost_for(allowed: set[str] | None) -> tuple[int, int]:
        ctx = ToolContext(allowed_tools=allowed)
        definitions = registry.request_tool_definitions(ctx, defer_loading=defer_loading)
        from agentos.engine.context_breakdown import tool_payload_chars

        chars = sum(tool_payload_chars(d) for d in definitions)
        return len(definitions), tokens_from_chars(chars)

    profiles: dict[str, tuple[int, int]] = {}
    for name in _PROFILE_ORDER:
        try:
            allowed = profile_allowlist(name, available)
        except ValueError:
            continue
        profiles[name] = cost_for(allowed)

    definitions = registry.request_tool_definitions(ToolContext(), defer_loading=defer_loading)
    return build_breakdown(tools=definitions, profiles=profiles)


def _deferred_count(defer_loading: bool = True) -> int:
    from agentos.tools.registry import get_default_registry
    from agentos.tools.types import ToolContext

    if not defer_loading:
        return 0
    return len(get_default_registry().deferred_tools(ToolContext()))


@app.callback(invoke_without_command=True)
def context(
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
    top: int = typer.Option(10, "--top", help="How many of the largest tools to list"),
) -> None:
    """Show the fixed per-request cost and what each tool profile would cost.

    This is the overhead every provider call carries before a single word of
    conversation: tool schemas above all. It is charged on every call in every
    turn, so it is usually the largest thing an operator can actually change.
    """

    defer_loading = _defer_loading_configured()
    breakdown = _measure(defer_loading)

    if json_output:
        print_json(breakdown.to_dict())
        return

    console.print()
    console.print("[bold]Fixed per-request cost[/bold]", style=ACCENT_HEADER)
    section_table = Table(show_header=True, header_style=ACCENT_HEADER)
    section_table.add_column("Section")
    section_table.add_column("Items", justify="right")
    section_table.add_column("Chars", justify="right")
    section_table.add_column("~Tokens", justify="right")
    for section in breakdown.sections:
        section_table.add_row(
            section.name,
            str(section.items or ""),
            f"{section.chars:,}",
            f"{section.tokens:,}",
        )
    console.print(section_table)
    deferred_count = _deferred_count(defer_loading)
    if deferred_count:
        console.print(
            f"{deferred_count} more tools are deferred: left out of each request until "
            "the model loads them with tool_search. Set tools.defer_loading = false "
            "to send every tool.",
            style="dim",
        )

    largest = breakdown.largest_tools(top)
    if largest:
        console.print()
        console.print(f"[bold]Largest {len(largest)} tool schemas[/bold]", style=ACCENT_HEADER)
        tool_table = Table(show_header=True, header_style=ACCENT_HEADER)
        tool_table.add_column("Tool")
        tool_table.add_column("Chars", justify="right")
        tool_table.add_column("~Tokens", justify="right")
        for tool in largest:
            tool_table.add_row(tool.name, f"{tool.chars:,}", f"{tool.tokens:,}")
        console.print(tool_table)

    profiles = breakdown.profiles
    if profiles:
        baseline = profiles.get("full", (0, 0))[1]
        console.print()
        # Escaped: rich would read a bare [tools] as a style tag and drop it.
        console.print(r"[bold]What \[tools] profile would cost[/bold]", style=ACCENT_HEADER)
        profile_table = Table(show_header=True, header_style=ACCENT_HEADER)
        profile_table.add_column("Profile")
        profile_table.add_column("Tools", justify="right")
        profile_table.add_column("~Tokens", justify="right")
        profile_table.add_column("Saved", justify="right")
        deferred = False
        for name in _PROFILE_ORDER:
            if name not in profiles:
                continue
            count, tokens = profiles[name]
            if count == 0:
                # Not a 100% saving — the profile's tools simply are not in the
                # base registry. Memory tools, for one, register per agent at
                # boot. Reporting a saving here would be a lie.
                deferred = True
                profile_table.add_row(name, "0*", "—", "—")
                continue
            saved = f"{(baseline - tokens) / baseline:.0%}" if baseline else "—"
            profile_table.add_row(name, str(count), f"{tokens:,}", saved)
        console.print(profile_table)
        console.print()
        console.print(
            r"Set one in agentos.toml under \[tools] profile. The set is fixed for the "
            "session, so narrowing it does not disturb the prompt cache.",
            style="dim",
        )
        if deferred:
            console.print(
                "* resolves to no tools in the base registry; those tools register "
                "per agent at runtime.",
                style="dim",
            )
    console.print()
