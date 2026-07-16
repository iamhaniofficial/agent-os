"""One-time migration of a free-form MEMORY.md into curated §-entries.

Before curated memory existed, agents wrote MEMORY.md as free-form Markdown
(headings, bullet lists, paragraphs). The ``CuratedMemoryStore`` treats
MEMORY.md as a §-delimited list of small entries and rejects on-disk content
that would not round-trip through its parser/serializer (the drift guard). A
pre-curated free-form file would therefore trip that guard on the first write.

``migrate_freeform_memory_md`` runs once, before the store's first load:

  * If MEMORY.md is missing/empty, or already round-trips as curated entries,
    it is a no-op (returns ``False``).
  * Otherwise the free-form text is split into entries (blank-line blocks;
    bullet blocks split per bullet; heading-only blocks dropped), kept in file
    order up to 80% of the char budget, with the remainder archived to
    ``memory/archive/memory-overflow.md`` (still picked up by the memory sync
    scanner, so nothing is lost — it stays searchable). MEMORY.md is rewritten
    as a clean §-delimited list. Returns ``True``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import structlog

from agentos.memory.curated import ENTRY_DELIMITER

log = structlog.get_logger(__name__)

# Keep migrated entries to this fraction of the budget so the agent has room to
# add/replace without immediately hitting the consolidation wall on first use.
_KEEP_BUDGET_FRACTION = 0.8

_BULLET_MARKERS = ("- ", "* ")


def _is_bullet_line(line: str) -> bool:
    return line.lstrip().startswith(_BULLET_MARKERS)


def _strip_bullet(line: str) -> str:
    stripped = line.lstrip()
    for marker in _BULLET_MARKERS:
        if stripped.startswith(marker):
            return stripped[len(marker) :].strip()
    return stripped.strip()


def _is_heading_only(block: str) -> bool:
    """True when every non-empty line of *block* is a Markdown heading."""
    lines = [ln for ln in block.splitlines() if ln.strip()]
    return bool(lines) and all(ln.lstrip().startswith("#") for ln in lines)


def _block_to_entries(block: str) -> list[str]:
    """Turn one blank-line-delimited block into zero or more entries."""
    block = block.strip()
    if not block or _is_heading_only(block):
        return []

    lines = [ln for ln in block.splitlines() if ln.strip()]
    # A block whose every line is a bullet becomes one entry per bullet.
    if lines and all(_is_bullet_line(ln) for ln in lines):
        return [e for e in (_strip_bullet(ln) for ln in lines) if e]

    return [block]


def _split_freeform(raw: str) -> list[str]:
    entries: list[str] = []
    for block in raw.split("\n\n"):
        entries.extend(_block_to_entries(block))
    return entries


def _round_trips_as_curated(raw: str, memory_char_limit: int) -> bool:
    """True when MEMORY.md is already a clean curated §-entry list.

    Two conditions, both required:

    1. Round-trip / entry-size clean — mirrors
       ``CuratedMemoryStore._detect_external_drift``: re-parsing on the
       §-delimiter and re-serializing reproduces the stripped bytes and no
       single parsed entry exceeds the char budget.
    2. Free-form-split stable — splitting the same bytes with the free-form
       parser yields the *same* entries as the §-parser. This distinguishes a
       genuinely curated file from a free-form file that happens to contain no
       §-delimiter (a heading, bullet list, or multi-paragraph note parses as
       one §-entry but is not curated — the free-form split would break it up).
    """
    parsed = [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]
    roundtrip = ENTRY_DELIMITER.join(parsed)
    max_entry_len = max((len(e) for e in parsed), default=0)
    if raw.strip() != roundtrip or max_entry_len > memory_char_limit:
        return False
    # A file that already contains the §-delimiter was written by the curated
    # store -- the clean round-trip above is sufficient. A file with no
    # delimiter is only curated if the free-form parser leaves it as-is (a
    # heading, bullet list, or multi-paragraph note would be broken up).
    if ENTRY_DELIMITER in raw:
        return True
    return _split_freeform(raw) == parsed


def _partition_by_budget(entries: list[str], budget: int) -> tuple[list[str], list[str]]:
    """Keep entries in file order while the joined length stays within *budget*."""
    kept: list[str] = []
    overflow: list[str] = []
    for entry in entries:
        candidate = ENTRY_DELIMITER.join([*kept, entry])
        if kept and len(candidate) > budget:
            overflow.append(entry)
        elif not kept and len(entry) > budget:
            # Single first entry already over budget -> archive it, keep going.
            overflow.append(entry)
        else:
            kept.append(entry)
    return kept, overflow


def migrate_freeform_memory_md(
    memory_dir: Path,
    memory_char_limit: int,
    *,
    today: str | None = None,
) -> bool:
    """Migrate a free-form MEMORY.md into curated §-entries. One-time, idempotent.

    Returns ``True`` when a migration was performed, ``False`` when MEMORY.md is
    missing, empty, or already curated (in which case the file is untouched).

    ``today`` overrides the archive-header date (ISO ``YYYY-MM-DD``); it exists
    so tests stay offline/deterministic. Defaults to ``date.today()``.
    """
    memory_path = memory_dir / "MEMORY.md"
    if not memory_path.is_file():
        return False
    try:
        raw = memory_path.read_text(encoding="utf-8")
    except OSError:
        return False
    if not raw.strip():
        return False

    if _round_trips_as_curated(raw, memory_char_limit):
        return False

    entries = _split_freeform(raw)
    if not entries:
        # Nothing salvageable (e.g. headings only). Leave the file untouched so
        # a human can decide -- rewriting to empty would silently drop content.
        return False

    budget = int(memory_char_limit * _KEEP_BUDGET_FRACTION)
    kept, overflow = _partition_by_budget(entries, budget)

    if overflow:
        _append_overflow(memory_dir, overflow, today or date.today().isoformat())

    memory_path.write_text(ENTRY_DELIMITER.join(kept), encoding="utf-8")
    log.info(
        "memory_md_migrated",
        kept=len(kept),
        overflow=len(overflow),
        char_limit=memory_char_limit,
    )
    return True


def _append_overflow(memory_dir: Path, overflow: list[str], today: str) -> None:
    archive_path = memory_dir / "memory" / "archive" / "memory-overflow.md"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"## Migrated from MEMORY.md {today}\n\n"
    body = "\n\n".join(overflow) + "\n"
    existing = ""
    if archive_path.is_file():
        try:
            existing = archive_path.read_text(encoding="utf-8").rstrip() + "\n\n"
        except OSError:
            existing = ""
    archive_path.write_text(existing + header + body, encoding="utf-8")
