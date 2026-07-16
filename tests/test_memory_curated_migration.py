"""One-time migration of free-form MEMORY.md into curated §-entries."""

from pathlib import Path

from agentos.memory.curated import ENTRY_DELIMITER, CuratedMemoryStore
from agentos.memory.curated_migration import migrate_freeform_memory_md


def test_freeform_memory_md_is_split_into_entries(tmp_path: Path):
    (tmp_path / "MEMORY.md").write_text(
        "# Notes\n\n- User prefers dark mode\n- Deploy via make deploy\n\nLong paragraph note.",
        encoding="utf-8",
    )
    assert migrate_freeform_memory_md(tmp_path, memory_char_limit=4000) is True
    store = CuratedMemoryStore(memory_dir=tmp_path)
    store.load_from_disk()
    entries = store.entries_for("memory")
    assert "User prefers dark mode" in ENTRY_DELIMITER.join(entries)
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8").count("§") == len(entries) - 1


def test_overflow_goes_to_archive(tmp_path: Path):
    (tmp_path / "MEMORY.md").write_text(
        "\n\n".join(f"note {i} " + "x" * 100 for i in range(20)), encoding="utf-8"
    )
    migrate_freeform_memory_md(tmp_path, memory_char_limit=500)
    overflow = tmp_path / "memory" / "archive" / "memory-overflow.md"
    assert overflow.exists()
    store = CuratedMemoryStore(memory_dir=tmp_path, memory_char_limit=500)
    store.load_from_disk()
    assert len(ENTRY_DELIMITER.join(store.entries_for("memory"))) <= 500


def test_already_curated_file_is_untouched(tmp_path: Path):
    curated = f"entry A{ENTRY_DELIMITER}entry B"
    (tmp_path / "MEMORY.md").write_text(curated, encoding="utf-8")
    assert migrate_freeform_memory_md(tmp_path, memory_char_limit=4000) is False
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8") == curated


def test_missing_file_is_noop(tmp_path: Path):
    assert migrate_freeform_memory_md(tmp_path, memory_char_limit=4000) is False


def test_empty_file_is_noop(tmp_path: Path):
    (tmp_path / "MEMORY.md").write_text("   \n\n  ", encoding="utf-8")
    assert migrate_freeform_memory_md(tmp_path, memory_char_limit=4000) is False


def test_heading_only_blocks_are_dropped(tmp_path: Path):
    (tmp_path / "MEMORY.md").write_text(
        "# Heading\n\nreal fact one\n\n## Another Heading", encoding="utf-8"
    )
    assert migrate_freeform_memory_md(tmp_path, memory_char_limit=4000) is True
    store = CuratedMemoryStore(memory_dir=tmp_path)
    store.load_from_disk()
    entries = store.entries_for("memory")
    assert entries == ["real fact one"]


def test_bullet_block_splits_per_bullet_stripping_marker(tmp_path: Path):
    (tmp_path / "MEMORY.md").write_text(
        "- first bullet\n* second bullet", encoding="utf-8"
    )
    migrate_freeform_memory_md(tmp_path, memory_char_limit=4000)
    store = CuratedMemoryStore(memory_dir=tmp_path)
    store.load_from_disk()
    entries = store.entries_for("memory")
    assert entries == ["first bullet", "second bullet"]


def test_file_order_preserved_and_overflow_header_present(tmp_path: Path):
    (tmp_path / "MEMORY.md").write_text(
        "\n\n".join(f"note {i} " + "x" * 100 for i in range(20)), encoding="utf-8"
    )
    migrate_freeform_memory_md(tmp_path, memory_char_limit=500, today="2026-07-15")
    store = CuratedMemoryStore(memory_dir=tmp_path, memory_char_limit=500)
    store.load_from_disk()
    kept = store.entries_for("memory")
    # Kept entries are the earliest in file order.
    assert kept[0].startswith("note 0")
    overflow = (tmp_path / "memory" / "archive" / "memory-overflow.md").read_text(
        encoding="utf-8"
    )
    assert "## Migrated from MEMORY.md 2026-07-15" in overflow
    # The last note overflowed and lives in the archive.
    assert "note 19" in overflow


def test_migration_is_idempotent(tmp_path: Path):
    (tmp_path / "MEMORY.md").write_text("- one\n- two\n\nparagraph", encoding="utf-8")
    assert migrate_freeform_memory_md(tmp_path, memory_char_limit=4000) is True
    first = (tmp_path / "MEMORY.md").read_text(encoding="utf-8")
    # Second run sees a now-curated file and is a no-op.
    assert migrate_freeform_memory_md(tmp_path, memory_char_limit=4000) is False
    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8") == first
