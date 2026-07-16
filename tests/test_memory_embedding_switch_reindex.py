"""Pin the reindex contract for migrated memory across an embedding switch.

Scenario: content arrives via ``agentos migrate`` from Hermes as plain
Markdown (a ``memory/imported-note.md`` plus a ``MEMORY.md``), gets indexed
under local embedding provider A, and then the user switches the local
embedding model (the BGE -> downloaded EmbeddingGemma path from Plan C).
Switching the model changes ``MemoryEmbeddingDecision.fingerprint``, which is
stamped onto the provider as ``_provider_fingerprint`` and, with the new
model, a different vector dimensionality.

Contract pinned here (the REAL behavior in ``LongTermMemoryStore`` +
``MemorySyncManager``):

* On the next gateway run, ``store.initialize()`` calls
  ``_check_meta_and_reindex()``. A fingerprint/dims mismatch DROPS the stale
  index (chunks / files / fts / ``chunks_vec``) and marks the store dirty. It
  does NOT re-embed by itself.
* The very next ``MemorySyncManager.sync()`` re-reads the on-disk Markdown
  (every file looks "new" because ``_mtimes`` starts empty) and re-embeds it
  with provider B via ``index_file``.

Net effect: reindex is AUTOMATIC on the next sync — no ``memory index
--force`` required — because the sync manager always rebuilds the index from
the canonical Markdown on disk. This test proves provider B actually did the
re-embedding (its ``embed_batch`` is called; stored vectors carry B's dims)
and that the migrated content stays searchable across the switch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentos.memory.store import LongTermMemoryStore
from agentos.memory.sync_manager import MemorySyncManager
from agentos.memory.types import MemorySource

# Distinctive tokens so FTS matches are unambiguous and can only come from the
# seeded migrated Markdown, never from incidental fixture text.
_IMPORTED_TOKEN = "cerulean_migrated_marker"
_MEMORY_TOKEN = "vermilion_root_marker"


class _FakeEmbeddingProvider:
    """Deterministic offline embedding provider with a pinned fingerprint/dims.

    Records every ``embed_batch`` / ``embed_query`` call so a test can prove
    which provider actually performed the (re)embedding. Vectors are a fixed
    length (``dims``) with a value derived from the text hash, so distinct
    chunks get distinct—but reproducible—vectors without any model download.
    """

    def __init__(self, *, fingerprint: str, dims: int, model: str) -> None:
        self._provider_fingerprint = fingerprint
        self._vector_dims = dims
        self._model = model
        self.batch_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def provider_id(self) -> str:
        return "local"

    @property
    def model(self) -> str:
        return self._model

    def _vector_for(self, text: str) -> list[float]:
        seed = int.from_bytes(text.encode("utf-8", "replace")[:4].ljust(4, b"\0"), "big")
        base = (seed % 997) / 997.0 or 0.5
        return [base + (i * 1e-3) for i in range(self._vector_dims)]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector_for(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(list(texts))
        return [self._vector_for(text) for text in texts]

    async def probe(self) -> tuple[bool, str | None]:
        return True, None


def _seed_hermes_migrated_memory(tmp_path: Path) -> tuple[Path, Path]:
    """Write Markdown that simulates ``agentos migrate`` output from Hermes."""
    workspace = tmp_path / "workspace"
    memory_dir = workspace / "memory"
    memory_dir.mkdir(parents=True)
    (workspace / "MEMORY.md").write_text(
        f"# Memory root\n\nThe project index keyword is {_MEMORY_TOKEN}.\n",
        encoding="utf-8",
    )
    (memory_dir / "imported-note.md").write_text(
        "# Imported from Hermes\n\n"
        f"This migrated note preserves the {_IMPORTED_TOKEN} decision "
        "and should survive an embedding model switch.\n",
        encoding="utf-8",
    )
    return workspace, memory_dir


async def _embedding_dims_in_store(store: LongTermMemoryStore) -> set[int]:
    """Return the distinct vector lengths stored in ``chunks.embedding``."""
    assert store._db is not None
    dims: set[int] = set()
    async with store._db.execute(
        "SELECT embedding FROM chunks WHERE embedding IS NOT NULL"
    ) as cur:
        for (embedding_json,) in await cur.fetchall():
            vector = json.loads(embedding_json)
            dims.add(len(vector))
    return dims


async def _chunks_with_embeddings(store: LongTermMemoryStore) -> int:
    assert store._db is not None
    async with store._db.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
    ) as cur:
        row = await cur.fetchone()
    return row[0] if row else 0


@pytest.mark.asyncio
async def test_migrated_memory_reindexes_across_embedding_model_switch(
    tmp_path: Path,
) -> None:
    workspace, memory_dir = _seed_hermes_migrated_memory(tmp_path)
    db_path = tmp_path / "memory.db"

    # --- Run 1: provider A (simulating bundled BGE), dims DA -------------------
    provider_a = _FakeEmbeddingProvider(
        fingerprint="fp-bge",
        dims=384,
        model="BAAI/bge-small-zh-v1.5",
    )
    store_a = LongTermMemoryStore(str(db_path), embedding_provider=provider_a)
    await store_a.initialize()
    try:
        sync_a = MemorySyncManager(
            store=store_a, workspace_dir=workspace, memory_dir=memory_dir
        )
        await sync_a.sync(reason="manual")

        # Migrated content indexed and searchable under provider A.
        imported_results, _mode = await store_a.search(_IMPORTED_TOKEN, min_score=0.0)
        memory_results, _mode = await store_a.search(_MEMORY_TOKEN, min_score=0.0)
        assert any(r.path == "memory/imported-note.md" for r in imported_results)
        assert any(r.path == "MEMORY.md" for r in memory_results)

        assert provider_a.batch_calls, "provider A should have embedded the chunks"
        assert await _embedding_dims_in_store(store_a) == {384}
        chunks_after_a = await _chunks_with_embeddings(store_a)
        assert chunks_after_a > 0
    finally:
        await store_a.close()

    # --- Run 2: provider B (simulating downloaded EmbeddingGemma), dims DB -----
    # Same db path, different fingerprint AND different dims -> the switch.
    provider_b = _FakeEmbeddingProvider(
        fingerprint="fp-gemma",
        dims=768,
        model="google/embeddinggemma-300m",
    )
    store_b = LongTermMemoryStore(str(db_path), embedding_provider=provider_b)
    # initialize() runs _check_meta_and_reindex() which detects the mismatch
    # and clears the stale index; it must NOT re-embed on its own.
    await store_b.initialize()
    try:
        assert store_b._dirty is True, "fingerprint change should mark store dirty"
        assert provider_b.batch_calls == [], (
            "initialize() must clear the stale index without re-embedding"
        )
        assert await _chunks_with_embeddings(store_b) == 0, (
            "stale vectors under old dims must be cleared, not left searchable"
        )

        # The next sync re-reads the on-disk Markdown and re-embeds with B.
        sync_b = MemorySyncManager(
            store=store_b, workspace_dir=workspace, memory_dir=memory_dir
        )
        await sync_b.sync(reason="manual")

        # Re-embedding actually happened via provider B, under the NEW dims.
        assert provider_b.batch_calls, "provider B must re-embed on the next sync"
        embedded_texts = [text for call in provider_b.batch_calls for text in call]
        assert any(_IMPORTED_TOKEN in text for text in embedded_texts)
        assert any(_MEMORY_TOKEN in text for text in embedded_texts)
        assert await _embedding_dims_in_store(store_b) == {768}, (
            "reindexed vectors must carry provider B's dimensionality"
        )

        # Migrated content is still searchable after the switch.
        imported_results, _mode = await store_b.search(_IMPORTED_TOKEN, min_score=0.0)
        memory_results, _mode = await store_b.search(_MEMORY_TOKEN, min_score=0.0)
        assert any(r.path == "memory/imported-note.md" for r in imported_results)
        assert any(r.path == "MEMORY.md" for r in memory_results)

        # And the source accounting reflects the migrated Markdown, not stale rows.
        counts = await store_b.source_counts()
        assert counts[MemorySource.memory.value]["files"] == 2
    finally:
        await store_b.close()
