"""Process-level singleton accessor for the local embedding provider.

The accessor lives outside memory/ because it is owned by the skill
retrieval subsystem; memory/embedding.py is the Protocol implementation.
Sharing a single embedder across skill filter and agentos_router avoids
loading the BGE weights twice.
"""

from __future__ import annotations

import threading

from agentos.memory.embedding import LocalEmbeddingProvider

_lock = threading.Lock()
_instances: dict[str, LocalEmbeddingProvider] = {}


def get_embedder(model_name: str | None = None) -> LocalEmbeddingProvider:
    """Return a process-wide LocalEmbeddingProvider keyed by model name.

    Lazy-constructs on first call per model. The underlying ONNX session
    is loaded by LocalEmbeddingProvider on first encode, not here.
    Raises nothing on its own; if onnxruntime / tokenizers / the
    bundled ONNX dir are missing, the corresponding ImportError or
    RuntimeError is surfaced when the caller invokes encode_sync /
    embed_query / embed_batch.
    """
    if model_name is None:
        # Lazy import to avoid a cycle: embedding_resolver imports from the
        # memory embedding module, and this module is imported during skill
        # retrieval setup. Prefer downloaded EmbeddingGemma over bundled BGE.
        from agentos.memory.embedding_resolver import preferred_local_model

        name = preferred_local_model()
    else:
        name = model_name
    with _lock:
        inst = _instances.get(name)
        if inst is None:
            inst = LocalEmbeddingProvider(name)
            _instances[name] = inst
        return inst
