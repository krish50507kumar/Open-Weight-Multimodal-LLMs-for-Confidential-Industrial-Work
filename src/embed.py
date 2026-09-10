"""
Embeddings module for SIH 26117 Workbench.
Lazy-loads the SentenceTransformer model on first actual query or PDF ingestion.
The login and registration screens start up instantly without downloading or loading models into RAM.
"""

from typing import Any, List

_model = None


def get_embed_model():
    """Return the SentenceTransformer model, loading/downloading it on first use."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


class _LazyModelWrapper:
    """Wrapper that defers model loading until an attribute or encode() is called."""

    def encode(self, *args, **kwargs):
        return get_embed_model().encode(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(get_embed_model(), name)


# Backwards-compatible model object (e.g. embed_model.encode(...))
model = _LazyModelWrapper()


def embed_chunks(chunks: List[str]):
    """Embed chunks of text using the sentence-transformers model."""
    return get_embed_model().encode(chunks)