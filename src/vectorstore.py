import chromadb
import os
import sys
from typing import List, Optional

# ── DB path resolution ────────────────────────────────────────────────────────

def _resolve_db_path() -> str:
    try:
        from src.config import DB_PATH
        return DB_PATH
    except ImportError:
        pass
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "..", "data", "chroma_db")


_DB_PATH = _resolve_db_path()
client = chromadb.PersistentClient(path=_DB_PATH)
collection = client.get_or_create_collection(name="industry_docs")


# ── Write ─────────────────────────────────────────────────────────────────────

def add_chunks(chunks: List[str], embeddings, source: str = "unknown"):
    ids = [f"{source}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source} for _ in chunks]
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas,
    )


# ── Read ──────────────────────────────────────────────────────────────────────

def query(
    question_embedding,
    n_results: int = 3,
    where: Optional[dict] = None,
) -> List[str]:
    """Query the vector store. Optionally filter by source metadata."""
    kwargs = dict(
        query_embeddings=[question_embedding.tolist()],
        n_results=n_results,
    )
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)
    return results["documents"][0]


def list_sources() -> List[str]:
    """Return sorted list of unique source names stored in ChromaDB."""
    try:
        result = collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in result.get("metadatas", [])}
        return sorted(sources)
    except Exception:
        return []


def get_chunk_count() -> int:
    """Total number of chunks stored."""
    try:
        return collection.count()
    except Exception:
        return 0


# ── Delete ────────────────────────────────────────────────────────────────────

def delete_source(source_name: str) -> int:
    """Delete all chunks belonging to source_name. Returns number of deleted chunks."""
    try:
        result = collection.get(where={"source": source_name}, include=["metadatas"])
        ids = result.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0