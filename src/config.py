"""
Central configuration — reads from .env file (or environment variables).
All other modules import from here instead of using os.getenv directly.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; fall back to env vars / defaults

# ── LLM ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemma2:2b")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ── RAG ──────────────────────────────────────────────────────────────────────
N_RESULTS: int = int(os.getenv("N_RESULTS", "3"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_FOLDER: Path = Path(os.getenv("PDF_FOLDER", "data/raw_pdfs"))
DB_PATH: str = os.getenv("DB_PATH", "data/chroma_db")
AUDIT_FILE: Path = Path(os.getenv("AUDIT_FILE", "data/audit_log.jsonl"))
SESSION_DB: str = os.getenv("SESSION_DB", "data/sessions.db")

# ── Auth ──────────────────────────────────────────────────────────────────────
# Format in .env:  USERS=admin:admin123,engineer:pass456
_users_raw: str = os.getenv("USERS", "admin:admin123,engineer:eng456")
USERS: dict[str, str] = {}
for _entry in _users_raw.split(","):
    _parts = _entry.strip().split(":", 1)
    if len(_parts) == 2:
        USERS[_parts[0].strip()] = _parts[1].strip()

# Ensure required directories exist
PDF_FOLDER.mkdir(parents=True, exist_ok=True)
Path(DB_PATH).mkdir(parents=True, exist_ok=True)
AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
Path(SESSION_DB).parent.mkdir(parents=True, exist_ok=True)
