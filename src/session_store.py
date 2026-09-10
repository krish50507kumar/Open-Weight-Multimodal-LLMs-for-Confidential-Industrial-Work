"""
SQLite-backed persistent chat session storage.
Each session stores messages that survive page refreshes.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict

from src.config import SESSION_DB


# ── DB initialisation ─────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(SESSION_DB)
    c.row_factory = sqlite3.Row
    return c


def _init():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                username    TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                sources     TEXT NOT NULL DEFAULT '[]',
                created_at  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        c.execute("PRAGMA foreign_keys = ON")
        c.commit()


_init()


# ── Public API ────────────────────────────────────────────────────────────────

def create_session(name: str, username: str = "") -> str:
    """Create a new session and return its ID."""
    sid = str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (id, name, username, created_at) VALUES (?,?,?,?)",
            (sid, name, username, datetime.now().isoformat()),
        )
        c.commit()
    return sid


def get_sessions(username: str = "") -> List[Dict]:
    """Return sessions for a user, newest first."""
    with _conn() as c:
        if username:
            rows = c.execute(
                "SELECT * FROM sessions WHERE username=? ORDER BY created_at DESC",
                (username,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def save_message(session_id: str, role: str, content: str, sources: List[str] = None):
    """Append a message to a session."""
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?,?,?,?,?)",
            (session_id, role, content, json.dumps(sources or []), datetime.now().isoformat()),
        )
        c.commit()


def get_messages(session_id: str) -> List[Dict]:
    """Load all messages for a session, oldest first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d.get("sources", "[]"))
        out.append(d)
    return out


def delete_session(session_id: str):
    """Delete a session and all its messages."""
    with _conn() as c:
        c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        c.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        c.commit()


def rename_session(session_id: str, new_name: str):
    with _conn() as c:
        c.execute("UPDATE sessions SET name=? WHERE id=?", (new_name, session_id))
        c.commit()
