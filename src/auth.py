"""
Authentication and User Management.
Supports both .env bootstrap accounts (admin/engineer) and self-registration stored in SQLite.
Uses built-in hashlib for secure credential storage with zero extra dependencies.
"""

import hashlib
import sqlite3
from typing import Dict, Optional

from src.config import SESSION_DB, USERS


def _hash_password(password: str) -> str:
    """Return SHA-256 hash of password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _get_db():
    conn = sqlite3.connect(SESSION_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _init_users_table():
    """Ensure the users table exists and seed initial .env accounts if empty."""
    with _get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username   TEXT PRIMARY KEY,
                password   TEXT NOT NULL,
                role       TEXT NOT NULL DEFAULT 'Engineer',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        # Seed default users from .env into DB if not present
        active_env_users = USERS if USERS else {"admin": "admin123", "engineer": "eng456"}
        for uname, pwd in active_env_users.items():
            role = "Administrator" if "admin" in uname.lower() else "Engineer"
            conn.execute(
                "INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
                (uname.strip(), _hash_password(pwd.strip()), role),
            )
        conn.commit()


# Run table init on import
try:
    _init_users_table()
except Exception:
    pass


def register_user(username: str, password: str, role: str = "Engineer") -> tuple[bool, str]:
    """
    Register a new user account in SQLite.
    Returns (success: bool, message: str).
    """
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    _init_users_table()
    try:
        with _get_db() as conn:
            # Check if user already exists
            existing = conn.execute("SELECT username FROM users WHERE LOWER(username) = LOWER(?)", (username,)).fetchone()
            if existing:
                return False, f"Username '{username}' already exists. Please choose another."

            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, _hash_password(password), role),
            )
            conn.commit()
        return True, f"Account '{username}' created successfully! You can now sign in."
    except Exception as e:
        return False, f"Registration failed: {str(e)}"


def check_credentials(username: str, password: str) -> bool:
    """Check credentials against SQLite database and .env fallback."""
    username = username.strip()
    if not username or not password:
        return False

    _init_users_table()
    try:
        with _get_db() as conn:
            user = conn.execute(
                "SELECT password FROM users WHERE LOWER(username) = LOWER(?)",
                (username,),
            ).fetchone()
            if user:
                # Compare SHA-256 hash
                return user["password"] == _hash_password(password)
    except Exception:
        pass

    # Fallback check against .env in case DB was not yet accessible
    active_env_users = USERS if USERS else {"admin": "admin123", "engineer": "eng456"}
    for u, p in active_env_users.items():
        if u.lower() == username.lower() and p == password:
            return True

    return False


def get_user_role(username: str) -> str:
    """Get the assigned role for a user (Administrator or Engineer)."""
    try:
        with _get_db() as conn:
            row = conn.execute("SELECT role FROM users WHERE LOWER(username) = LOWER(?)", (username.strip(),)).fetchone()
            if row:
                return row["role"]
    except Exception:
        pass
    return "Administrator" if "admin" in username.lower() else "Engineer"
