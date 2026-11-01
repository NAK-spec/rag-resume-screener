"""Simple user accounts: sign up / log in with salted, hashed passwords (SQLite)."""
import hashlib
import hmac
import os
import re
import sqlite3
from contextlib import closing

from .config import ROOT

DB_PATH = ROOT / "users.db"
ITERATIONS = 200_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")
MIN_PASSWORD = 8


def _connect(db_path=None):
    return sqlite3.connect(str(db_path or DB_PATH))


def init_db(db_path=None):
    with closing(_connect(db_path)) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "username TEXT PRIMARY KEY COLLATE NOCASE, "
            "salt BLOB NOT NULL, pw_hash BLOB NOT NULL, "
            "created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
        con.commit()


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)


def create_user(username: str, password: str, db_path=None):
    """Returns (ok, message)."""
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        return False, "Username must be 3-30 characters: letters, numbers or underscore."
    if len(password or "") < MIN_PASSWORD:
        return False, f"Password must be at least {MIN_PASSWORD} characters."
    salt = os.urandom(16)
    try:
        with closing(_connect(db_path)) as con:
            con.execute("INSERT INTO users (username, salt, pw_hash) VALUES (?, ?, ?)",
                        (username, salt, _hash(password, salt)))
            con.commit()
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created. You can log in now."


def verify_user(username: str, password: str, db_path=None):
    """Returns the stored username if the credentials are right, else None."""
    username = (username or "").strip()
    with closing(_connect(db_path)) as con:
        row = con.execute("SELECT username, salt, pw_hash FROM users WHERE username = ?",
                          (username,)).fetchone()
    if row is None:
        _hash(password or "", b"\0" * 16)  # keep timing similar for unknown users
        return None
    name, salt, stored = row
    if hmac.compare_digest(_hash(password or "", salt), stored):
        return name
    return None
