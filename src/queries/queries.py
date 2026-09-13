from __future__ import annotations

from datetime import datetime
from typing import Optional
import psycopg2
from helpers.config import get_settings
from psycopg2.extras import RealDictRow
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL PRIMARY KEY,
    google_id     TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    access_token  TEXT NOT NULL,
    refresh_token TEXT,
    token_expiry  TIMESTAMPTZ,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""
settings = get_settings()
DB_ ={
        "host": settings.POSTGRES_HOST,
        "port": settings.POSTGRES_PORT,
        "dbname": settings.POSTGRES_DB,
        "user": settings.POSTGRES_USER,
        "password": settings.POSTGRES_PASSWORD,
    }
def get_connection():
    return psycopg2.connect(**DB_,cursor_factory=RealDictCursor)

def create_data_base():
    with get_connection() as conn:
        with conn.cursor() as cur:
           cur.execute(
               "SELECT 1 FROM pg_database WHERE datname=%s",
                (settings.POSTGRES_DB,)
            )
           exists = cur.fetchone()
           if not exists:
                cur.execute(
                sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(settings.POSTGRES_DB)
                )
            )
                print(f"Database {settings.POSTGRES_DB} created")
           else:
                print(f"Database {settings.POSTGRES_DB} already exists")          

    pass
def create_users_table() -> None:
    """Create the `users` table if it does not already exist."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_USERS_TABLE)
        

def user_exists(email: str) -> bool:
    """Return True if a user with the given email already exists."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM users WHERE email = %s)", (email,))
            row = cur.fetchone()
    return bool(row["exists"]) if row else False


def find_user_by_email(email: str) -> Optional[RealDictRow]:
    """Return the user row for `email`, or None if not found."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            return cur.fetchone()


def find_user_by_google_id(google_id: str) -> Optional[RealDictRow]:
    """Return the user row for `google_id`, or None if not found."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE google_id = %s", (google_id,))
            return cur.fetchone()


def create_user(
    *,
    google_id: str,
    name: str,
    email: str,
    access_token: str,
    refresh_token: Optional[str],
    token_expiry: Optional[datetime],
) -> RealDictRow:
    """Insert a new user and return the created row."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users
                    (google_id, name, email, access_token, refresh_token, token_expiry)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (google_id, name, email, access_token, refresh_token, token_expiry),
            )
            return cur.fetchone()


def update_user_tokens(
    user_id: int,
    *,
    access_token: str,
    refresh_token: Optional[str],
    token_expiry: Optional[datetime],
) -> RealDictRow:
    """
    Refresh a user's OAuth credentials. A fresh `refresh_token` from Google
    replaces the stored one; otherwise the existing value is preserved.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET access_token = %s,
                    refresh_token = COALESCE(%s, refresh_token),
                    token_expiry = %s,
                    updated_at = NOW()
                WHERE id = %s
                RETURNING *
                """,
                (access_token, refresh_token, token_expiry, user_id),
            )
            return cur.fetchone()


def update_user_profile(user_id: int, *, name: str) -> RealDictRow:
    """Update mutable profile fields (name) for a user."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET name = %s, updated_at = NOW()
                WHERE id = %s
                RETURNING *
                """,
                (name, user_id),
            )
            return cur.fetchone()
