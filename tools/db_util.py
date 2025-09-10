"""
tools/db_util.py

Helpers to create an async Postgres connection pool using psycopg3, with
optional loading of a specific .env file you provide by absolute path.

Requires: pip install 'psycopg[binary,pool]' python-dotenv
"""
from __future__ import annotations

from typing import Optional
import os

try:
    from dotenv import load_dotenv, dotenv_values
except Exception as e:  # pragma: no cover
    raise SystemExit("Install python-dotenv: pip install python-dotenv")

try:
    # psycopg3 async pool
    from psycopg_pool import AsyncConnectionPool
except Exception as e:  # pragma: no cover
    raise SystemExit("Install psycopg: pip install 'psycopg[binary,pool]'")


def get_async_pool(url: str, max_size: int = 5, autocommit: bool = False) -> AsyncConnectionPool:
    """Create an AsyncConnectionPool WITHOUT opening it (avoids deprecation warning).

    Call `await pool.open()` before first use, or use `create_open_pool()`.
    """
    if not url:
        raise ValueError("Database URL is required")
    try:
        # Newer psycopg_pool supports "open=False" to avoid deprecated auto-open.
        return AsyncConnectionPool(url, max_size=max_size, kwargs={"autocommit": autocommit}, open=False)  # type: ignore[arg-type]
    except TypeError:
        # Fallback for older versions: constructor may auto-open and warn.
        return AsyncConnectionPool(url, max_size=max_size, kwargs={"autocommit": autocommit})


async def create_open_pool(url: str, max_size: int = 5, autocommit: bool = False) -> AsyncConnectionPool:
    """Create and open a pool (preferred in async code)."""
    pool = get_async_pool(url, max_size=max_size, autocommit=autocommit)
    try:
        await pool.open()
    except Exception:
        # If already open or unsupported, ignore.
        pass
    return pool


async def ping(pool: AsyncConnectionPool) -> bool:
    """Run a simple SELECT 1 to verify connectivity."""
    try:
        try:
            await pool.open()
        except Exception:
            pass
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                await cur.fetchone()
        return True
    except Exception:
        return False


# ----- Optional helpers for .env absolute path loading -----

def load_env_file(env_path: str, override: bool = True) -> None:
    """Load a specific .env file (absolute path recommended) into os.environ.

    Use when you want to control exactly which .env is read.
    """
    if not env_path:
        raise ValueError("env_path is required")
    load_dotenv(env_path, override=override)


# .env파일 절대 경로 찾는 함수
def get_abs_env_file(env_path: str) -> dict[str, str | None]:
    """Return DATABASE_URL from a specific .env file without mutating os.environ."""
    if not env_path:
        raise ValueError("env_path is required")
    values = dotenv_values(env_path)
    if not values:
        raise ValueError("env_path is not exist")
    return values


def get_async_pool_from_env_file(env_path: str, max_size: int = 5, autocommit: bool = False) -> AsyncConnectionPool:
    """Create a pool using DATABASE_URL loaded from the provided .env path."""
    url = get_abs_env_file(env_path)
    return get_async_pool(url, max_size=max_size, autocommit=autocommit)
