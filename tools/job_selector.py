"""
tools/job_selector.py

Selects one pending job from "AudioStatus" with status in [3..7], with a
pluggable priority order that you can customize now or later.

Key functions:
- build_order_by_clause(rules): Build an ORDER BY SQL snippet and params.
- peek_next_audio_job(pool, rules): Read-only select of the next job.
- claim_next_audio_job(pool, rules, next_status_map): Transactionally lock and
  update the job to its next status (e.g., 3->4, 5->6, 7->8).

Usage example (with .env absolute path):
  from tools.db_util import get_database_url_from_env_file, create_open_pool
  from tools.job_selector import PriorityRules, peek_next_audio_job

  url = get_database_url_from_env_file("/abs/path/.env")
  pool = await create_open_pool(url)
  rules = PriorityRules(status_order=[3,4,5,6,7], prefer_usercodes=[1001, 2002])
  job = await peek_next_audio_job(pool, rules)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from psycopg.rows import dict_row


@dataclass
class PriorityRules:
    """Describe how to order jobs when selecting one.

    - status_order: earlier items are higher priority (e.g., [3,4,5,6,7]).
    - prefer_usercodes: prioritize these usercodes in order (optional).
    - secondary_order: fallback order-by, default by primary key ascending.
    - custom_order_sql: full ORDER BY fragment to override everything.
    """

    status_order: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7])
    prefer_usercodes: List[int] = field(default_factory=list)
    secondary_order: str = '"no" ASC'
    custom_order_sql: Optional[str] = None


def build_order_by_clause(rules: PriorityRules) -> Tuple[str, List[Any]]:
    """Build an SQL ORDER BY clause and its parameters based on rules.

    Returns (order_sql, params). The caller can inject it into a query as:
      f"... WHERE ... ORDER BY {order_sql} LIMIT 1"
    and pass params to execute().
    """
    if rules.custom_order_sql:
        return rules.custom_order_sql, []

    parts: List[str] = []
    params: List[Any] = []

    # 1) CASE for status priority
    if rules.status_order:
        when_clauses = []
        for idx, st in enumerate(rules.status_order):
            when_clauses.append(f"WHEN status = %s THEN %s")
            params.extend([st, idx])
        status_case = "CASE " + " ".join(when_clauses) + " ELSE 999 END"
        parts.append(status_case)

    # 2) CASE for preferred usercodes priority
    if rules.prefer_usercodes:
        when_user = []
        for idx, uc in enumerate(rules.prefer_usercodes):
            when_user.append("WHEN usercode = %s THEN %s")
            params.extend([uc, idx])
        user_case = "CASE " + " ".join(when_user) + " ELSE 999 END"
        parts.append(user_case)

    # 3) Secondary/fallback deterministic order
    parts.append(rules.secondary_order)

    order_sql = ", ".join(parts)
    return order_sql, params


async def peek_next_audio_job(pool, rules: PriorityRules) -> Optional[Dict[str, Any]]:
    """Return the next job by priority (read-only, no locks/updates).

    Returns a dict row with keys: no, status, filename, usercode (if available).
    Returns None if no matching job.
    """
    order_sql, params = build_order_by_clause(rules)
    sql = (
        'SELECT "no", status, filename, usercode '
        'FROM "AudioStatus" '
        'WHERE status BETWEEN 3 AND 7 '
        f'ORDER BY {order_sql} '
        'LIMIT 1'
    )
    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            row = await cur.fetchone()
            return row  # type: ignore[return-value]


async def claim_next_audio_job(
    pool,
    rules: PriorityRules,
    next_status_map: Optional[Dict[int, int]] = None,
) -> Optional[Dict[str, Any]]:
    """Transactionally select-and-mark one job using FOR UPDATE SKIP LOCKED.

    - next_status_map maps current status -> next status (e.g., {3:4, 5:6, 7:8}).
    - If map is provided and a job is found, updates the status inside the
      transaction, then commits and returns the selected row (with original
      fields). If map is None, it just locks and returns the row.
    """
    if next_status_map is None:
        next_status_map = {}
    order_sql, params = build_order_by_clause(rules)
    select_sql = (
        'SELECT "no", status, filename, usercode '
        'FROM "AudioStatus" '
        'WHERE status BETWEEN 3 AND 7 '
        'ORDER BY ' + order_sql + ' '
        'LIMIT 1 '
        'FOR UPDATE SKIP LOCKED'
    )
    async with pool.connection() as conn:
        async with conn.transaction():
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(select_sql, params)
                row = await cur.fetchone()
                if not row:
                    return None
                # Optional status advance (e.g., 3->4, 5->6, 7->8)
                cur_status = int(row.get("status"))
                if cur_status in next_status_map:
                    nxt = int(next_status_map[cur_status])
                    await cur.execute(
                        'UPDATE "AudioStatus" SET status = %s WHERE "no" = %s',
                        (nxt, int(row["no"]))
                    )
                return row

