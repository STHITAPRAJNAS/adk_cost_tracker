"""
Storage backends for LLM usage records.

SQLiteStore   — stdlib only, zero extra deps.  Good for local dev.
PostgresStore — async via asyncpg.  Shares the same Postgres instance
                (but NOT the same database) as ADK's DatabaseSessionService.
                Creates table `llm_usage_log` with app_name column so you
                can GROUP BY / filter per app without needing a JOIN.

Factory:
    from adk_cost_tracker.store import make_store

    store = make_store()                                      # SQLite default
    store = make_store("postgresql://user:pw@host:5432/db")  # Postgres
    store = make_store(db_path=Path("/tmp/costs.db"))         # SQLite custom path
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SQLITE_DEFAULT = Path.home() / ".adk_cost_tracker" / "usage.db"

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS llm_usage_log (
    id            INTEGER  PRIMARY KEY AUTOINCREMENT,
    ts            TEXT     NOT NULL,
    provider      TEXT     NOT NULL DEFAULT '',
    agent         TEXT     NOT NULL DEFAULT '',
    model         TEXT     NOT NULL DEFAULT '',
    session_id    TEXT     NOT NULL DEFAULT '',
    app_name      TEXT     NOT NULL DEFAULT '',
    input_tokens  INTEGER  NOT NULL DEFAULT 0,
    output_tokens INTEGER  NOT NULL DEFAULT 0,
    cached_tokens INTEGER  NOT NULL DEFAULT 0,
    cost_usd      REAL     NOT NULL DEFAULT 0,
    tags          TEXT     NOT NULL DEFAULT '',
    meta          TEXT     NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS llm_pricing (
    model_key           TEXT  PRIMARY KEY,
    provider            TEXT  NOT NULL,
    input_per_m         REAL  NOT NULL,
    output_per_m        REAL  NOT NULL,
    cached_input_per_m  REAL  NOT NULL DEFAULT 0,
    updated_at          TEXT  NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""
_SQLITE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS _ct_ts  ON llm_usage_log(ts)",
    "CREATE INDEX IF NOT EXISTS _ct_app ON llm_usage_log(app_name)",
    "CREATE INDEX IF NOT EXISTS _ct_prv ON llm_usage_log(provider)",
]

_PG_DDL = """
CREATE TABLE IF NOT EXISTS llm_usage_log (
    id            BIGSERIAL    PRIMARY KEY,
    ts            TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    provider      TEXT         NOT NULL DEFAULT '',
    agent         TEXT         NOT NULL DEFAULT '',
    model         TEXT         NOT NULL DEFAULT '',
    session_id    TEXT         NOT NULL DEFAULT '',
    app_name      TEXT         NOT NULL DEFAULT '',
    input_tokens  INTEGER      NOT NULL DEFAULT 0,
    output_tokens INTEGER      NOT NULL DEFAULT 0,
    cached_tokens INTEGER      NOT NULL DEFAULT 0,
    cost_usd      NUMERIC(14,8)NOT NULL DEFAULT 0,
    tags          TEXT         NOT NULL DEFAULT '',
    meta          JSONB        NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS llm_pricing (
    model_key           TEXT           PRIMARY KEY,
    provider            TEXT           NOT NULL,
    input_per_m         NUMERIC(14,8)  NOT NULL,
    output_per_m        NUMERIC(14,8)  NOT NULL,
    cached_input_per_m  NUMERIC(14,8)  NOT NULL DEFAULT 0,
    updated_at          TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);
"""
_PG_INDEXES = [
    "CREATE INDEX IF NOT EXISTS _ct_ts  ON llm_usage_log(ts)",
    "CREATE INDEX IF NOT EXISTS _ct_app ON llm_usage_log(app_name)",
    "CREATE INDEX IF NOT EXISTS _ct_prv ON llm_usage_log(provider)",
    "CREATE INDEX IF NOT EXISTS _ct_agt ON llm_usage_log(agent)",
]


@dataclass
class CallRecord:
    ts: str
    provider: str       # "gemini" | "openai" | "bedrock" | custom
    agent: str          # ADK agent name or custom identifier
    model: str          # exact model string as returned by provider
    session_id: str     # ADK session id (empty for non-ADK calls)
    app_name: str       # ADK app_name — groups records per application
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    tags: str           # "key=value,key=value" static labels
    meta: dict          # free-form extras (reasoning_tokens, stop_reason, …)


# ── Abstract interface ─────────────────────────────────────────────────────

class BaseStore(ABC):
    @abstractmethod
    async def insert(self, record: CallRecord) -> None: ...

    @abstractmethod
    async def summary(self, since: str | None = None) -> list[dict]: ...

    @abstractmethod
    async def totals(self, since: str | None = None) -> dict: ...

    @abstractmethod
    async def recent(self, n: int = 20) -> list[dict]: ...

    @abstractmethod
    async def get_all_prices(self) -> dict[str, dict]: ...

    @abstractmethod
    async def update_price(
        self,
        model_key: str,
        provider: str,
        input_per_m: float,
        output_per_m: float,
        cached_input_per_m: float = 0.0,
    ) -> None: ...

    async def health(self) -> bool:
        """Return True if the store is reachable."""
        try:
            await self.totals()
            return True
        except Exception:
            return False


# ── PostgreSQL ─────────────────────────────────────────────────────────────

class PostgresStore(BaseStore):
    """
    Async Postgres store backed by an asyncpg connection pool.

    Args:
        dsn:       Connection string.  SQLAlchemy-style prefixes are stripped.
        min_size:  Minimum pool connections (default 2).
        max_size:  Maximum pool connections (default 10).
        max_retries: How many times to retry a failed insert (default 3).
    """

    def __init__(
        self,
        dsn: str,
        min_size: int = 2,
        max_size: int = 10,
        max_retries: int = 3,
    ) -> None:
        # Strip SQLAlchemy driver prefixes — asyncpg wants plain postgresql://
        self._dsn = (
            dsn.replace("postgresql+asyncpg://", "postgresql://")
               .replace("postgresql+psycopg2://", "postgresql://")
        )
        self._min = min_size
        self._max = max_size
        self._max_retries = max_retries
        self._pool: Any = None
        self._init_lock = asyncio.Lock()

    # ── Pool management ────────────────────────────────────────────────────

    async def _get_pool(self):
        if self._pool is not None:
            return self._pool
        async with self._init_lock:
            if self._pool is not None:
                return self._pool
            try:
                import asyncpg  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "asyncpg is required for PostgresStore. "
                    "Install: pip install 'adk-cost-tracker[postgres]'"
                ) from exc
            self._pool = await asyncpg.create_pool(
                self._dsn, min_size=self._min, max_size=self._max
            )
            await self._setup(self._pool)
        return self._pool

    async def _setup(self, pool) -> None:
        async with pool.acquire() as conn:
            await conn.execute(_PG_DDL)
            for idx in _PG_INDEXES:
                await conn.execute(idx)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ── Write ──────────────────────────────────────────────────────────────

    async def insert(self, record: CallRecord) -> None:
        pool = await self._get_pool()
        sql = """
            INSERT INTO llm_usage_log
                (ts,provider,agent,model,session_id,app_name,
                 input_tokens,output_tokens,cached_tokens,cost_usd,tags,meta)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                async with pool.acquire() as conn:
                    await conn.execute(
                        sql,
                        record.ts, record.provider, record.agent, record.model,
                        record.session_id, record.app_name,
                        record.input_tokens, record.output_tokens,
                        record.cached_tokens, record.cost_usd,
                        record.tags, json.dumps(record.meta),
                    )
                return
            except Exception as exc:
                if attempt == self._max_retries:
                    logger.error(
                        "[CostTracker/Postgres] insert failed after %d attempts: %s",
                        self._max_retries, exc
                    )
                    raise
                wait = 0.1 * (2 ** attempt)   # 0.2s, 0.4s, 0.8s …
                logger.warning(
                    "[CostTracker/Postgres] insert attempt %d failed, retrying in %.1fs: %s",
                    attempt, wait, exc
                )
                await asyncio.sleep(wait)

    # ── Read ───────────────────────────────────────────────────────────────

    async def summary(self, since: str | None = None) -> list[dict]:
        where = f"WHERE ts >= '{since}'" if since else ""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT provider, agent, model,
                       COUNT(*)          AS calls,
                       SUM(input_tokens) AS total_input,
                       SUM(output_tokens)AS total_output,
                       SUM(cost_usd)     AS total_cost
                FROM llm_usage_log {where}
                GROUP BY provider, agent, model
                ORDER BY total_cost DESC
            """)
        return [dict(r) for r in rows]

    async def totals(self, since: str | None = None) -> dict:
        where = f"WHERE ts >= '{since}'" if since else ""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT COUNT(*)          AS calls,
                       SUM(input_tokens) AS total_input,
                       SUM(output_tokens)AS total_output,
                       SUM(cost_usd)     AS total_cost
                FROM llm_usage_log {where}
            """)
        return dict(row) if row else {}

    async def recent(self, n: int = 20) -> list[dict]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM llm_usage_log ORDER BY id DESC LIMIT $1", n
            )
        return [dict(r) for r in rows]

    # ── Pricing ────────────────────────────────────────────────────────────

    async def get_all_prices(self) -> dict[str, dict]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM llm_pricing")
        return {r["model_key"]: dict(r) for r in rows}

    async def update_price(
        self,
        model_key: str,
        provider: str,
        input_per_m: float,
        output_per_m: float,
        cached_input_per_m: float = 0.0,
    ) -> None:
        pool = await self._get_pool()
        sql = """
            INSERT INTO llm_pricing
                (model_key, provider, input_per_m, output_per_m, cached_input_per_m, updated_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (model_key) DO UPDATE SET
                provider = EXCLUDED.provider,
                input_per_m = EXCLUDED.input_per_m,
                output_per_m = EXCLUDED.output_per_m,
                cached_input_per_m = EXCLUDED.cached_input_per_m,
                updated_at = NOW()
        """
        async with pool.acquire() as conn:
            await conn.execute(
                sql, model_key, provider, input_per_m, output_per_m, cached_input_per_m
            )

    async def health(self) -> bool:
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False


# ── SQLite ─────────────────────────────────────────────────────────────────

class SQLiteStore(BaseStore):
    """
    Sync SQLite store wrapped for async interface via run_in_executor.
    Zero extra dependencies — stdlib only.
    """

    def __init__(self, db_path: Path = SQLITE_DEFAULT) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._conn() as conn:
            # executescript allows multiple statements (including CREATE TABLEs)
            conn.executescript(_SQLITE_DDL)
            for idx in _SQLITE_INDEXES:
                conn.execute(idx)

    def _insert_sync(self, record: CallRecord) -> None:
        sql = """
            INSERT INTO llm_usage_log
                (ts,provider,agent,model,session_id,app_name,
                 input_tokens,output_tokens,cached_tokens,cost_usd,tags,meta)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._lock, self._conn() as conn:
            conn.execute(sql, (
                record.ts, record.provider, record.agent, record.model,
                record.session_id, record.app_name,
                record.input_tokens, record.output_tokens,
                record.cached_tokens, record.cost_usd,
                record.tags, json.dumps(record.meta),
            ))

    def _query_sync(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    async def _run(self, fn, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)

    async def insert(self, record: CallRecord) -> None:
        await self._run(self._insert_sync, record)

    async def summary(self, since: str | None = None) -> list[dict]:
        where, params = ("WHERE ts >= ?", (since,)) if since else ("", ())
        sql = f"""
            SELECT provider, agent, model,
                   COUNT(*)           AS calls,
                   SUM(input_tokens)  AS total_input,
                   SUM(output_tokens) AS total_output,
                   SUM(cost_usd)      AS total_cost
            FROM llm_usage_log {where}
            GROUP BY provider, agent, model
            ORDER BY total_cost DESC
        """
        return await self._run(self._query_sync, sql, params)

    async def totals(self, since: str | None = None) -> dict:
        where, params = ("WHERE ts >= ?", (since,)) if since else ("", ())
        sql = f"""
            SELECT COUNT(*)           AS calls,
                   SUM(input_tokens)  AS total_input,
                   SUM(output_tokens) AS total_output,
                   SUM(cost_usd)      AS total_cost
            FROM llm_usage_log {where}
        """
        rows = await self._run(self._query_sync, sql, params)
        return rows[0] if rows else {}

    async def recent(self, n: int = 20) -> list[dict]:
        sql = "SELECT * FROM llm_usage_log ORDER BY id DESC LIMIT ?"
        return await self._run(self._query_sync, sql, (n,))

    # ── Pricing ────────────────────────────────────────────────────────────

    async def get_all_prices(self) -> dict[str, dict]:
        sql = "SELECT * FROM llm_pricing"
        rows = await self._run(self._query_sync, sql)
        return {r["model_key"]: dict(r) for r in rows}

    def _update_price_sync(
        self,
        model_key: str,
        provider: str,
        input_per_m: float,
        output_per_m: float,
        cached_input_per_m: float,
    ) -> None:
        sql = """
            INSERT INTO llm_pricing
                (model_key, provider, input_per_m, output_per_m, cached_input_per_m)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (model_key) DO UPDATE SET
                provider = excluded.provider,
                input_per_m = excluded.input_per_m,
                output_per_m = excluded.output_per_m,
                cached_input_per_m = excluded.cached_input_per_m,
                updated_at = CURRENT_TIMESTAMP
        """
        with self._lock, self._conn() as conn:
            conn.execute(sql, (
                model_key, provider, input_per_m, output_per_m, cached_input_per_m
            ))

    async def update_price(
        self,
        model_key: str,
        provider: str,
        input_per_m: float,
        output_per_m: float,
        cached_input_per_m: float = 0.0,
    ) -> None:
        await self._run(
            self._update_price_sync,
            model_key, provider, input_per_m, output_per_m, cached_input_per_m
        )


# ── Factory ────────────────────────────────────────────────────────────────

def make_store(dsn: str | None = None, **kwargs) -> BaseStore:
    """
    Create the right store from a DSN or keyword args.

      make_store()                                   → SQLiteStore at default path
      make_store("postgresql://user:pw@host/db")     → PostgresStore
      make_store("postgresql+asyncpg://user:pw@h/d") → PostgresStore (strips prefix)
      make_store(db_path=Path("/tmp/dev.db"))        → SQLiteStore at custom path

    Pass the same DSN you use for ADK's DatabaseSessionService if you want
    Postgres, but consider a dedicated database for cross-app cost visibility.
    """
    if dsn and ("postgresql" in dsn or "postgres" in dsn):
        return PostgresStore(dsn, **kwargs)
    db_path = kwargs.get("db_path")
    return SQLiteStore(db_path) if db_path else SQLiteStore()
