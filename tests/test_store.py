"""Tests for SQLiteStore (always runs) and PostgresStore (skipped if no DB)."""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from adk_cost_tracker.store import (
    CallRecord,
    SQLiteStore,
    PostgresStore,
    make_store,
)

# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_record(**overrides) -> CallRecord:
    base = dict(
        ts=datetime.now(timezone.utc).isoformat(),
        provider="gemini",
        agent="test_agent",
        model="gemini-2.5-flash",
        session_id="sess-abc123",
        app_name="test_app",
        input_tokens=1000,
        output_tokens=500,
        cached_tokens=100,
        cost_usd=0.000425,
        tags="env=test",
        meta={},
    )
    base.update(overrides)
    return CallRecord(**base)


@pytest.fixture
def sqlite_store(tmp_path) -> SQLiteStore:
    return SQLiteStore(db_path=tmp_path / "test_usage.db")


# ── SQLiteStore tests ──────────────────────────────────────────────────────

class TestSQLiteStore:
    @pytest.mark.asyncio
    async def test_insert_and_recent(self, sqlite_store):
        rec = _make_record()
        await sqlite_store.insert(rec)
        rows = await sqlite_store.recent(10)
        assert len(rows) == 1
        assert rows[0]["model"] == "gemini-2.5-flash"
        assert rows[0]["agent"] == "test_agent"

    @pytest.mark.asyncio
    async def test_multiple_inserts(self, sqlite_store):
        for i in range(5):
            await sqlite_store.insert(_make_record(agent=f"agent_{i}"))
        rows = await sqlite_store.recent(10)
        assert len(rows) == 5

    @pytest.mark.asyncio
    async def test_summary_groups_correctly(self, sqlite_store):
        await sqlite_store.insert(_make_record(provider="gemini", model="gemini-2.5-flash", cost_usd=0.001))
        await sqlite_store.insert(_make_record(provider="gemini", model="gemini-2.5-flash", cost_usd=0.002))
        await sqlite_store.insert(_make_record(provider="openai", model="gpt-4o", cost_usd=0.010))
        rows = await sqlite_store.summary()
        assert len(rows) == 2
        models = {r["model"] for r in rows}
        assert "gemini-2.5-flash" in models
        assert "gpt-4o" in models

    @pytest.mark.asyncio
    async def test_totals(self, sqlite_store):
        await sqlite_store.insert(_make_record(cost_usd=0.001, input_tokens=100, output_tokens=50))
        await sqlite_store.insert(_make_record(cost_usd=0.002, input_tokens=200, output_tokens=100))
        t = await sqlite_store.totals()
        assert t["calls"] == 2
        assert abs(t["total_cost"] - 0.003) < 1e-9
        assert t["total_input"] == 300
        assert t["total_output"] == 150

    @pytest.mark.asyncio
    async def test_since_filter(self, sqlite_store):
        await sqlite_store.insert(_make_record(ts="2026-01-01T00:00:00+00:00", cost_usd=0.001))
        await sqlite_store.insert(_make_record(ts="2026-06-01T00:00:00+00:00", cost_usd=0.002))
        rows = await sqlite_store.summary(since="2026-03-01")
        assert len(rows) == 1
        t = await sqlite_store.totals(since="2026-03-01")
        assert t["calls"] == 1

    @pytest.mark.asyncio
    async def test_recent_limit(self, sqlite_store):
        for i in range(10):
            await sqlite_store.insert(_make_record())
        rows = await sqlite_store.recent(n=3)
        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_meta_roundtrip(self, sqlite_store):
        meta = {"reasoning_tokens": 42, "stop_reason": "end_turn"}
        await sqlite_store.insert(_make_record(meta=meta))
        rows = await sqlite_store.recent(1)
        import json
        stored_meta = json.loads(rows[0]["meta"])
        assert stored_meta["reasoning_tokens"] == 42

    @pytest.mark.asyncio
    async def test_health_returns_true(self, sqlite_store):
        assert await sqlite_store.health() is True


# ── make_store factory ─────────────────────────────────────────────────────

class TestMakeStore:
    def test_no_args_returns_sqlite(self):
        store = make_store()
        assert isinstance(store, SQLiteStore)

    def test_db_path_kwarg(self, tmp_path):
        store = make_store(db_path=tmp_path / "x.db")
        assert isinstance(store, SQLiteStore)

    def test_postgres_dsn_returns_postgres(self):
        store = make_store("postgresql://user:pw@localhost/db")
        assert isinstance(store, PostgresStore)

    def test_postgres_asyncpg_dsn_stripped(self):
        store = make_store("postgresql+asyncpg://user:pw@localhost/db")
        assert isinstance(store, PostgresStore)
        assert "+asyncpg" not in store._dsn

    def test_sqlalchemy_psycopg2_prefix_stripped(self):
        store = make_store("postgresql+psycopg2://user:pw@localhost/db")
        assert isinstance(store, PostgresStore)
        assert "+psycopg2" not in store._dsn


# ── PostgresStore (integration, skipped if no DB available) ───────────────

PG_DSN = os.environ.get("TEST_PG_DSN", "")

@pytest.mark.skipif(not PG_DSN, reason="Set TEST_PG_DSN env var to run Postgres tests")
class TestPostgresStore:
    @pytest_asyncio.fixture
    async def pg_store(self):
        store = PostgresStore(PG_DSN)
        # Clean slate
        pool = await store._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM llm_usage_log WHERE tags = 'env=pytest'")
        yield store
        await store.close()

    @pytest.mark.asyncio
    async def test_insert_and_recent(self, pg_store):
        rec = _make_record(tags="env=pytest")
        await pg_store.insert(rec)
        rows = await pg_store.recent(5)
        assert any(r["tags"] == "env=pytest" for r in rows)

    @pytest.mark.asyncio
    async def test_health(self, pg_store):
        assert await pg_store.health() is True

    @pytest.mark.asyncio
    async def test_totals(self, pg_store):
        await pg_store.insert(_make_record(tags="env=pytest", cost_usd=0.005))
        t = await pg_store.totals()
        assert t["calls"] >= 1
