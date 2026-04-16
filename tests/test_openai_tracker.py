"""Tests for OpenAI tracker — uses mock objects, no real API calls."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adk_cost_tracker.store import CallRecord, SQLiteStore
from adk_cost_tracker.trackers.openai_tracker import track_openai, _make_record


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_oai_response(
    model="gpt-4o-mini",
    input_tokens=100,
    output_tokens=50,
    cached_tokens=0,
    reasoning_tokens=0,
):
    """Build a fake OpenAI response object matching the real SDK shape."""
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )
    return SimpleNamespace(
        model=model,
        usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(content="Hello!"))],
    )


def _make_legacy_response(model="gpt-4o", prompt_tokens=200, completion_tokens=80):
    """Older Chat API shape uses prompt_tokens / completion_tokens."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        input_tokens=None,   # not present in legacy
        output_tokens=None,
        input_tokens_details=None,
        output_tokens_details=None,
    )
    return SimpleNamespace(model=model, usage=usage)


# ── _make_record unit tests ────────────────────────────────────────────────

class TestMakeRecord:
    def test_basic_record(self):
        resp = _make_oai_response()
        rec = _make_record(resp, app_name="app", agent="agent1", tags="env=test")
        assert rec is not None
        assert rec.provider == "openai"
        assert rec.model == "gpt-4o-mini"
        assert rec.input_tokens == 100
        assert rec.output_tokens == 50
        assert rec.cached_tokens == 0
        assert rec.cost_usd > 0

    def test_cached_tokens_extracted(self):
        resp = _make_oai_response(cached_tokens=40)
        rec = _make_record(resp, "app", "agent", "")
        assert rec.cached_tokens == 40

    def test_reasoning_tokens_in_meta(self):
        resp = _make_oai_response(reasoning_tokens=30)
        rec = _make_record(resp, "app", "agent", "")
        assert rec.meta.get("reasoning_tokens") == 30

    def test_legacy_response_shape(self):
        resp = _make_legacy_response()
        rec = _make_record(resp, "app", "agent", "")
        assert rec.input_tokens == 200
        assert rec.output_tokens == 80

    def test_no_usage_returns_none(self):
        resp = SimpleNamespace(model="gpt-4o", usage=None)
        assert _make_record(resp, "app", "agent", "") is None

    def test_cost_is_nonzero_for_known_model(self):
        resp = _make_oai_response(model="gpt-4o", input_tokens=1_000_000, output_tokens=0)
        rec = _make_record(resp, "app", "agent", "")
        assert rec.cost_usd == pytest.approx(2.50, rel=1e-4)  # $2.50/M input


# ── Sync wrapper integration ───────────────────────────────────────────────

class TestSyncTracker:
    @pytest.fixture
    def store(self, tmp_path):
        return SQLiteStore(db_path=tmp_path / "test.db")

    def test_create_records_on_call(self, store):
        """tracked client.chat.completions.create() records a row."""
        fake_completions = MagicMock()
        fake_completions.create.return_value = _make_oai_response()

        fake_chat = MagicMock()
        fake_chat.completions = fake_completions

        fake_client = MagicMock()
        fake_client.chat = fake_chat

        # Mark create as sync (not a coroutine)
        fake_completions.create.__wrapped__ = None

        client = track_openai(fake_client, store=store, app_name="myapp", agent="bot")
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[])

        # Verify the original call was made
        fake_completions.create.assert_called_once()
        # Verify response passed through
        assert resp.model == "gpt-4o-mini"

    def test_passthrough_unknown_attr(self, store):
        """Non-chat attributes pass through to the underlying client."""
        fake_client = MagicMock()
        fake_client.api_key = "sk-test"
        fake_client.chat = MagicMock()
        fake_client.chat.completions = MagicMock()

        client = track_openai(fake_client, store=store)
        assert client.api_key == "sk-test"


# ── Async wrapper integration ──────────────────────────────────────────────

class TestAsyncTracker:
    @pytest.fixture
    def store(self, tmp_path):
        return SQLiteStore(db_path=tmp_path / "async_test.db")

    @pytest.mark.asyncio
    async def test_async_create_records(self, store):
        fake_completions = AsyncMock()
        fake_completions.create = AsyncMock(return_value=_make_oai_response())

        fake_chat = MagicMock()
        fake_chat.completions = fake_completions

        fake_client = MagicMock()
        fake_client.chat = fake_chat

        client = track_openai(fake_client, store=store, app_name="app", agent="async-bot")
        resp = await client.chat.completions.create(model="gpt-4o-mini", messages=[])

        assert resp.model == "gpt-4o-mini"
        # Give the background task time to complete
        await asyncio.sleep(0.05)
        rows = await store.recent(5)
        assert len(rows) == 1
        assert rows[0]["agent"] == "async-bot"
        assert rows[0]["cost_usd"] > 0
