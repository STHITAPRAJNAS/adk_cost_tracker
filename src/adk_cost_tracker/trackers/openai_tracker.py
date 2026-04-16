"""
OpenAI cost tracker — wraps the OpenAI client so every chat/completion
call is automatically recorded.

Supports both sync and async OpenAI clients.

Usage (sync):
    import openai
    from adk_cost_tracker.trackers.openai_tracker import track_openai

    client = track_openai(openai.OpenAI(), store=store, app_name="my_app")
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )

Usage (async):
    client = track_openai(openai.AsyncOpenAI(), store=store, app_name="my_app")
    resp = await client.chat.completions.create(...)

The wrapper is transparent — the returned object behaves exactly like
the real OpenAI client. Your existing code needs zero changes beyond
wrapping the client once at startup.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from functools import wraps
from typing import Any

from ..pricing import calculate_cost, get_price
from ..store import BaseStore, CallRecord

logger = logging.getLogger(__name__)


def _make_record(
    response: Any,
    app_name: str,
    agent: str,
    tags: str,
) -> CallRecord | None:
    """Extract usage from an OpenAI response object and return a CallRecord."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    model = getattr(response, "model", "") or ""

    # New Responses API: input_tokens / output_tokens
    # Old Chat API:      prompt_tokens / completion_tokens
    input_tokens  = (
        getattr(usage, "input_tokens", None)
        or getattr(usage, "prompt_tokens", 0)
        or 0
    )
    output_tokens = (
        getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", 0)
        or 0
    )

    # Cached token details (input_tokens_details.cached_tokens)
    details       = getattr(usage, "input_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", 0) or 0

    # Reasoning tokens go into meta (billed as output)
    out_details     = getattr(usage, "output_tokens_details", None)
    reasoning_toks  = getattr(out_details, "reasoning_tokens", 0) or 0

    cost  = calculate_cost(model, input_tokens, output_tokens, cached_tokens)
    price = get_price(model)

    return CallRecord(
        ts=datetime.now(timezone.utc).isoformat(),
        provider=price.provider if price.provider != "unknown" else "openai",
        agent=agent,
        model=model,
        session_id="",
        app_name=app_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cost_usd=cost,
        tags=tags,
        meta={"reasoning_tokens": reasoning_toks} if reasoning_toks else {},
    )


class _TrackedCompletions:
    """Proxy for client.chat.completions that records every call."""

    def __init__(self, real_completions, store: BaseStore, app_name: str,
                 agent: str, tags: str, is_async: bool):
        self._real = real_completions
        self._store = store
        self._app_name = app_name
        self._agent = agent
        self._tags = tags
        self._is_async = is_async

    def create(self, *args, **kwargs):
        if self._is_async:
            return self._acreate(*args, **kwargs)
        return self._screate(*args, **kwargs)

    def _screate(self, *args, **kwargs):
        response = self._real.create(*args, **kwargs)
        record = _make_record(response, self._app_name, self._agent, self._tags)
        if record:
            try:
                # Run async insert in a new event loop slice
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self._store.insert(record))
            except Exception as exc:
                logger.warning("[CostTracker/OpenAI] insert failed: %s", exc)
        return response

    async def _acreate(self, *args, **kwargs):
        response = await self._real.create(*args, **kwargs)
        record = _make_record(response, self._app_name, self._agent, self._tags)
        if record:
            asyncio.create_task(_safe_insert(self._store, record))
        return response

    def __getattr__(self, name):
        return getattr(self._real, name)


class _TrackedChat:
    def __init__(self, real_chat, **kw):
        self.completions = _TrackedCompletions(real_chat.completions, **kw)

    def __getattr__(self, name):
        return getattr(self._real, name)


class TrackedOpenAIClient:
    """
    Thin proxy around an OpenAI (sync or async) client.
    Intercepts chat.completions.create() calls to record cost.
    All other attributes/methods pass through unchanged.
    """

    def __init__(
        self,
        client: Any,
        store: BaseStore,
        app_name: str = "",
        agent: str = "openai-agent",
        tags: dict[str, str] | None = None,
    ):
        self._client = client
        is_async = asyncio.iscoroutinefunction(
            getattr(client.chat.completions, "create", None)
        )
        tag_str = ",".join(f"{k}={v}" for k, v in (tags or {}).items())
        self.chat = _TrackedChat(
            client.chat,
            store=store,
            app_name=app_name,
            agent=agent,
            tags=tag_str,
            is_async=is_async,
        )

    def __getattr__(self, name):
        return getattr(self._client, name)


def track_openai(
    client: Any,
    store: BaseStore,
    app_name: str = "",
    agent: str = "openai-agent",
    tags: dict[str, str] | None = None,
) -> TrackedOpenAIClient:
    """
    Wrap an OpenAI client with cost tracking. Drop-in replacement.

        client = track_openai(openai.OpenAI(), store=store, app_name="search_app")
    """
    return TrackedOpenAIClient(client, store, app_name, agent, tags)


async def _safe_insert(store: BaseStore, record: CallRecord):
    try:
        await store.insert(record)
    except Exception as exc:
        logger.warning("[CostTracker/OpenAI] store insert failed: %s", exc)
