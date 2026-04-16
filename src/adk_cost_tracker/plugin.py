"""
CostTrackerPlugin — Google ADK plugin.
Intercepts every Gemini LLM response via after_model_callback,
calculates cost, and persists to the configured store (SQLite or Postgres).

Usage:
    from adk_cost_tracker import CostTrackerPlugin
    from adk_cost_tracker.store import make_store

    # Share the same DB as DatabaseSessionService:
    # This enables cross-app cost visibility and centralized pricing.
    store = make_store("postgresql://user:pw@host/mydb")

    plugin = CostTrackerPlugin(
        store=store, 
        app_name="my_app", 
        sync_pricing=True,  # Syncs local pricing with DB at startup
        verbose=True
    )

    runner = Runner(
        agent=agent,
        app_name="my_app",
        session_service=DatabaseSessionService("postgresql://user:pw@host/mydb"),
        plugins=[plugin],
    )
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin

from .pricing import calculate_cost, get_price
from .store import BaseStore, CallRecord, SQLiteStore

logger = logging.getLogger(__name__)


class CostTrackerPlugin(BasePlugin):
    """
    Drop-in ADK plugin — tracks every Gemini LLM call automatically.

    Args:
        store:    Storage backend. Defaults to local SQLite.
                  Pass a PostgresStore to share the ADK database.
        app_name: Your ADK app_name — stored on each record so you can
                  JOIN llm_usage_log with ADK's sessions table.
        tags:     Static labels attached to every record {"env": "prod"}.
        sync_pricing: If True, syncs the global pricing registry with the store
                  at startup (seeds DB with defaults if empty).
        verbose:  Log a cost line per call (uses logging.INFO).
    """

    def __init__(
        self,
        store: Optional[BaseStore] = None,
        app_name: str = "",
        tags: dict[str, str] | None = None,
        sync_pricing: bool = False,
        verbose: bool = False,
    ):
        super().__init__(name="cost_tracker")
        self._store = store or SQLiteStore()
        self._app_name = app_name
        self._tags = ",".join(f"{k}={v}" for k, v in (tags or {}).items())
        self._verbose = verbose
        if sync_pricing:
            # We fire this off — it won't block the plugin creation
            from .pricing import registry
            asyncio.create_task(registry.sync_with_store(self._store))

    async def initialize(self):
        """
        Optional: call this if you want to ensure pricing is synced 
        before any LLM calls are made.
        """
        from .pricing import registry
        await registry.sync_with_store(self._store)

    async def after_model_callback(
        self,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        """Async callback — never blocks the agent."""
        try:
            meta = llm_response.usage_metadata
            if meta is None:
                return None   # streaming partial — no metadata yet

            input_tokens  = getattr(meta, "prompt_token_count", 0) or 0
            output_tokens = getattr(meta, "candidates_token_count", 0) or 0
            cached_tokens = getattr(meta, "cached_content_token_count", 0) or 0
            model = getattr(llm_response, "model_version", "") or ""

            if not model:
                model = getattr(
                    callback_context.agent_context, "model", "unknown"
                )

            cost = calculate_cost(model, input_tokens, output_tokens, cached_tokens)
            price = get_price(model)

            record = CallRecord(
                ts=datetime.now(timezone.utc).isoformat(),
                provider=price.provider,
                agent=callback_context.agent_name or "unknown",
                model=model,
                session_id=str(getattr(callback_context, "session_id", "")),
                app_name=self._app_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                cost_usd=cost,
                tags=self._tags,
                meta={},
            )

            # Fire-and-forget so we never delay the agent response
            asyncio.create_task(self._safe_insert(record))

            if self._verbose:
                logger.info(
                    "[CostTracker] %s | %s | in=%d out=%d cached=%d | $%.6f",
                    record.agent, record.model,
                    input_tokens, output_tokens, cached_tokens, cost,
                )

        except Exception as exc:
            logger.warning("[CostTracker] failed to record usage: %s", exc)

        return None   # pass response through unchanged

    async def _safe_insert(self, record: CallRecord):
        try:
            await self._store.insert(record)
        except Exception as exc:
            logger.warning("[CostTracker] store insert failed: %s", exc)
