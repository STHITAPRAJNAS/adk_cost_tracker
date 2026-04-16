"""
Optional: BudgetGuard — a before_model_callback companion that blocks
LLM calls when cumulative spend exceeds a configurable threshold.

Useful for dev environments where you want a hard cap.

Usage:
    runner = Runner(
        agent=agent,
        plugins=[
            CostTrackerPlugin(store=shared_store, verbose=True),
            BudgetGuard(store=shared_store, limit_usd=5.00),
        ],
    )
"""

from __future__ import annotations

import logging

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai.types import Content, Part

from .store import BaseStore

logger = logging.getLogger(__name__)


class BudgetExceeded(Exception):
    pass


class BudgetGuard(BasePlugin):
    """
    Blocks LLM calls once cumulative spend exceeds limit_usd.

    Args:
        store:      The same BaseStore instance used by CostTrackerPlugin
        limit_usd:  Hard spend cap in USD
        since:      Optional ISO date string — only count spend from this date
        raise_exc:  If True, raise BudgetExceeded instead of returning stub response
    """

    def __init__(
        self,
        store: BaseStore,
        limit_usd: float,
        since: str | None = None,
        raise_exc: bool = False,
    ):
        super().__init__(name="budget_guard")
        self._store = store
        self._limit = limit_usd
        self._since = since
        self._raise = raise_exc

    async def before_model_callback(
        self,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        totals = await self._store.totals(self._since)
        spent = totals.get("total_cost") or 0.0

        if spent >= self._limit:
            msg = (
                f"[BudgetGuard] Spend cap ${self._limit:.4f} reached "
                f"(spent ${spent:.4f}). LLM call blocked."
            )
            logger.warning(msg)

            if self._raise:
                raise BudgetExceeded(msg)

            # Return a stub response so the agent gets *something* back
            return LlmResponse(
                content=Content(
                    role="model",
                    parts=[
                        Part(
                            text=f"[Budget cap of ${self._limit:.2f} reached. Request blocked.]"
                        )
                    ],
                )
            )

        return None  # allow the call through
