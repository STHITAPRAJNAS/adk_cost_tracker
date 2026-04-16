"""
adk_cost_tracker — self-hosted, multi-provider LLM cost tracking.

Supports: Google ADK (Gemini) · OpenAI · AWS Bedrock
Storage:  SQLite (local dev) · PostgreSQL (production, shared schema)

Quick start:
    from adk_cost_tracker import CostTrackerPlugin, make_store
    from adk_cost_tracker.trackers.openai_tracker import track_openai
    from adk_cost_tracker.trackers.bedrock_tracker import track_bedrock
"""

from .plugin import CostTrackerPlugin
from .pricing import calculate_cost, get_price, PRICES
from .store import make_store, BaseStore, PostgresStore, SQLiteStore, CallRecord
from .report import print_summary, print_recent

__all__ = [
    # ADK plugin
    "CostTrackerPlugin",
    # Store factory + backends
    "make_store",
    "BaseStore",
    "PostgresStore",
    "SQLiteStore",
    "CallRecord",
    # Pricing
    "calculate_cost",
    "get_price",
    "PRICES",
    # Reporting
    "print_summary",
    "print_recent",
]
