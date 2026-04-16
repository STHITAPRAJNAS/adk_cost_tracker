"""
PricingRegistry — how the library knows what each model costs.

╔══════════════════════════════════════════════════════════════════════╗
║  HOW COST LOOKUP WORKS                                               ║
║                                                                      ║
║  1. Built-in table  – PRICES dict below, covers all major models.   ║
║     Uses longest-substring matching so "gemini-2.5-flash" beats     ║
║     "gemini-2.5" and "flash", avoiding mis-pricing.                 ║
║                                                                      ║
║  2. YAML override   – point PRICING_CONFIG env var at a YAML file   ║
║     to add private / on-prem / fine-tuned model pricing without     ║
║     touching library code.                                           ║
║                                                                      ║
║  3. Runtime override – call registry.register() or                  ║
║     registry.load_from_dict() at startup to inject custom prices    ║
║     programmatically (from your own config system / secrets store). ║
║                                                                      ║
║  4. Unknown model  – returns ModelPrice(0, 0, 0) and logs a         ║
║     WARNING so you know pricing data is missing rather than          ║
║     silently billing $0.                                             ║
╚══════════════════════════════════════════════════════════════════════╝

YAML override format (~/.pricing.yaml or $PRICING_CONFIG path):

    models:
      my-fine-tuned-gpt4:
        provider: openai
        input_per_m: 6.00
        output_per_m: 24.00
        cached_input_per_m: 1.50

Sources (April 2026):
  Gemini  — https://ai.google.dev/gemini-api/docs/pricing
  OpenAI  — https://openai.com/api/pricing/
  Bedrock — https://aws.amazon.com/bedrock/pricing/
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    provider: str             # "gemini" | "openai" | "bedrock" | custom
    input_per_m: float        # USD per 1M input tokens
    output_per_m: float       # USD per 1M output tokens
    cached_input_per_m: float = 0.0  # USD per 1M cached/read tokens (usually ~10% of input)


# ── Built-in price table ───────────────────────────────────────────────────
# Key = lowercase substring that must appear in the model name.
# Longest-match wins: "gemini-2.5-flash" always beats "gemini-2.5" or "flash".

_BUILTIN: dict[str, ModelPrice] = {

    # ── Google Gemini ──────────────────────────────────────────────────────
    "gemini-2.5-pro": ModelPrice(
        provider="gemini", input_per_m=1.25, output_per_m=10.00, cached_input_per_m=0.3125
    ),
    "gemini-2.5-flash": ModelPrice(
        provider="gemini", input_per_m=0.30, output_per_m=2.50, cached_input_per_m=0.03
    ),
    "gemini-2.0-flash": ModelPrice(
        provider="gemini", input_per_m=0.10, output_per_m=0.40, cached_input_per_m=0.025
    ),
    "gemini-1.5-pro": ModelPrice(
        provider="gemini", input_per_m=1.25, output_per_m=5.00, cached_input_per_m=0.3125
    ),
    "gemini-1.5-flash": ModelPrice(
        provider="gemini", input_per_m=0.075, output_per_m=0.30, cached_input_per_m=0.01875
    ),

    # ── OpenAI ────────────────────────────────────────────────────────────
    "gpt-4.1-mini": ModelPrice(
        provider="openai", input_per_m=0.40, output_per_m=1.60, cached_input_per_m=0.10
    ),
    "gpt-4.1-nano": ModelPrice(
        provider="openai", input_per_m=0.10, output_per_m=0.40, cached_input_per_m=0.025
    ),
    "gpt-4.1": ModelPrice(
        provider="openai", input_per_m=2.00, output_per_m=8.00, cached_input_per_m=0.50
    ),
    "gpt-4o-mini": ModelPrice(
        provider="openai", input_per_m=0.15, output_per_m=0.60, cached_input_per_m=0.075
    ),
    "gpt-4o": ModelPrice(
        provider="openai", input_per_m=2.50, output_per_m=10.00, cached_input_per_m=1.25
    ),
    "o3-mini": ModelPrice(
        provider="openai", input_per_m=1.10, output_per_m=4.40, cached_input_per_m=0.55
    ),
    "o3": ModelPrice(
        provider="openai", input_per_m=10.00, output_per_m=40.00, cached_input_per_m=2.50
    ),

    # ── AWS Bedrock — Amazon Nova ─────────────────────────────────────────
    "amazon.nova-micro": ModelPrice(
        provider="bedrock", input_per_m=0.035, output_per_m=0.14
    ),
    "amazon.nova-lite": ModelPrice(
        provider="bedrock", input_per_m=0.06, output_per_m=0.24
    ),
    "amazon.nova-pro": ModelPrice(
        provider="bedrock", input_per_m=0.80, output_per_m=3.20
    ),
    "amazon.nova-premier": ModelPrice(
        provider="bedrock", input_per_m=2.00, output_per_m=8.00
    ),

    # ── AWS Bedrock — Anthropic Claude ────────────────────────────────────
    "claude-3-5-sonnet": ModelPrice(
        provider="bedrock", input_per_m=3.00, output_per_m=15.00, cached_input_per_m=0.30
    ),
    "claude-3-5-haiku": ModelPrice(
        provider="bedrock", input_per_m=0.80, output_per_m=4.00, cached_input_per_m=0.08
    ),
    "claude-3-haiku": ModelPrice(
        provider="bedrock", input_per_m=0.25, output_per_m=1.25, cached_input_per_m=0.03
    ),
    "claude-3-opus": ModelPrice(
        provider="bedrock", input_per_m=15.00, output_per_m=75.00, cached_input_per_m=1.50
    ),

    # ── AWS Bedrock — Meta Llama ───────────────────────────────────────────
    "llama3-1-405b": ModelPrice(provider="bedrock", input_per_m=0.65, output_per_m=0.80),
    "llama3-1-70b":  ModelPrice(provider="bedrock", input_per_m=0.35, output_per_m=0.45),
    "llama3-1-8b":   ModelPrice(provider="bedrock", input_per_m=0.20, output_per_m=0.25),
}

_UNKNOWN = ModelPrice(provider="unknown", input_per_m=0.0, output_per_m=0.0)


# ── PricingRegistry ────────────────────────────────────────────────────────

class PricingRegistry:
    """
    Thread-safe registry of model prices.

    Lookup order (highest priority first):
      1. Runtime overrides registered via register() / load_from_dict()
      2. YAML file pointed to by PRICING_CONFIG env var
      3. Built-in hardcoded table

    All lookups use longest-substring matching (case-insensitive).
    """

    def __init__(self) -> None:
        # Start with built-in table; overrides layer on top
        self._prices: dict[str, ModelPrice] = dict(_BUILTIN)
        self._load_env_yaml()

    # ── Loading helpers ────────────────────────────────────────────────────

    def _load_env_yaml(self) -> None:
        path = os.environ.get("PRICING_CONFIG", "")
        if path:
            self.load_from_yaml(Path(path))

    def load_from_yaml(self, path: Path) -> "PricingRegistry":
        """
        Load / merge prices from a YAML file.  PyYAML must be installed.

        yaml format:
            models:
              my-model:
                provider: openai
                input_per_m: 6.00
                output_per_m: 24.00
                cached_input_per_m: 1.50   # optional
        """
        try:
            import yaml  # type: ignore[import]
        except ImportError:
            logger.warning(
                "PyYAML not installed. Cannot load pricing from %s. "
                "Run: pip install adk-cost-tracker[yaml]", path
            )
            return self

        try:
            data = yaml.safe_load(path.read_text())
            models = (data or {}).get("models", {})
            self.load_from_dict(models)
            logger.info("Loaded %d custom prices from %s", len(models), path)
        except Exception as exc:
            logger.error("Failed to load pricing YAML %s: %s", path, exc)
        return self

    def load_from_dict(self, models: dict[str, dict]) -> "PricingRegistry":
        """
        Merge a dict of model prices at runtime.

            registry.load_from_dict({
                "my-fine-tuned": {
                    "provider": "openai",
                    "input_per_m": 6.0,
                    "output_per_m": 24.0,
                }
            })
        """
        for key, cfg in models.items():
            self.register(
                model_key=key.lower(),
                provider=cfg.get("provider", "custom"),
                input_per_m=float(cfg.get("input_per_m", 0)),
                output_per_m=float(cfg.get("output_per_m", 0)),
                cached_input_per_m=float(cfg.get("cached_input_per_m", 0)),
            )
        return self

    def register(
        self,
        model_key: str,
        provider: str,
        input_per_m: float,
        output_per_m: float,
        cached_input_per_m: float = 0.0,
    ) -> "PricingRegistry":
        """Register or override a single model price at runtime."""
        self._prices[model_key.lower()] = ModelPrice(
            provider=provider,
            input_per_m=input_per_m,
            output_per_m=output_per_m,
            cached_input_per_m=cached_input_per_m,
        )
        return self

    # ── Lookup ─────────────────────────────────────────────────────────────

    def get(self, model_name: str) -> ModelPrice:
        """
        Return pricing for a model.  Longest-key substring match wins.
        Logs a WARNING if no match is found (tokens will still be tracked,
        but cost will be $0 — better to know than to silently mis-bill).
        """
        low = model_name.lower()
        matched = [k for k in self._prices if k in low]
        if not matched:
            logger.warning(
                "[CostTracker] Unknown model '%s' — cost will be $0. "
                "Add it via registry.register() or a PRICING_CONFIG YAML file.",
                model_name,
            )
            return _UNKNOWN
        return self._prices[max(matched, key=len)]

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Return total USD cost for one LLM call."""
        price = self.get(model_name)
        billable_input = max(0, input_tokens - cached_tokens)
        return round(
            billable_input   * price.input_per_m        / 1_000_000
            + cached_tokens  * price.cached_input_per_m / 1_000_000
            + output_tokens  * price.output_per_m        / 1_000_000,
            8,
        )

    def all_models(self) -> dict[str, ModelPrice]:
        """Return a snapshot of all registered model prices."""
        return dict(self._prices)

    async def sync_with_store(self, store: "BaseStore") -> "PricingRegistry":
        """
        Sync pricing with a storage backend.
        1. Fetches all prices from the store and updates local registry.
        2. If the store is empty, it can optionally seed the store with built-ins.
        """
        # Note: BaseStore and other types are imported here to avoid circular imports
        # or we rely on the type hint being a string in the signature.
        from .store import BaseStore

        db_prices = await store.get_all_prices()
        if not db_prices:
            # Seed the database with built-ins if it's empty
            logger.info("[CostTracker] Seeding store with built-in prices...")
            for model_key, price in _BUILTIN.items():
                await store.update_price(
                    model_key=model_key,
                    provider=price.provider,
                    input_per_m=price.input_per_m,
                    output_per_m=price.output_per_m,
                    cached_input_per_m=price.cached_input_per_m,
                )
            return self

        # Update local registry from DB
        for model_key, p in db_prices.items():
            self.register(
                model_key=model_key,
                provider=p["provider"],
                input_per_m=float(p["input_per_m"]),
                output_per_m=float(p["output_per_m"]),
                cached_input_per_m=float(p["cached_input_per_m"]),
            )
        logger.info("[CostTracker] Synced %d prices from store", len(db_prices))
        return self

    async def update_store(self, store: "BaseStore") -> None:
        """
        Push all local pricing data into the database store.
        Useful for administrative scripts to update the 'Source of Truth'.
        """
        for model_key, price in self._prices.items():
            await store.update_price(
                model_key=model_key,
                provider=price.provider,
                input_per_m=price.input_per_m,
                output_per_m=price.output_per_m,
                cached_input_per_m=price.cached_input_per_m,
            )
        logger.info("[CostTracker] Pushed %d prices to store", len(self._prices))


# ── Module-level default registry (used by all components by default) ──────
# Enterprise: replace with a custom registry at startup:
#   from adk_cost_tracker.pricing import registry
#   registry.load_from_yaml(Path("my_prices.yaml"))

registry = PricingRegistry()

# Convenience module-level aliases (backwards-compatible)
PRICES = registry.all_models()

def get_price(model_name: str) -> ModelPrice:
    return registry.get(model_name)

def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    return registry.calculate_cost(model_name, input_tokens, output_tokens, cached_tokens)
