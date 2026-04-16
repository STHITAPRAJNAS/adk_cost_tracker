# ADK Cost Tracker

[![PyPI - Version](https://img.shields.io/pypi/v/adk-cost-tracker)](https://pypi.org/project/adk-cost-tracker/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/STHITAPRAJNAS/adk_cost_tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/STHITAPRAJNAS/adk_cost_tracker/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Centralized, provider-agnostic LLM cost tracking and pricing management for Google ADK.

ADK Cost Tracker is a lightweight observability layer for Python 3.10+ that intercepts LLM usage and persists it to a shared database. It solves the "distributed pricing" problem by using your database as the Source of Truth for model costs across all your services.

---

## Feature Matrix

| Feature | Description |
| :--- | :--- |
| **Agnostic Core** | Treats providers and models as arbitrary keys-no hard SDK dependencies. |
| **Centralized Pricing** | Store LLM prices in a shared DB; update once, sync everywhere. |
| **Shared Storage** | Drop-in support for the same PostgreSQL instance used by ADK's `SessionService`. |
| **Auto-Seeding** | Automatically populates new databases with current market pricing. |
| **CLI Reporting** | Generate cross-app cost summaries directly from your terminal. |
| **Zero-Config Plugin** | Seamless integration with Google ADK agents via `CostTrackerPlugin`. |

---

## Installation

```bash
# Core only (SQLite + Python 3.10+)
pip install adk-cost-tracker

# With PostgreSQL support (Recommended for production)
pip install "adk-cost-tracker[postgres]"
```

---

## Agnostic Design

ADK Cost Tracker is designed to be **provider-blind**. It does not include logic for specific LLM SDKs (like OpenAI or Bedrock). Instead, it provides a clean interface that expects:
1.  **Model Key**: A string identifier (e.g., `"gpt-4o"`, `"gemini-1.5-pro"`).
2.  **Usage Metadata**: Token counts (input, output, cached).

This allows the library to stay extremely lightweight and future-proof as new providers emerge.

---

## 🚦 Quick Start

### 1. Integrate with Google ADK
The library provides a first-class plugin for Google ADK that automatically captures usage metadata from model responses.

```python
from adk_cost_tracker import CostTrackerPlugin
from adk_cost_tracker.store import make_store

# Connect to your shared ADK database
store = make_store("postgresql://user:pw@host/mydb")

# Initialize the plugin
plugin = CostTrackerPlugin(
    store=store,
    app_name="finance_agent",
    sync_pricing=True, # Syncs local memory with DB values at startup
    verbose=True
)

# Add to your ADK Runner
runner = Runner(
    agent=agent,
    app_name="finance_agent",
    plugins=[plugin]
)
```

### 2. Centralized Pricing Management
You can update pricing globally for all your apps by updating the `llm_pricing` table in your database.

```python
from adk_cost_tracker.pricing import registry
from adk_cost_tracker.store import make_store

async def update_global_prices():
    store = make_store("postgresql://user:pw@host/mydb")
    
    # Update a specific model price
    await store.update_price(
        model_key="gpt-4o",
        provider="openai",
        input_per_m=2.50,
        output_per_m=10.00
    )
    
    # All ADK apps will reflect this change on their next sync/restart.
```

---

## 📊 Comparison

| Feature | ADK Cost Tracker | Microsoft Presidio / Others |
| :--- | :--- | :--- |
| **Focus** | FinOps & Cost Observability | PII Redaction / Content Safety |
| **Storage** | Self-hosted (Postgres/SQLite) | Cloud-specific or SaaS |
| **Pricing** | Centralized in YOUR DB | Hardcoded or API-polled |
| **Integration** | Native ADK Plugin | Generic Wrappers |

---

## Developer Info

### Running Tests
```bash
# Install dev dependencies
uv sync --dev

# Run pytest
uv run pytest tests
```

### Linting
```bash
uv run ruff check src tests
```

---

## 📜 License

Distributed under the **Apache License, Version 2.0**. See `LICENSE` for more information.

Copyright © 2026 **Sthitaprajna Sahoo**
