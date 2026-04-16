# ADK Cost Tracker

[![PyPI version](https://img.shields.io/pypi/v/adk-cost-tracker.svg)](https://pypi.org/project/adk-cost-tracker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/STHITAPRAJNAS/adk_cost_tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/STHITAPRAJNAS/adk_cost_tracker/actions/workflows/ci.yml)

**Centralized, database-driven LLM cost tracking for Google ADK environments.**

ADK Cost Tracker is a lightweight, provider-agnostic library for Python 3.10+ that logs token usage and costs into a shared database. While it includes utilities for popular SDKs, its primary goal is to provide a seamless observability layer for **Google ADK** agents.

## Features

- 🎯 **Seamless ADK Integration**: First-class `CostTrackerPlugin` for Google ADK.
- 💰 **Centralized Pricing**: Store LLM prices in your shared database—update once, sync across all apps.
- 🗄️ **Shared Storage**: Use the same PostgreSQL instance as ADK's `SessionService` for unified logs.
- 📊 **Provider Agnostic**: Core logic treats models and providers as simple keys—works with any LLM.
- 🪶 **Zero Core Dependencies**: Core library has no external requirements.

## Installation

```bash
# Core only (SQLite + Python 3.10+)
pip install adk-cost-tracker
```

## How It Works: Centralized Pricing

A common challenge in LLM apps is keeping cost data up-to-date across multiple microservices. ADK Cost Tracker solves this by using your database as the **Source of Truth** for pricing.

1.  **Apps sync at startup**: ADK apps call `registry.sync_with_store(db)`.
2.  **Seeding**: If the database is new, the library automatically seeds it with standard market rates.
3.  **Updates**: Update a price in the `llm_pricing` table via SQL or the library's API, and all apps will reflect the change.

## Quick Start (Google ADK)

```python
from adk_cost_tracker import CostTrackerPlugin
from adk_cost_tracker.store import make_store

# Share your ADK database
store = make_store("postgresql://user:pw@host/mydb")

# Use the plugin in your runner
plugin = CostTrackerPlugin(
    store=store,
    app_name="finance_agent",
    sync_pricing=True, # Syncs local memory with DB values
    verbose=True
)
```

### Tracking OpenAI Calls

```python
import openai
from adk_cost_tracker.trackers.openai_tracker import track_openai
from adk_cost_tracker.store import make_store

store = make_store()
client = track_openai(openai.OpenAI(), store=store, app_name="standalone_app")

# Use exactly like the standard client
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Usage and cost ($) are now recorded in SQLite!
```

## CLI Reporting

Generate reports from your terminal:

```bash
# Summary of all apps
adk-cost-report

# Filter by app and date
adk-cost-report --app my_rag_app --since 2026-04-01

# Specify database DSN
adk-cost-report --db postgresql://user:pw@host/mydb
```

## Centralized Pricing

You can manage LLM prices centrally in your database. `adk-cost-tracker` will automatically seed the database with current market rates if it's empty, and you can override them via SQL or YAML.

```python
# Force a sync from store
await registry.sync_with_store(store)
```

## License

MIT © [STHITAPRAJNAS](https://github.com/STHITAPRAJNAS)
