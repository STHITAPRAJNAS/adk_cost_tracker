# ADK Cost Tracker

[![PyPI version](https://img.shields.io/pypi/v/adk-cost-tracker.svg)](https://pypi.org/project/adk-cost-tracker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/STHITAPRAJNAS/adk_cost_tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/STHITAPRAJNAS/adk_cost_tracker/actions/workflows/ci.yml)

**Self-hosted, multi-provider LLM cost tracking for Google ADK, OpenAI, and AWS Bedrock.**

ADK Cost Tracker is a lightweight, zero-dependency (core) Python library that intercepts LLM calls to track token usage and calculate costs in real-time. It supports centralized pricing and storage across multiple applications.

## Features

- 🎯 **Seamless ADK Integration**: Drop-in plugin for Google ADK.
- 🔌 **Multi-Provider Support**: Built-in trackers for OpenAI and AWS Bedrock (Converse API).
- 🗄️ **Flexible Storage**: Store usage logs in local SQLite or shared PostgreSQL.
- 💰 **Centralized Pricing**: Sync model prices from a shared database across all your apps.
- 📊 **CLI Reporting**: Generate beautiful cost reports from your terminal.
- 🪶 **Zero Core Dependencies**: Only install what you need (e.g., `pip install adk-cost-tracker[postgres,openai]`).

## Installation

```bash
# Core only (SQLite + Gemini built-in)
pip install adk-cost-tracker

# With Postgres and OpenAI support
pip install "adk-cost-tracker[postgres,openai]"

# Install everything
pip install "adk-cost-tracker[all]"
```

## Quick Start

### Using with Google ADK

```python
from adk_cost_tracker import CostTrackerPlugin
from adk_cost_tracker.store import make_store

# 1. Setup a shared store (Postgres recommended for production)
store = make_store("postgresql://user:pw@host/mydb")

# 2. Add the plugin to your ADK Runner
plugin = CostTrackerPlugin(
    store=store,
    app_name="my_rag_app",
    sync_pricing=True,  # Automatically sync/seed pricing from DB
    verbose=True
)

runner = Runner(
    agent=agent,
    app_name="my_rag_app",
    plugins=[plugin]
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
