"""
CLI report generator.

    python -m adk_cost_tracker.report
    python -m adk_cost_tracker.report --since 2026-04-01
    python -m adk_cost_tracker.report --since 2026-04-01 --app my_app
    python -m adk_cost_tracker.report --recent 20
    python -m adk_cost_tracker.report --db postgresql://user:pw@host/db
"""

from __future__ import annotations

import argparse
import asyncio

from .store import BaseStore, make_store

BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RESET  = "\033[0m"


def _h(t):  return f"{BOLD}{CYAN}{t}{RESET}"
def _money(v: float) -> str:
    v = float(v or 0)
    c = GREEN if v < 0.01 else (YELLOW if v < 1.0 else RED)
    return f"{c}${v:.6f}{RESET}"


async def _print_summary(store: BaseStore, since: str | None, app: str | None):
    rows   = await store.summary(since)
    totals = await store.totals(since)

    # Filter by app_name if requested
    if app:
        rows = [r for r in rows if r.get("app_name", "") == app or
                r.get("agent", "") == app]

    period = f"since {since}" if since else "all time"
    title  = f"app={app}" if app else "all apps"
    print(f"\n{_h('LLM Cost Tracker')} — {title} | {period}\n")

    if not rows:
        print("  No records found.")
        return

    col = "{:<10} {:<22} {:<28} {:>7} {:>13} {:>13} {:>14}"
    print(col.format("Provider", "Agent", "Model", "Calls",
                     "Input tok", "Output tok", "Cost (USD)"))
    print("─" * 111)

    for r in rows:
        print(col.format(
            str(r.get("provider",""))[:10],
            str(r.get("agent",""))[:22],
            str(r.get("model",""))[:28],
            r.get("calls", 0),
            f"{int(r.get('total_input') or 0):,}",
            f"{int(r.get('total_output') or 0):,}",
            _money(r.get("total_cost") or 0),
        ))

    print("─" * 111)
    col2 = "{:<64} {:>7} {:>13} {:>13} {:>14}"
    print(col2.format(
        "TOTAL", totals.get("calls", 0),
        f"{int(totals.get('total_input') or 0):,}",
        f"{int(totals.get('total_output') or 0):,}",
        _money(totals.get("total_cost") or 0),
    ))
    print()


async def _print_recent(store: BaseStore, n: int):
    rows = await store.recent(n)
    print(f"\n{_h('LLM Cost Tracker')} — last {n} calls\n")

    if not rows:
        print("  No records found.")
        return

    col = "{:<26} {:<10} {:<18} {:<24} {:>10} {:>10} {:>14}"
    print(col.format("Timestamp", "Provider", "Agent", "Model",
                     "In tok", "Out tok", "Cost (USD)"))
    print("─" * 115)

    for r in rows:
        print(col.format(
            str(r.get("ts",""))[:26],
            str(r.get("provider",""))[:10],
            str(r.get("agent",""))[:18],
            str(r.get("model",""))[:24],
            f"{int(r.get('input_tokens') or 0):,}",
            f"{int(r.get('output_tokens') or 0):,}",
            _money(r.get("cost_usd") or 0),
        ))
    print()


# --- Public helpers ---

def print_summary(store: BaseStore, since: str | None = None,
                  app: str | None = None):
    asyncio.run(_print_summary(store, since, app))


def print_recent(store: BaseStore, n: int = 20):
    asyncio.run(_print_recent(store, n))


# --- CLI entry point ---

def main():
    parser = argparse.ArgumentParser(description="LLM Cost Tracker report")
    parser.add_argument("--since",  metavar="YYYY-MM-DD")
    parser.add_argument("--app",    metavar="APP_NAME",
                        help="Filter by app_name")
    parser.add_argument("--recent", type=int, metavar="N",
                        help="Show N most recent calls")
    parser.add_argument("--db",     metavar="DSN_OR_PATH",
                        help="PostgreSQL DSN or path to usage.db")
    args = parser.parse_args()

    store = make_store(args.db) if args.db else make_store()

    if args.recent:
        asyncio.run(_print_recent(store, args.recent))
    else:
        asyncio.run(_print_summary(store, args.since, args.app))


if __name__ == "__main__":
    main()
