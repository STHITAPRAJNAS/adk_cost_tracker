"""
Minimal end-to-end example showing CostTrackerPlugin wired into an ADK agent.

Run:
    export GOOGLE_API_KEY=your_key
    python -m adk_cost_tracker.example_agent
    python -m adk_cost_tracker.report
"""

import asyncio
import os

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from adk_cost_tracker import CostTrackerPlugin


# ── 1. Build your agent as usual ──────────────────────────────────────────
agent = LlmAgent(
    name="demo_agent",
    model="gemini-2.5-flash",
    instruction="You are a helpful assistant. Be concise.",
)

# ── 2. Create the plugin — one line, all done ─────────────────────────────
cost_plugin = CostTrackerPlugin(
    tags={"env": "dev", "team": "platform"},
    verbose=True,   # print per-call cost to stdout
)

# ── 3. Register plugin on Runner ───────────────────────────────────────────
session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name="demo_app",
    session_service=session_service,
    plugins=[cost_plugin],          # <── only change from your existing setup
)


async def main():
    session = await session_service.create_session(
        app_name="demo_app", user_id="user_1"
    )

    prompts = [
        "What is the capital of France?",
        "Explain transformers in one sentence.",
        "Write a haiku about Python.",
    ]

    for prompt in prompts:
        print(f"\nUser: {prompt}")
        message = Content(role="user", parts=[Part(text=prompt)])
        async for event in runner.run_async(
            user_id="user_1",
            session_id=session.id,
            new_message=message,
        ):
            if event.is_final_response() and event.content:
                print(f"Agent: {event.content.parts[0].text}")

    # ── 4. Print a summary directly from Python ───────────────────────────
    print("\n" + "=" * 60)
    from adk_cost_tracker import print_summary
    print_summary(cost_plugin._store)


if __name__ == "__main__":
    asyncio.run(main())
