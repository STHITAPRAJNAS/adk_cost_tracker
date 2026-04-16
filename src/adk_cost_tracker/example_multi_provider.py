"""
Multi-provider example — Gemini (ADK) + OpenAI + Bedrock all writing
to the same store (PostgreSQL or SQLite).

    export GOOGLE_API_KEY=...
    export OPENAI_API_KEY=...
    # AWS credentials via ~/.aws or env vars
    python -m adk_cost_tracker.example_multi_provider
"""

import asyncio
import os

# ── 1. Create ONE shared store ─────────────────────────────────────────────
from adk_cost_tracker import make_store, CostTrackerPlugin, print_summary

# For local dev — SQLite
store = make_store()

# For production — point at ANY postgres instance (not necessarily ADK's):
# store = make_store("postgresql://user:pw@host:5432/llm_costs")


# ── 2. Google ADK agent ────────────────────────────────────────────────────
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

async def run_adk_agent():
    agent = LlmAgent(
        name="search_agent",
        model="gemini-2.5-flash",
        instruction="Be concise.",
    )
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="demo",
        session_service=session_service,
        plugins=[CostTrackerPlugin(store=store, app_name="demo", verbose=True)],
    )
    session = await session_service.create_session(app_name="demo", user_id="u1")
    msg = Content(role="user", parts=[Part(text="What is 2+2?")])
    async for event in runner.run_async(
        user_id="u1", session_id=session.id, new_message=msg
    ):
        if event.is_final_response() and event.content:
            print(f"[ADK] {event.content.parts[0].text}")


# ── 3. OpenAI client ───────────────────────────────────────────────────────
from adk_cost_tracker.trackers.openai_tracker import track_openai

def run_openai():
    try:
        import openai
    except ImportError:
        print("[OpenAI] openai package not installed — skipping")
        return

    raw_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    client = track_openai(
        raw_client,
        store=store,
        app_name="demo",
        agent="openai-chat",
        tags={"env": "dev"},
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Name one planet."}],
    )
    print(f"[OpenAI] {resp.choices[0].message.content}")


# ── 4. AWS Bedrock client ──────────────────────────────────────────────────
from adk_cost_tracker.trackers.bedrock_tracker import track_bedrock

def run_bedrock():
    try:
        import boto3
    except ImportError:
        print("[Bedrock] boto3 not installed — skipping")
        return

    raw_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    client = track_bedrock(
        raw_client,
        store=store,
        app_name="demo",
        agent="bedrock-nova",
        tags={"env": "dev"},
    )
    resp = client.converse(
        modelId="amazon.nova-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": "Name one planet."}]}],
    )
    text = resp["output"]["message"]["content"][0]["text"]
    print(f"[Bedrock] {text}")


# ── 5. Run all, then print unified report ─────────────────────────────────
async def main():
    print("Running ADK agent...")
    try:
        await run_adk_agent()
    except Exception as e:
        print(f"[ADK] skipped: {e}")

    print("\nRunning OpenAI...")
    run_openai()

    print("\nRunning Bedrock...")
    run_bedrock()

    print("\n" + "=" * 60)
    print_summary(store)   # unified view across all three providers


if __name__ == "__main__":
    asyncio.run(main())
