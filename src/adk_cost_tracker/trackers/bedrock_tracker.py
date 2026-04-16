"""
AWS Bedrock cost tracker — wraps a boto3 bedrock-runtime client.
Uses the Converse API which returns normalised usage across all models.

Usage:
    import boto3
    from adk_cost_tracker.trackers.bedrock_tracker import track_bedrock

    boto_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    client = track_bedrock(boto_client, store=store, app_name="my_app")

    # Use exactly like the real boto3 client:
    resp = client.converse(
        modelId="amazon.nova-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": "Hello"}]}],
    )

Note: Only converse() and converse_stream() are intercepted.
      invoke_model() is model-specific (no unified usage schema).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from ..pricing import calculate_cost, get_price
from ..store import BaseStore, CallRecord

logger = logging.getLogger(__name__)


def _parse_model_id(model_id: str) -> str:
    """
    Bedrock model IDs look like:
      "amazon.nova-lite-v1:0"
      "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
      "meta.llama3-1-70b-instruct-v1:0"

    Strip region prefix and version suffix so pricing lookup works.
    """
    # Drop cross-region prefix (us., eu., ap.)
    for prefix in ("us.", "eu.", "ap."):
        if model_id.startswith(prefix):
            model_id = model_id[len(prefix):]
    # Drop version suffix after the colon
    model_id = model_id.split(":")[0]
    return model_id.lower()


def _make_record(
    response: dict,
    model_id: str,
    app_name: str,
    agent: str,
    tags: str,
) -> CallRecord | None:
    """Extract usage from a Bedrock Converse response dict."""
    usage = response.get("usage", {})
    if not usage:
        return None

    input_tokens  = usage.get("inputTokens", 0) or 0
    output_tokens = usage.get("outputTokens", 0) or 0
    # Bedrock doesn't report cached tokens in usage block — may be added later
    cached_tokens = 0

    clean_model = _parse_model_id(model_id)
    cost  = calculate_cost(clean_model, input_tokens, output_tokens, cached_tokens)
    price = get_price(clean_model)

    return CallRecord(
        ts=datetime.now(timezone.utc).isoformat(),
        provider=price.provider if price.provider != "unknown" else "bedrock",
        agent=agent,
        model=clean_model,
        session_id="",
        app_name=app_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cost_usd=cost,
        tags=tags,
        meta={
            "raw_model_id": model_id,
            "stop_reason": response.get("stopReason", ""),
        },
    )


class TrackedBedrockClient:
    """
    Thin proxy around a boto3 bedrock-runtime client.
    Intercepts converse() and converse_stream() to record cost.
    All other methods pass through unchanged.
    """

    def __init__(
        self,
        client: Any,
        store: BaseStore,
        app_name: str = "",
        agent: str = "bedrock-agent",
        tags: dict[str, str] | None = None,
    ):
        self._client = client
        self._store = store
        self._app_name = app_name
        self._agent = agent
        self._tags = ",".join(f"{k}={v}" for k, v in (tags or {}).items())

    def converse(self, modelId: str, **kwargs) -> dict:
        response = self._client.converse(modelId=modelId, **kwargs)
        record = _make_record(
            response, modelId, self._app_name, self._agent, self._tags
        )
        if record:
            self._fire_insert(record)
        return response

    def converse_stream(self, modelId: str, **kwargs):
        """
        Streams response chunks. Usage appears in the final metadata event.
        We buffer and record after the stream is fully consumed.
        """
        response = self._client.converse_stream(modelId=modelId, **kwargs)
        return _UsageCapturingStream(
            response, modelId, self._app_name, self._agent,
            self._tags, self._store
        )

    def _fire_insert(self, record: CallRecord):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_safe_insert(self._store, record))
            else:
                loop.run_until_complete(self._store.insert(record))
        except Exception as exc:
            logger.warning("[CostTracker/Bedrock] insert failed: %s", exc)

    def __getattr__(self, name):
        return getattr(self._client, name)


class _UsageCapturingStream:
    """
    Wraps the boto3 streaming response for converse_stream().
    Captures the 'metadata' event that carries usage at end of stream.
    """

    def __init__(self, raw_response, model_id, app_name, agent, tags, store):
        self._raw = raw_response
        self._model_id = model_id
        self._app_name = app_name
        self._agent = agent
        self._tags = tags
        self._store = store

    def __iter__(self):
        usage_event = {}
        stream = self._raw.get("stream", [])
        for event in stream:
            yield event
            if "metadata" in event and "usage" in event["metadata"]:
                usage_event = {
                    "usage": event["metadata"]["usage"],
                    "stopReason": event.get("messageStop", {}).get("stopReason", ""),
                }
        if usage_event:
            record = _make_record(
                usage_event, self._model_id,
                self._app_name, self._agent, self._tags
            )
            if record:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(_safe_insert(self._store, record))
                    else:
                        loop.run_until_complete(self._store.insert(record))
                except Exception as exc:
                    logger.warning("[CostTracker/Bedrock] stream insert failed: %s", exc)

    def __getattr__(self, name):
        return getattr(self._raw, name)


def track_bedrock(
    client: Any,
    store: BaseStore,
    app_name: str = "",
    agent: str = "bedrock-agent",
    tags: dict[str, str] | None = None,
) -> TrackedBedrockClient:
    """
    Wrap a boto3 bedrock-runtime client with cost tracking.

        boto_client = boto3.client("bedrock-runtime", region_name="us-east-1")
        client = track_bedrock(boto_client, store=store, app_name="rag_pipeline")
    """
    return TrackedBedrockClient(client, store, app_name, agent, tags)


async def _safe_insert(store: BaseStore, record: CallRecord):
    try:
        await store.insert(record)
    except Exception as exc:
        logger.warning("[CostTracker/Bedrock] store insert failed: %s", exc)
