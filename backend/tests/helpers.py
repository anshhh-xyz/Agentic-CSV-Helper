"""Shared test helpers: a scripted fake LLM and a Groq SDK client on a mock HTTP transport."""

import json
import logging
import os
import sys
import uuid
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402
from groq import Groq  # noqa: E402

logging.disable(logging.CRITICAL)  # keep test output readable
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(BASE, "data", "sample_sales.csv")


def tc(name, args, call_id=None):
    """A tool call as returned by the SDK (function.arguments is a JSON string)."""
    raw = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=call_id or f"call_{uuid.uuid4().hex[:8]}", type="function",
                           function=SimpleNamespace(name=name, arguments=raw))


def msg(content=None, calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


class FakeClient:
    """Mimics client.chat.completions.create(); `policy(messages, kwargs)` returns a message
    (or raises). Every request is recorded in .requests."""

    def __init__(self, policy):
        self.policy = policy
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        out = self.policy(kwargs["messages"], kwargs)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(choices=[SimpleNamespace(message=out)])


def tool_results(messages):
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def completion_json(content=None, tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                                 for c in tool_calls]
    return {"id": "cmpl-test", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def groq_on_mock(handler):
    """A real Groq SDK client whose HTTP layer is replaced by `handler(request)->httpx.Response`."""
    return Groq(api_key="test-key", base_url="https://groq.invalid",
                http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)
