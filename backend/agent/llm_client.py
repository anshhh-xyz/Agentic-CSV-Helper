"""
llm_client.py -- thin wrapper around the Groq SDK (native tool calling).

Groq is the LLM provider. The SDK already retries transient errors and
honours Retry-After on 429s; this module turns whatever is left into a small
set of exceptions the orchestrator can handle with friendly messages.
"""

from __future__ import annotations

import logging
import os

import groq
from groq import Groq

from agent.config import groq_model

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Message is safe to show to the user."""


class ToolCallFormatError(LLMError):
    """The model produced a malformed tool call (Groq: tool_use_failed)."""


class LLMRateLimitError(LLMError):
    pass


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set on the server. Get a free key at https://console.groq.com "
            "and put it in backend/.env (see .env.example)."
        )
    return Groq(api_key=api_key)


def _is_tool_use_failed(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("code") == "tool_use_failed":
            return True
    return "tool_use_failed" in str(exc)


def chat_completion(client, messages: list, tools: list = None, model: str = None, tool_choice: str = "auto"):
    """One call to the LLM. tool_choice: "auto" (default) or "none" (force a text answer)."""
    kwargs = {"model": model or groq_model(), "messages": messages, "temperature": 0.1}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    try:
        return client.chat.completions.create(**kwargs)
    except groq.BadRequestError as e:
        if _is_tool_use_failed(e):
            raise ToolCallFormatError("The model produced a malformed tool call.") from e
        logger.error("Groq rejected the request: %s", e)
        raise LLMError("The AI service rejected the request. Try rephrasing your question.") from e
    except groq.RateLimitError as e:
        logger.warning("Groq rate limit: %s", e)
        raise LLMRateLimitError("The AI service is rate-limited right now. Wait a few seconds and try again.") from e
    except groq.AuthenticationError as e:
        raise LLMError("The Groq API key was rejected. Check GROQ_API_KEY on the server.") from e
    except groq.APIConnectionError as e:
        raise LLMError("Could not reach the Groq API. Check the server's internet connection.") from e
    except groq.APIStatusError as e:
        logger.error("Groq API error %s: %s", e.status_code, e)
        raise LLMError(f"The AI service returned an error ({e.status_code}). Please try again.") from e
