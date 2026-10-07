"""
orchestrator.py -- the agent loop.

    user question
      -> Groq picks tool(s) via native function calling
      -> executor runs them (independent calls in parallel, dependent ones
         across turns) on the real DataFrame
      -> results go back to Groq, which calls more tools or answers

Groq never computes anything: it chooses tools, reads their results, and
writes the final answer. Tool results are the only source of numbers.

run_agent() returns
    {
      "answer":     str,
      "tool_trace": [ {round, tool, category, arguments, ok, error, summary,
                       duration_ms, parallel}, ... ],   # in execution order
      "plot_files": [absolute paths],
      "rounds":     int,
    }
"""

from __future__ import annotations

import json
import logging
import uuid

import pandas as pd

from agent import tool_registry
from agent.config import PLOTS_DIR, max_tool_rounds
from agent.executor import Outcome, execute_calls
from agent.llm_client import LLMError, ToolCallFormatError, chat_completion, get_client
from agent.prompts import build_system_prompt
from agent.tools._base import ToolContext, get_tool

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 24          # hard cap on tool calls per question
MAX_FORMAT_RETRIES = 2       # retries after a malformed tool call from the model
MAX_RESULT_CHARS = 6000      # cap on one tool result fed back to the model

FORMAT_NUDGE = ("Your last tool call was malformed. Call the tool again with valid JSON arguments "
                "that match its schema exactly, or answer without tools if none is needed.")
FINAL_NUDGE = ("Tool budget reached. Answer the user now using ONLY the tool results above. "
               "If something is missing, say what is missing.")
FALLBACK_ANSWER = "I couldn't put together an answer for that. Try rephrasing or asking a narrower question."


# ------------------------------------ helpers ------------------------------------

def _extract_calls(message) -> list[dict]:
    calls = []
    for tc in getattr(message, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        if fn is None or not getattr(fn, "name", None):
            continue
        args = getattr(fn, "arguments", None)
        calls.append({
            "id": getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:12]}",
            "name": fn.name,
            "arguments": args if isinstance(args, str) else json.dumps(args or {}),
        })
    return calls


def _summarize(result: dict) -> str:
    """One short line describing a tool result, for the UI trace."""
    if "error" in result:
        return str(result["error"])[:160]
    if "chart" in result:
        return f"{result['chart']} chart: {result.get('title', '')}"[:120]
    if "value" in result and not isinstance(result["value"], (list, dict)):
        return f"{result.get('statistic', 'value')}({result.get('column', '')}) = {result['value']}"
    if isinstance(result.get("rows"), list):
        return f"{len(result['rows'])} rows"
    if "match_count" in result:
        return f"{result['match_count']} matching rows"
    if "trend_direction" in result:
        return f"{result['trend_direction']} over {result.get('n_periods', '?')} periods"
    return "done"


def _feedback(outcome: Outcome, plot_files: list) -> str:
    """The JSON the model sees for a tool result (file paths stay server-side)."""
    result = dict(outcome.result)
    path = result.pop("file_path", None)
    if path:
        plot_files.append(path)
        result["chart_created"] = "The chart is shown to the user automatically."
    text = json.dumps(result, default=str)
    if len(text) > MAX_RESULT_CHARS:
        text = json.dumps({"warning": "Result too large and was cut off. Narrow it with filters or top_n.",
                           "partial": text[:MAX_RESULT_CHARS]})
    return text


def _trace_entry(round_no: int, o: Outcome) -> dict:
    tool = get_tool(o.name)
    return {
        "round": round_no, "tool": o.name, "category": tool.category if tool else None,
        "arguments": o.arguments, "ok": o.ok,
        "error": o.result.get("error") if not o.ok else None,
        "summary": _summarize(o.result), "duration_ms": o.duration_ms, "parallel": o.parallel,
    }


def _force_final(client, messages, tools) -> str:
    """Ask for a plain-text answer without allowing more tool calls."""
    try:
        response = chat_completion(client, messages + [{"role": "user", "content": FINAL_NUDGE}],
                                   tools=tools, tool_choice="none")
        return ((response.choices[0].message.content or "").strip()) or FALLBACK_ANSWER
    except (LLMError, IndexError, AttributeError) as e:
        logger.warning("Forced final answer failed: %s", e)
        return FALLBACK_ANSWER


# ------------------------------------- main -------------------------------------

def run_agent(df: pd.DataFrame, user_question: str, verbose: bool = False,
              client=None, extra_context: str | None = None) -> dict:
    """`client` is injectable for tests; `extra_context` is where a future
    memory layer can add text to the system prompt."""
    client = client or get_client()
    tools = tool_registry.get_tool_schemas()
    ctx = ToolContext(df=df, plots_dir=PLOTS_DIR)   # cleaning tools swap ctx.df; `df` is never mutated

    messages = [
        {"role": "system", "content": build_system_prompt(df, extra_context)},
        {"role": "user", "content": user_question},
    ]
    trace: list[dict] = []
    plot_files: list[str] = []
    calls_made = 0
    format_retries = 0

    def result(answer: str, rounds: int) -> dict:
        return {"answer": answer, "tool_trace": trace, "plot_files": plot_files, "rounds": rounds}

    for round_no in range(1, max_tool_rounds() + 1):
        try:
            response = chat_completion(client, messages, tools=tools)
        except ToolCallFormatError:
            format_retries += 1
            if format_retries > MAX_FORMAT_RETRIES:
                return result("I had trouble forming a valid analysis step for that request. "
                              "Try rephrasing it or splitting it into smaller questions.", round_no)
            messages.append({"role": "user", "content": FORMAT_NUDGE})
            continue

        try:
            message = response.choices[0].message
        except (IndexError, AttributeError, TypeError):
            raise LLMError("The AI service returned an unexpected response. Please try again.")

        calls = _extract_calls(message)
        if not calls:
            answer = (getattr(message, "content", None) or "").strip()
            if not answer:
                answer = _force_final(client, messages, tools) if trace else FALLBACK_ANSWER
            return result(answer, round_no)

        remaining = MAX_TOOL_CALLS - calls_made
        if remaining <= 0:
            break
        calls = calls[:remaining]
        calls_made += len(calls)

        messages.append({
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls],
        })

        for outcome in execute_calls(calls, ctx):
            trace.append(_trace_entry(round_no, outcome))
            if verbose:
                mark = "ok " if outcome.ok else "ERR"
                par = " [parallel]" if outcome.parallel else ""
                print(f"  [{mark}] round {round_no}: {outcome.name}({json.dumps(outcome.arguments)[:100]}){par}")
            messages.append({"role": "tool", "tool_call_id": outcome.call_id, "name": outcome.name,
                             "content": _feedback(outcome, plot_files)})

    return result(_force_final(client, messages, tools), max_tool_rounds())
