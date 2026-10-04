"""
executor.py -- runs the tool calls the model requested in ONE turn.

How dependencies are handled:
  * Tool calls the model emits together in a single turn are, by contract,
    independent (the prompt tells it to emit dependent steps one turn at a
    time, after reading earlier results). Those run in PARALLEL on a thread
    pool.
  * Dependent steps therefore arrive in later turns -> run SEQUENTIALLY,
    each seeing the previous result via the conversation.
  * Safety net: tools that change the working data (mutates=True) never run
    concurrently with anything else. They act as barriers, so calls before
    them finish first and calls after them see the new data.

Results come back in the same order as the calls (required by the API).
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from agent import tool_registry
from agent.config import max_parallel_workers
from agent.tools._base import ToolContext, get_tool

logger = logging.getLogger(__name__)


@dataclass
class Outcome:
    call_id: str
    name: str
    arguments: dict
    result: dict
    ok: bool
    duration_ms: int
    parallel: bool


def _parse_args(raw):
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    try:
        args = json.loads(raw)
    except (TypeError, ValueError):
        return None, "Tool arguments were not valid JSON. Send a JSON object that matches the tool's schema."
    if not isinstance(args, dict):
        return None, "Tool arguments must be a JSON object."
    return args, None


def _run_one(call: dict, ctx: ToolContext, parallel: bool) -> Outcome:
    started = time.perf_counter()
    args, err = _parse_args(call.get("arguments"))
    result = {"error": err} if err else tool_registry.dispatch(call["name"], args, ctx)
    ms = int((time.perf_counter() - started) * 1000)
    return Outcome(call["id"], call["name"], args or {}, result, "error" not in result, ms, parallel)


def plan_batches(calls: list[dict]) -> list[list[dict]]:
    """Group consecutive read-only calls; data-mutating calls run alone."""
    batches, current = [], []
    for call in calls:
        tool = get_tool(call["name"])
        if tool is not None and tool.mutates:
            if current:
                batches.append(current)
                current = []
            batches.append([call])
        else:
            current.append(call)
    if current:
        batches.append(current)
    return batches


def execute_calls(calls: list[dict], ctx: ToolContext) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for batch in plan_batches(calls):
        if len(batch) == 1:
            outcomes.append(_run_one(batch[0], ctx, parallel=False))
            continue
        workers = max(1, min(max_parallel_workers(), len(batch)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes.extend(pool.map(lambda c: _run_one(c, ctx, True), batch))
    return outcomes
