"""
_base.py -- the tool framework every tools module builds on.

A tool is a plain function `fn(ctx, **args) -> dict` registered with the
@tool decorator, which records its name, description and JSON-schema
parameters. The LLM sees each function as its own tool; the modules
(mathematical_operations.py, graphs.py, ...) are only for organisation.

Conventions (kept identical across all tools):
  * `ctx` is a ToolContext holding the working DataFrame for this request.
  * Tools that analyse a subset accept an optional `filters` argument.
  * Tools return JSON-friendly dicts; expected problems raise ToolError and
    become an {"error": ...} result the model can read and recover from.
  * `mutates=True` marks tools that replace ctx.df (cleaning tools). The
    executor never runs those in parallel with anything else.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd


class ToolError(Exception):
    """A predictable, fixable problem. The message is shown to the LLM."""


@dataclass
class ToolContext:
    df: pd.DataFrame  # working copy for this request (never the stored dataset)
    plots_dir: str
    scope: str = "global"  # dataset_id for this request; used by memory tools


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable
    category: str
    mutates: bool = False


_REGISTRY: dict[str, Tool] = {}


def tool(*, name: str, description: str, parameters: dict, category: str, mutates: bool = False):
    def decorator(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"Duplicate tool name: {name}")
        _REGISTRY[name] = Tool(name, description, parameters, fn, category, mutates)
        return fn

    return decorator


def get_tool(name: str) -> Optional[Tool]:
    return _REGISTRY.get(name)


def all_tools() -> list[Tool]:
    return list(_REGISTRY.values())


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def suggest_tool_names(name: str) -> list[str]:
    return difflib.get_close_matches(str(name), list(_REGISTRY), n=3, cutoff=0.5)


# ---- JSON-schema builders (keep every tool's schema shape consistent) ----

def obj(props: dict, required=()) -> dict:
    schema = {"type": "object", "properties": props}
    if required:
        schema["required"] = list(required)
    return schema


def p_str(desc: str, enum=None) -> dict:
    d = {"type": "string", "description": desc}
    if enum:
        d["enum"] = list(enum)
    return d


def p_num(desc: str, minimum=None, maximum=None) -> dict:
    d = {"type": "number", "description": desc}
    if minimum is not None:
        d["minimum"] = minimum
    if maximum is not None:
        d["maximum"] = maximum
    return d


def p_int(desc: str, minimum=None, maximum=None) -> dict:
    d = {"type": "integer", "description": desc}
    if minimum is not None:
        d["minimum"] = minimum
    if maximum is not None:
        d["maximum"] = maximum
    return d


def p_bool(desc: str) -> dict:
    return {"type": "boolean", "description": desc}


def p_list(items: dict, desc: str) -> dict:
    return {"type": "array", "items": items, "description": desc}
