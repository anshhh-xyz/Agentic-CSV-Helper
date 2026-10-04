"""
tool_registry.py

Bridge between the LLM and the tool functions:

  get_tool_schemas()  -> Groq/OpenAI `tools` list (native function calling)
  dispatch(name, args, ctx)
        -> validates the tool name and arguments against the tool's own
           JSON schema, runs it, and ALWAYS returns a JSON-safe dict
           ({"error": ...} on any problem, never a traceback).

Only functions registered with @tool can run; the model cannot execute
arbitrary code.
"""

from __future__ import annotations

import json
import logging

import agent.tools  # noqa: F401  (registers all tools)
from agent.tools._base import Tool, ToolContext, ToolError, all_tools, get_tool, suggest_tool_names
from agent.tools._helpers import normalize_filters, to_jsonable

logger = logging.getLogger(__name__)

__all__ = ["get_tool_schemas", "dispatch", "validate_args", "list_tools", "get_tool", "ToolContext"]


def get_tool_schemas() -> list:
    return [
        {"type": "function",
         "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in all_tools()
    ]


def list_tools() -> list:
    return [{"name": t.name, "category": t.category, "description": t.description,
             "mutates_data": t.mutates, "parameters": list(t.parameters["properties"])}
            for t in all_tools()]


# ------------------------------ argument validation ------------------------------

def _coerce(schema: dict, value, path: str):
    """Coerce/validate one value against a (small) JSON-schema subset.
    Lenient about common LLM slips: numbers sent as strings, scalars where a
    list is expected, enum case. Raises ValueError with a readable message."""
    t = schema.get("type")
    if t is None:
        return value

    if t == "string":
        if isinstance(value, (dict, list, bool)):
            raise ValueError(f"'{path}' must be a string")
        v = str(value)
        enum = schema.get("enum")
        if enum:
            match = next((e for e in enum if e.lower() == v.strip().lower()), None)
            if match is None:
                raise ValueError(f"'{path}' must be one of {enum}, got '{v}'")
            v = match
        return v

    if t in ("integer", "number"):
        if isinstance(value, bool):
            raise ValueError(f"'{path}' must be a {t}")
        try:
            num = float(str(value).replace(",", "")) if not isinstance(value, (int, float)) else float(value)
        except ValueError:
            raise ValueError(f"'{path}' must be a {t}, got '{value}'")
        if t == "integer":
            if not num.is_integer():
                raise ValueError(f"'{path}' must be a whole number, got {value}")
            num = int(num)
        elif num.is_integer() and isinstance(value, int):
            num = value
        if "minimum" in schema and num < schema["minimum"]:
            raise ValueError(f"'{path}' must be >= {schema['minimum']}")
        if "maximum" in schema and num > schema["maximum"]:
            raise ValueError(f"'{path}' must be <= {schema['maximum']}")
        return num

    if t == "boolean":
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in ("true", "yes", "1"):
            return True
        if str(value).strip().lower() in ("false", "no", "0"):
            return False
        raise ValueError(f"'{path}' must be true or false")

    if t == "array":
        if isinstance(value, str):
            text = value.strip()
            value = json.loads(text) if text.startswith("[") else [value]
        elif not isinstance(value, (list, tuple)):
            value = [value]
        return [_coerce(schema.get("items", {}), v, f"{path}[{i}]") for i, v in enumerate(value)]

    if t == "object":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                raise ValueError(f"'{path}' must be an object")
        if not isinstance(value, dict):
            raise ValueError(f"'{path}' must be an object")
        props = schema.get("properties")
        if props is None:
            extra = schema.get("additionalProperties")
            return {k: _coerce(extra, v, f"{path}.{k}") for k, v in value.items()} if extra else value
        for req in schema.get("required", []):
            if req not in value:
                raise ValueError(f"'{path}.{req}' is required")
        return {k: (_coerce(props[k], v, f"{path}.{k}") if k in props else v) for k, v in value.items()}

    return value


def validate_args(tool: Tool, args) -> tuple:
    """Returns (clean_args, error_message_or_None)."""
    if not isinstance(args, dict):
        return None, "arguments must be a JSON object"
    schema = tool.parameters
    props, required = schema["properties"], schema.get("required", [])

    unknown = [k for k in args if k not in props]
    if unknown:
        return None, f"unknown parameter(s) {unknown}. Valid parameters: {list(props)}"

    clean = {}
    try:
        for key, value in args.items():
            if value is None:
                continue  # treat null as "not provided"
            if key == "filters":
                clean[key] = normalize_filters(value)
            else:
                clean[key] = _coerce(props[key], value, key)
    except ToolError as e:
        return None, str(e)
    except (ValueError, TypeError) as e:
        return None, str(e)

    missing = [r for r in required if r not in clean]
    if missing:
        return None, f"missing required parameter(s) {missing}. Valid parameters: {list(props)}"
    return clean, None


# ------------------------------------ dispatch ------------------------------------

def dispatch(name: str, args, ctx: ToolContext) -> dict:
    tool = get_tool(name)
    if tool is None:
        hint = suggest_tool_names(name)
        return {"error": f"Unknown tool '{name}'." + (f" Did you mean: {', '.join(hint)}?" if hint else "")
                + " Only the provided tools can be used."}

    clean, err = validate_args(tool, args if args is not None else {})
    if err:
        return {"error": f"Invalid arguments for '{name}': {err}"}

    try:
        result = tool.func(ctx, **clean)
    except ToolError as e:
        return {"error": str(e)}
    except Exception:
        logger.exception("Tool '%s' failed unexpectedly (args=%s)", name, clean)
        return {"error": f"'{name}' could not process these arguments. Check column names and types, "
                         "or try a different tool."}

    if not isinstance(result, dict):
        result = {"value": result}
    return to_jsonable(result)
