"""
memory.py -- lets the agent remember small facts about a dataset across
questions: term definitions, corrections, preferences, named filters.

NOT a cache of data or conversation history -- just a short list
of things the user explicitly stated that should change future answers.
Storage is agent/memory/store.py (plain SQLite, one row per fact).
"""

from agent.memory import store
from agent.tools._base import ToolError, obj, p_str, tool

CATEGORY = "memory"

@tool(
    name="remember",
    description=(
        "PERSISTENT MEMORY WRITE: You MUST call this tool whenever the "
        "user explicitly asks to remember, save, or store a definition, "
        "correction, standing preference, or named filter for future "
        "questions. Saving requires executing this tool; saying 'I will "
        "remember' is not sufficient. Use a short stable key, preserve "
        "the user's intended meaning, and choose the appropriate kind. "
        "Do not use for temporary facts or one-off analysis results."
        
    ),

    category=CATEGORY,
    parameters=obj(
        {
            "key": p_str("Short identifier for this fact, e.g. 'active_customers' or 'summary_frequency'."),
            "value": p_str("The fact itself, in plain words (e.g. 'excludes cancelled accounts')."),
            "kind": p_str("Type of fact.", ["definition", "correction", "preference", "named_filter"]),
        },
        ["key", "value", "kind"],
    ),
)
def remember(ctx, key, value, kind):
    key = (key or "").strip()
    if not key:
        raise ToolError("key cannot be empty.")
    store.upsert(ctx.scope, kind, key, value)
    return {"remembered": key, "kind": kind, "value": value}

@tool(
    name="forget",
    description="Remove a previously remembered fact by its key.",
    category=CATEGORY,
    parameters=obj({"key": p_str("The key to forget.")}, ["key"]),
)
def forget(ctx, key):
    key = (key or "").strip()
    if not key:
        raise ToolError("key cannot be empty.")
    removed = store.delete(ctx.scope, key)
    if not removed:
        raise ToolError(f"Nothing is remembered under '{key}'.")
    return {"forgotten": key}

@tool(
    name="list_memory",
    description="List all facts currently remembered for this dataset.",
    category=CATEGORY,
    parameters=obj({}),
)
def list_memory(ctx):
    rows = store.get_all(ctx.scope)
    return {"count": len(rows), "memory": rows}




