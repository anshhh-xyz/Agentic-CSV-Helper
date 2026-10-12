"""
config.py -- paths and env-driven settings. Getters are lazy so a .env file
loaded after import (e.g. in cli.py) is still honoured.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
MEMORY_DB_PATH = os.path.join(BASE_DIR, "data", "memory.db")


def groq_model() -> str:
    return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def max_tool_rounds() -> int:
    """Max LLM turns that may request tools before a final answer is forced."""
    return int(os.environ.get("MAX_TOOL_ROUNDS", "8"))


def max_parallel_workers() -> int:
    return int(os.environ.get("MAX_PARALLEL_TOOLS", "4"))
