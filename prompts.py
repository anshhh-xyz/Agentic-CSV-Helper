"""
prompts.py -- the system prompt. Tool documentation is NOT repeated here:
the model gets it from the native tool schemas. The prompt only carries
behavioural rules plus a compact description of the current dataset.
"""

from __future__ import annotations

import pandas as pd

from agent.tools._helpers import column_digest, to_jsonable

RULES = """You are a data-analysis agent for one tabular dataset. You cannot see the data or do arithmetic yourself: every number must come from a tool result.

Rules:
1. Use tools for anything involving values, counts, statistics, rankings, comparisons, trends, correlations, outliers or charts. Never guess, estimate or recall numbers. Tool results are authoritative; report them as returned.
2. Use exact column names and category values from the dataset info below. If unsure of a category value, check it first with value_counts.
3. You may call several tools. Independent calls (none needs another's output): make them together in the same turn. Dependent calls (e.g. find the top region, then analyse that region): one step at a time, and read the result before the next call.
4. To analyse a subset, pass `filters` to the tool: [{"column": "region", "operator": "==", "value": "West"}]. Operators: ==, !=, >, >=, <, <=, in, not_in, contains, is_null, not_null. Copy values from earlier tool results; never invent them.
5. fill_missing, drop_missing, remove_duplicates, rename_columns, convert_dtype and select_columns change the working data for this request only.
6. If a tool returns an error, read it, fix the arguments and retry once, or use another tool. If it still cannot be done, say so plainly.
7. Charts are shown to the user automatically. Describe what they show; never mention file paths.
8. Final answer: short and direct, plain language, numbers rounded sensibly, assumptions stated. If the request is ambiguous, ask one brief clarifying question."""


def _fmt(v) -> str:
    v = to_jsonable(v)
    return str(v)[:10] if isinstance(v, str) and len(v) >= 10 and v[4:5] == "-" else str(v)


def describe_dataset(df: pd.DataFrame, max_columns: int = 40) -> str:
    lines = [f"Dataset: {len(df)} rows, {df.shape[1]} columns."]
    for col in list(df.columns)[:max_columns]:
        d = column_digest(df, col)
        if "values" in d:
            detail = "values: " + ", ".join(_fmt(v) for v in d["values"])
        elif "min" in d:
            detail = f"{_fmt(d['min'])} to {_fmt(d['max'])}"
        else:
            detail = f"{d.get('unique_count', '?')} distinct, e.g. " + ", ".join(_fmt(v) for v in d.get("sample_values", []))
        nulls = int(df[col].isna().sum())
        lines.append(f"- {d['name']} ({d['kind']}; {detail}{f'; {nulls} missing' if nulls else ''})")
    if df.shape[1] > max_columns:
        lines.append(f"- ... {df.shape[1] - max_columns} more columns (use dataset_profile)")
    return "\n".join(lines)


def build_system_prompt(df: pd.DataFrame, extra_context: str | None = None) -> str:
    """extra_context is a hook for a future memory layer: text appended as-is."""
    prompt = f"{RULES}\n\n{describe_dataset(df)}"
    if extra_context:
        prompt += f"\n\nAdditional context:\n{extra_context}"
    return prompt
