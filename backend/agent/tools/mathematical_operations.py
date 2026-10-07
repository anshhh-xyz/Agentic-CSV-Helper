"""
mathematical_operations.py

Tools: mean, median, sum, min, max, std, variance, count, quantile,
       group_aggregate

All numbers are computed with pandas; the LLM only picks the tool.
"""

from __future__ import annotations

import math

import pandas as pd

from agent.tools._base import ToolError, obj, p_bool, p_int, p_list, p_num, p_str, tool
from agent.tools._helpers import FILTERS, need_col, records, scope, subset

CATEGORY = "mathematical_operations"

# name, description, allowed column kind, function over a non-null Series
_SINGLE_STATS = [
    ("mean", "Average of a numeric column.", "numeric", lambda s: s.mean()),
    ("median", "Median (middle value) of a numeric column.", "numeric", lambda s: s.median()),
    ("sum", "Total of a numeric column.", "numeric", lambda s: s.sum()),
    ("min", "Smallest value of a numeric or date column.", "numeric_or_datetime", lambda s: s.min()),
    ("max", "Largest value of a numeric or date column.", "numeric_or_datetime", lambda s: s.max()),
    ("std", "Sample standard deviation of a numeric column.", "numeric", lambda s: s.std()),
    ("variance", "Sample variance of a numeric column.", "numeric", lambda s: s.var()),
]


def _register_single_stat(name, description, kind, fn):
    @tool(
        name=name,
        description=description,
        category=CATEGORY,
        parameters=obj({"column": p_str("Column name."), "filters": FILTERS}, ["column"]),
    )
    def _run(ctx, column, filters=None):
        col = need_col(ctx.df, column, kind)
        df, conds = subset(ctx, filters)
        s = df[col].dropna()
        if s.empty:
            raise ToolError(f"'{col}' has no non-null values for the selected rows.")
        value = fn(s)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ToolError(f"Cannot compute {name} of '{col}' (needs at least 2 values).")
        return {"statistic": name, "column": col, "value": value,
                "non_null_count": int(len(s)), **scope(df, conds)}

    return _run


for _spec in _SINGLE_STATS:
    _register_single_stat(*_spec)


@tool(
    name="count",
    description="Count rows, non-null values of a column, or distinct values of a column.",
    category=CATEGORY,
    parameters=obj({
        "column": p_str("Column to count. Omit to count rows."),
        "distinct": p_bool("Count distinct values instead of non-null values."),
        "filters": FILTERS,
    }),
)
def count(ctx, column=None, distinct=False, filters=None):
    df, conds = subset(ctx, filters, allow_empty=True)  # 0 matches is a valid answer
    if column is None:
        return {"statistic": "row_count", "value": int(len(df)), **scope(df, conds)}
    col = need_col(ctx.df, column)
    if distinct:
        return {"statistic": "distinct_count", "column": col,
                "value": int(df[col].nunique(dropna=True)), **scope(df, conds)}
    return {"statistic": "non_null_count", "column": col,
            "value": int(df[col].notna().sum()), **scope(df, conds)}


@tool(
    name="quantile",
    description="Quantiles/percentiles of a numeric column (q between 0 and 1, e.g. 0.9 = 90th percentile).",
    category=CATEGORY,
    parameters=obj({
        "column": p_str("Numeric column."),
        "q": p_list({"type": "number"}, "Quantiles, e.g. [0.25, 0.5, 0.75]."),
        "filters": FILTERS,
    }, ["column", "q"]),
)
def quantile(ctx, column, q, filters=None):
    col = need_col(ctx.df, column, "numeric")
    df, conds = subset(ctx, filters)
    qs = [float(x) for x in q]
    if not qs:
        raise ToolError("Give at least one quantile, e.g. q=[0.5].")
    if any(x > 1 for x in qs):  # tolerate percentages like 90
        qs = [x / 100 for x in qs]
    if any(x < 0 or x > 1 for x in qs):
        raise ToolError("Quantiles must be between 0 and 1 (or 0-100 as percentages).")
    s = df[col].dropna()
    if s.empty:
        raise ToolError(f"'{col}' has no non-null values for the selected rows.")
    values = {f"p{round(x * 100, 2):g}": s.quantile(x) for x in qs}
    return {"column": col, "percentiles": values, "non_null_count": int(len(s)), **scope(df, conds)}


AGGS = ["mean", "median", "sum", "min", "max", "count", "std", "variance", "nunique"]
_PANDAS_NAME = {"variance": "var"}


@tool(
    name="group_aggregate",
    description=("Group rows by one or more columns and aggregate a column per group "
                 "(e.g. average revenue by region). Rows come back sorted by value."),
    category=CATEGORY,
    parameters=obj({
        "group_by": p_list({"type": "string"}, "Column(s) to group by."),
        "agg": p_str("Aggregation to apply.", AGGS),
        "column": p_str("Column to aggregate. Optional only when agg='count' (counts rows)."),
        "sort": p_str("Sort order of the result by value. Default desc.", ["desc", "asc"]),
        "top_n": p_int("Return only the first N groups after sorting.", minimum=1),
        "filters": FILTERS,
    }, ["group_by", "agg"]),
)
def group_aggregate(ctx, group_by, agg, column=None, sort="desc", top_n=None, filters=None):
    if not group_by:
        raise ToolError("group_by needs at least one column.")
    gcols = [need_col(ctx.df, g) for g in group_by]

    col = None
    if not (agg == "count" and column is None):
        if column is None:
            raise ToolError(f"agg='{agg}' needs a 'column' to aggregate.")
        kind = ("any" if agg in ("count", "nunique")
                else "numeric_or_datetime" if agg in ("min", "max") else "numeric")
        col = need_col(ctx.df, column, kind)

    df, conds = subset(ctx, filters)
    grouped = df.groupby(gcols, dropna=True, observed=True)
    result = grouped.size() if col is None else grouped[col].agg(_PANDAS_NAME.get(agg, agg))

    vk, nk = ("agg_value", "group_rows") if {"value", "n"} & set(gcols) else ("value", "n")
    out = result.rename(vk).to_frame().join(grouped.size().rename(nk)).reset_index()
    out = out.dropna(subset=[vk]).sort_values(vk, ascending=(sort == "asc"), kind="stable")
    if out.empty:
        raise ToolError("No groups to report after grouping (all values were missing).")

    limit = top_n or 50
    return {
        "group_by": gcols, "agg": agg, "column": col,
        "rows": records(out, limit=limit),
        "groups_total": int(len(out)),
        "truncated": bool(len(out) > limit),
        **scope(df, conds),
    }
