"""
data_summary.py

Tools: dataset_profile, statistical_summary, missing_value_summary,
       unique_value_summary, value_counts, distribution_info
"""

from __future__ import annotations

import numpy as np

from agent.tools._base import ToolError, obj, p_bool, p_int, p_list, p_str, tool
from agent.tools._helpers import (
    FILTERS, column_digest, need_col, numeric_columns, scope, subset,
)

CATEGORY = "data_summary"
MAX_COLS = 60


@tool(
    name="dataset_profile",
    description="Overview of the whole dataset: size, duplicate rows, and per column type, missing count and value info.",
    category=CATEGORY,
    parameters=obj({}),
)
def dataset_profile(ctx):
    df = ctx.df
    cols = []
    for c in list(df.columns)[:MAX_COLS]:
        d = column_digest(df, c)
        d["missing"] = int(df[c].isna().sum())
        d["missing_pct"] = round(100 * d["missing"] / max(len(df), 1), 2)
        cols.append(d)
    return {"n_rows": int(len(df)), "n_columns": int(df.shape[1]),
            "duplicate_rows": int(df.duplicated().sum()), "columns": cols,
            "columns_truncated": bool(df.shape[1] > MAX_COLS)}


@tool(
    name="statistical_summary",
    description="Descriptive statistics (count, mean, std, min, quartiles, max) for numeric columns.",
    category=CATEGORY,
    parameters=obj({
        "columns": p_list({"type": "string"}, "Numeric columns. Default all numeric columns."),
        "filters": FILTERS,
    }),
)
def statistical_summary(ctx, columns=None, filters=None):
    df, conds = subset(ctx, filters)
    cols = [need_col(ctx.df, c, "numeric") for c in columns] if columns else numeric_columns(df)
    if not cols:
        raise ToolError("No numeric columns available to summarise.")
    desc = df[cols[:MAX_COLS]].describe().T
    desc = desc.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
    return {"summary": {c: row.to_dict() for c, row in desc.iterrows()}, **scope(df, conds)}


@tool(
    name="missing_value_summary",
    description="Which columns have missing values, how many, and what percentage.",
    category=CATEGORY,
    parameters=obj({"filters": FILTERS}),
)
def missing_value_summary(ctx, filters=None):
    df, conds = subset(ctx, filters)
    miss = df.isna().sum()
    with_missing = {str(c): {"count": int(n), "pct": round(100 * n / len(df), 2)}
                    for c, n in miss[miss > 0].sort_values(ascending=False).items()}
    return {"columns_with_missing": with_missing, "columns_without_missing": int((miss == 0).sum()),
            "total_missing_cells": int(miss.sum()), **scope(df, conds)}


@tool(
    name="unique_value_summary",
    description="Number of distinct values per column, with the values themselves for low-cardinality columns.",
    category=CATEGORY,
    parameters=obj({
        "columns": p_list({"type": "string"}, "Columns to inspect. Default all."),
        "filters": FILTERS,
    }),
)
def unique_value_summary(ctx, columns=None, filters=None):
    df, conds = subset(ctx, filters)
    cols = [need_col(ctx.df, c) for c in columns] if columns else list(df.columns)[:MAX_COLS]
    out = {}
    for c in cols:
        u = df[c].dropna().unique()
        entry = {"unique_count": int(len(u))}
        if len(u) <= 15:
            entry["values"] = sorted(u.tolist(), key=str)
        out[c] = entry
    return {"columns": out, **scope(df, conds)}


@tool(
    name="value_counts",
    description="Frequency of each value in a column (most common first), with percentages.",
    category=CATEGORY,
    parameters=obj({
        "column": p_str("Column to count values of."),
        "top_n": p_int("Return only the N most common values (default 10).", minimum=1, maximum=50),
        "include_missing": p_bool("Also count missing values."),
        "filters": FILTERS,
    }, ["column"]),
)
def value_counts(ctx, column, top_n=10, include_missing=False, filters=None):
    col = need_col(ctx.df, column)
    df, conds = subset(ctx, filters)
    vc = df[col].value_counts(dropna=not include_missing)
    total = int(vc.sum())
    top = vc.head(top_n or 10)
    return {"column": col,
            "counts": [{"value": v, "count": int(n), "pct": round(100 * n / total, 2)} for v, n in top.items()],
            "distinct_values": int(len(vc)), "truncated": bool(len(vc) > len(top)), **scope(df, conds)}


@tool(
    name="distribution_info",
    description="Shape of a numeric column: spread, skewness, quartiles and a 10-bin histogram table.",
    category=CATEGORY,
    parameters=obj({"column": p_str("Numeric column."), "filters": FILTERS}, ["column"]),
)
def distribution_info(ctx, column, filters=None):
    col = need_col(ctx.df, column, "numeric")
    df, conds = subset(ctx, filters)
    s = df[col].dropna().astype(float)
    if len(s) < 2:
        raise ToolError(f"'{col}' needs at least 2 non-null values.")
    counts, edges = np.histogram(s, bins=10)
    skew = float(s.skew()) if len(s) > 2 else 0.0
    shape = ("approximately symmetric" if abs(skew) < 0.5
             else "right-skewed (long tail of high values)" if skew > 0 else "left-skewed (long tail of low values)")
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return {
        "column": col, "count": int(len(s)), "mean": s.mean(), "median": med, "std": s.std(),
        "min": s.min(), "max": s.max(), "q1": q1, "q3": q3, "iqr": q3 - q1,
        "skewness": skew, "kurtosis": float(s.kurt()) if len(s) > 3 else None, "shape": shape,
        "histogram": [{"from": edges[i], "to": edges[i + 1], "count": int(counts[i])} for i in range(len(counts))],
        **scope(df, conds),
    }
