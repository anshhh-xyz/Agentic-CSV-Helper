"""
outlier_analysis.py

Tools: iqr_outliers, zscore_outliers, outlier_summary

(Isolation Forest was left out on purpose: it needs scikit-learn and the two
statistical methods above already cover typical questions.)
"""

from __future__ import annotations

import pandas as pd

from agent.tools._base import ToolError, obj, p_int, p_list, p_num, p_str, tool
from agent.tools._helpers import FILTERS, need_col, numeric_columns, records, scope, subset

CATEGORY = "outlier_analysis"


def _iqr_mask(s: pd.Series, k: float):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (s < lo) | (s > hi), {"lower_bound": lo, "upper_bound": hi}


def _z_mask(s: pd.Series, threshold: float):
    std = s.std()
    if not std or pd.isna(std):
        raise ToolError("Standard deviation is zero, so z-scores cannot be computed.")
    z = (s - s.mean()) / std
    return z.abs() > threshold, {"mean": s.mean(), "std": std}


def _report(df, col, mask, bounds, method, param_name, param, limit):
    clean = df[col].dropna()
    flagged = df.loc[clean[mask].index]
    flagged = flagged.reindex(flagged[col].abs().sort_values(ascending=False).index) if len(flagged) else flagged
    return {
        "column": col, "method": method, param_name: param, **bounds,
        "outlier_count": int(mask.sum()), "total_checked": int(len(clean)),
        "outlier_pct": round(100 * mask.sum() / max(len(clean), 1), 2),
        "rows": records(flagged, limit=limit, with_index=True),
        "rows_shown": int(min(len(flagged), limit)),
    }


@tool(
    name="iqr_outliers",
    description="Find outliers in a numeric column using the IQR rule (beyond Q1/Q3 by k*IQR).",
    category=CATEGORY,
    parameters=obj({
        "column": p_str("Numeric column."),
        "multiplier": p_num("IQR multiplier k (default 1.5).", minimum=0.1),
        "limit": p_int("Outlier rows to show (default 10, max 20).", minimum=1, maximum=20),
        "filters": FILTERS,
    }, ["column"]),
)
def iqr_outliers(ctx, column, multiplier=1.5, limit=10, filters=None):
    col = need_col(ctx.df, column, "numeric")
    df, conds = subset(ctx, filters)
    s = df[col].dropna().astype(float)
    if len(s) < 4:
        raise ToolError(f"'{col}' needs at least 4 non-null values for IQR outliers.")
    mask, bounds = _iqr_mask(s, multiplier or 1.5)
    return {**_report(df, col, mask, bounds, "iqr", "multiplier", multiplier or 1.5, limit or 10), **scope(df, conds)}


@tool(
    name="zscore_outliers",
    description="Find outliers in a numeric column using z-scores (|z| above a threshold).",
    category=CATEGORY,
    parameters=obj({
        "column": p_str("Numeric column."),
        "threshold": p_num("Absolute z-score cutoff (default 3).", minimum=0.5),
        "limit": p_int("Outlier rows to show (default 10, max 20).", minimum=1, maximum=20),
        "filters": FILTERS,
    }, ["column"]),
)
def zscore_outliers(ctx, column, threshold=3.0, limit=10, filters=None):
    col = need_col(ctx.df, column, "numeric")
    df, conds = subset(ctx, filters)
    s = df[col].dropna().astype(float)
    if len(s) < 3:
        raise ToolError(f"'{col}' needs at least 3 non-null values for z-scores.")
    mask, bounds = _z_mask(s, threshold or 3.0)
    return {**_report(df, col, mask, bounds, "zscore", "threshold", threshold or 3.0, limit or 10), **scope(df, conds)}


@tool(
    name="outlier_summary",
    description="Outlier counts for several numeric columns at once, to see where the anomalies are.",
    category=CATEGORY,
    parameters=obj({
        "columns": p_list({"type": "string"}, "Numeric columns. Default all numeric columns."),
        "method": p_str("Detection method (default iqr).", ["iqr", "zscore"]),
        "filters": FILTERS,
    }),
)
def outlier_summary(ctx, columns=None, method="iqr", filters=None):
    df, conds = subset(ctx, filters)
    cols = [need_col(ctx.df, c, "numeric") for c in columns] if columns else numeric_columns(df)
    if not cols:
        raise ToolError("No numeric columns to check.")
    rows = []
    for c in cols:
        s = df[c].dropna().astype(float)
        if len(s) < 4 or s.std() == 0:
            continue
        mask, _ = _iqr_mask(s, 1.5) if method != "zscore" else _z_mask(s, 3.0)
        rows.append({"column": c, "outlier_count": int(mask.sum()),
                     "outlier_pct": round(100 * mask.sum() / len(s), 2), "checked": int(len(s))})
    if not rows:
        raise ToolError("None of the columns had enough varying values to check.")
    rows.sort(key=lambda r: r["outlier_count"], reverse=True)
    return {"method": method or "iqr", "columns": rows, **scope(df, conds)}
