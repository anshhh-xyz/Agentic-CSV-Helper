"""
correlation_analysis.py

Tools: correlation_matrix, pairwise_correlation, covariance,
       linear_regression, trend_analysis

trend_analysis (time series) lives here because it is the "how does X move
with time" counterpart of correlation: it also reports a fitted slope.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from agent.tools._base import ToolError, obj, p_int, p_list, p_str, tool
from agent.tools._helpers import (
    FILTERS, FREQ_LABELS, TIME_AGGS, need_col, numeric_columns, scope, subset, time_aggregate,
)

CATEGORY = "correlation_analysis"
METHODS = ["pearson", "spearman"]


def _strength(r: float) -> str:
    a = abs(r)
    label = ("negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5
             else "strong" if a < 0.7 else "very strong")
    return f"{label} {'positive' if r > 0 else 'negative'}" if a >= 0.1 else label


def _numeric_frame(ctx, columns, filters, minimum=2):
    df, conds = subset(ctx, filters)
    cols = [need_col(ctx.df, c, "numeric") for c in columns] if columns else numeric_columns(df)
    cols = list(dict.fromkeys(cols))
    if len(cols) < minimum:
        raise ToolError(f"Need at least {minimum} numeric columns; found {len(cols)}.")
    return df, conds, cols


@tool(
    name="correlation_matrix",
    description="Correlation matrix between numeric columns, plus the strongest pairs.",
    category=CATEGORY,
    parameters=obj({
        "columns": p_list({"type": "string"}, "Numeric columns. Default all numeric columns."),
        "method": p_str("Default pearson.", METHODS),
        "filters": FILTERS,
    }),
)
def correlation_matrix(ctx, columns=None, method="pearson", filters=None):
    df, conds, cols = _numeric_frame(ctx, columns, filters)
    corr = df[cols].astype(float).corr(method=method or "pearson")
    pairs = [{"column_a": a, "column_b": b, "correlation": corr.loc[a, b], "strength": _strength(corr.loc[a, b])}
             for a, b in itertools.combinations(cols, 2) if pd.notna(corr.loc[a, b])]
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return {"method": method or "pearson", "matrix": corr.to_dict(), "strongest_pairs": pairs[:5], **scope(df, conds)}


@tool(
    name="pairwise_correlation",
    description="Correlation between two numeric columns with p-value and strength label.",
    category=CATEGORY,
    parameters=obj({
        "column_a": p_str("First numeric column."),
        "column_b": p_str("Second numeric column."),
        "method": p_str("Default pearson.", METHODS),
        "filters": FILTERS,
    }, ["column_a", "column_b"]),
)
def pairwise_correlation(ctx, column_a, column_b, method="pearson", filters=None):
    a, b = need_col(ctx.df, column_a, "numeric"), need_col(ctx.df, column_b, "numeric")
    df, conds = subset(ctx, filters)
    pair = df[[a, b]].dropna().astype(float) if a != b else None
    if pair is None:
        raise ToolError("Choose two different columns.")
    if len(pair) < 3:
        raise ToolError("Need at least 3 rows with both values present.")
    if pair[a].std() == 0 or pair[b].std() == 0:
        raise ToolError("One column is constant, so correlation is undefined.")
    fn = stats.spearmanr if method == "spearman" else stats.pearsonr
    r, p = fn(pair[a], pair[b])
    return {"column_a": a, "column_b": b, "method": method or "pearson", "correlation": r,
            "p_value": p, "strength": _strength(float(r)), "pairs_used": int(len(pair)), **scope(df, conds)}


@tool(
    name="covariance",
    description="Covariance between numeric columns (a matrix; give exactly two columns for a single pair).",
    category=CATEGORY,
    parameters=obj({
        "columns": p_list({"type": "string"}, "Numeric columns. Default all numeric columns."),
        "filters": FILTERS,
    }),
)
def covariance(ctx, columns=None, filters=None):
    df, conds, cols = _numeric_frame(ctx, columns, filters)
    cov = df[cols].astype(float).cov()
    out = {"matrix": cov.to_dict(), **scope(df, conds)}
    if len(cols) == 2:
        out["covariance"] = cov.loc[cols[0], cols[1]]
    return out


@tool(
    name="linear_regression",
    description="Simple linear regression of y on x: slope, intercept, R-squared and p-value.",
    category=CATEGORY,
    parameters=obj({
        "x": p_str("Numeric predictor column."),
        "y": p_str("Numeric outcome column."),
        "filters": FILTERS,
    }, ["x", "y"]),
)
def linear_regression(ctx, x, y, filters=None):
    xc, yc = need_col(ctx.df, x, "numeric"), need_col(ctx.df, y, "numeric")
    df, conds = subset(ctx, filters)
    pair = df[[xc, yc]].dropna().astype(float)
    if xc == yc or len(pair) < 3 or pair[xc].std() == 0:
        raise ToolError("Need two different columns, at least 3 complete rows, and a non-constant x.")
    fit = stats.linregress(pair[xc], pair[yc])
    return {"x": xc, "y": yc, "slope": fit.slope, "intercept": fit.intercept, "r_squared": fit.rvalue ** 2,
            "p_value": fit.pvalue, "pairs_used": int(len(pair)),
            "interpretation": f"Each +1 in {xc} is associated with {fit.slope:+.4g} in {yc}.", **scope(df, conds)}


@tool(
    name="trend_analysis",
    description=("How a numeric column changes over time: aggregates per day/week/month/quarter/year, "
                 "reports overall % change, fitted slope, best/worst period and a rolling average."),
    category=CATEGORY,
    parameters=obj({
        "date_column": p_str("Column with dates."),
        "value_column": p_str("Numeric column to track. Optional only when agg='count'."),
        "freq": p_str("Period size (default M).", list(FREQ_LABELS)),
        "agg": p_str("How to combine values inside a period (default sum).", TIME_AGGS),
        "filters": FILTERS,
    }, ["date_column"]),
)
def trend_analysis(ctx, date_column, value_column=None, freq="M", agg="sum", filters=None):
    freq, agg = freq or "M", agg or "sum"
    dcol = need_col(ctx.df, date_column, "datetime")
    if value_column is None and agg != "count":
        raise ToolError(f"agg='{agg}' needs a value_column.")
    vcol = need_col(ctx.df, value_column, "numeric") if value_column else None
    df, conds = subset(ctx, filters)

    series = time_aggregate(df, dcol, vcol, freq, agg)["value"]
    valid = series.dropna()
    if valid.empty:
        raise ToolError("No values to analyse after aggregation.")

    first, last = float(valid.iloc[0]), float(valid.iloc[-1])
    pct = round(100 * (last - first) / abs(first), 2) if first not in (0, 0.0) else None
    slope = float(np.polyfit(np.arange(len(valid)), valid.values.astype(float), 1)[0]) if len(valid) >= 2 else 0.0
    tol = 0.01 * (abs(float(valid.mean())) or 1.0)
    direction = "increasing" if slope > tol else "decreasing" if slope < -tol else "roughly flat"
    rolling = series.rolling(3, min_periods=1).mean()

    shown = series.tail(60)
    return {
        "date_column": dcol, "value_column": vcol, "freq": freq, "agg": agg,
        "n_periods": int(len(series)), "trend_direction": direction,
        "overall_pct_change_first_to_last": pct, "slope_per_period": slope,
        "best_period": {"period": valid.idxmax(), "value": valid.max()},
        "worst_period": {"period": valid.idxmin(), "value": valid.min()},
        "periods": [{"period": str(idx.date()), "value": v, "rolling_avg_3": rolling[idx]} for idx, v in shown.items()],
        "periods_truncated": bool(len(series) > 60),
        "note": "The first or last period may be partial if data does not cover it fully.",
        **scope(df, conds),
    }
