"""
_helpers.py -- data helpers shared by the tool modules (not tools themselves).

  * filters   : one structured filter format used by every analytical tool
  * columns   : forgiving column lookup with helpful errors
  * jsonable  : turns numpy/pandas values (NaN, Timestamps...) into clean JSON
  * time      : period-based time aggregation (works on pandas 2.x and 3.x)
"""

from __future__ import annotations

import difflib
import json
import math
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd

from agent.tools._base import ToolContext, ToolError

# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------

OPERATORS = ["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "is_null", "not_null"]

_OP_ALIASES = {
    "=": "==", "eq": "==", "equals": "==", "is": "==",
    "ne": "!=", "<>": "!=", "not_equals": "!=",
    "gt": ">", "gte": ">=", "ge": ">=", "lt": "<", "lte": "<=", "le": "<=",
    "not in": "not_in", "nin": "not_in",
    "isnull": "is_null", "is null": "is_null", "notnull": "not_null", "not null": "not_null",
    "like": "contains", "includes": "contains",
}

# Schema fragment reused by every tool that can be scoped to a subset of rows.
FILTERS = {
    "type": "array",
    "description": "Optional row conditions (all must match): [{column, operator, value}].",
    "items": {"type": "object"},
}


def normalize_filters(raw) -> list[dict]:
    """Accepts the canonical list of conditions, a single condition, or the
    {"column": value} shorthand; returns a validated list of conditions."""
    if raw is None or raw == [] or raw == {}:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raise ToolError("filters must be a list of {column, operator, value} objects.")
    if isinstance(raw, dict):
        if "column" in raw:
            raw = [raw]
        else:
            raw = [
                {"column": k, "operator": "in" if isinstance(v, list) else "==", "value": v}
                for k, v in raw.items()
            ]
    if not isinstance(raw, list):
        raise ToolError("filters must be a list of {column, operator, value} objects.")

    out = []
    for cond in raw:
        if not isinstance(cond, dict) or "column" not in cond:
            raise ToolError("Each filter needs 'column', 'operator' and 'value'.")
        op = str(cond.get("operator", "==")).strip().lower()
        op = _OP_ALIASES.get(op, op)
        if op not in OPERATORS:
            raise ToolError(f"Unsupported filter operator '{cond.get('operator')}'. Use one of: {', '.join(OPERATORS)}.")
        if op not in ("is_null", "not_null") and "value" not in cond:
            raise ToolError(f"Filter on '{cond['column']}' is missing 'value'.")
        out.append({"column": str(cond["column"]), "operator": op, "value": cond.get("value")})
    return out


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def _norm_text(v) -> str:
    return str(v).strip().casefold()


def _coerce_value(series: pd.Series, value, column: str):
    """Make a comparison value match the column's type (LLMs often send "3000")."""
    if pd.api.types.is_bool_dtype(series):
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in ("true", "1", "yes"):
            return True
        if str(value).strip().lower() in ("false", "0", "no"):
            return False
        raise ToolError(f"Column '{column}' is boolean; '{value}' is not true/false.")
    if _is_numeric(series):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        try:
            f = float(str(value).replace(",", ""))
        except ValueError:
            raise ToolError(f"Column '{column}' is numeric; '{value}' is not a number.")
        return int(f) if f.is_integer() and pd.api.types.is_integer_dtype(series) else f
    if _is_datetime(series):
        try:
            return pd.Timestamp(value)
        except (ValueError, TypeError):
            raise ToolError(f"Column '{column}' holds dates; '{value}' is not a valid date (use YYYY-MM-DD).")
    return value


def _condition_mask(series: pd.Series, cond: dict) -> pd.Series:
    op, value, column = cond["operator"], cond["value"], cond["column"]
    if op == "is_null":
        return series.isna()
    if op == "not_null":
        return series.notna()
    if op == "contains":
        return series.astype("string").str.contains(str(value), case=False, na=False, regex=False).astype(bool)

    typed = _is_numeric(series) or _is_datetime(series) or pd.api.types.is_bool_dtype(series)

    if op in ("in", "not_in"):
        vals = value if isinstance(value, (list, tuple)) else [value]
        if typed:
            mask = series.isin([_coerce_value(series, v, column) for v in vals])
        else:  # text: trimmed, case-insensitive
            norm = {_norm_text(v) for v in vals}
            mask = series.astype("string").str.strip().str.casefold().isin(norm).fillna(False).astype(bool)
        return mask if op == "in" else ~mask

    if isinstance(value, (list, dict)):
        raise ToolError(f"Operator '{op}' needs a single value for '{column}'. Use 'in' for several values.")

    if typed:
        v = _coerce_value(series, value, column)
        cmp = {"==": series == v, "!=": series != v}
        if op in cmp:
            return cmp[op].fillna(False).astype(bool)
        if pd.api.types.is_bool_dtype(series):
            raise ToolError(f"Operator '{op}' does not apply to boolean column '{column}'.")
        ordered = {">": series > v, ">=": series >= v, "<": series < v, "<=": series <= v}
        return ordered[op].fillna(False).astype(bool)

    if op not in ("==", "!="):
        raise ToolError(f"Operator '{op}' needs a numeric or date column; '{column}' is text. Use ==, !=, in or contains.")
    s = series.astype("string").str.strip().str.casefold()
    mask = (s == _norm_text(value)).fillna(False).astype(bool)
    return mask if op == "==" else ~mask & series.notna()


def apply_conditions(df: pd.DataFrame, conds: list[dict]) -> pd.DataFrame:
    if not conds:
        return df
    mask = pd.Series(True, index=df.index)
    for cond in conds:
        col = resolve_column(df, cond["column"])
        cond = {**cond, "column": col}
        mask &= _condition_mask(df[col], cond)
    return df[mask]


def _empty_message(conds: list[dict]) -> str:
    if not conds:
        return "The dataset has no rows to analyse."
    desc = "; ".join(f"{c['column']} {c['operator']} {c['value']}" for c in conds)
    return (f"No rows match the filters ({desc}). Check the exact category values "
            "with value_counts, or loosen the filters.")


def subset(ctx: ToolContext, filters, allow_empty: bool = False):
    """Returns (filtered_df, normalized_conditions). Raises ToolError on an
    empty result unless allow_empty (e.g. counting matches: 0 is an answer)."""
    conds = normalize_filters(filters)
    df = apply_conditions(ctx.df, conds)
    if df.empty and not allow_empty:
        raise ToolError(_empty_message(conds))
    return df, conds


def scope(df: pd.DataFrame, conds: list[dict]) -> dict:
    """Standard 'what data was used' block echoed in results."""
    out = {"rows_used": int(len(df))}
    if conds:
        out["filters"] = conds
    return out


# --------------------------------------------------------------------------
# Columns
# --------------------------------------------------------------------------

def cols_preview(df: pd.DataFrame, limit: int = 40) -> str:
    names = [str(c) for c in df.columns]
    text = ", ".join(names[:limit])
    return text + (f", ... (+{len(names) - limit} more)" if len(names) > limit else "")


def resolve_column(df: pd.DataFrame, name) -> str:
    """Find a column tolerating case/spacing differences; else explain."""
    if name in df.columns:
        return name
    key = str(name).strip().casefold()
    lookup = {str(c).strip().casefold(): c for c in df.columns}
    for candidate in (key, key.replace(" ", "_"), key.replace("_", " ")):
        if candidate in lookup:
            return lookup[candidate]
    close = difflib.get_close_matches(str(name), [str(c) for c in df.columns], n=3, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ToolError(f"Column '{name}' not found.{hint} Available columns: {cols_preview(df)}")


def need_col(df: pd.DataFrame, name, kind: str = "any") -> str:
    """Resolve a column and check its type. kind: any | numeric | datetime | numeric_or_datetime"""
    col = resolve_column(df, name)
    s = df[col]
    if kind == "numeric" and not _is_numeric(s) and not pd.api.types.is_bool_dtype(s):
        raise ToolError(f"Column '{col}' is {s.dtype}, not numeric. For text/category columns use "
                        "count, value_counts or group_aggregate with agg='count'.")
    if kind == "numeric_or_datetime" and not (_is_numeric(s) or _is_datetime(s) or pd.api.types.is_bool_dtype(s)):
        raise ToolError(f"Column '{col}' is {s.dtype}; this needs a numeric or date column.")
    if kind == "datetime" and not _is_datetime(s):
        as_datetime(df, col)  # raises a clear error if it cannot be parsed
    return col


def as_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    """The column as datetimes; parses text columns, rejects numeric ones."""
    s = df[col]
    if _is_datetime(s):
        return s
    if _is_numeric(s):
        raise ToolError(f"Column '{col}' is numeric, not a date column.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce")
    if s.notna().sum() == 0 or parsed.notna().sum() < 0.8 * s.notna().sum():
        raise ToolError(f"Column '{col}' does not look like dates.")
    return parsed


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if _is_numeric(df[c])]


def column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if _is_numeric(s):
        return "numeric"
    if _is_datetime(s):
        return "datetime"
    n = s.nunique(dropna=True)
    return "categorical" if n <= max(20, len(s) * 0.05) else "text"


def column_digest(df: pd.DataFrame, col: str, max_values: int = 12) -> dict:
    """Compact facts about a column, shared by the system prompt and profile tool."""
    s = df[col]
    kind = column_kind(s)
    info = {"name": str(col), "dtype": str(s.dtype), "kind": kind}
    nn = s.dropna()
    if kind == "numeric" and len(nn):
        info["min"], info["max"] = nn.min(), nn.max()
    elif kind == "datetime" and len(nn):
        info["min"], info["max"] = nn.min(), nn.max()
    elif kind in ("categorical", "boolean"):
        uniques = nn.unique()
        if len(uniques) <= max_values:
            info["values"] = sorted(uniques.tolist(), key=str)
        else:
            info["unique_count"] = int(len(uniques))
            info["sample_values"] = nn.value_counts().head(5).index.tolist()
    else:
        info["unique_count"] = int(nn.nunique())
        info["sample_values"] = nn.unique()[:3].tolist()
    return info


# --------------------------------------------------------------------------
# JSON sanitising
# --------------------------------------------------------------------------

def _round_float(f: float):
    if not math.isfinite(f):
        return None
    return round(f, 4) if abs(f) >= 1e-3 or f == 0 else float(f"{f:.4g}")


def to_jsonable(o):
    """Recursively convert numpy/pandas values to plain JSON types
    (NaN/inf -> null, Timestamps -> ISO strings)."""
    if o is None or o is pd.NaT:
        return None
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, str):
        return o
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return _round_float(float(o))
    if isinstance(o, (pd.Timestamp, datetime, date)):
        return o.isoformat()
    if isinstance(o, (pd.Period, pd.Timedelta)):
        return str(o)
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, pd.DataFrame):
        return [to_jsonable(r) for r in o.to_dict(orient="records")]
    if isinstance(o, pd.Series):
        return to_jsonable(o.to_dict())
    if isinstance(o, (list, tuple, set, np.ndarray, pd.Index)):
        return [to_jsonable(x) for x in list(o)]
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return str(o)


def records(df: pd.DataFrame, columns=None, limit: int = 20, with_index: bool = False) -> list[dict]:
    view = df[columns] if columns else df
    view = view.head(limit)
    out = []
    for idx, row in view.iterrows():
        rec = {str(k): row[k] for k in view.columns}
        if with_index:
            rec = {"row_index": idx, **rec}
        out.append(to_jsonable(rec))
    return out


# --------------------------------------------------------------------------
# Time aggregation
# --------------------------------------------------------------------------

FREQ_LABELS = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}
TIME_AGGS = ["sum", "mean", "median", "min", "max", "count"]


def time_aggregate(df: pd.DataFrame, date_col: str, value_col, freq: str, agg: str, group_col=None) -> pd.DataFrame:
    """Aggregate value_col per calendar period. Returns a frame indexed by
    period-start timestamps with one column per group ('value' if ungrouped).
    Empty periods are filled (0 for sum/count, NaN otherwise)."""
    work = df.copy()
    work[date_col] = as_datetime(work, date_col)
    work = work.dropna(subset=[date_col])
    if work.empty:
        raise ToolError(f"No valid dates in '{date_col}'.")

    keys = [work[date_col].dt.to_period(freq).rename("_period")]
    if group_col:
        keys.append(work[group_col].astype("string").fillna("(missing)").rename("_group"))
    grouped = work.groupby(keys, observed=True)

    if agg == "count":
        res = grouped[value_col].count() if value_col else grouped.size()
    else:
        res = grouped[value_col].agg(agg)

    res = res.unstack("_group") if group_col else res.to_frame("value")
    full = pd.period_range(res.index.min(), res.index.max(), freq=freq)
    res = res.reindex(full)
    if agg in ("sum", "count"):
        res = res.fillna(0)
    res.index = res.index.to_timestamp()
    res.columns = [str(c) for c in res.columns]
    return res
