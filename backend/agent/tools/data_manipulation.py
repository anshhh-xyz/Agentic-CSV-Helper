"""
data_manipulation.py

Read-only views : filter_rows, sort_rows, sample_rows
Cleaning tools  : fill_missing, drop_missing, remove_duplicates,
                  rename_columns, convert_dtype, select_columns

Cleaning tools are marked mutates=True: they replace the working copy of the
data for the REST OF THIS REQUEST ONLY (the stored dataset is never changed),
and the executor runs them one at a time, in order.
"""

from __future__ import annotations

import pandas as pd

from agent.tools._base import ToolError, obj, p_int, p_list, p_str, tool
from agent.tools._helpers import (
    FILTERS, need_col, numeric_columns, records, resolve_column, scope, subset,
)

CATEGORY = "data_manipulation"


# ------------------------------ read-only views ------------------------------

@tool(
    name="filter_rows",
    description="Find rows matching conditions. Returns the match count and a preview of up to 20 rows.",
    category=CATEGORY,
    parameters=obj({
        "filters": FILTERS,
        "columns": p_list({"type": "string"}, "Columns to show in the preview. Default all."),
        "limit": p_int("Preview rows to return (default 10, max 20).", minimum=1, maximum=20),
    }, ["filters"]),
)
def filter_rows(ctx, filters, columns=None, limit=10):
    df, conds = subset(ctx, filters, allow_empty=True)
    if not conds:
        raise ToolError("filter_rows needs at least one condition in 'filters'.")
    cols = [need_col(ctx.df, c) for c in columns] if columns else None
    return {"match_count": int(len(df)), "total_rows": int(len(ctx.df)),
            "rows": records(df, cols, limit=limit or 10, with_index=True), **scope(df, conds)}


@tool(
    name="sort_rows",
    description="Sort rows by a column and preview the top rows (e.g. top 5 orders by revenue).",
    category=CATEGORY,
    parameters=obj({
        "sort_by": p_str("Column to sort by."),
        "order": p_str("Default desc.", ["desc", "asc"]),
        "limit": p_int("Rows to return (default 10, max 20).", minimum=1, maximum=20),
        "columns": p_list({"type": "string"}, "Columns to show. Default all."),
        "filters": FILTERS,
    }, ["sort_by"]),
)
def sort_rows(ctx, sort_by, order="desc", limit=10, columns=None, filters=None):
    col = need_col(ctx.df, sort_by)
    df, conds = subset(ctx, filters)
    cols = [need_col(ctx.df, c) for c in columns] if columns else None
    ordered = df.sort_values(col, ascending=(order == "asc"), kind="stable", na_position="last")
    return {"sorted_by": col, "order": order,
            "rows": records(ordered, cols, limit=limit or 10, with_index=True), **scope(df, conds)}


@tool(
    name="sample_rows",
    description="Return a random sample of rows to get a feel for the data.",
    category=CATEGORY,
    parameters=obj({
        "n": p_int("Rows to sample (default 5, max 20).", minimum=1, maximum=20),
        "seed": p_int("Random seed for repeatable samples."),
        "filters": FILTERS,
    }),
)
def sample_rows(ctx, n=5, seed=None, filters=None):
    df, conds = subset(ctx, filters)
    sample = df.sample(n=min(n or 5, len(df)), random_state=seed)
    return {"rows": records(sample, limit=20, with_index=True), **scope(df, conds)}


# ------------------------------ cleaning tools ------------------------------

def _columns_now(df):
    return {"columns_now": [str(c) for c in df.columns], "rows_now": int(len(df))}


@tool(
    name="fill_missing",
    description="Fill missing values in the working data (mean/median for numeric, mode, a constant, or forward/backward fill).",
    category=CATEGORY, mutates=True,
    parameters=obj({
        "strategy": p_str("How to fill.", ["mean", "median", "mode", "constant", "forward_fill", "backward_fill"]),
        "columns": p_list({"type": "string"}, "Columns to fill. Default: all columns the strategy applies to."),
        "value": {"description": "Fill value; required when strategy='constant'."},
    }, ["strategy"]),
)
def fill_missing(ctx, strategy, columns=None, value=None):
    df = ctx.df.copy()
    cols = [need_col(df, c) for c in columns] if columns else list(df.columns)
    if strategy in ("mean", "median"):
        pool = numeric_columns(df)
        if columns:
            bad = [c for c in cols if c not in pool]
            if bad:
                raise ToolError(f"Strategy '{strategy}' needs numeric columns; not numeric: {bad}.")
        cols = [c for c in cols if c in pool]
    if strategy == "constant" and value is None:
        raise ToolError("strategy='constant' needs a 'value'.")

    filled = {}
    for c in cols:
        before = int(df[c].isna().sum())
        if before == 0:
            continue
        if strategy == "mean":
            df[c] = df[c].fillna(df[c].mean())
        elif strategy == "median":
            df[c] = df[c].fillna(df[c].median())
        elif strategy == "mode":
            mode = df[c].mode(dropna=True)
            if mode.empty:
                continue
            df[c] = df[c].fillna(mode.iloc[0])
        elif strategy == "constant":
            try:
                fill = pd.Series([value]).astype(df[c].dtype).iloc[0] if pd.api.types.is_numeric_dtype(df[c]) else value
            except (ValueError, TypeError):
                raise ToolError(f"Cannot use '{value}' to fill column '{c}' ({df[c].dtype}).")
            df[c] = df[c].fillna(fill)
        elif strategy == "forward_fill":
            df[c] = df[c].ffill()
        else:
            df[c] = df[c].bfill()
        filled[c] = before - int(df[c].isna().sum())

    ctx.df = df
    return {"strategy": strategy, "values_filled": filled, "total_filled": int(sum(filled.values())),
            "missing_remaining": int(df.isna().sum().sum()), **_columns_now(df)}


@tool(
    name="drop_missing",
    description="Drop rows with missing values from the working data.",
    category=CATEGORY, mutates=True,
    parameters=obj({
        "columns": p_list({"type": "string"}, "Only look at these columns. Default all."),
        "how": p_str("'any' drops a row if any checked value is missing; 'all' only if all are. Default any.", ["any", "all"]),
    }),
)
def drop_missing(ctx, columns=None, how="any"):
    cols = [need_col(ctx.df, c) for c in columns] if columns else None
    before = len(ctx.df)
    new = ctx.df.dropna(subset=cols, how=how or "any")
    if new.empty:
        raise ToolError("Dropping these rows would remove every row; nothing was changed.")
    ctx.df = new
    return {"rows_removed": int(before - len(new)), **_columns_now(new)}


@tool(
    name="remove_duplicates",
    description="Remove duplicate rows from the working data.",
    category=CATEGORY, mutates=True,
    parameters=obj({
        "columns": p_list({"type": "string"}, "Judge duplicates by these columns only. Default all."),
        "keep": p_str("Which duplicate to keep. Default first.", ["first", "last"]),
    }),
)
def remove_duplicates(ctx, columns=None, keep="first"):
    cols = [need_col(ctx.df, c) for c in columns] if columns else None
    before = len(ctx.df)
    ctx.df = ctx.df.drop_duplicates(subset=cols, keep=keep or "first")
    return {"duplicates_removed": int(before - len(ctx.df)), **_columns_now(ctx.df)}


@tool(
    name="rename_columns",
    description="Rename columns in the working data. Use the mapping {old_name: new_name}.",
    category=CATEGORY, mutates=True,
    parameters=obj({
        "mapping": {"type": "object", "description": "Old name -> new name.",
                    "additionalProperties": {"type": "string"}},
    }, ["mapping"]),
)
def rename_columns(ctx, mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ToolError("mapping must be a non-empty object like {\"old\": \"new\"}.")
    resolved = {resolve_column(ctx.df, old): str(new).strip() for old, new in mapping.items()}
    if any(not n for n in resolved.values()):
        raise ToolError("New column names cannot be empty.")
    result = ctx.df.rename(columns=resolved)
    if result.columns.duplicated().any():
        raise ToolError("That rename would create duplicate column names; nothing was changed.")
    ctx.df = result
    return {"renamed": resolved, **_columns_now(result)}


_DTYPES = ["int", "float", "string", "category", "datetime", "bool"]


@tool(
    name="convert_dtype",
    description="Convert a column's type in the working data (int, float, string, category, datetime, bool).",
    category=CATEGORY, mutates=True,
    parameters=obj({
        "column": p_str("Column to convert."),
        "dtype": p_str("Target type.", _DTYPES),
    }, ["column", "dtype"]),
)
def convert_dtype(ctx, column, dtype):
    col = need_col(ctx.df, column)
    s = ctx.df[col]
    if dtype in ("int", "float"):
        cleaned = s.astype("string").str.replace(",", "", regex=False) if not pd.api.types.is_numeric_dtype(s) else s
        new = pd.to_numeric(cleaned, errors="coerce")
        if dtype == "float":
            new = new.astype("float64")
        if dtype == "int":
            if (new.dropna() % 1 != 0).any():
                raise ToolError(f"'{col}' has non-integer values; convert to float instead.")
            new = new.astype("Int64") if new.isna().any() else new.astype("int64")
    elif dtype == "datetime":
        new = pd.to_datetime(s, errors="coerce")
    elif dtype == "bool":
        mapping = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}
        new = s.astype("string").str.strip().str.lower().map(mapping).astype("boolean")
    elif dtype == "category":
        new = s.astype("category")
    else:
        new = s.astype("string")

    newly_null = int(new.isna().sum() - s.isna().sum())
    if newly_null > 0.5 * max(int(s.notna().sum()), 1):
        raise ToolError(f"Converting '{col}' to {dtype} would lose {newly_null} of {int(s.notna().sum())} "
                        "values; nothing was changed.")
    df = ctx.df.copy()
    df[col] = new
    ctx.df = df
    return {"column": col, "new_dtype": str(df[col].dtype), "values_that_failed_to_convert": max(newly_null, 0),
            **_columns_now(df)}


@tool(
    name="select_columns",
    description="Keep only the listed columns in the working data.",
    category=CATEGORY, mutates=True,
    parameters=obj({"columns": p_list({"type": "string"}, "Columns to keep.")}, ["columns"]),
)
def select_columns(ctx, columns):
    if not columns:
        raise ToolError("Give at least one column to keep.")
    cols = []
    for c in columns:
        r = need_col(ctx.df, c)
        if r not in cols:
            cols.append(r)
    ctx.df = ctx.df[cols]
    return {"kept": cols, **_columns_now(ctx.df)}
