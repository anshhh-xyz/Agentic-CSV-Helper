"""
data_loader.py -- loads any CSV into a DataFrame and auto-detects date columns
so date-based tools work on datasets the agent has never seen.

A text column becomes datetime if (a) its name contains "date" or "time" and
it parses, or (b) every sampled value is a strict ISO date/datetime
(e.g. 2024-03-01) -- so non-date text is never converted by accident.
"""

import warnings

import pandas as pd


def load_csv(path_or_buffer) -> pd.DataFrame:
    """Accepts a filesystem path or a file-like object (e.g. a Flask upload)."""
    df = pd.read_csv(path_or_buffer)
    return _auto_parse_dates(df)


def _parses(values: pd.Series, **kwargs) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.to_datetime(values, errors="raise", **kwargs)
        return True
    except (ValueError, TypeError):
        return False


def _auto_parse_dates(df: pd.DataFrame, sample_size: int = 30) -> pd.DataFrame:
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        sample = s.dropna().astype(str).head(sample_size)
        if sample.empty:
            continue
        named_like_date = any(k in str(col).lower() for k in ("date", "time"))
        if (named_like_date and _parses(sample)) or _parses(sample, format="ISO8601"):
            df[col] = pd.to_datetime(s, errors="coerce")
    return df
