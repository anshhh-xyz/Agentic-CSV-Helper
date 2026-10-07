"""
graphs.py

Tools: line_plot, bar_plot, scatter_plot, histogram, box_plot, heatmap

Every graph accepts the same optional `filters` as the analysis tools, so a
chart can be scoped to a subset found in an earlier step. Figures use
matplotlib's object-oriented API (no pyplot global state), which makes them
safe to render concurrently when the model requests several charts at once.

Each tool returns {"file_path": ...} (picked up by the orchestrator and
served to the UI) plus a small numeric summary the model can describe.
"""

from __future__ import annotations

import os
import uuid

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from agent.tools._base import ToolError, obj, p_bool, p_int, p_list, p_str, tool  # noqa: E402
from agent.tools._helpers import (  # noqa: E402
    FILTERS, FREQ_LABELS, TIME_AGGS, as_datetime, need_col, numeric_columns, scope, subset, time_aggregate,
)

CATEGORY = "graphs"
PALETTE = ["#3B6FB6", "#E8A33D", "#4FA39A", "#C4524F", "#7B6BB5", "#8C8C8C", "#6DA34D", "#D17FA6"]
MAX_SERIES = 8
BAR_AGGS = ["mean", "sum", "median", "min", "max", "count"]


def _figure(width=8.0, height=4.8):
    fig = Figure(figsize=(width, height), dpi=110)
    ax = fig.subplots()
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def _save(fig, ctx, kind: str) -> str:
    os.makedirs(ctx.plots_dir, exist_ok=True)
    path = os.path.join(ctx.plots_dir, f"{kind}_{uuid.uuid4().hex[:10]}.png")
    fig.tight_layout()
    fig.savefig(path)
    return path


def _result(path, chart, title, **extra):
    return {"file_path": path, "chart": chart, "title": title, **extra}


def _title(custom, default):
    return (custom or default)[:120]


# ------------------------------- line_plot -------------------------------

@tool(
    name="line_plot",
    description=("Line chart of y over x. For time trends give a date x and a resample period "
                 "(e.g. monthly totals); group_by draws one line per category."),
    category=CATEGORY,
    parameters=obj({
        "x": p_str("Date or numeric column for the x-axis."),
        "y": p_str("Numeric column for the y-axis. Optional only when agg='count'."),
        "resample": p_str("Aggregate per period before plotting (needs a date x).", list(FREQ_LABELS)),
        "agg": p_str("Aggregation used with resample (default sum).", TIME_AGGS),
        "group_by": p_str("Category column: one line per value (max 8 lines)."),
        "title": p_str("Chart title."),
        "filters": FILTERS,
    }, ["x"]),
)
def line_plot(ctx, x, y=None, resample=None, agg="sum", group_by=None, title=None, filters=None):
    agg = agg or "sum"
    xc = need_col(ctx.df, x)
    if y is None and not (resample and agg == "count"):
        raise ToolError("line_plot needs a 'y' column (or resample with agg='count').")
    yc = need_col(ctx.df, y, "numeric") if y else None
    gc = need_col(ctx.df, group_by) if group_by else None
    df, conds = subset(ctx, filters)

    fig, ax = _figure()
    summary = {}
    if resample:
        wide = time_aggregate(df, xc, yc, resample, agg, gc)
        if wide.shape[1] > MAX_SERIES:
            keep = wide.abs().sum().sort_values(ascending=False).head(MAX_SERIES).index
            wide = wide[keep]
        for i, name in enumerate(wide.columns):
            ax.plot(wide.index, wide[name], marker="o", ms=4, lw=1.8, color=PALETTE[i % len(PALETTE)], label=name)
            v = wide[name].dropna()
            summary[name] = {"points": int(len(v)), "first": v.iloc[0], "last": v.iloc[-1], "min": v.min(), "max": v.max()}
        ylabel = f"{agg} of {yc}" if yc else "rows"
        xlabel = f"{xc} (per {FREQ_LABELS[resample]})"
        fig.autofmt_xdate()
    else:
        work = df[[xc, yc] + ([gc] if gc else [])].dropna(subset=[xc, yc]).copy()
        try:
            work[xc] = as_datetime(work, xc)
        except ToolError:
            if not pd.api.types.is_numeric_dtype(work[xc]):
                raise ToolError(f"x column '{xc}' must be a date or numeric column; for categories use bar_plot.")
        if not pd.api.types.is_datetime64_any_dtype(work[xc]) and pd.api.types.is_numeric_dtype(work[xc]) is False:
            raise ToolError(f"x column '{xc}' must be a date or numeric column.")
        work = work.sort_values(xc)
        groups = [(None, work)] if not gc else list(work.groupby(work[gc].astype("string"), observed=True))[:MAX_SERIES]
        for i, (name, part) in enumerate(groups):
            ax.plot(part[xc], part[yc], lw=1.5, color=PALETTE[i % len(PALETTE)], label=str(name) if name else None)
            summary[str(name) if name else yc] = {"points": int(len(part)), "min": part[yc].min(), "max": part[yc].max()}
        if pd.api.types.is_datetime64_any_dtype(work[xc]):
            fig.autofmt_xdate()
        ylabel, xlabel = yc, xc

    label = _title(title, f"{ylabel} over {xc}")
    ax.set_title(label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if gc or len(summary) > 1:
        ax.legend(frameon=False, fontsize=8)
    return _result(_save(fig, ctx, "line"), "line", label, series=summary, **scope(df, conds))


# ------------------------------- bar_plot -------------------------------

@tool(
    name="bar_plot",
    description="Bar chart comparing categories: an aggregate of y per x category (or a row count per category).",
    category=CATEGORY,
    parameters=obj({
        "x": p_str("Category column."),
        "y": p_str("Numeric column to aggregate. Omit to count rows per category."),
        "agg": p_str("Aggregation (default mean when y is given, otherwise count).", BAR_AGGS),
        "top_n": p_int("Show only the top N categories (default 15).", minimum=1, maximum=40),
        "sort": p_str("Sort bars by value. Default desc.", ["desc", "asc"]),
        "horizontal": p_bool("Draw horizontal bars (good for long labels)."),
        "title": p_str("Chart title."),
        "filters": FILTERS,
    }, ["x"]),
)
def bar_plot(ctx, x, y=None, agg=None, top_n=15, sort="desc", horizontal=False, title=None, filters=None):
    xc = need_col(ctx.df, x)
    yc = need_col(ctx.df, y, "numeric") if y else None
    agg = agg or ("mean" if yc else "count")
    if agg != "count" and not yc:
        raise ToolError(f"agg='{agg}' needs a 'y' column.")
    df, conds = subset(ctx, filters)

    grouped = df.groupby(df[xc].astype("string"), observed=True)
    values = grouped.size() if agg == "count" and not yc else grouped[yc].agg(agg)
    values = values.dropna().sort_values(ascending=(sort == "asc"), kind="stable")
    total = len(values)
    values = values.head(top_n or 15)
    if values.empty:
        raise ToolError("Nothing to plot after grouping.")

    fig, ax = _figure(8, max(3.6, 0.38 * len(values) + 1.6) if horizontal else 4.8)
    labels = [str(i) for i in values.index]
    if horizontal:
        ax.barh(labels[::-1], values.values[::-1], color=PALETTE[0])
        ax.set_xlabel(f"{agg} of {yc}" if yc else "rows")
    else:
        ax.bar(labels, values.values, color=PALETTE[0])
        ax.set_ylabel(f"{agg} of {yc}" if yc else "rows")
        ax.tick_params(axis="x", labelrotation=30)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    label = _title(title, f"{agg} of {yc} by {xc}" if yc else f"Rows per {xc}")
    ax.set_title(label)
    return _result(_save(fig, ctx, "bar"), "bar", label,
                   bars=[{"label": l, "value": v} for l, v in zip(labels, values.values)],
                   categories_total=int(total), **scope(df, conds))


# ------------------------------ scatter_plot ------------------------------

@tool(
    name="scatter_plot",
    description="Scatter plot of two numeric columns, optionally coloured by a category and with a fitted trend line.",
    category=CATEGORY,
    parameters=obj({
        "x": p_str("Numeric column for the x-axis."),
        "y": p_str("Numeric column for the y-axis."),
        "color_by": p_str("Category column to colour points by (max 8 categories)."),
        "trendline": p_bool("Overlay a least-squares line."),
        "title": p_str("Chart title."),
        "filters": FILTERS,
    }, ["x", "y"]),
)
def scatter_plot(ctx, x, y, color_by=None, trendline=False, title=None, filters=None):
    xc, yc = need_col(ctx.df, x, "numeric"), need_col(ctx.df, y, "numeric")
    cc = need_col(ctx.df, color_by) if color_by else None
    df, conds = subset(ctx, filters)
    work = df[[xc, yc] + ([cc] if cc else [])].dropna(subset=[xc, yc])
    if work.empty:
        raise ToolError("No rows with both x and y values to plot.")
    if len(work) > 3000:
        work = work.sample(3000, random_state=0)

    fig, ax = _figure()
    if cc:
        for i, (name, part) in enumerate(list(work.groupby(work[cc].astype("string"), observed=True))[:MAX_SERIES]):
            ax.scatter(part[xc], part[yc], s=16, alpha=0.65, color=PALETTE[i % len(PALETTE)], label=str(name))
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.scatter(work[xc], work[yc], s=16, alpha=0.6, color=PALETTE[0])

    extra = {}
    if trendline and len(work) >= 3 and work[xc].std() > 0:
        slope, intercept = np.polyfit(work[xc].astype(float), work[yc].astype(float), 1)
        xs = np.array([work[xc].min(), work[xc].max()], dtype=float)
        ax.plot(xs, slope * xs + intercept, color="#222222", lw=1.4, ls="--")
        extra["trendline"] = {"slope": slope, "intercept": intercept}
    label = _title(title, f"{yc} vs {xc}")
    ax.set_title(label)
    ax.set_xlabel(xc)
    ax.set_ylabel(yc)
    return _result(_save(fig, ctx, "scatter"), "scatter", label, points_plotted=int(len(work)), **extra, **scope(df, conds))


# ------------------------------- histogram -------------------------------

@tool(
    name="histogram",
    description="Histogram showing the distribution of a numeric column.",
    category=CATEGORY,
    parameters=obj({
        "column": p_str("Numeric column."),
        "bins": p_int("Number of bins (default 20).", minimum=2, maximum=100),
        "title": p_str("Chart title."),
        "filters": FILTERS,
    }, ["column"]),
)
def histogram(ctx, column, bins=20, title=None, filters=None):
    col = need_col(ctx.df, column, "numeric")
    df, conds = subset(ctx, filters)
    s = df[col].dropna().astype(float)
    if s.empty:
        raise ToolError(f"'{col}' has no values to plot.")
    fig, ax = _figure()
    ax.hist(s, bins=bins or 20, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    label = _title(title, f"Distribution of {col}")
    ax.set_title(label)
    ax.set_xlabel(col)
    ax.set_ylabel("frequency")
    return _result(_save(fig, ctx, "hist"), "histogram", label, values_plotted=int(len(s)),
                   min=s.min(), max=s.max(), mean=s.mean(), median=s.median(), **scope(df, conds))


# -------------------------------- box_plot --------------------------------

@tool(
    name="box_plot",
    description="Box plot of a numeric column, optionally one box per category (shows median, spread, outliers).",
    category=CATEGORY,
    parameters=obj({
        "column": p_str("Numeric column."),
        "group_by": p_str("Category column: one box per value (max 12)."),
        "title": p_str("Chart title."),
        "filters": FILTERS,
    }, ["column"]),
)
def box_plot(ctx, column, group_by=None, title=None, filters=None):
    col = need_col(ctx.df, column, "numeric")
    gc = need_col(ctx.df, group_by) if group_by else None
    df, conds = subset(ctx, filters)
    fig, ax = _figure()
    style = dict(patch_artist=True, medianprops={"color": "#222222"})
    medians = {}
    if gc:
        parts = [(str(n), p[col].dropna().astype(float)) for n, p in df.groupby(df[gc].astype("string"), observed=True)]
        parts = [(n, v) for n, v in parts if len(v)][:12]
        if not parts:
            raise ToolError("No groups with values to plot.")
        boxes = ax.boxplot([v for _, v in parts], tick_labels=[n for n, _ in parts], **style)
        medians = {n: v.median() for n, v in parts}
        ax.tick_params(axis="x", labelrotation=30)
    else:
        s = df[col].dropna().astype(float)
        if s.empty:
            raise ToolError(f"'{col}' has no values to plot.")
        boxes = ax.boxplot([s], tick_labels=[col], **style)
        medians = {col: s.median()}
    for i, patch in enumerate(boxes["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)] + "88")
    label = _title(title, f"{col} by {gc}" if gc else f"Spread of {col}")
    ax.set_title(label)
    ax.set_ylabel(col)
    return _result(_save(fig, ctx, "box"), "box", label, medians=medians, **scope(df, conds))


# --------------------------------- heatmap ---------------------------------

@tool(
    name="heatmap",
    description=("Heatmap. mode='correlation' (default) shows correlations between numeric columns; "
                 "mode='pivot' shows an aggregate of `value` across x categories (columns) and y categories (rows)."),
    category=CATEGORY,
    parameters=obj({
        "mode": p_str("Default correlation.", ["correlation", "pivot"]),
        "columns": p_list({"type": "string"}, "correlation mode: numeric columns (default all)."),
        "method": p_str("correlation mode: default pearson.", ["pearson", "spearman"]),
        "x": p_str("pivot mode: category column for the horizontal axis."),
        "y": p_str("pivot mode: category column for the vertical axis."),
        "value": p_str("pivot mode: numeric column to aggregate."),
        "agg": p_str("pivot mode: aggregation (default mean).", BAR_AGGS),
        "title": p_str("Chart title."),
        "filters": FILTERS,
    }),
)
def heatmap(ctx, mode="correlation", columns=None, method="pearson", x=None, y=None, value=None,
            agg="mean", title=None, filters=None):
    df, conds = subset(ctx, filters)
    mode = mode or "correlation"

    if mode == "correlation":
        cols = [need_col(ctx.df, c, "numeric") for c in columns] if columns else numeric_columns(df)[:20]
        if len(cols) < 2:
            raise ToolError("Need at least 2 numeric columns for a correlation heatmap.")
        table = df[cols].astype(float).corr(method=method or "pearson")
        cmap, vmin, vmax, default_title = "RdBu_r", -1, 1, f"{(method or 'pearson').title()} correlation"
    else:
        if not (x and y):
            raise ToolError("pivot mode needs both 'x' and 'y' category columns.")
        xc, yc = need_col(ctx.df, x), need_col(ctx.df, y)
        agg = agg or "mean"
        vc = need_col(ctx.df, value, "numeric") if value else None
        if agg != "count" and not vc:
            raise ToolError(f"pivot mode with agg='{agg}' needs a numeric 'value' column.")
        keys = [df[yc].astype("string"), df[xc].astype("string")]
        g = df.groupby(keys, observed=True)
        cells = g.size() if agg == "count" and not vc else g[vc].agg(agg)
        table = cells.unstack().iloc[:25, :25]
        if table.empty:
            raise ToolError("Nothing to plot after grouping.")
        cmap, vmin, vmax = "YlOrRd", None, None
        default_title = f"{agg} of {vc} by {yc} and {xc}" if vc else f"Rows by {yc} and {xc}"

    n_rows, n_cols = table.shape
    fig, ax = _figure(max(6.0, 0.7 * n_cols + 3), max(4.5, 0.55 * n_rows + 2))
    ax.grid(False)
    im = ax.imshow(table.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_cols), labels=[str(c) for c in table.columns], rotation=35, ha="right")
    ax.set_yticks(range(n_rows), labels=[str(i) for i in table.index])
    fig.colorbar(im, ax=ax, shrink=0.85)
    if n_rows * n_cols <= 144:
        finite = table.values.astype(float)
        lo, hi = (-1.0, 1.0) if mode == "correlation" else (np.nanmin(finite), np.nanmax(finite))
        for i in range(n_rows):
            for j in range(n_cols):
                v = finite[i, j]
                if np.isfinite(v):
                    frac = (v - lo) / ((hi - lo) or 1.0)
                    # white text only on the darkest cells of the colour scale
                    dark = (abs(frac - 0.5) > 0.3) if mode == "correlation" else (frac > 0.6)
                    ax.text(j, i, f"{v:.2f}" if abs(v) < 1000 else f"{v:,.0f}", ha="center", va="center",
                            fontsize=8, color="white" if dark else "#222222")
    label = _title(title, default_title)
    ax.set_title(label)
    return _result(_save(fig, ctx, "heatmap"), "heatmap", label, mode=mode,
                   table=table.round(3).where(table.notna(), None).to_dict(), **scope(df, conds))
