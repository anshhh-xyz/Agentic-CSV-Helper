"""
demo_tools_only.py -- runs a sample of tools directly, with NO LLM and no API key.
Shows the deterministic layer works on its own, including a manual version of
the chained example ("top region -> its monthly trend").

    python demo_tools_only.py [--data path/to.csv]
"""

import argparse
import json

from agent import tool_registry
from agent.config import PLOTS_DIR
from agent.data_loader import load_csv
from agent.tools._base import ToolContext


def show(title, result):
    print(f"--- {title} ---")
    print(json.dumps(result, indent=2, default=str)[:900], "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_sales.csv")
    df = load_csv(parser.parse_args().data)
    ctx = ToolContext(df=df, plots_dir=PLOTS_DIR)
    run = lambda name, **args: tool_registry.dispatch(name, args, ctx)

    print(f"{len(tool_registry.list_tools())} tools registered.\n")
    show("dataset_profile", run("dataset_profile"))

    nums = [c for c in df.columns if str(df[c].dtype).startswith(("int", "float")) and not str(c).lower().endswith("id")]
    cats = [c for c in df.columns if c not in nums and not str(df[c].dtype).startswith("datetime")]
    dates = [c for c in df.columns if str(df[c].dtype).startswith("datetime")]
    if nums:
        show(f"mean({nums[0]})", run("mean", column=nums[0]))
        show("iqr_outliers", run("iqr_outliers", column=nums[0]))
    if cats and nums:
        top = run("group_aggregate", group_by=[cats[0]], agg="mean", column=nums[0], top_n=1)
        show("step 1: group_aggregate", top)
        if dates and top.get("rows"):
            group = top["rows"][0][cats[0]]
            filt = [{"column": cats[0], "operator": "==", "value": group}]
            show("step 2: trend_analysis (filtered)", run("trend_analysis", date_column=dates[0],
                 value_column=nums[0], freq="M", filters=filt))
            show("step 3: line_plot (filtered)", run("line_plot", x=dates[0], y=nums[0], resample="M", filters=filt))


if __name__ == "__main__":
    main()
