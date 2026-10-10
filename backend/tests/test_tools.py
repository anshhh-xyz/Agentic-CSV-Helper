"""Tool layer: registration, schemas, numeric correctness vs plain pandas, filters, errors."""

import json
import os
import re
import tempfile
import unittest

import numpy as np
import pandas as pd

from tests.helpers import SAMPLE_CSV  # noqa: F401  (also fixes sys.path)

from agent import tool_registry as tr
from agent.data_loader import load_csv
from agent.tools._base import ToolContext, ToolError, all_tools, tool, unregister

EXPECTED = {
    "mathematical_operations": {"mean", "median", "sum", "min", "max", "count", "std", "variance",
                                "quantile", "group_aggregate"},
    "data_manipulation": {"filter_rows", "sort_rows", "fill_missing", "drop_missing", "remove_duplicates",
                          "rename_columns", "convert_dtype", "select_columns", "sample_rows"},
    "data_summary": {"dataset_profile", "statistical_summary", "missing_value_summary",
                     "unique_value_summary", "value_counts", "distribution_info"},
    "graphs": {"line_plot", "bar_plot", "scatter_plot", "histogram", "box_plot", "heatmap"},
    "correlation_analysis": {"correlation_matrix", "pairwise_correlation", "covariance",
                             "linear_regression", "trend_analysis"},
    "outlier_analysis": {"iqr_outliers", "zscore_outliers", "outlier_summary"},
    "memory": {"remember", "forget", "list_memory"},
}
MUTATING = {"fill_missing", "drop_missing", "remove_duplicates", "rename_columns", "convert_dtype", "select_columns"}
W = [{"column": "region", "operator": "==", "value": "West"}]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_csv(SAMPLE_CSV)
        cls.plots = tempfile.mkdtemp()

    def ctx(self, df=None):
        return ToolContext(df=self.df if df is None else df, plots_dir=self.plots)

    def run_tool(self, name, df=None, **args):
        return tr.dispatch(name, args, self.ctx(df))


class TestRegistration(Base):
    def test_every_module_registers_exactly_the_requested_tools(self):
        by_cat = {}
        for t in all_tools():
            by_cat.setdefault(t.category, set()).add(t.name)
        self.assertEqual(by_cat, EXPECTED)

    def test_names_unique_and_groq_safe(self):
        names = [t.name for t in all_tools()]
        self.assertEqual(len(names), len(set(names)))
        for n in names:
            self.assertRegex(n, r"^[a-zA-Z0-9_-]{1,64}$")

    def test_mutating_flags(self):
        self.assertEqual({t.name for t in all_tools() if t.mutates}, MUTATING)

    def test_schemas_well_formed(self):
        schemas = tr.get_tool_schemas()
        self.assertEqual(len(schemas), sum(len(v) for v in EXPECTED.values()))
        for s in schemas:
            self.assertEqual(s["type"], "function")
            fn = s["function"]
            self.assertTrue(fn["description"] and len(fn["description"]) <= 220, fn["name"])
            p = fn["parameters"]
            self.assertEqual(p["type"], "object")
            props = p["properties"]
            for req in p.get("required", []):
                self.assertIn(req, props, f"{fn['name']}: required '{req}' not a property")
            for k, v in props.items():
                if k in ("value",):  # deliberately typeless (any JSON value) -- only inside filter items
                    continue
                self.assertIn("type", v, f"{fn['name']}.{k} has no type")
                if v["type"] == "array":
                    self.assertIn("items", v, f"{fn['name']}.{k}")
                if "enum" in v:
                    self.assertTrue(all(isinstance(e, str) for e in v["enum"]))
            json.dumps(s)  # serialisable

    def test_filters_param_is_uniform(self):
        """Every tool that offers filters uses the identical schema fragment."""
        frags = {json.dumps(t.parameters["properties"]["filters"], sort_keys=True)
                 for t in all_tools() if "filters" in t.parameters["properties"]}
        self.assertEqual(len(frags), 1)
        no_filters = {t.name for t in all_tools() if "filters" not in t.parameters["properties"]}
        # cleaning tools / profile take no filters on purpose (they would be meaningless there)
        self.assertTrue(MUTATING <= no_filters and "dataset_profile" in no_filters)

    def test_schema_size_guard(self):
        size = len(json.dumps(tr.get_tool_schemas(), separators=(",", ":")))
        print(f"\n   [info] compact tool-schema size: {size} chars (~{int(size / 3.5)} tokens)")
        self.assertLess(size, 26000, "tool schemas are bloating every LLM request")

    def test_duplicate_registration_rejected(self):
        with self.assertRaises(ValueError):
            tool(name="mean", description="x", parameters={"type": "object", "properties": {}}, category="x")(lambda ctx: {})


class TestMathCorrectness(Base):
    def test_single_stats_match_pandas(self):
        r = self.df["revenue"]
        for name, expected in [("mean", r.mean()), ("median", r.median()), ("sum", r.sum()),
                               ("min", r.min()), ("max", r.max()), ("std", r.std()), ("variance", r.var())]:
            out = self.run_tool(name, column="revenue")
            self.assertNotIn("error", out, name)
            self.assertAlmostEqual(out["value"], round(float(expected), 4), places=3, msg=name)

    def test_min_max_on_dates(self):
        self.assertTrue(self.run_tool("min", column="order_date")["value"].startswith(str(self.df.order_date.min().date())))

    def test_count_variants(self):
        self.assertEqual(self.run_tool("count")["value"], len(self.df))
        self.assertEqual(self.run_tool("count", column="rating")["value"], int(self.df.rating.notna().sum()))
        self.assertEqual(self.run_tool("count", column="category", distinct=True)["value"], self.df.category.nunique())
        self.assertEqual(self.run_tool("count", filters=W)["value"], int((self.df.region == "West").sum()))
        zero = self.run_tool("count", filters=[{"column": "region", "operator": "==", "value": "Mars"}])
        self.assertEqual(zero["value"], 0)  # 0 matches is a valid answer, not an error

    def test_quantile_and_percent_style(self):
        out = self.run_tool("quantile", column="revenue", q=[0.5, 0.9])
        self.assertAlmostEqual(out["percentiles"]["p50"], round(self.df.revenue.quantile(0.5), 4), places=3)
        self.assertAlmostEqual(out["percentiles"]["p90"], round(self.df.revenue.quantile(0.9), 4), places=3)
        self.assertAlmostEqual(self.run_tool("quantile", column="revenue", q=90)["percentiles"]["p90"],
                               round(self.df.revenue.quantile(0.9), 4), places=3)

    def test_group_aggregate_matches_pandas_and_sorts(self):
        out = self.run_tool("group_aggregate", group_by=["region"], agg="mean", column="revenue")
        expected = self.df.groupby("region")["revenue"].mean().sort_values(ascending=False)
        self.assertEqual([r["region"] for r in out["rows"]], list(expected.index))
        self.assertAlmostEqual(out["rows"][0]["value"], round(expected.iloc[0], 4), places=3)
        self.assertEqual(out["rows"][0]["n"], int((self.df.region == expected.index[0]).sum()))
        top1 = self.run_tool("group_aggregate", group_by=["region"], agg="mean", column="revenue", top_n=1)
        self.assertEqual(len(top1["rows"]), 1)
        self.assertTrue(top1["truncated"])

    def test_group_aggregate_count_and_multi_key(self):
        out = self.run_tool("group_aggregate", group_by=["region", "category"], agg="count")
        self.assertEqual(sum(r["value"] for r in out["rows"]), len(self.df))
        asc = self.run_tool("group_aggregate", group_by=["region"], agg="sum", column="quantity", sort="asc")
        vals = [r["value"] for r in asc["rows"]]
        self.assertEqual(vals, sorted(vals))

    def test_group_aggregate_name_collision_with_value_column(self):
        df = pd.DataFrame({"value": ["a", "a", "b"], "x": [1, 2, 3]})
        out = self.run_tool("group_aggregate", df=df, group_by=["value"], agg="sum", column="x")
        self.assertEqual({r["value"]: r["agg_value"] for r in out["rows"]}, {"a": 3, "b": 3})


class TestOtherToolCorrectness(Base):
    def test_trend_analysis_matches_manual_monthly_sum(self):
        out = self.run_tool("trend_analysis", date_column="order_date", value_column="revenue", freq="M", filters=W)
        manual = self.df[self.df.region == "West"].set_index("order_date")["revenue"].resample("MS").sum()
        self.assertEqual([round(v, 4) for v in manual], [p["value"] for p in out["periods"]])
        self.assertEqual([p["period"] for p in out["periods"]], [str(d.date()) for d in manual.index])
        self.assertIn(out["trend_direction"], ("increasing", "decreasing", "roughly flat"))

    def test_correlation_tools_match_pandas(self):
        m = self.run_tool("correlation_matrix", columns=["revenue", "quantity", "rating"])
        self.assertAlmostEqual(m["matrix"]["revenue"]["quantity"], round(self.df.revenue.corr(self.df.quantity), 4), places=3)
        p = self.run_tool("pairwise_correlation", column_a="revenue", column_b="unit_price", method="spearman")
        self.assertAlmostEqual(p["correlation"], round(self.df.revenue.corr(self.df.unit_price, method="spearman"), 4), places=3)
        c = self.run_tool("covariance", columns=["revenue", "quantity"])
        self.assertAlmostEqual(c["covariance"], round(self.df.revenue.cov(self.df.quantity), 4), places=2)

    def test_linear_regression_matches_polyfit(self):
        out = self.run_tool("linear_regression", x="unit_price", y="revenue")
        slope, intercept = np.polyfit(self.df.unit_price, self.df.revenue, 1)
        self.assertAlmostEqual(out["slope"], round(slope, 4), places=3)
        self.assertAlmostEqual(out["intercept"], round(intercept, 4), places=2)

    def test_outliers_match_manual(self):
        s = self.df.revenue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        n_iqr = int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())
        self.assertEqual(self.run_tool("iqr_outliers", column="revenue")["outlier_count"], n_iqr)
        n_z = int((((s - s.mean()) / s.std()).abs() > 3).sum())
        self.assertEqual(self.run_tool("zscore_outliers", column="revenue")["outlier_count"], n_z)
        summ = self.run_tool("outlier_summary")
        self.assertEqual({r["column"] for r in summ["columns"]} >= {"revenue"}, True)

    def test_summary_tools(self):
        prof = self.run_tool("dataset_profile")
        self.assertEqual(prof["n_rows"], len(self.df))
        self.assertEqual(next(c for c in prof["columns"] if c["name"] == "rating")["missing"], int(self.df.rating.isna().sum()))
        vc = self.run_tool("value_counts", column="region")
        self.assertEqual({r["value"]: r["count"] for r in vc["counts"]}, self.df.region.value_counts().to_dict())
        self.assertEqual(self.run_tool("missing_value_summary")["columns_with_missing"]["rating"]["count"], int(self.df.rating.isna().sum()))
        self.assertEqual(self.run_tool("unique_value_summary", columns=["region"])["columns"]["region"]["unique_count"], 4)
        d = self.run_tool("distribution_info", column="revenue")
        self.assertEqual(sum(b["count"] for b in d["histogram"]), len(self.df))
        self.assertIn("skewed", d["shape"])  # injected outliers make revenue right-skewed
        ss = self.run_tool("statistical_summary", columns=["revenue"])
        self.assertAlmostEqual(ss["summary"]["revenue"]["mean"], round(self.df.revenue.mean(), 4), places=3)

    def test_manipulation_views(self):
        f = self.run_tool("filter_rows", filters=[{"column": "revenue", "operator": ">", "value": "3000"}])
        self.assertEqual(f["match_count"], int((self.df.revenue > 3000).sum()))
        s = self.run_tool("sort_rows", sort_by="revenue", limit=3)
        self.assertEqual([r["revenue"] for r in s["rows"]], [round(v, 4) for v in self.df.revenue.nlargest(3)])
        self.assertEqual(len(self.run_tool("sample_rows", n=4, seed=1)["rows"]), 4)


class TestGraphs(Base):
    def test_every_graph_writes_a_png_with_summary(self):
        cases = [
            ("line_plot", dict(x="order_date", y="revenue", resample="M", filters=W)),
            ("line_plot", dict(x="order_date", y="revenue", resample="M", group_by="region")),
            ("line_plot", dict(x="order_date", resample="W", agg="count")),
            ("bar_plot", dict(x="category", y="revenue", agg="sum", filters=W)),
            ("bar_plot", dict(x="region")),
            ("scatter_plot", dict(x="unit_price", y="revenue", color_by="region", trendline=True)),
            ("histogram", dict(column="revenue", filters=W)),
            ("box_plot", dict(column="revenue", group_by="category")),
            ("heatmap", dict()),
            ("heatmap", dict(mode="pivot", x="category", y="region", value="revenue", agg="sum")),
        ]
        for name, args in cases:
            out = self.run_tool(name, **args)
            self.assertNotIn("error", out, f"{name} {args}: {out}")
            self.assertTrue(os.path.getsize(out["file_path"]) > 5000, name)
            with open(out["file_path"], "rb") as fh:
                self.assertEqual(fh.read(4), b"\x89PNG")

    def test_filtered_line_plot_uses_only_that_subset(self):
        out = self.run_tool("line_plot", x="order_date", y="revenue", resample="M", filters=W)
        self.assertEqual(out["rows_used"], int((self.df.region == "West").sum()))
        manual = self.df[self.df.region == "West"].set_index("order_date")["revenue"].resample("MS").sum()
        self.assertAlmostEqual(out["series"]["value"]["max"], round(manual.max(), 4), places=3)

    def test_graph_errors(self):
        self.assertIn("error", self.run_tool("bar_plot", x="category", agg="sum"))          # sum needs y
        self.assertIn("error", self.run_tool("line_plot", x="category", y="revenue"))       # text x
        self.assertIn("error", self.run_tool("histogram", column="region"))                 # non-numeric
        self.assertIn("error", self.run_tool("heatmap", mode="pivot", x="category"))        # missing y


class TestFilters(Base):
    def f(self, *conds, **extra):
        return self.run_tool("count", filters=list(conds), **extra)["value"]

    def test_text_is_case_and_space_insensitive(self):
        n = int((self.df.region == "West").sum())
        self.assertEqual(self.f({"column": "region", "operator": "==", "value": "west"}), n)
        self.assertEqual(self.f({"column": "Region", "operator": "==", "value": " WEST "}), n)

    def test_in_not_in_contains_null(self):
        self.assertEqual(self.f({"column": "region", "operator": "in", "value": ["West", "east"]}),
                         int(self.df.region.isin(["West", "East"]).sum()))
        self.assertEqual(self.f({"column": "region", "operator": "not_in", "value": ["West"]}), int((self.df.region != "West").sum()))
        self.assertEqual(self.f({"column": "category", "operator": "contains", "value": "elec"}), int((self.df.category == "Electronics").sum()))
        self.assertEqual(self.f({"column": "rating", "operator": "is_null", "value": None}), int(self.df.rating.isna().sum()))
        self.assertEqual(self.f({"column": "rating", "operator": "not_null"}), int(self.df.rating.notna().sum()))

    def test_type_coercion_numeric_and_date(self):
        self.assertEqual(self.f({"column": "revenue", "operator": ">", "value": "3000"}), int((self.df.revenue > 3000).sum()))
        self.assertEqual(self.f({"column": "quantity", "operator": "==", "value": "3"}), int((self.df.quantity == 3).sum()))
        self.assertEqual(self.f({"column": "order_date", "operator": ">=", "value": "2024-03-01"}),
                         int((self.df.order_date >= "2024-03-01").sum()))

    def test_multiple_conditions_and_shorthand_and_aliases(self):
        exp = int(((self.df.region == "West") & (self.df.revenue > 1000)).sum())
        self.assertEqual(self.f({"column": "region", "operator": "==", "value": "West"},
                                {"column": "revenue", "operator": "gt", "value": 1000}), exp)
        self.assertEqual(self.run_tool("count", filters={"region": "West", "category": "Books"})["value"],
                         int(((self.df.region == "West") & (self.df.category == "Books")).sum()))
        self.assertEqual(self.f({"column": "revenue", "operator": ">=", "value": "1,000"}), int((self.df.revenue >= 1000).sum()))

    def test_filters_do_not_leak_between_calls(self):
        a = self.run_tool("mean", column="revenue", filters=W)["value"]
        b = self.run_tool("mean", column="revenue")["value"]
        self.assertNotEqual(a, b)


class TestErrorsAndValidation(Base):
    def err(self, name, **args):
        out = self.run_tool(name, **args)
        self.assertIn("error", out, f"{name}{args} should fail")
        self.assertNotIn("Traceback", out["error"])
        return out["error"]

    def test_bad_columns_give_suggestions(self):
        e = self.err("mean", column="revenu")
        self.assertIn("revenue", e)
        self.assertIn("Available columns", e)
        self.assertNotIn("error", self.run_tool("mean", column="REVENUE"))   # case-insensitive match

    def test_wrong_types(self):
        self.assertIn("not numeric", self.err("mean", column="region"))
        self.assertIn("not a number", self.err("count", filters=[{"column": "revenue", "operator": ">", "value": "lots"}]))
        self.assertIn("valid date", self.err("count", filters=[{"column": "order_date", "operator": ">", "value": "soon"}]))
        self.assertIn("text", self.err("count", filters=[{"column": "region", "operator": ">", "value": "A"}]))

    def test_invalid_filters(self):
        self.assertIn("Unsupported filter operator", self.err("count", filters=[{"column": "region", "operator": "~~", "value": 1}]))
        self.assertIn("missing 'value'", self.err("count", filters=[{"column": "region", "operator": "=="}]))
        self.assertIn("not found", self.err("count", filters=[{"column": "nope", "operator": "==", "value": 1}]))
        self.assertIn("No rows match", self.err("mean", column="revenue", filters=[{"column": "region", "operator": "==", "value": "Mars"}]))

    def test_argument_validation(self):
        self.assertIn("Unknown tool", self.err("meen", column="revenue")) if False else None
        out = tr.dispatch("meen", {"column": "revenue"}, self.ctx())
        self.assertIn("Did you mean: mean", out["error"])
        self.assertIn("unknown parameter", self.err("mean", column="revenue", bogus=1))
        self.assertIn("missing required", self.err("mean"))
        self.assertIn("must be one of", self.err("group_aggregate", group_by=["region"], agg="average", column="revenue"))
        self.assertIn("must be a whole number", self.err("sort_rows", sort_by="revenue", limit=2.5))
        self.assertIn("<= 20", self.err("sort_rows", sort_by="revenue", limit=500))
        self.assertIn("arguments must be a JSON object", tr.dispatch("mean", ["x"], self.ctx())["error"])

    def test_lenient_coercions_that_llms_need(self):
        self.assertNotIn("error", self.run_tool("sort_rows", sort_by="revenue", limit="3"))          # "3" -> 3
        self.assertNotIn("error", self.run_tool("group_aggregate", group_by="region", agg="MEAN", column="revenue"))  # scalar->list, enum case
        self.assertNotIn("error", self.run_tool("quantile", column="revenue", q=0.5))                # scalar -> [scalar]
        self.assertNotIn("error", self.run_tool("count", column=None, distinct="false"))             # null + bool string

    def test_unsupported_operations(self):
        self.assertIn("at least one condition", self.err("filter_rows", filters=[]))
        self.assertIn("constant", self.err("fill_missing", strategy="constant"))
        self.assertIn("needs a 'column'", self.err("group_aggregate", group_by=["region"], agg="mean"))
        self.assertIn("Choose two different", self.err("pairwise_correlation", column_a="revenue", column_b="revenue"))

    def test_unexpected_exception_is_masked(self):
        @tool(name="_boom", description="test", parameters={"type": "object", "properties": {}}, category="test")
        def _boom(ctx):
            raise RuntimeError("secret internal detail /etc/passwd")
        try:
            out = tr.dispatch("_boom", {}, self.ctx())
        finally:
            unregister("_boom")
        self.assertIn("could not process", out["error"])
        self.assertNotIn("secret", out["error"])
        self.assertNotIn("RuntimeError", out["error"])

    def test_results_are_strict_json(self):
        df = pd.DataFrame({"a": [1.0, np.nan, np.inf], "b": ["x", None, "z"], "d": pd.to_datetime(["2024-01-01", None, "2024-01-03"])})
        for name, args in [("dataset_profile", {}), ("filter_rows", {"filters": [{"column": "a", "operator": "not_null"}]}),
                           ("sample_rows", {"n": 3}), ("sort_rows", {"sort_by": "a"})]:
            out = self.run_tool(name, df=df, **args)
            json.dumps(out, allow_nan=False)  # raises on NaN/inf


class TestDataManipulation(Base):
    def test_fill_missing_variants(self):
        ctx = self.ctx()
        out = tr.dispatch("fill_missing", {"strategy": "mean", "columns": ["rating"]}, ctx)
        self.assertEqual(out["missing_remaining"], 0)
        self.assertAlmostEqual(ctx.df.rating.mean(), self.df.rating.mean(), places=6)
        self.assertEqual(int(self.df.rating.isna().sum()) > 0, True)     # original untouched
        ctx = self.ctx()
        tr.dispatch("fill_missing", {"strategy": "constant", "columns": ["rating"], "value": 0}, ctx)
        self.assertEqual(int(ctx.df.rating.isna().sum()), 0)
        self.assertAlmostEqual(ctx.df.rating.sum(), self.df.rating.sum(), places=6)
        bad = tr.dispatch("fill_missing", {"strategy": "mean", "columns": ["region"]}, self.ctx())
        self.assertIn("numeric", bad["error"])

    def test_drop_duplicates_rename_select_convert(self):
        df = pd.DataFrame({"a": [1, 1, 2, None], "b": ["x", "x", "y", "z"]})
        ctx = self.ctx(df)
        self.assertEqual(tr.dispatch("remove_duplicates", {}, ctx)["duplicates_removed"], 1)
        self.assertEqual(tr.dispatch("drop_missing", {"columns": ["a"]}, ctx)["rows_removed"], 1)
        self.assertEqual(len(ctx.df), 2)
        self.assertEqual(tr.dispatch("rename_columns", {"mapping": {"A": "alpha"}}, ctx)["columns_now"], ["alpha", "b"])
        self.assertIn("duplicate", tr.dispatch("rename_columns", {"mapping": {"alpha": "b"}}, ctx)["error"])
        self.assertEqual(tr.dispatch("select_columns", {"columns": ["b"]}, ctx)["kept"], ["b"])
        ctx2 = self.ctx()
        self.assertEqual(tr.dispatch("convert_dtype", {"column": "quantity", "dtype": "float"}, ctx2)["new_dtype"], "float64")
        refuse = tr.dispatch("convert_dtype", {"column": "region", "dtype": "int"}, ctx2)
        self.assertIn("nothing was changed", refuse["error"])
        self.assertEqual(str(ctx2.df.region.dtype), str(self.df.region.dtype))

    def test_drop_all_rows_refused(self):
        ctx = self.ctx(pd.DataFrame({"a": [None, None]}))
        self.assertIn("every row", tr.dispatch("drop_missing", {}, ctx)["error"])


class TestDataLoader(unittest.TestCase):
    def test_iso_dates_detected_regardless_of_name_and_text_not_converted(self):
        import io
        csv = "when,label,code\n2024-01-05,alpha,1-2\n2024-02-07,beta,3-4\n2024-03-09,gamma,5-6\n"
        df = load_csv(io.StringIO(csv))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["when"]))
        self.assertFalse(pd.api.types.is_datetime64_any_dtype(df["label"]))
        self.assertFalse(pd.api.types.is_datetime64_any_dtype(df["code"]))


if __name__ == "__main__":
    unittest.main()
