"""Agent layer: chaining, parallel execution, recovery, Groq SDK wire format, Flask API."""

import io
import json
import os
import tempfile
import time
import unittest
from unittest import mock

import httpx
import pandas as pd

from tests.helpers import (FakeClient, SAMPLE_CSV, completion_json, groq_on_mock, msg, tc, tool_results)

from agent import orchestrator, tool_registry as tr
from agent.config import PLOTS_DIR
from agent.data_loader import load_csv
from agent.executor import execute_calls, plan_batches
from agent.llm_client import LLMError, LLMRateLimitError, ToolCallFormatError
from agent.orchestrator import FALLBACK_ANSWER, run_agent
from agent.prompts import build_system_prompt
from agent.tools._base import ToolContext, tool, unregister

DF = load_csv(SAMPLE_CSV)
TOP_REGION = DF.groupby("region")["revenue"].mean().idxmax()


def west_filter(region):
    return [{"column": "region", "operator": "==", "value": region}]


class TestChainedExecution(unittest.TestCase):
    """'Find the region with the highest average revenue and plot its monthly trend.'"""

    def policy(self, messages, kw):
        results = tool_results(messages)
        if not results:                                    # step 1
            return msg(calls=[tc("group_aggregate", {"group_by": ["region"], "agg": "mean", "column": "revenue"})])
        if len(results) == 1:                              # step 2 depends on step 1's RESULT
            top = results[0]["rows"][0]["region"]
            return msg(calls=[tc("line_plot", {"x": "order_date", "y": "revenue", "resample": "M",
                                               "filters": west_filter(top), "title": f"{top} monthly revenue"})])
        top = results[0]["rows"][0]["region"]
        return msg(content=f"{top} has the highest average revenue; its monthly trend is charted.")

    def test_two_step_dependency_uses_first_result(self):
        client = FakeClient(self.policy)
        out = run_agent(DF, "Find the region with the highest average revenue and plot its monthly trend.", client=client)

        self.assertIn(TOP_REGION, out["answer"])
        self.assertEqual([t["tool"] for t in out["tool_trace"]], ["group_aggregate", "line_plot"])
        self.assertEqual([t["round"] for t in out["tool_trace"]], [1, 2])          # sequential, in separate turns
        self.assertTrue(all(t["ok"] and not t["parallel"] for t in out["tool_trace"]))
        # the filter given to step 2 came from step 1's output
        self.assertEqual(out["tool_trace"][1]["arguments"]["filters"][0]["value"], TOP_REGION)
        # ...and the chart was really scoped to that region
        self.assertEqual(len(out["plot_files"]), 1)
        self.assertTrue(os.path.exists(out["plot_files"][0]))
        step2 = tool_results(client.requests[2]["messages"])[1]
        self.assertEqual(step2["rows_used"], int((DF.region == TOP_REGION).sum()))
        self.assertEqual(step2["series"]["value"]["max"], round(DF[DF.region == TOP_REGION].set_index("order_date")["revenue"].resample("MS").sum().max(), 4))
        self.assertEqual(out["rounds"], 3)

    def test_message_protocol_is_valid(self):
        client = FakeClient(self.policy)
        run_agent(DF, "q", client=client)
        final_msgs = client.requests[-1]["messages"]
        assert final_msgs[0]["role"] == "system" and final_msgs[1]["role"] == "user"
        for i, m in enumerate(final_msgs):
            if m.get("tool_calls"):
                ids = [c["id"] for c in m["tool_calls"]]
                following = [x["tool_call_id"] for x in final_msgs[i + 1:i + 1 + len(ids)]]
                self.assertEqual(ids, following)                  # every call answered, same order
        self.assertTrue(all("tools" in r and r["tool_choice"] == "auto" for r in client.requests))

    def test_model_never_sees_file_paths(self):
        client = FakeClient(self.policy)
        run_agent(DF, "q", client=client)
        blob = json.dumps(client.requests[-1]["messages"])
        self.assertNotIn(".png", blob)
        self.assertNotIn("/home/", blob)
        self.assertIn("chart_created", blob)

    def test_three_step_chain_with_cleaning(self):
        def policy(messages, kw):
            r = tool_results(messages)
            if len(r) == 0:
                return msg(calls=[tc("fill_missing", {"strategy": "constant", "columns": ["rating"], "value": 0})])
            if len(r) == 1:
                return msg(calls=[tc("mean", {"column": "rating"})])
            return msg(content=f"mean rating after filling: {r[1]['value']}")
        out = run_agent(DF, "fill missing ratings with 0 then average them", client=FakeClient(policy))
        self.assertAlmostEqual(float(out["answer"].split(": ")[1]), DF.rating.fillna(0).mean(), places=3)
        self.assertEqual(int(DF.rating.isna().sum()) > 0, True)   # stored dataset not modified


class TestParallelExecution(unittest.TestCase):
    def ctx(self):
        return ToolContext(df=DF, plots_dir=tempfile.mkdtemp())

    def test_independent_calls_in_one_turn_run_in_parallel_and_in_order(self):
        def policy(messages, kw):
            if not tool_results(messages):
                return msg(calls=[tc("mean", {"column": "revenue"}), tc("median", {"column": "revenue"}),
                                  tc("count", {"column": "customer_age", "distinct": True}),
                                  tc("line_plot", {"x": "order_date", "y": "revenue", "resample": "M"})])
            return msg(content="done")
        out = run_agent(DF, "avg, median, distinct ages and plot", client=FakeClient(policy))
        trace = out["tool_trace"]
        self.assertEqual([t["tool"] for t in trace], ["mean", "median", "count", "line_plot"])
        self.assertEqual({t["round"] for t in trace}, {1})
        self.assertTrue(all(t["parallel"] and t["ok"] for t in trace))
        self.assertEqual(len(out["plot_files"]), 1)
        self.assertEqual(out["rounds"], 2)

    def test_parallel_is_actually_concurrent(self):
        @tool(name="_slow_a", description="t", parameters={"type": "object", "properties": {}}, category="test")
        def _a(ctx):
            time.sleep(0.4)
            return {"value": 1}

        @tool(name="_slow_b", description="t", parameters={"type": "object", "properties": {}}, category="test")
        def _b(ctx):
            time.sleep(0.4)
            return {"value": 2}
        try:
            calls = [{"id": "1", "name": "_slow_a", "arguments": "{}"}, {"id": "2", "name": "_slow_b", "arguments": "{}"}]
            t0 = time.perf_counter()
            outs = execute_calls(calls, self.ctx())
            elapsed = time.perf_counter() - t0
        finally:
            unregister("_slow_a")
            unregister("_slow_b")
        self.assertLess(elapsed, 0.7, f"took {elapsed:.2f}s -- not concurrent")
        self.assertEqual([o.result["value"] for o in outs], [1, 2])
        self.assertTrue(all(o.parallel for o in outs))

    def test_concurrent_charts_do_not_interfere(self):
        calls = [{"id": str(i), "name": "histogram", "arguments": json.dumps({"column": c})}
                 for i, c in enumerate(["revenue", "rating", "unit_price", "quantity", "customer_age"] * 2)]
        outs = execute_calls(calls, self.ctx())
        self.assertTrue(all(o.ok for o in outs), [o.result for o in outs if not o.ok])
        self.assertEqual(len({o.result["file_path"] for o in outs}), len(outs))     # unique files

    def test_mutating_tools_are_barriers_never_parallel(self):
        calls = [{"id": "1", "name": "mean", "arguments": json.dumps({"column": "rating"})},
                 {"id": "2", "name": "fill_missing", "arguments": json.dumps({"strategy": "constant", "columns": ["rating"], "value": 0})},
                 {"id": "3", "name": "mean", "arguments": json.dumps({"column": "rating"})},
                 {"id": "4", "name": "count", "arguments": json.dumps({"column": "rating"})}]
        self.assertEqual([[c["id"] for c in b] for b in plan_batches(calls)], [["1"], ["2"], ["3", "4"]])
        before, fill, after, cnt = execute_calls(calls, self.ctx())
        self.assertAlmostEqual(before.result["value"], DF.rating.mean(), places=3)          # ran BEFORE the fill
        self.assertAlmostEqual(after.result["value"], DF.rating.fillna(0).mean(), places=3)  # ran AFTER the fill
        self.assertEqual(cnt.result["value"], len(DF))
        self.assertEqual([before.parallel, fill.parallel, after.parallel, cnt.parallel], [False, False, True, True])

    def test_one_failure_does_not_break_the_others(self):
        calls = [{"id": "1", "name": "mean", "arguments": json.dumps({"column": "nope"})},
                 {"id": "2", "name": "median", "arguments": json.dumps({"column": "revenue"})}]
        bad, good = execute_calls(calls, self.ctx())
        self.assertFalse(bad.ok)
        self.assertTrue(good.ok)


class TestRecovery(unittest.TestCase):
    def test_unknown_tool_then_recovers(self):
        def policy(messages, kw):
            r = tool_results(messages)
            if not r:
                return msg(calls=[tc("average", {"column": "revenue"})])
            if "error" in r[-1]:
                self.assertIn("Only the provided tools", r[-1]["error"])
                return msg(calls=[tc("mean", {"column": "revenue"})])
            return msg(content=f"mean is {r[-1]['value']}")
        out = run_agent(DF, "q", client=FakeClient(policy))
        self.assertEqual([t["ok"] for t in out["tool_trace"]], [False, True])
        self.assertIn(str(round(DF.revenue.mean(), 4)), out["answer"])

    def test_malformed_json_arguments_recovers(self):
        def policy(messages, kw):
            r = tool_results(messages)
            if not r:
                return msg(calls=[tc("mean", '{"column": "revenue"')])          # truncated JSON
            if "error" in r[-1]:
                self.assertIn("not valid JSON", r[-1]["error"])
                return msg(calls=[tc("mean", {"column": "revenue"})])
            return msg(content="ok")
        out = run_agent(DF, "q", client=FakeClient(policy))
        self.assertEqual([t["ok"] for t in out["tool_trace"]], [False, True])

    def test_bad_column_error_reaches_model_with_hint(self):
        seen = {}
        def policy(messages, kw):
            r = tool_results(messages)
            if not r:
                return msg(calls=[tc("mean", {"column": "revenu"})])
            seen["error"] = r[-1].get("error", "")
            return msg(content="sorry")
        run_agent(DF, "q", client=FakeClient(policy))
        self.assertIn("Did you mean: revenue", seen["error"])

    def test_tool_use_failed_is_retried_with_nudge(self):
        state = {"n": 0}
        def policy(messages, kw):
            state["n"] += 1
            if state["n"] == 1:
                return ToolCallFormatError("bad call")
            self.assertIn("malformed", messages[-1]["content"])
            return msg(content="recovered")
        self.assertEqual(run_agent(DF, "q", client=FakeClient(policy))["answer"], "recovered")

    def test_repeated_format_failures_give_friendly_message(self):
        out = run_agent(DF, "q", client=FakeClient(lambda m, k: ToolCallFormatError("bad")))
        self.assertIn("trouble forming", out["answer"])

    def test_round_limit_forces_a_final_answer_without_tools(self):
        def policy(messages, kw):
            if kw.get("tool_choice") == "none":
                return msg(content="Here is what I found so far.")
            return msg(calls=[tc("count", {})])
        with mock.patch.dict(os.environ, {"MAX_TOOL_ROUNDS": "3"}):
            client = FakeClient(policy)
            out = run_agent(DF, "q", client=client)
        self.assertEqual(out["answer"], "Here is what I found so far.")
        self.assertEqual(len(out["tool_trace"]), 3)
        self.assertEqual(client.requests[-1]["tool_choice"], "none")

    def test_empty_model_reply(self):
        self.assertEqual(run_agent(DF, "q", client=FakeClient(lambda m, k: msg(content="")))["answer"], FALLBACK_ANSWER)
        self.assertEqual(run_agent(DF, "q", client=FakeClient(lambda m, k: msg(content=None)))["answer"], FALLBACK_ANSWER)

    def test_empty_content_after_tools_asks_again_for_text(self):
        def policy(messages, kw):
            if kw.get("tool_choice") == "none":
                return msg(content="final text")
            return msg(calls=[tc("count", {})]) if not tool_results(messages) else msg(content="")
        self.assertEqual(run_agent(DF, "q", client=FakeClient(policy))["answer"], "final text")

    def test_unexpected_response_shape(self):
        client = FakeClient(lambda m, k: None)
        client.chat.completions.create = lambda **kw: type("R", (), {"choices": []})()
        with self.assertRaises(LLMError):
            run_agent(DF, "q", client=client)

    def test_tool_call_cap(self):
        def policy(messages, kw):
            if kw.get("tool_choice") == "none":
                return msg(content="stopped")
            return msg(calls=[tc("count", {}) for _ in range(30)])
        out = run_agent(DF, "q", client=FakeClient(policy))
        self.assertLessEqual(len(out["tool_trace"]), orchestrator.MAX_TOOL_CALLS)


class TestPrompt(unittest.TestCase):
    def test_prompt_is_compact_and_informative(self):
        p = build_system_prompt(DF)
        self.assertLess(len(p), 3500)
        for needle in ("region", "West", "order_date", "2024-01", "never guess", "same turn", "one step at a time",
                       "authoritative", "filters"):
            self.assertIn(needle.lower(), p.lower(), needle)
        self.assertNotIn("group_aggregate", p)        # tool docs come from the native schemas, not the prompt
        self.assertIn("MEMORY-HOOK", build_system_prompt(DF, "MEMORY-HOOK"))


class TestGroqSdkWire(unittest.TestCase):
    """Runs the loop through the REAL groq SDK with the HTTP layer mocked (no network, no key)."""

    def test_full_flow_payload_and_parsing(self):
        seen = []

        def handler(request: httpx.Request):
            body = json.loads(request.content)
            seen.append(body)
            self.assertTrue(request.url.path.endswith("/chat/completions"), request.url.path)
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            if not any(m["role"] == "tool" for m in body["messages"]):
                return httpx.Response(200, json=completion_json(tool_calls=[
                    {"id": "call_a", "name": "group_aggregate",
                     "args": {"group_by": ["region"], "agg": "mean", "column": "revenue", "top_n": 1}}]))
            if len([m for m in body["messages"] if m["role"] == "tool"]) == 1:
                top = json.loads([m for m in body["messages"] if m["role"] == "tool"][0]["content"])["rows"][0]["region"]
                return httpx.Response(200, json=completion_json(tool_calls=[
                    {"id": "call_b", "name": "trend_analysis",
                     "args": {"date_column": "order_date", "value_column": "revenue", "freq": "M", "filters": west_filter(top)}},
                    {"id": "call_c", "name": "line_plot",
                     "args": {"x": "order_date", "y": "revenue", "resample": "M", "filters": west_filter(top)}}]))
            return httpx.Response(200, json=completion_json(content="All done."))

        out = run_agent(DF, "top region trend", client=groq_on_mock(handler))
        self.assertEqual(out["answer"], "All done.")
        self.assertEqual([t["tool"] for t in out["tool_trace"]], ["group_aggregate", "trend_analysis", "line_plot"])
        self.assertEqual([t["parallel"] for t in out["tool_trace"]], [False, True, True])
        self.assertEqual(out["rounds"], 3)

        first = seen[0]
        self.assertEqual(first["tool_choice"], "auto")
        self.assertEqual(len(first["tools"]), 39)
        self.assertEqual(first["model"], "llama-3.3-70b-versatile")
        self.assertEqual(first["messages"][0]["role"], "system")
        # assistant tool_calls + tool results were serialised the way the API requires
        last = seen[-1]["messages"]
        assistant = [m for m in last if m["role"] == "assistant"]
        self.assertEqual(assistant[1]["tool_calls"][0]["function"]["name"], "trend_analysis")
        self.assertIsInstance(assistant[1]["tool_calls"][0]["function"]["arguments"], str)
        tool_msgs = [m for m in last if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["call_a", "call_b", "call_c"])

    def test_400_tool_use_failed_is_recovered(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            body = json.loads(request.content)
            if calls["n"] == 1:
                return httpx.Response(400, json={"error": {"message": "Failed to call a function.", "type": "invalid_request_error",
                                                            "code": "tool_use_failed", "failed_generation": "<function=mean>"}})
            self.assertIn("malformed", body["messages"][-1]["content"])
            return httpx.Response(200, json=completion_json(content="fine now"))
        self.assertEqual(run_agent(DF, "q", client=groq_on_mock(handler))["answer"], "fine now")

    def test_http_errors_map_to_friendly_exceptions(self):
        def resp(status, body=None, headers=None):
            return lambda request: httpx.Response(status, json=body or {"error": {"message": "x", "type": "e"}}, headers=headers or {})
        with self.assertRaises(LLMRateLimitError):
            run_agent(DF, "q", client=groq_on_mock(resp(429, headers={"retry-after": "1"})))
        with self.assertRaises(LLMError) as ctx:
            run_agent(DF, "q", client=groq_on_mock(resp(401)))
        self.assertIn("GROQ_API_KEY", str(ctx.exception))
        with self.assertRaises(LLMError):
            run_agent(DF, "q", client=groq_on_mock(resp(500)))
        with self.assertRaises(LLMError):
            run_agent(DF, "q", client=groq_on_mock(resp(400)))

        def connection_error(request):
            raise httpx.ConnectError("boom", request=request)
        with self.assertRaises(LLMError) as ctx:
            run_agent(DF, "q", client=groq_on_mock(connection_error))
        self.assertNotIn("boom", str(ctx.exception))

    def test_model_returning_no_choices(self):
        with self.assertRaises(LLMError):
            run_agent(DF, "q", client=groq_on_mock(lambda r: httpx.Response(200, json={**completion_json(content="x"), "choices": []})))


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from api.app import create_app
        cls.app = create_app()
        cls.client = cls.app.test_client()

    def with_llm(self, handler):
        return mock.patch("agent.orchestrator.get_client", return_value=groq_on_mock(handler))

    def test_health_tools_schema(self):
        self.assertEqual(self.client.get("/api/health").get_json(), {"status": "ok"})
        tools = self.client.get("/api/tools").get_json()
        self.assertEqual(tools["count"], 39)
        self.assertEqual({t["category"] for t in tools["tools"]}, {"mathematical_operations", "data_manipulation", "data_summary",
                                                                    "graphs", "correlation_analysis", "outlier_analysis"})
        sch = self.client.get("/api/datasets/sample/schema").get_json()
        self.assertEqual(sch["schema"]["n_rows"], 400)
        self.assertIn({"name": "region", "dtype": "str"}["name"], [c["name"] for c in sch["schema"]["columns"]])
        self.assertTrue(all("name" in c and "dtype" in c for c in sch["schema"]["columns"]))   # what script.js reads
        self.assertEqual(self.client.get("/api/datasets/nope/schema").status_code, 404)

    def test_upload_validation(self):
        post = lambda data: self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        ok = post({"file": (io.BytesIO(b"a,b\n1,x\n2,y\n"), "t.csv")})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.get_json()["schema"]["n_rows"], 2)
        self.assertEqual(post({}).status_code, 400)
        self.assertEqual(post({"file": (io.BytesIO(b"x"), "t.txt")}).status_code, 400)
        empty = post({"file": (io.BytesIO(b""), "e.csv")})
        self.assertEqual(empty.status_code, 400)
        self.assertNotIn("Traceback", json.dumps(empty.get_json()))

    def test_ask_end_to_end_contract_and_plot_serving(self):
        def handler(request):
            body = json.loads(request.content)
            if not any(m["role"] == "tool" for m in body["messages"]):
                return httpx.Response(200, json=completion_json(tool_calls=[
                    {"id": "c1", "name": "mean", "args": {"column": "revenue"}},
                    {"id": "c2", "name": "bar_plot", "args": {"x": "category", "y": "revenue"}}]))
            return httpx.Response(200, json=completion_json(content="Average revenue computed and charted."))
        with self.with_llm(handler):
            r = self.client.post("/api/ask", json={"dataset_id": "sample", "question": "avg revenue and a chart"})
        data = r.get_json()
        self.assertEqual(r.status_code, 200, data)
        self.assertEqual(set(data), {"answer", "tool_trace", "plot_urls", "rounds"})
        self.assertEqual([t["tool"] for t in data["tool_trace"]], ["mean", "bar_plot"])
        self.assertTrue(all({"tool", "arguments", "ok", "summary", "parallel", "round", "duration_ms"} <= set(t) for t in data["tool_trace"]))
        self.assertEqual(len(data["plot_urls"]), 1)
        img = self.client.get(data["plot_urls"][0])
        self.assertEqual((img.status_code, img.data[:4]), (200, b"\x89PNG"))
        img.close()

    def test_ask_errors_are_friendly(self):
        ask = lambda body: self.client.post("/api/ask", json=body)
        self.assertEqual(ask({}).status_code, 400)
        self.assertEqual(ask({"dataset_id": "sample", "question": "x" * 5000}).status_code, 400)
        self.assertEqual(ask({"dataset_id": "zzz", "question": "hi"}).status_code, 404)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            r = ask({"dataset_id": "sample", "question": "hi"})
        self.assertEqual(r.status_code, 500)
        self.assertIn("GROQ_API_KEY", r.get_json()["error"])
        with self.with_llm(lambda req: httpx.Response(429, json={"error": {"message": "x"}})):
            self.assertEqual(ask({"dataset_id": "sample", "question": "hi"}).status_code, 429)
        with self.with_llm(lambda req: httpx.Response(500, json={"error": {"message": "x"}})):
            self.assertEqual(ask({"dataset_id": "sample", "question": "hi"}).status_code, 502)
        with mock.patch("api.routes.run_agent", side_effect=ValueError("secret /etc stuff")):
            r = ask({"dataset_id": "sample", "question": "hi"})
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("secret", r.get_json()["error"])

    def test_cors_headers_for_frontend(self):
        r = self.client.options("/api/ask", headers={"Origin": "http://localhost:8080", "Access-Control-Request-Method": "POST"})
        self.assertEqual(r.headers.get("Access-Control-Allow-Origin"), "http://localhost:8080")


if __name__ == "__main__":
    unittest.main()
