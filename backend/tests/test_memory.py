"""Memory layer: SQLite store, remember/forget/list_memory tools, and end-to-end context injection."""

import tempfile
import unittest

import pandas as pd
from tests.helpers import SAMPLE_CSV
from tests.helpers import FakeClient, msg, tc, tool_results

from agent.memory import store
from agent.tools._base import ToolContext
from agent import tool_registry as tr
from agent.data_loader import load_csv
from agent.orchestrator import run_agent

from unittest import mock


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.patcher = mock.patch("agent.memory.store.MEMORY_DB_PATH", self.tmp.name)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        import os
        self.tmp.close()
        os.unlink(self.tmp.name)

    def test_upsert_then_get_all(self):
        store.upsert("ds1", "preference", "freq", "weekly")
        rows = store.get_all("ds1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["key"], "freq")
        self.assertEqual(rows[0]["value"], "weekly")
        self.assertEqual(rows[0]["kind"], "preference")

    def test_upsert_overwrites_same_key(self):
        store.upsert("ds1", "preference", "freq", "weekly")
        store.upsert("ds1", "preference", "freq", "daily")
        rows = store.get_all("ds1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "daily")

    def test_scopes_are_isolated(self):
        store.upsert("ds1", "definition", "active", "not cancelled")
        store.upsert("ds2", "definition", "active", "something else")
        self.assertEqual(len(store.get_all("ds1")), 1)
        self.assertEqual(len(store.get_all("ds2")), 1)
        self.assertNotEqual(store.get_all("ds1")[0]["value"], store.get_all("ds2")[0]["value"])

    def test_delete_returns_true_then_false(self):
        store.upsert("ds1", "preference", "freq", "weekly")
        self.assertTrue(store.delete("ds1", "freq"))
        self.assertFalse(store.delete("ds1", "freq"))
        self.assertEqual(store.get_all("ds1"), [])

    def test_empty_scope_raises(self):
        with self.assertRaises(ValueError):
            store.upsert("", "preference", "freq", "weekly")
        with self.assertRaises(ValueError):
            store.get_all("")

class TestMemoryTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.patcher = mock.patch("agent.memory.store.MEMORY_DB_PATH", self.tmp.name)
        self.patcher.start()
        self.ctx = ToolContext(df=pd.DataFrame(), plots_dir=tempfile.mkdtemp(), scope="ds_test")

    def tearDown(self):
        self.patcher.stop()
        import os
        self.tmp.close()
        os.unlink(self.tmp.name)

    def test_remember_dispatch_writes_to_store(self):
        out = tr.dispatch("remember", {"key": "active", "value": "not cancelled", "kind": "definition"}, self.ctx)
        self.assertNotIn("error", out)
        self.assertEqual(out["remembered"], "active")
        self.assertEqual(store.get_all("ds_test")[0]["key"], "active")

    def test_remember_empty_key_errors(self):
        out = tr.dispatch("remember", {"key": "  ", "value": "x", "kind": "preference"}, self.ctx)
        self.assertIn("error", out)

    def test_forget_missing_key_errors(self):
        out = tr.dispatch("forget", {"key": "nope"}, self.ctx)
        self.assertIn("error", out)
        self.assertIn("nope", out["error"])

    def test_forget_existing_key_removes_it(self):
        tr.dispatch("remember", {"key": "active", "value": "x", "kind": "definition"}, self.ctx)
        out = tr.dispatch("forget", {"key": "active"}, self.ctx)
        self.assertNotIn("error", out)
        self.assertEqual(store.get_all("ds_test"), [])

    def test_list_memory_reflects_store(self):
        tr.dispatch("remember", {"key": "a", "value": "1", "kind": "preference"}, self.ctx)
        tr.dispatch("remember", {"key": "b", "value": "2", "kind": "preference"}, self.ctx)
        out = tr.dispatch("list_memory", {}, self.ctx)
        self.assertEqual(out["count"], 2)

    def test_scope_isolation_through_dispatch(self):
        other_ctx = ToolContext(df=pd.DataFrame(), plots_dir=tempfile.mkdtemp(), scope="other_ds")
        tr.dispatch("remember", {"key": "x", "value": "1", "kind": "preference"}, self.ctx)
        out = tr.dispatch("list_memory", {}, other_ctx)
        self.assertEqual(out["count"], 0)

class TestMemoryEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.patcher = mock.patch("agent.memory.store.MEMORY_DB_PATH", self.tmp.name)
        self.patcher.start()
        self.df = load_csv(SAMPLE_CSV)  # import SAMPLE_CSV from tests.helpers at top

    def tearDown(self):
        self.patcher.stop()
        import os
        self.tmp.close()
        os.unlink(self.tmp.name)

    def test_remember_tool_call_lands_in_store_with_correct_scope(self):
        def policy(messages, kw):
            if not tool_results(messages):
                return msg(calls=[tc("remember", {"key": "active_customers", "value": "excludes cancelled", "kind": "definition"})])
            return msg(content="noted")

        out = run_agent(self.df, "remember this", dataset_id="ds_e2e", client=FakeClient(policy))
        self.assertEqual(out["answer"], "noted")
        rows = store.get_all("ds_e2e")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["key"], "active_customers")

    def test_extra_context_reaches_system_prompt(self):
        captured = {}

        def policy(messages, kw):
            captured["system"] = messages[0]["content"]
            return msg(content="ok")

        run_agent(self.df, "anything", dataset_id="ds_e2e", client=FakeClient(policy),
                  extra_context="Remembered facts for this dataset:\n- (definition) active_customers: excludes cancelled")
        self.assertIn("active_customers", captured["system"])
        self.assertIn("excludes cancelled", captured["system"])