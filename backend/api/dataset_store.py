"""
dataset_store.py
-------------------
A minimal in-memory store mapping dataset_id -> DataFrame, so the API can
hold multiple uploaded datasets across requests without a database. This
is intentionally simple for a demo/interview project:

  - Data lives only in server RAM and is lost on restart.
  - Not safe for concurrent multi-process deployment (fine for `flask run`
    / a single dev process).

A production version would swap this for Redis, a temp-file cache keyed
by id, or an actual database -- the rest of the API doesn't need to
change since it only calls .add() / .get() / .get_name().
"""

import threading
import uuid
import pandas as pd


class DatasetStore:
    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()

    def add(self, df: pd.DataFrame, name: str) -> str:
        dataset_id = str(uuid.uuid4())
        with self._lock:
            self._store[dataset_id] = {"df": df, "name": name}
        return dataset_id

    def add_with_id(self, dataset_id: str, df: pd.DataFrame, name: str) -> None:
        with self._lock:
            self._store[dataset_id] = {"df": df, "name": name}

    def get(self, dataset_id: str):
        with self._lock:
            entry = self._store.get(dataset_id)
        return entry["df"] if entry else None

    def get_name(self, dataset_id: str):
        with self._lock:
            entry = self._store.get(dataset_id)
        return entry["name"] if entry else None

    def exists(self, dataset_id: str) -> bool:
        with self._lock:
            return dataset_id in self._store


# Single shared instance used by the whole Flask app
store = DatasetStore()
