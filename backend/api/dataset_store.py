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
        self._by_name = {}   # filename -> most recent dataset_id
        self._stash = {}     # content_hash -> {df, name}, awaiting same-vs-new resolution
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

 # ---- filename -> most-recent dataset_id, for re-upload detection ----

    def find_by_name(self, filename: str):
        with self._lock:
            return self._by_name.get(filename)

    def record_name(self, filename: str, dataset_id: str) -> None:
        with self._lock:
            self._by_name[filename] = dataset_id

    # ---- a new upload held here until the user resolves same-vs-new ----

    def stash(self, content_hash: str, df: pd.DataFrame, name: str) -> None:
        with self._lock:
            self._stash[content_hash] = {"df": df, "name": name}

    def pop_stash(self, content_hash: str):
        with self._lock:
            return self._stash.pop(content_hash, None)

    # ---- pending edits from the most recent /api/ask, awaiting commit ----

    def set_pending(self, dataset_id: str, df: pd.DataFrame) -> None:
        with self._lock:
            if dataset_id in self._store:
                self._store[dataset_id]["pending_df"] = df

    def get_pending(self, dataset_id: str):
        with self._lock:
            entry = self._store.get(dataset_id)
        return entry.get("pending_df") if entry else None

    def commit(self, dataset_id: str) -> bool:
        """Promote the pending (edited) DataFrame to be the dataset's main
        DataFrame. Returns False if there's nothing pending to commit."""
        with self._lock:
            entry = self._store.get(dataset_id)
            if not entry or entry.get("pending_df") is None:
                return False
            entry["df"] = entry["pending_df"]
            entry["pending_df"] = None
            return True


# Single shared instance used by the whole Flask app
store = DatasetStore()






   