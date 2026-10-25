"""Persistent caches for embeddings and generations (SQLite).

Free-tier API limits mean a full run takes many sessions. Every embedding and
every model response is stored by a hash of its exact inputs, so an interrupted
run resumes where it stopped, and identical prompts shared by several
experiments (e.g. the K=3 / no-noise baseline) are only paid for once.
"""
import hashlib
import json
import os
import sqlite3
import threading

import numpy as np


def key_of(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class Cache:
    def __init__(self, path: str):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        self.db.execute("CREATE TABLE IF NOT EXISTS gen (k TEXT PRIMARY KEY, v TEXT)")
        self.db.commit()

    # embeddings ---------------------------------------------------------------
    def get_emb(self, k):
        with self._lock:
            row = self.db.execute("SELECT v FROM emb WHERE k=?", (k,)).fetchone()
        return None if row is None else np.frombuffer(row[0], dtype=np.float32)

    def put_embs(self, items):
        with self._lock:
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                                [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items])
            self.db.commit()

    # generations --------------------------------------------------------------
    def get_gen(self, k):
        with self._lock:
            row = self.db.execute("SELECT v FROM gen WHERE k=?", (k,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put_gen(self, k, value: dict):
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO gen VALUES (?, ?)", (k, json.dumps(value)))
            self.db.commit()

    def has_gen(self, k) -> bool:
        with self._lock:
            return self.db.execute("SELECT 1 FROM gen WHERE k=?", (k,)).fetchone() is not None
