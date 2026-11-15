"""Offline search eval: build the real index from the data files (no DB) and check judged queries.

    pytest tests/search_eval                       # lexical (default run) + semantic when vectors exist
    python -m tests.search_eval [--semantic PATH.npz] [--fusion or|rrf] [-v]   # report + latency table

Semantic vectors: `python -m scripts.load_places --offline-embed PATH.npz` (needs fastembed), then
point SEARCH_EVAL_VECTORS at the file (and SEARCH_EMBED_MODEL at the model that made it).
"""
from __future__ import annotations

import math
import os
import re
import statistics
import threading
import time
from datetime import datetime

USC = (34.0224, -118.2851)
# Fixed clock so open_now / hours-derived checks are reproducible: a Saturday, 1 pm.
NOW = datetime(2026, 11, 14, 13, 0)

_lock = threading.Lock()
_cache = {}


def load_rows():
    with _lock:
        if "rows" not in _cache:
            from app import create_app
            from scripts.load_places import collect_rows

            app = create_app(TESTING=True)
            with app.app_context():
                batches = collect_rows(app.config["DATA_DIR"], log=lambda *a: None)
            _cache["rows"] = [r for _, rows, _ in batches for r in rows]
        return _cache["rows"]


def load_vectors(path):
    import numpy as np

    z = np.load(path, allow_pickle=False)
    return {str(i): v for i, v in zip(z["ids"], z["vectors"])}, str(z["model"])


def build_index(vectors_path=None):
    from app.services.search_index import SearchIndex

    key = ("index", vectors_path)
    with _lock:
        if key in _cache:
            return _cache[key]
    rows = load_rows()
    vectors, model = load_vectors(vectors_path) if vectors_path else (None, None)
    t = time.perf_counter()
    idx = SearchIndex(rows, vectors=vectors, semantic_model=model)
    idx.build_seconds = time.perf_counter() - t
    with _lock:
        _cache[key] = idx
    return idx


def eval_vectors_path():
    p = os.getenv("SEARCH_EVAL_VECTORS")
    return p if p and os.path.exists(p) else None


class Runner:
    """Runs a case against an index, optionally with a query encoder (semantic)."""

    def __init__(self, idx, encoder=None, fusion="or"):
        from app.services.opening_hours import is_open

        self.idx, self.encoder, self.fusion = idx, encoder, fusion
        self.is_open = is_open
        self.rows = {r["id"]: r for r in load_rows()}

    def run(self, case):
        pq = self.idx.parse(case.q) if case.q else None
        qvec = None
        if self.encoder is not None and pq is not None and pq.semantic_text:
            qvec = self.encoder.encode(pq.semantic_text)
        return self.idx.search(case.q, near=case.near, categories=case.categories, limit=case.limit,
                               now=NOW, qvec=qvec, is_open_fn=self.is_open, fusion=self.fusion, explain=True)

    def check(self, case):
        res = self.run(case)
        fails = [msg for c in case.checks for ok, msg in [c(res, self)] if not ok]
        return res, fails


# ---------------------------------------------------------------------------------- expectations

def _text(r, runner):
    row = runner.rows.get(r["id"], {})
    return " | ".join([r.get("category") or "", r.get("subcategory") or "", " ".join(r.get("_labels") or []),
                       (row.get("detail") or {}).get("poi_type") or ""])


def _is(r, runner, pattern):
    return re.search(pattern, _text(r, runner), re.I) is not None


def _names(res, k):
    return [r["name"] for r in res[:k]]


def top_type(k, pattern, frac=1.0):
    """At least frac of the top k are of the type (category / subcategory / labels match pattern)."""
    def check(res, runner):
        top = res[:k]
        n = sum(_is(r, runner, pattern) for r in top)
        ok = len(top) >= min(k, 1) and n >= math.ceil(frac * k)
        return ok, f"top{k} type /{pattern}/: {n}/{len(top)} match {_names(res, k)}"
    return check


def top_type_or_name(k, pattern, frac=1.0):
    """Like top_type, but the name counts too ("Mary's Tacos" is a taco place whatever its label)."""
    def check(res, runner):
        top = res[:k]
        n = sum(_is(r, runner, pattern) or re.search(pattern, r["name"], re.I) is not None for r in top)
        return bool(top) and n >= math.ceil(frac * k), f"top{k} type-or-name /{pattern}/: {n} match {_names(res, k)}"
    return check


def in_top_id(k, pid):
    def check(res, runner):
        return any(r["id"] == pid for r in res[:k]), f"{pid} in top{k}: got {[r['id'] for r in res[:k]]}"
    return check


def first_fields(**expected):
    """The top result has these field values (room results: floor, pathing, verified, ...)."""
    def check(res, runner):
        got = {k: res[0].get(k) for k in expected} if res else None
        return bool(res) and got == expected, f"top1 {expected}: got {got}"
    return check


def second_id(pid):
    def check(res, runner):
        return len(res) > 1 and res[1]["id"] == pid, f"top2 id {pid}: got {[r['id'] for r in res[:2]]}"
    return check


def no_rooms(k=5):
    def check(res, runner):
        bad = [r["id"] for r in res[:k] if r.get("kind") == "room"]
        return not bad, f"no room results in top{k}: got {bad}"
    return check


def rest_rooms_of(code, prefix, start=1):
    """Results after `start` are rooms of `code` whose number starts with `prefix` (and there is one)."""
    def check(res, runner):
        rest = res[start:]
        ok = bool(rest) and all(r.get("kind") == "room" and r.get("code") == code
                                and str(r.get("room", "")).startswith(prefix) for r in rest)
        return ok, f"results[{start}:] rooms {code} {prefix}*: got {[r['name'] for r in rest[:6]]}"
    return check


def top_name(k, pattern, frac=1.0):
    def check(res, runner):
        n = sum(re.search(pattern, r["name"], re.I) is not None for r in res[:k])
        return n >= math.ceil(frac * k), f"top{k} name /{pattern}/: {n} match {_names(res, k)}"
    return check


def first_name(pattern):
    def check(res, runner):
        ok = bool(res) and re.search(pattern, res[0]["name"], re.I) is not None
        return ok, f"top1 name /{pattern}/: got {_names(res, 1)}"
    return check


def in_top_name(k, pattern):
    def check(res, runner):
        ok = any(re.search(pattern, r["name"], re.I) for r in res[:k])
        return ok, f"/{pattern}/ in top{k}: got {_names(res, k)}"
    return check


def not_in_top_name(k, pattern):
    def check(res, runner):
        bad = [r["name"] for r in res[:k] if re.search(pattern, r["name"], re.I)]
        return not bad, f"/{pattern}/ must not be in top{k}: got {bad}"
    return check


def first_id(pid):
    def check(res, runner):
        return bool(res) and res[0]["id"] == pid, f"top1 id {pid}: got {[r['id'] for r in res[:2]]}"
    return check


def min_results(n):
    def check(res, runner):
        return len(res) >= n, f">= {n} results: got {len(res)}"
    return check


def max_results(n):
    def check(res, runner):
        return len(res) <= n, f"<= {n} results: got {len(res)} {_names(res, 3)}"
    return check


def median_km(k, km):
    def check(res, runner):
        ds = [r["distance_m"] for r in res[:k] if "distance_m" in r]
        med = statistics.median(ds) / 1000 if ds else float("inf")
        return med <= km, f"top{k} median distance <= {km} km: got {med:.2f} km"
    return check


def first_within_km(km):
    def check(res, runner):
        d = res[0]["distance_m"] / 1000 if res else float("inf")
        return d <= km, f"top1 within {km} km: got {d:.2f} km ({_names(res, 1)})"
    return check


def top_address(k, pattern, frac=1.0):
    def check(res, runner):
        n = sum(re.search(pattern, r.get("address") or "", re.I) is not None for r in res[:k])
        return n >= math.ceil(frac * k), f"top{k} address /{pattern}/: {n} match {[r.get('address') for r in res[:k]]}"
    return check


def top_near_point(k, lat, lng, km):
    from app.services.search_index import haversine_m

    def check(res, runner):
        ds = [float(haversine_m(lat, lng, r["lat"], r["lng"])) / 1000 for r in res[:k]]
        return bool(ds) and max(ds) <= km, f"top{k} within {km} km of ({lat},{lng}): got {[round(d, 2) for d in ds]}"
    return check


def top_has_restroom(k, frac=1.0):
    def check(res, runner):
        def ok(r):
            row = runner.rows.get(r["id"], {})
            attrs = " ".join(o for g in (row.get("detail") or {}).get("attributes") or [] if isinstance(g, dict)
                             for o in g.get("options") or [])
            return bool(re.search(r"restroom|bathroom|toilet", attrs + " " + _text(r, runner) + " " + r["name"], re.I))
        n = sum(ok(r) for r in res[:k])
        return n >= math.ceil(frac * k), f"top{k} with a restroom: {n} {_names(res, k)}"
    return check


def top_open_late(k, frac):
    from app.services.search_index import hours_tags

    def check(res, runner):
        n = sum("latenight" in hours_tags(runner.rows.get(r["id"], {}).get("hours")) for r in res[:k])
        return n >= math.ceil(frac * k), f"top{k} open late: {n} {_names(res, k)}"
    return check


class Case:
    def __init__(self, q, *checks, near=USC, categories=None, limit=20, semantic_only=False, held_out=False, why=""):
        self.q, self.checks, self.near, self.categories, self.limit = q, checks, near, categories, limit
        self.semantic_only = semantic_only     # only expected to pass with the semantic signal
        self.held_out = held_out               # not used for tuning; reported separately
        self.why = why

    def __repr__(self):
        return f"Case({self.q!r})"


# ---------------------------------------------------------------------------------- report

def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def latency(idx, cases, encoder=None, repeats=3):
    """p50/p95 (ms) of index search (warm parse cache excluded: parse is timed too, cold), and query embeds."""
    from app.services.opening_hours import is_open

    from app.services.search_semantic import get_embedder

    search_ms, embed_ms = [], []
    for rep in range(repeats):
        idx._qcache = type(idx._qcache)(idx._qcache.n)          # cold query-understanding cache
        idx._fcache = type(idx._fcache)(idx._fcache.n)
        if encoder is not None:
            get_embedder()._cache.clear()                        # cold query-embedding cache
        for c in cases:
            qvec = None
            t0 = time.perf_counter()
            pq = idx.parse(c.q) if c.q else None
            t1 = time.perf_counter()
            if encoder is not None and pq is not None and pq.semantic_text:
                qvec = encoder.encode(pq.semantic_text)
                if rep == 0:
                    embed_ms.append((time.perf_counter() - t1) * 1000)
            t2 = time.perf_counter()
            idx.search(c.q, near=c.near, categories=c.categories, limit=c.limit, now=NOW, qvec=qvec, is_open_fn=is_open)
            search_ms.append(((t1 - t0) + (time.perf_counter() - t2)) * 1000)
    return search_ms, embed_ms
