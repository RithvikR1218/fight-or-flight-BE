"""Place search service: one in-memory index per worker process, plus place detail / categories /
reverse geocoding (those still read the DB).

Retrieval and ranking live in services/search_index.py (pure numpy, no DB) and are unit-tested and
evaluated offline (tests/search_eval). This module only:

* loads the index lazily on the first search (all searchable fields of every place, ~10k rows,
  plus stored embeddings when present) and keeps it per worker process;
* every SEARCH_INDEX_TTL seconds (default 60) runs one cheap query for search_meta.places_version
  (bumped by scripts/load_places at the end of a load) and rebuilds in the background of the
  request that notices a change, while other threads keep serving the old index;
* embeds the query for the semantic signal when available (services/search_semantic.py).

Memory per worker: index ~40 MB (Python objects) + 15 MB vectors; the ONNX embedding model adds
~100-130 MB once loaded. Thread-safe: the index is immutable and swapped atomically.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import current_app, has_app_context

from ..db_utils import fetch_all
from ..extensions import db
from .search_index import SearchIndex, haversine_m
from .text_norm import normalize

log = logging.getLogger(__name__)
LA = ZoneInfo("America/Los_Angeles")

INDEX_COLUMNS = ("id, kind, name, aliases, category, subcategory, categories, address, lat, lng, rating, "
                 "review_count, popularity, hours, detail")
DEFAULT_TTL_S = 60.0

_state = {"index": None, "version": None, "checked": 0.0, "meta": {}}
_state_lock = threading.Lock()
_build_lock = threading.Lock()
_encoder = None
_warned = set()


def _warn_once(key, msg, *args):
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


def _config(key, default):
    if has_app_context():
        return current_app.config.get(key, default)
    return default


# ---------------------------------------------------------------- loading the index

def read_meta():
    """search_meta as a dict ({} when the table does not exist yet, i.e. before migration 0005)."""
    try:
        rows = fetch_all("SELECT key, value FROM search_meta")
        return {r["key"]: r["value"] for r in rows}
    except Exception:
        db.session.rollback()
        return {}


def current_version(meta=None):
    meta = read_meta() if meta is None else meta
    if meta.get("places_version"):
        return meta["places_version"]
    # pre-0005 fallback: changes whenever rows are added or removed
    row = fetch_all("SELECT count(*)::text AS n, coalesce(max(id), '') AS m FROM places")[0]
    return f"count:{row['n']}:{row['m']}"


def load_rows():
    """Every searchable place, with its embedding when the column exists."""
    try:
        return fetch_all(f"SELECT {INDEX_COLUMNS}, embedding FROM places")
    except Exception as e:   # migration 0005 not applied yet: no embedding column
        db.session.rollback()
        _warn_once("no-embedding-col", "places.embedding unavailable (%s); semantic search off", type(e).__name__)
        return fetch_all(f"SELECT {INDEX_COLUMNS} FROM places")


def build_index_from_db():
    meta = read_meta()
    version = current_version(meta)
    t0 = time.perf_counter()
    rows = load_rows()
    idx = SearchIndex(rows, semantic_model=meta.get("embedding_model"))
    log.info("search index: %d places in %.1fs (version %s, vectors %s)", idx.N, time.perf_counter() - t0,
             version, "yes" if idx.vectors is not None else "no")
    return idx, version, meta


def get_index():
    """The worker's index; built on first use, refreshed when the loader bumped places_version."""
    ttl = float(_config("SEARCH_INDEX_TTL", DEFAULT_TTL_S))
    idx = _state["index"]
    now = time.monotonic()
    if idx is not None and now - _state["checked"] < ttl:
        return idx
    if idx is None:
        with _build_lock:                       # first request(s) wait for the first build
            if _state["index"] is None:
                _install(*build_index_from_db())
        return _state["index"]
    # stale check: one thread looks (one small query); a changed version is rebuilt in a background
    # thread while every request keeps using the current index
    if not _build_lock.acquire(blocking=False):
        return idx
    started = False
    try:
        with _state_lock:
            _state["checked"] = now
        try:
            version = current_version()
        except Exception as e:
            db.session.rollback()
            _warn_once("version", "search index version check failed: %s", e)
            return idx
        if version != _state["version"]:
            app = current_app._get_current_object()
            threading.Thread(target=_rebuild, args=(app,), name="search-index-rebuild", daemon=True).start()
            started = True
    finally:
        if not started:
            _build_lock.release()
    return idx


def _rebuild(app):
    try:
        with app.app_context():
            _install(*build_index_from_db())
    except Exception as e:
        log.warning("search index rebuild failed (keeping the old one): %s", e)
    finally:
        _build_lock.release()


def _install(idx, version, meta):
    with _state_lock:
        _state.update(index=idx, version=version, meta=meta, checked=time.monotonic())


def set_index(idx, version="test"):
    """Install an index directly (tests, offline tools)."""
    _install(idx, version, {})


def warm():
    """Build the index (and load the embedding model) now; gunicorn calls this per worker at start."""
    try:
        idx = get_index()
        enc = encoder()
        if enc is not None and idx.vectors is not None:
            enc.encode("coffee")
    except Exception as e:
        log.warning("search warm-up failed (first search will retry): %s", e)


def encoder():
    """QueryEncoder or None (SEARCH_SEMANTIC=off / fastembed missing)."""
    global _encoder
    if _encoder is None:
        from .search_semantic import QueryEncoder

        _encoder = QueryEncoder(mode=str(_config("SEARCH_SEMANTIC", "auto")).lower())
    return None if _encoder.disabled_reason else _encoder


def _query_vector(idx, pq):
    if pq is None or not pq.semantic_text or idx.vectors is None:
        return None
    enc = encoder()
    if enc is None:
        return None
    stored = _state["meta"].get("embedding_model") or idx.semantic_model
    if stored and enc.model_name and stored != enc.model_name:
        _warn_once("model-mismatch", "stored vectors are from %s but SEARCH_EMBED_MODEL is %s; semantic off "
                   "(re-run scripts.load_places)", stored, enc.model_name)
        return None
    v = enc.encode(pq.semantic_text)
    if v is not None and v.shape[-1] != idx.vectors.shape[1]:
        _warn_once("dim-mismatch", "query vector dim %s != stored %s; semantic off", v.shape[-1], idx.vectors.shape[1])
        return None
    return v


# ---------------------------------------------------------------- search

RESULT_FIELDS = ("id", "kind", "name", "category", "subcategory", "address", "lat", "lng",
                 "distance_m", "rating", "review_count", "open_now", "match")
ROOM_FIELDS = ("floor", "floor_label", "building_id", "building_name", "code", "room", "pathing", "verified")


def to_result(r):
    out = {k: r[k] for k in RESULT_FIELDS if k in r}
    if out.get("open_now") is None:
        out.pop("open_now", None)
    if r.get("kind") == "room":
        out.update({k: r.get(k) for k in ROOM_FIELDS})
    return out


def search(q=None, near=None, bbox=None, categories=None, open_now=False, limit=20, now=None):
    idx = get_index()
    if idx is None or not idx.N:
        return []
    pq = idx.parse(q) if q and normalize(q) else None
    qvec = _query_vector(idx, pq)
    ranked = idx.search(q, near=near, bbox=bbox, categories=categories, open_now=open_now, limit=limit,
                        now=now or datetime.now(LA), qvec=qvec, is_open_fn=_is_open)
    return [to_result(r) for r in ranked]


def _is_open(hours, now):
    try:
        from .opening_hours import is_open
    except ImportError:
        return None
    return is_open(hours, now)


WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _hhmm(minutes):
    minutes = int(minutes) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def hours_by_weekday(hours):
    """{"0": [[480, 1320]], ...} (Monday=0, local minutes, close may pass 1440) ->
    {"monday": [{"open": "08:00", "close": "22:00"}], ...}; a day with no intervals is closed ([]).
    Intervals that end after midnight get "overnight": true; round-the-clock ones "all_day": true."""
    if not hours:
        return None
    out = {}
    for i, day in enumerate(WEEKDAYS):
        spans = []
        for o, c in hours.get(str(i)) or hours.get(i) or []:
            span = {"open": _hhmm(o), "close": _hhmm(c)}
            if c >= 1440:
                span["overnight"] = True
            if o <= 0 and c - o >= 1440:
                span["all_day"] = True
            spans.append(span)
        out[day] = spans
    return out


# ---------------------------------------------------------------- categories, detail, reverse geocode

def _type_label(slug):
    return (slug or "").replace("_", " ").capitalize()


def category_tree(rows=None):
    """[{"id","label","count","types":[{"id","label","count"}]}] in taxonomy order, empty ones omitted."""
    from .place_taxonomy import CATEGORIES

    if rows is None:
        rows = fetch_all("""
            SELECT category, subcategory, count(*)::int AS count,
                   mode() WITHIN GROUP (ORDER BY categories[1]) AS label
            FROM places WHERE category IS NOT NULL AND kind <> 'room'
            GROUP BY category, subcategory""")
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    out = []
    known = [c["id"] for c in CATEGORIES]
    labels = {c["id"]: c["label"] for c in CATEGORIES}
    for cid in known + sorted(set(by_cat) - set(known)):
        group = by_cat.get(cid)
        if not group:
            continue
        types = [{"id": r["subcategory"],
                  "label": (r.get("label") if r.get("label") and normalize(r["label"]).replace(" ", "_") == r["subcategory"]
                            else _type_label(r["subcategory"])),
                  "count": r["count"]}
                 for r in group if r.get("subcategory")]
        types.sort(key=lambda t: (-t["count"], t["label"]))
        out.append({"id": cid, "label": labels.get(cid, _type_label(cid)),
                    "count": sum(r["count"] for r in group), "types": types})
    return out


DETAIL_KEYS = ("phone", "website", "photo_url", "rating_breakdown", "attributes", "description",
               "reviews", "plus_code")
ROOM_DETAIL_KEYS = ("floor", "floor_label", "floor_confidence", "building_id", "building_name", "code", "room",
                    "campus", "pathing", "capacity", "classroom_type", "furniture", "equipment", "url")


def place_detail(place_id, near=None, now=None):
    rows = fetch_all("""
        SELECT id, kind, name, aliases, category, subcategory, categories, address, lat, lng,
               rating, review_count, hours, detail, source
        FROM places WHERE id = :id""", id=place_id)
    if not rows:
        return synthesized_room_detail(place_id, near)
    r = rows[0]
    detail = r.pop("detail") or {}
    hours = r.pop("hours")
    out = {k: r.get(k) for k in ("id", "kind", "name", "category", "subcategory", "address", "lat", "lng",
                                 "rating", "review_count")}
    if near and r.get("lat") is not None:
        out["distance_m"] = int(round(float(haversine_m(near[0], near[1], r["lat"], r["lng"]))))
    open_now = _is_open(hours, now or datetime.now(LA)) if hours else None
    if open_now is not None:
        out["open_now"] = open_now
    out["hours"] = hours_by_weekday(hours)
    out["hours_text"] = detail.get("hours_text")
    out.update({k: detail.get(k) for k in DETAIL_KEYS})
    out["aliases"] = r.get("aliases") or []
    out["categories"] = r.get("categories") or []
    out["source"] = r.get("source")
    if r.get("kind") == "room":
        out.update({k: detail.get(k) for k in ROOM_DETAIL_KEYS})
        out["pathing"] = bool(out.get("pathing"))
        out["verified"] = True
    return out


def synthesized_room_detail(place_id, near=None):
    """/places/room:RTH-108 for a room that is not in the dataset: placed at its building, floor from
    the room number (same as the search result). None unless "<code> <room>" is a valid room."""
    if not str(place_id).startswith("room:") or "-" not in place_id:
        return None
    from .rooms import parse_room_query

    code, room = place_id[5:].split("-", 1)
    idx = get_index()
    if idx is None or not parse_room_query(f"{code} {room}", idx.known_codes):
        return None
    r = idx.room_result(code, room, near)
    if r is None:
        return None
    r.pop("match", None)
    r.update(hours=None, hours_text=None, aliases=[], categories=[], source="synthesized",
             **{k: None for k in DETAIL_KEYS})
    return r


REVERSE_RADIUS_M = 60.0


def _road_name(name):
    """osmnx stores multi-name edges as a stringified list: "['A St', 'B St']" -> "A St"."""
    if not name:
        return None
    s = str(name).strip()
    if s.startswith("["):
        s = s.strip("[]").split(",")[0].strip().strip("'\"")
    return s or None


def reverse_geocode(lat, lng):
    """Nearest named place within ~60 m (buildings preferred), else the nearest named road."""
    pt = "ST_SetSRID(ST_MakePoint(:lng, :lat), 4326)"
    # ~0.0008 deg latitude is ~89 m; the box is a cheap GiST prefilter, the exact cut is in metres
    rows = fetch_all(f"""
        SELECT id, kind, name, address, ST_DistanceSphere(geom, {pt}) AS distance_m
        FROM places
        WHERE geom && ST_Expand({pt}, 0.0009) AND name IS NOT NULL AND kind <> 'room'
          AND ST_DistanceSphere(geom, {pt}) <= :radius
        ORDER BY (kind = 'building') DESC, geom <-> {pt}
        LIMIT 1""", lat=lat, lng=lng, radius=REVERSE_RADIUS_M)
    if rows:
        r = rows[0]
        return {"name": r["name"], "address": r["address"], "id": r["id"], "kind": r["kind"],
                "distance_m": int(round(r["distance_m"]))}
    rows = fetch_all(f"""
        SELECT name, ST_DistanceSphere(ST_ClosestPoint(geom, {pt}), {pt}) AS distance_m
        FROM road_edges
        WHERE name IS NOT NULL AND name <> '' AND geom && ST_Expand({pt}, 0.003)
        ORDER BY geom <-> {pt}
        LIMIT 1""", lat=lat, lng=lng)
    if rows and _road_name(rows[0]["name"]):
        return {"name": _road_name(rows[0]["name"]), "address": None, "id": None, "kind": "road",
                "distance_m": int(round(rows[0]["distance_m"]))}
    return None
