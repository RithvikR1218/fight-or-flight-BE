"""Safe walking routes: the route planner in pathfinding/ running on backend data.

pathfinding/ (see pathfinding/README.md) plans a route that keeps H3 cell risk low,
passes USC blue-light phones and stays within a detour budget. Standalone, it
downloads its own OSM graph from the public Overpass API and reads a risk cache
file. Here it runs on what the backend already has, with its code unchanged:

* graph: road_edges in PostGIS (the OSM walk network from data/raw/*.osm.pbf),
  split into straight segments, which is exactly the shape pathfinding.graph.WalkGraph
  expects (an edge's geometry is its two endpoints, length = haversine).
* risk: the risk model's heatmap for the requested date (risk_model.heatmap,
  memoised per date) wrapped in pathfinding.risk.RiskGrid's interface.

The graph covers the study circle (RADIUS_KM, the same 5 miles as the risk model)
plus ROAD_MARGIN_KM.
"""
import datetime as dt
import json
import logging
import math
import threading
import time
from functools import lru_cache

import h3

from ..db_utils import fetch_all
from . import demo_incidents, risk_model

log = logging.getLogger(__name__)
MAX_SNAP_M = 300.0          # start/end farther than this from a walkable street -> 400
MAX_TRIP_KM = 12.0          # straight-line cap per request (planner cost grows with distance)

_graph_lock = threading.Lock()
_graph = None


class RouteError(ValueError):
    """Bad request for routing (outside the area, too far from a street, ...)."""


def _graph_area():
    """(center, radius_m) of the walk graph: study circle + ROAD_MARGIN_KM."""
    from flask import current_app
    c = current_app.config
    return (c["USC_LAT"], c["USC_LNG"]), (c["RADIUS_KM"] + c["ROAD_MARGIN_KM"]) * 1000


def explode_edges(geometries):
    """[[lng, lat], ...] polylines -> node lat/lon lists and undirected straight
    segments (u, v) between consecutive vertices. Vertices are merged by
    coordinate (7 decimals), so polylines that share an endpoint connect."""
    index, lat, lon, u, v, seen = {}, [], [], [], [], set()

    def node(x, y):
        key = (round(y, 7), round(x, 7))
        i = index.get(key)
        if i is None:
            i = index[key] = len(lat)
            lat.append(key[0])
            lon.append(key[1])
        return i

    for coords in geometries:
        if len(coords) < 2:
            continue
        prev = node(*coords[0][:2])
        for c in coords[1:]:
            cur = node(*c[:2])
            if cur != prev:
                e = (prev, cur) if prev < cur else (cur, prev)
                if e not in seen:
                    seen.add(e)
                    u.append(e[0])
                    v.append(e[1])
            prev = cur
    return lat, lon, u, v


def walk_graph():
    """pathfinding.graph.WalkGraph built from road_edges (once per process, ~10 s)."""
    global _graph
    if _graph is None:
        with _graph_lock:
            if _graph is None:
                from pathfinding.graph import WalkGraph

                t = time.time()
                (lat0, lon0), r = _graph_area()
                dlat = r / 111_320.0
                dlon = r / (111_320.0 * math.cos(math.radians(lat0)))
                rows = fetch_all("""
                    SELECT ST_AsGeoJSON(geom, 7) AS g FROM road_edges
                    WHERE geom && ST_MakeEnvelope(:w, :s, :e, :n, 4326)
                """, w=lon0 - dlon, s=lat0 - dlat, e=lon0 + dlon, n=lat0 + dlat)
                if not rows:
                    raise risk_model.ModelUnavailable("road_edges is empty: run `python -m scripts.load_roads`")
                lat, lon, u, v = explode_edges(json.loads(r_["g"])["coordinates"] for r_ in rows)
                _graph = WalkGraph(lat, lon, u, v)
                log.info("walk graph: %d road edges -> %d nodes, %d segments in %.1fs",
                         len(rows), len(lat), len(u), time.time() - t)
    return _graph


@lru_cache(maxsize=4)
def risk_grid(date_iso: str):
    """pathfinding.risk.RiskGrid over the backend's heatmap for `date_iso`."""
    from pathfinding.risk import NO_DATA_RISK, RiskGrid

    cache = risk_model.heatmap(dt.date.fromisoformat(date_iso))

    class BackendRiskGrid(RiskGrid):
        def __init__(self):  # same attributes as RiskGrid, from the in-memory heatmap instead of a file
            self.date = cache["date"]
            self.res = int(cache["h3_res"])
            self.step_m = h3.average_hexagon_edge_length(self.res, unit="m") / 3
            self.no_data_risk = NO_DATA_RISK
            self.cells = {c: v["risk"] for c, v in cache["cells"].items()}
            self.no_data = frozenset(cache.get("no_data_cells", []))
            self._init_cost(cache)   # routing cost = calibrated probability (pathfinding/risk.py)

    return BackendRiskGrid()


def warm():
    try:
        walk_graph()
        risk_grid(risk_model.local_now().date().isoformat())
        log.info("route planner warm")
    except Exception as exc:  # noqa: BLE001  (routing is optional; don't kill the worker)
        log.warning("route planner not warmed: %s", exc)


def _route_json(r):
    out = r.summary()
    out["coords"] = r.coords                                       # [[lat, lon], ...]
    out["geometry"] = {"type": "LineString", "coordinates": [[lon, lat] for lat, lon in r.coords]}
    return out


def risk_time(at=None, date=None, hour=None):
    """(local date, hour, UTC instant) whose hourly risk prediction a route uses.

    The model predicts one risk per clock hour. A trip with no time is planned for
    now; a trip with a time (`at`) uses the prediction for the nearest clock hour
    (1:44 AM -> 2 AM, 1:20 AM -> 1 AM; 11:40 PM -> midnight of the next day).
    `date`/`hour` without `at` name a prediction hour directly."""
    if at is None and date is None and hour is None:
        at = risk_model.local_now()
    if at is not None:
        local = at.astimezone(risk_model.LA)
        at = (local + dt.timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    d, h, instant, _ = risk_model.resolve_time(at, date, hour)
    return d, h, instant


def _demo_incidents():
    """Active simulated incidents when DEMO_MODE is on, else None."""
    if not demo_incidents.enabled():
        return None
    try:
        return demo_incidents.active()
    except Exception as exc:  # noqa: BLE001  (a demo extra must never break routing)
        log.warning("demo incidents unavailable: %s", exc)
        return None


def plan(start, end, at=None, date=None, hour=None, **params) -> dict:
    """Safe + shortest route between (lat, lon) points. params: planner knobs
    (max_detour, phone_weight, pass_radius_m)."""
    from pathfinding.graph import haversine_np
    from pathfinding.planner import plan_route

    trip_km = float(haversine_np(start[0], start[1], end[0], end[1])) / 1000
    if trip_km > MAX_TRIP_KM:
        raise RouteError(f"start and end are {trip_km:.1f} km apart; the limit is {MAX_TRIP_KM:g} km")
    d, h, instant = risk_time(at, date, hour)
    g = walk_graph()
    for name, (la, lo) in (("start", start), ("end", end)):
        _, snap = g.nearest_node(la, lo)
        if snap > MAX_SNAP_M:
            raise RouteError(f"{name} is {snap:.0f} m from the nearest walkable street in the route area")
    rg = risk_grid(d.isoformat())
    demo = _demo_incidents()
    if demo:   # demo mode: steer around simulated incidents (services/demo_incidents.py)
        from flask import current_app
        rg = demo_incidents.overlay(rg, [(i["lat"], i["lng"]) for i in demo], current_app.config["DEMO_INCIDENT_RADIUS_M"])
    try:
        res = plan_route(start, end, hour=h, graph=g, risk=rg, **params)
    except ValueError as exc:  # same node / no path
        raise RouteError(str(exc)) from exc
    res["date"] = d.isoformat()
    res["risk_at"] = instant.astimezone(risk_model.LA).isoformat()   # the prediction hour used
    if demo is not None:
        res["demo_incidents"] = len(demo)
    res["safe"] = _route_json(res["safe"])
    res["shortest"] = _route_json(res["shortest"])
    return res
