"""Safe walking routes: low cell risk, past blue-light phones, bounded detour.

How a route is chosen
---------------------
1. Snap A and B to the nearest walkable OSM nodes and find the plain shortest
   path (A*, by length). Its length L0 sets the budget: no route may be longer
   than (1 + max_detour) x L0.
2. Prune the graph to the ellipse  d(A, n) + d(n, B) <= budget. A node outside
   it can't be on any route within budget (straight-line distance is a lower
   bound on walking distance), so this is exact and makes the search small.
3. Give every edge a risk (length-weighted mean of the H3 cells it crosses at
   the chosen hour, from the scoring cache) and a phone protection
   (geo.bluelight_protection of its midpoint: 1 at a phone, ~0 beyond 600 m).
4. Generate candidates with a small sweep (Lagrangian-style) over two weights:
        cost(edge) = length x (1 + lam x risk) x (1 - beta x protection)
   lam trades length for lower risk, beta pulls the route toward phones (beta < 1
   keeps every cost positive, so Dijkstra stays valid). For each (lam, beta)
   we run Dijkstra from A and from B over the pruned graph, which gives
     - the direct cheapest route A -> B, and
     - for every phone inside the ellipse, the route A -> phone -> B
       (the two trees joined at the phone), so routes that actually walk past
       phones are always on the table.
5. Drop candidates longer than the budget, then pick the one with the lowest
        score = exposure - min(phone_weight x phones_passed, PHONE_CREDIT_CAP x exposure)
                + detour_weight x detour
   exposure = sum(edge length x edge risk) / L0, i.e. the risk the walk adds up,
   in units of the shortest path. (Scoring on mean risk instead rewarded padding
   a route with extra metres of zero-risk campus: the detour lowered the average
   without making any part of the walk safer.)
   A phone counts as passed when the route comes within pass_radius_m of it.
   Phones can cut a route's score by at most PHONE_CREDIT_CAP of its own
   exposure, so a riskier route can never win on phone count alone (without
   the cap, leaving a 0-risk campus walk to pass 10 street phones won).
   Only phones outside the zero-risk zones (campus.py) earn the score credit:
   on campus the risk is already 0, so a phone there adds no safety and
   chasing it would only add detour. n_phones still reports every phone passed.
   The plain shortest path is always a candidate, so a valid answer exists.
"""
import math
from dataclasses import dataclass, field, asdict, replace

import numpy as np

from algorithm.common import geo

from .campus import in_zone_point
from .graph import haversine_np, load_graph, path_to
from .risk import load_risk

# ---- Tunable defaults -----------------------------------------------------
MAX_DETOUR = 0.20          # route may be at most 20% longer than the shortest path
PHONE_WEIGHT = 0.05        # one extra phone passed is worth 0.05 of mean risk
DETOUR_WEIGHT = 0.05       # small tie-break toward shorter routes
PHONE_CREDIT_CAP = 0.5     # phones can lower a route's score by at most half its exposure
PASS_RADIUS_M = 35.0       # within this distance of a phone = "passed" it
PHONE_SNAP_M = 80.0        # phones farther than this from any street are not via targets
LAMBDAS = (0.0, 1.0, 3.0, 8.0, 20.0)   # risk weights swept
BETAS = (0.0, 0.5, 0.85)                # phone-pull weights swept
HIGH_RISK = 0.8            # risk at or above this counts toward high_risk_m


@dataclass
class RouteStats:
    label: str
    length_m: float
    detour_pct: float
    mean_risk: float
    exposure: float            # sum(length x risk) / shortest length; what the score uses
    max_risk: float
    high_risk_m: float
    no_data_pct: float
    phones_passed: list = field(default_factory=list)   # [{"name", "lat", "lon", "dist_m"}]
    coords: list = field(default_factory=list)          # [[lat, lon], ...]
    lam: float | None = None
    beta: float | None = None
    via_phone: str | None = None
    score: float | None = None

    @property
    def n_phones(self):
        return len(self.phones_passed)

    @property
    def n_phones_scored(self):
        """Phones passed outside the zero-risk zones (the ones the score rewards)."""
        return sum(not in_zone_point(p["lat"], p["lon"], 0.0) for p in self.phones_passed)

    def summary(self):
        d = asdict(self)
        d.pop("coords")
        d["n_phones"] = self.n_phones
        d["n_phones_scored"] = self.n_phones_scored
        return d


def _local_xy(lat, lon, lat0, lon0):
    """Equirectangular metres around (lat0, lon0); fine at city scale."""
    k = 111_320.0
    return (np.asarray(lon) - lon0) * k * math.cos(math.radians(lat0)), (np.asarray(lat) - lat0) * k


def _phones_near_route(coords, phones, radius_m, step_m=10.0):
    """Phones within radius_m of the route polyline (densified every ~step_m)."""
    if not phones or len(coords) < 1:
        return []
    c = np.asarray(coords)
    lat0, lon0 = c[:, 0].mean(), c[:, 1].mean()
    x, y = _local_xy(c[:, 0], c[:, 1], lat0, lon0)
    pts_x, pts_y = [x[:1]], [y[:1]]
    for i in range(len(c) - 1):
        seg = math.hypot(x[i + 1] - x[i], y[i + 1] - y[i])
        n = max(1, int(seg // step_m))
        t = np.arange(1, n + 1) / n
        pts_x.append(x[i] + (x[i + 1] - x[i]) * t)
        pts_y.append(y[i] + (y[i + 1] - y[i]) * t)
    px, py = np.concatenate(pts_x), np.concatenate(pts_y)
    plat = np.array([p[0] for p in phones])
    plon = np.array([p[1] for p in phones])
    fx, fy = _local_xy(plat, plon, lat0, lon0)
    d = np.sqrt((fx[:, None] - px[None, :]) ** 2 + (fy[:, None] - py[None, :]) ** 2).min(axis=1)
    return [{"name": phones[i][2], "lat": phones[i][0], "lon": phones[i][1], "dist_m": round(float(d[i]), 1)}
            for i in np.flatnonzero(d <= radius_m)]


def plan_route(start, end, hour=22, max_detour=MAX_DETOUR, phone_weight=PHONE_WEIGHT,
               detour_weight=DETOUR_WEIGHT, pass_radius_m=PASS_RADIUS_M,
               lambdas=LAMBDAS, betas=BETAS, graph=None, risk=None):
    """Plan a safe walking route from start=(lat, lon) to end=(lat, lon).

    hour: local clock hour 0-23 (risk changes a lot by hour).
    max_detour: hard cap, 0.20 = at most 20% longer than the shortest path.
    phone_weight: how much one passed phone is worth against mean risk.
    Returns a dict with 'safe' and 'shortest' RouteStats, the graph mask used
    and counts of candidates considered."""
    g = graph or load_graph()
    rg = risk or load_risk()
    phones = geo.bluelight_phones()

    a, da = g.nearest_node(*start)
    b, db = g.nearest_node(*end)
    if a == b:
        raise ValueError("start and end snap to the same street node")

    # 1. Plain shortest path (A* with straight-line heuristic).
    hb = haversine_np(g.lat, g.lon, g.lat[b], g.lon[b]).tolist()
    dist, pn, pe = g.dijkstra(a, g.length, target=b, heuristic=hb.__getitem__)
    if math.isinf(dist[b]):
        raise ValueError("no walking path between these points in the graph")
    L0 = dist[b]
    budget = (1.0 + max_detour) * L0

    # 2. Ellipse pruning: exact, since straight-line <= walking distance.
    ha = haversine_np(g.lat, g.lon, g.lat[a], g.lon[a])
    allowed = (ha + np.asarray(hb)) <= budget + 1.0
    emask = allowed[g.eu] & allowed[g.ev]
    eidx = np.flatnonzero(emask)

    # 3. Edge risk and phone protection (only for edges inside the ellipse).
    n_e = len(g.eu)
    erisk = np.full(n_e, rg.no_data_risk)
    enodata = np.zeros(n_e)
    for k in eidx.tolist():
        u, v = g.eu[k], g.ev[k]
        erisk[k], enodata[k] = rg.edge_risk(g.lat[u], g.lon[u], g.lat[v], g.lon[v], g.length[k], hour)
    mid_lat = (g.lat[g.eu[eidx]] + g.lat[g.ev[eidx]]) / 2
    mid_lon = (g.lon[g.eu[eidx]] + g.lon[g.ev[eidx]]) / 2
    plat = np.array([p[0] for p in phones])
    plon = np.array([p[1] for p in phones])
    eprot = np.zeros(n_e)
    if len(eidx):
        dmin = haversine_np(mid_lat[:, None], mid_lon[:, None], plat[None, :], plon[None, :]).min(axis=1)
        eprot[eidx] = np.exp(-dmin / geo.config.BLUELIGHT_RANGE_M)  # == geo.bluelight_protection

    # Phones usable as via points: inside the ellipse and close to a street.
    via = []
    for i, (la, lo, name) in enumerate(phones):
        n_i, d_i = g.nearest_node(la, lo)
        if d_i <= PHONE_SNAP_M and allowed[n_i]:
            via.append((n_i, name))

    def stats(nodes, edges, label, **extra):
        lens = g.length[edges]
        L = float(lens.sum())
        r = erisk[edges]
        coords = [[float(g.lat[x]), float(g.lon[x])] for x in nodes]
        coords = [list(start)] + coords + [list(end)]
        return RouteStats(
            label=label, length_m=round(L + da + db, 1),
            detour_pct=round(max(100 * (L / L0 - 1), 0.0), 2),
            mean_risk=round(float((lens * r).sum() / L), 4),
            exposure=round(float((lens * r).sum() / L0), 4),
            max_risk=round(float(r.max()), 4),
            high_risk_m=round(float(lens[r >= HIGH_RISK].sum()), 1),
            no_data_pct=round(100 * float((lens * enodata[edges]).sum() / L), 1),
            phones_passed=_phones_near_route(coords, phones, pass_radius_m),
            coords=coords, **extra)

    sp_nodes, sp_edges = path_to(pn, pe, a, b)
    shortest = stats(sp_nodes, np.array(sp_edges), "shortest", lam=0.0, beta=0.0)

    # 4. Candidate sweep.
    seen, cands = set(), [shortest]
    seen.add(tuple(sp_nodes))
    for lam in lambdas:
        for beta in betas:
            cost = g.length * (1 + lam * erisk) * (1 - beta * eprot)
            da_, pa_n, pa_e = g.dijkstra(a, cost, allowed=allowed)
            db_, pb_n, pb_e = g.dijkstra(b, cost, allowed=allowed)
            routes = [(path_to(pa_n, pa_e, a, b), None)]
            for p, name in via:
                if p in (a, b) or math.isinf(da_[p]) or math.isinf(db_[p]):
                    continue
                first, back = path_to(pa_n, pa_e, a, p), path_to(pb_n, pb_e, b, p)
                if first and back:
                    nodes = first[0] + back[0][::-1][1:]
                    edges = first[1] + back[1][::-1]
                    routes.append(((nodes, edges), name))
            for route, name in routes:
                if route is None:
                    continue
                nodes, edges = route
                key = tuple(nodes)
                if key in seen:
                    continue
                seen.add(key)
                edges = np.array(edges)
                if g.length[edges].sum() > budget:   # 5. hard detour cap
                    continue
                cands.append(stats(nodes, edges, "safe", lam=lam, beta=beta, via_phone=name))

    for c in cands:
        c.score = round(c.exposure - min(phone_weight * c.n_phones_scored, PHONE_CREDIT_CAP * c.exposure)
                        + detour_weight * max(c.detour_pct, 0) / 100, 4)
    best = replace(min(cands, key=lambda c: (c.score, c.length_m)), label="safe")
    return {
        "start": list(start), "end": list(end), "hour": int(hour),
        "risk_date": rg.date,
        "params": {"max_detour": max_detour, "phone_weight": phone_weight,
                   "detour_weight": detour_weight, "pass_radius_m": pass_radius_m,
                   "lambdas": list(lambdas), "betas": list(betas)},
        "shortest": shortest, "safe": best,
        "n_candidates": len(seen), "n_feasible": len(cands),
        "search_nodes": int(allowed.sum()), "search_edges": int(len(eidx)),
        "snap_m": [round(da, 1), round(db, 1)],
    }
