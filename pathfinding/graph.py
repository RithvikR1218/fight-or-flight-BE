"""Walking street graph for the SafeRoute area, built from OpenStreetMap.

The first call downloads every walkable way inside the model area's bounding
box (config.BBOX, the 5-mile circle around USC) from the Overpass API and
caches a compact copy in pathfinding/cache/walk_graph.json.gz (a few MB).
Later calls load the cache in a couple of seconds.

Nodes are OSM nodes; each edge is one straight segment between two
consecutive nodes of a way, so its geometry is just its two endpoints.
Walking ignores one-way tags, so the graph is undirected.

    python -m pathfinding.graph            # download (or refresh) the cache
"""
import gzip
import heapq
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests

from algorithm.common import config

CACHE_DIR = Path(__file__).resolve().parent / "cache"
GRAPH_PATH = CACHE_DIR / "walk_graph.json.gz"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Same exclusions as OSMnx's "walk" network: no motorways, construction,
# private service roads or ways tagged foot=no.
WALK_FILTER = (
    '["highway"]["area"!~"yes"]'
    '["highway"!~"abandoned|bus_guideway|construction|cycleway|motor|planned|'
    'platform|proposed|raceway|bridleway|corridor|elevator|escalator"]'
    '["foot"!~"no"]["service"!~"private"]["access"!~"private"]'
)
# A way explicitly open to pedestrians is walkable even if it is a cycleway or private:
# USC's main walks (Trousdale Pkwy, Childs Way, Watt Way, ...) are highway=cycleway + foot=designated.
FOOT_OK_FILTER = (
    '["highway"]["area"!~"yes"]["foot"~"^(yes|designated|permissive)$"]'
    '["highway"!~"abandoned|bus_guideway|construction|motor|planned|platform|proposed|raceway"]'
)
# Every path on campus / in the Village is open to pedestrians whatever its access tags
# (Downey Way is access=private), so ways touching those outlines are fetched unfiltered.
ZONE_FILTER = '["highway"]["area"!~"yes"]["foot"!~"no"]["highway"!~"motorway|construction|proposed"]'
TILES = 3  # the bbox is fetched as TILES x TILES pieces to stay under Overpass limits

M_PER_DEG_LAT = 111_320.0


def _zone_ways():
    from .campus import ZERO_RISK_ZONES

    polys = (" ".join(f"{la} {lo}" for la, lo in pts) for pts in ZERO_RISK_ZONES.values())
    return "".join(f'way{ZONE_FILTER}(poly:"{p}");' for p in polys)


def _fetch_tile(s, w, n, e, retries=4):
    q = (f"[out:json][timeout:180];"
         f"(way{WALK_FILTER}({s},{w},{n},{e});way{FOOT_OK_FILTER}({s},{w},{n},{e});{_zone_ways()});"
         f"(._;>;);out skel qt;")
    for attempt in range(retries):
        try:
            r = requests.post(OVERPASS_URL, data={"data": q}, timeout=240,
                              headers={"User-Agent": "saferoute-hackathon/0.1"})
            if r.status_code == 200:
                return r.json()["elements"]
            print(f"  Overpass HTTP {r.status_code}, retrying")
        except requests.RequestException as exc:
            print(f"  Overpass error {exc!r}, retrying")
        time.sleep(10 * (attempt + 1))
    raise RuntimeError("Overpass API kept failing; try again in a few minutes")


def download_graph(path: Path = GRAPH_PATH) -> Path:
    s, w, n, e = config.BBOX
    nodes, edges = {}, set()
    for i in range(TILES):
        for j in range(TILES):
            ts = s + (n - s) * i / TILES
            tn = s + (n - s) * (i + 1) / TILES
            tw = w + (e - w) * j / TILES
            te = w + (e - w) * (j + 1) / TILES
            print(f"Fetching OSM walk ways, tile {i * TILES + j + 1}/{TILES * TILES} ...")
            for el in _fetch_tile(ts, tw, tn, te):
                if el["type"] == "node":
                    nodes[el["id"]] = (el["lat"], el["lon"])
                elif el["type"] == "way":
                    refs = el["nodes"]
                    for a, b in zip(refs, refs[1:]):
                        if a != b:
                            edges.add((a, b) if a < b else (b, a))
            time.sleep(2)  # be polite to the public Overpass server
    edges = [(a, b) for a, b in edges if a in nodes and b in nodes]
    used = sorted({x for ab in edges for x in ab})
    idx = {osm: k for k, osm in enumerate(used)}
    data = {
        "source": "OpenStreetMap contributors (ODbL), via Overpass API",
        "bbox": list(config.BBOX),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "osm_ids": used,
        "lat": [round(nodes[o][0], 7) for o in used],
        "lon": [round(nodes[o][1], 7) for o in used],
        "u": [idx[a] for a, _ in edges],
        "v": [idx[b] for _, b in edges],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"Wrote {len(used):,} nodes and {len(edges):,} edges to {path}")
    return path


class WalkGraph:
    """Undirected street graph held as flat arrays plus an adjacency list.

    adj[u] is a list of (v, edge_index); edge arrays (eu, ev, length_m) are
    indexed by edge_index, so per-edge costs can be swapped in as one array."""

    def __init__(self, lat, lon, u, v):
        from .campus import closed_side_mask

        self.lat = np.asarray(lat, dtype=float)
        self.lon = np.asarray(lon, dtype=float)
        eu = np.asarray(u, dtype=np.int64)
        ev = np.asarray(v, dtype=np.int64)
        # Drop segments that cut through a fenced side of campus away from its gates
        # (campus.CLOSED_SIDES: Vermont is closed except at Downey Way).
        keep = ~closed_side_mask(self.lat[eu], self.lon[eu], self.lat[ev], self.lon[ev])
        self.n_closed = int((~keep).sum())
        self.eu, self.ev = eu[keep], ev[keep]
        self.length = haversine_np(self.lat[self.eu], self.lon[self.eu],
                                   self.lat[self.ev], self.lon[self.ev])
        self.adj = [[] for _ in range(len(self.lat))]
        for k, (a, b) in enumerate(zip(self.eu.tolist(), self.ev.tolist())):
            self.adj[a].append((b, k))
            self.adj[b].append((a, k))

    @property
    def n_nodes(self):
        return len(self.lat)

    def nearest_node(self, lat, lon):
        """Index of the closest node and its distance in metres."""
        d = haversine_np(lat, lon, self.lat, self.lon)
        i = int(np.argmin(d))
        return i, float(d[i])

    def dijkstra(self, src, cost, allowed=None, target=None, heuristic=None):
        """Single-source Dijkstra (A* when `heuristic` is given with `target`).

        cost: per-edge cost array (positive). allowed: optional bool array of
        nodes the search may enter. Returns (dist, parent_node, parent_edge)
        as lists; unreached nodes have dist = inf and parent -1."""
        n = self.n_nodes
        inf = math.inf
        dist = [inf] * n
        pnode = [-1] * n
        pedge = [-1] * n
        done = [False] * n
        cost = cost.tolist() if isinstance(cost, np.ndarray) else cost
        ok = allowed.tolist() if isinstance(allowed, np.ndarray) else allowed
        dist[src] = 0.0
        h = heuristic or (lambda _i: 0.0)
        pq = [(h(src), 0.0, src)]
        adj = self.adj
        while pq:
            _, d, x = heapq.heappop(pq)
            if done[x]:
                continue
            done[x] = True
            if x == target:
                break
            for y, k in adj[x]:
                if done[y] or (ok is not None and not ok[y]):
                    continue
                nd = d + cost[k]
                if nd < dist[y]:
                    dist[y] = nd
                    pnode[y] = x
                    pedge[y] = k
                    heapq.heappush(pq, (nd + h(y), nd, y))
        return dist, pnode, pedge


def path_to(pnode, pedge, src, dst):
    """(nodes, edges) from src to dst in a Dijkstra parent tree, or None."""
    if dst != src and pnode[dst] == -1:
        return None
    nodes, edges = [dst], []
    x = dst
    while x != src:
        edges.append(pedge[x])
        x = pnode[x]
        nodes.append(x)
    return nodes[::-1], edges[::-1]


def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in metres."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6_371_000.0 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


@lru_cache(maxsize=1)
def load_graph(path: Path = GRAPH_PATH) -> WalkGraph:
    """The cached walk graph; downloads it first if the cache is missing."""
    if not Path(path).exists():
        download_graph(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    return WalkGraph(d["lat"], d["lon"], d["u"], d["v"])


if __name__ == "__main__":
    download_graph()
    g = load_graph()
    print(f"Graph ready: {g.n_nodes:,} nodes, {len(g.eu):,} edges, "
          f"{g.length.sum() / 1000:,.0f} km of walkable segments")
