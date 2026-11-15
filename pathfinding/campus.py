"""Zones that routing treats as zero risk: USC University Park Campus and USC Village.

Routing tests each sampled point of a walk edge against the outlines
(in_zone_point): a point inside an outline, or within BOUNDARY_M of it (the
campus-side sidewalk of Vermont, Jefferson, ...), is risk 0 at every hour.
This is exact at any H3 resolution; a cell-centre test is not (at res 9 only 6
whole cells have their centre on campus, so most campus paths kept model risk).
zero_risk_cells() (cells whose centre is inside) is still what the risk map
paints.

The outlines follow street centrelines taken from the OpenStreetMap walk graph
in pathfinding/cache/walk_graph.json.gz (Overpass was unreachable when this was
written, so they were traced from that OSM street data, not from OSM's
amenity=university outline). Points are (lat, lon), clockwise from the NW corner.

    python -m pathfinding.campus            # list the zero-risk cells at the configured resolution
"""
import math
from functools import lru_cache

import h3

BOUNDARY_M = 20.0   # points this close to an outline count as inside (boundary streets' campus side)

TOMMY_TROJAN = (34.0206, -118.2856)

ZERO_RISK_ZONES = {
    # Vermont Ave (W), Jefferson Blvd (N), Figueroa St (E), Exposition Blvd (S).
    "USC University Park Campus": [
        (34.0255, -118.2915),   # Vermont / Jefferson
        (34.0255, -118.2885),   # Jefferson, before it bends ESE
        (34.0251, -118.2872),   # Jefferson / McClintock roundabout
        (34.0238, -118.2840),   # Jefferson / Hoover
        (34.0220, -118.2802),   # Jefferson / Figueroa
        (34.0183, -118.2825),   # Figueroa / Exposition
        (34.0183, -118.2915),   # Exposition / Vermont
    ],
    # North of Jefferson: McClintock Ave (W), W 30th St (N), Hoover St (E), Jefferson Blvd (S).
    "USC Village": [
        (34.0251, -118.2872),   # Jefferson / McClintock roundabout
        (34.0265, -118.2863),   # McClintock, curving north-east
        (34.0272, -118.2857),
        (34.0274, -118.2850),   # 30th St
        (34.0274, -118.2839),   # 30th / Hoover
        (34.0238, -118.2840),   # Hoover / Jefferson
    ],
}


def _poly(points):
    return h3.LatLngPoly(list(points))


@lru_cache(maxsize=8)
def zone_cells(res: int) -> dict:
    """{zone name: frozenset of H3 cells (centre inside the zone)} at `res`."""
    return {name: frozenset(h3.polygon_to_cells(_poly(pts), res))
            for name, pts in ZERO_RISK_ZONES.items()}


@lru_cache(maxsize=8)
def zero_risk_cells(res: int) -> frozenset:
    """All cells at `res` that routing scores as risk 0."""
    return frozenset().union(*zone_cells(res).values())


def in_zero_risk_zone(lat: float, lon: float, res: int) -> bool:
    return h3.latlng_to_cell(lat, lon, res) in zero_risk_cells(res)


_K = 111_320.0
_COS = math.cos(math.radians(TOMMY_TROJAN[0]))


def _xy(lat, lon):
    return (lon - TOMMY_TROJAN[1]) * _K * _COS, (lat - TOMMY_TROJAN[0]) * _K


@lru_cache(maxsize=1)
def _zones_xy():
    out = []
    for pts in ZERO_RISK_ZONES.values():
        xy = [_xy(la, lo) for la, lo in pts]
        xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        bbox = (min(xs) - BOUNDARY_M, min(ys) - BOUNDARY_M, max(xs) + BOUNDARY_M, max(ys) + BOUNDARY_M)
        out.append((xy, bbox))
    return out


def _inside(x, y, poly):
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def _near(x, y, poly, r):
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy or 1.0)))
        if math.hypot(x - x1 - t * dx, y - y1 - t * dy) <= r:
            return True
    return False


# Fenced sides: walls you can only cross at a gate. Any walk edge that crosses a
# wall more than gate_radius_m from every gate is removed from the graph
# (WalkGraph does this), so routes can't cut through a fence or a building.
# The Vermont side of campus is closed except at Downey Way (= W 36th Place). The wall
# runs along the inner edge of the Vermont east sidewalk (lon -118.2913), so walking along that
# sidewalk is still fine; it stops short of Jefferson and Exposition so the corners stay open.
CLOSED_SIDES = [
    {"name": "Vermont Ave (west side of campus)",
     "wall": [(34.01865, -118.29130), (34.02515, -118.29130)],
     "gates": [(34.02205, -118.29130)],     # Downey Way / W 36th Place
     "gate_radius_m": 25.0},
]


def closed_side_mask(lat1, lon1, lat2, lon2):
    """Bool array: True for segments (arrays of endpoints) that cross a closed side away from its gates."""
    import numpy as np

    lat1, lon1, lat2, lon2 = (np.asarray(a, dtype=float) for a in (lat1, lon1, lat2, lon2))
    x1, y1 = _xy(lat1, lon1)
    x2, y2 = _xy(lat2, lon2)
    out = np.zeros(x1.shape, dtype=bool)
    for side in CLOSED_SIDES:
        wall = [_xy(la, lo) for la, lo in side["wall"]]
        for (ax, ay), (bx, by) in zip(wall, wall[1:]):
            # proper segment intersection, vectorised over the edges
            def orient(px, py, qx, qy, rx, ry):
                return np.sign((qx - px) * (ry - py) - (qy - py) * (rx - px))
            hit = ((orient(ax, ay, bx, by, x1, y1) != orient(ax, ay, bx, by, x2, y2))
                   & (orient(x1, y1, x2, y2, ax, ay) != orient(x1, y1, x2, y2, bx, by)))
            if not hit.any():
                continue
            # crossing point, then distance to the nearest gate
            dx, dy = x2 - x1, y2 - y1
            den = dx * (by - ay) - dy * (bx - ax)
            t = np.where(den != 0, ((ax - x1) * (by - ay) - (ay - y1) * (bx - ax)) / np.where(den == 0, 1, den), 0.5)
            cx, cy = x1 + t * dx, y1 + t * dy
            near_gate = np.zeros_like(hit)
            for gla, glo in side["gates"]:
                gx, gy = _xy(gla, glo)
                near_gate |= np.hypot(cx - gx, cy - gy) <= side["gate_radius_m"]
            out |= hit & ~near_gate
    return out


def in_zone_point(lat: float, lon: float, boundary_m: float = BOUNDARY_M) -> bool:
    """True if (lat, lon) is inside a zero-risk zone or within boundary_m of its outline."""
    x, y = _xy(lat, lon)
    for poly, (x0, y0, x1, y1) in _zones_xy():
        if x0 <= x <= x1 and y0 <= y <= y1 and (_inside(x, y, poly) or _near(x, y, poly, boundary_m)):
            return True
    return False


if __name__ == "__main__":
    from algorithm.common import config

    res = config.H3_RES
    print(f"H3 res {res}")
    for name, cells in zone_cells(res).items():
        print(f"  {name}: {len(cells)} cells")
        for c in sorted(cells):
            print(f"    {c}")
    print(f"Tommy Trojan cell {h3.latlng_to_cell(*TOMMY_TROJAN, res)} "
          f"zero-risk: {in_zero_risk_zone(*TOMMY_TROJAN, res)}")
