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
