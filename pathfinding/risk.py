"""Per-cell risk lookup for routing, read from the scoring cache.

Uses algorithm/scoring/cache/risk_cache.json (every modelled H3 cell x
24 hours, at whatever resolution algorithm.common.config.H3_RES says, built by `python -m algorithm.scoring.score --build-cache`), so
routing never loads the model and each lookup is a dict access.

Cells with no LAPD data (other police jurisdictions) and cells outside the
model area get NO_DATA_RISK (0.5, neutral), so a route never prefers a place
just because nothing is known about it. Routes report how much of their
length ran through such cells.

Points inside the USC University Park Campus and USC Village outlines
(campus.py, plus a BOUNDARY_M margin for the boundary streets) are risk 0 at
every hour, whatever the cache says. The test is per point, not per cell, so
it is exact at res 9 and res 10 alike.
"""
import json
import warnings
from functools import lru_cache
from pathlib import Path

import h3
import numpy as np

from algorithm.common import config

from .campus import in_zone_point, zero_risk_cells

NO_DATA_RISK = 0.5
COST_REF_QUANTILE = 0.9   # routing cost 1.0 = the 2023 90th-percentile cell-hour probability


def _cost_ref():
    """(p_ref, p_median): incident probabilities at the 2023 90th and 50th percentile
    of cell-hours, from model_meta.json risk_quantiles. None if unavailable."""
    try:
        qs = np.asarray(json.loads(Path(config.META_PATH).read_text())["risk_quantiles"], dtype=float)
    except Exception:  # noqa: BLE001
        return None
    grid = np.linspace(0, 1, len(qs))
    return float(np.interp(COST_REF_QUANTILE, grid, qs)), float(np.interp(0.5, grid, qs))


class RiskGrid:
    def __init__(self, cache_path: Path = config.CACHE_PATH, no_data_risk: float = NO_DATA_RISK):
        cache = json.loads(Path(cache_path).read_text())
        self.date = cache["date"]
        # Follow the model's resolution; never hardcode it here.
        self.res = int(cache.get("h3_res", config.H3_RES))
        if self.res != config.H3_RES:
            warnings.warn(f"risk cache is H3 res {self.res} but config.H3_RES is {config.H3_RES}; "
                          "rebuild it with `python -m algorithm.scoring.score --build-cache`")
        # Sample edges at ~1/3 of a cell edge so no cell is skipped (res 9: ~67 m, res 10: ~25 m).
        self.step_m = h3.average_hexagon_edge_length(self.res, unit="m") / 3
        self.no_data_risk = no_data_risk
        self.cells = {c: v["risk"] for c, v in cache["cells"].items()}
        self.no_data = frozenset(cache.get("no_data_cells", []))
        self._init_cost(cache)

    def _init_cost(self, cache):
        """Routing cost per cell-hour: calibrated incident probability / p_ref.

        `risk` is a percentile rank, so a 99th-percentile block is only ~2x a median
        one, while its probability is often 10x or more. Routing sums cost (expected
        incidents, in units of a 90th-percentile cell-hour) so hot spots weigh what
        they should; `risk` stays the display value. Caches without `prob` fall back
        to cost = risk."""
        ref = _cost_ref()
        cells = cache["cells"]
        has_prob = bool(cells) and all("prob" in v for v in cells.values())
        if ref is None or not has_prob:
            self.cost = {c: list(map(float, r)) for c, r in self.cells.items()}
            self.no_data_cost, self.cost_unit = self.no_data_risk, "risk"
            return
        p_ref, p_med = ref
        self.cost = {c: (np.asarray(v["prob"], dtype=float) / p_ref).tolist() for c, v in cells.items()}
        self.no_data_cost = p_med / p_ref      # neutral = a median cell-hour, like risk 0.5
        self.cost_unit = f"probability / {p_ref:.4g} (2023 p90 cell-hour)"

    def cell(self, lat: float, lon: float) -> str:
        return h3.latlng_to_cell(lat, lon, self.res)

    def risk(self, h3_id: str, hour: int):
        """(risk 0..1, known?) for a cell at a clock hour. Campus/Village cells are 0."""
        if h3_id in zero_risk_cells(self.res):
            return 0.0, True
        r = self.cells.get(h3_id)
        if r is None:
            return self.no_data_risk, False
        return float(r[int(hour) % 24]), True

    def risk_at(self, lat, lon, hour):
        """(risk, known?) at a point: 0 inside a zero-risk zone, else the point's cell."""
        if in_zone_point(lat, lon):
            return 0.0, True
        r = self.cells.get(self.cell(lat, lon))
        if r is None:
            return self.no_data_risk, False
        return float(r[int(hour) % 24]), True

    def point(self, lat, lon, hour):
        """(risk, cost, known?) at a point: 0 inside a zero-risk zone."""
        if in_zone_point(lat, lon):
            return 0.0, 0.0, True
        c = self.cell(lat, lon)
        r = self.cells.get(c)
        if r is None:
            return self.no_data_risk, self.no_data_cost, False
        h = int(hour) % 24
        return float(r[h]), float(self.cost[c][h]), True

    def edge_risk_cost(self, lat1, lon1, lat2, lon2, length_m, hour, step_m=None):
        """(mean risk, share without data, mean routing cost) along a straight segment."""
        n = max(1, int(np.ceil(length_m / (step_m or self.step_m))))
        t = (np.arange(n) + 0.5) / n
        r, c, k = zip(*(self.point(lat1 + (lat2 - lat1) * f, lon1 + (lon2 - lon1) * f, hour) for f in t))
        return float(np.mean(r)), 1.0 - float(np.mean(k)), float(np.mean(c))

    def edge_risk(self, lat1, lon1, lat2, lon2, length_m, hour, step_m=None):
        """Length-weighted mean risk along a straight segment, sampled every
        ~step_m, plus the share of the segment that had no data."""
        n = max(1, int(np.ceil(length_m / (step_m or self.step_m))))
        t = (np.arange(n) + 0.5) / n
        vals, known = zip(*(self.risk_at(lat1 + (lat2 - lat1) * f, lon1 + (lon2 - lon1) * f, hour)
                            for f in t))
        return float(np.mean(vals)), 1.0 - float(np.mean(known))


@lru_cache(maxsize=1)
def load_risk() -> RiskGrid:
    return RiskGrid()
