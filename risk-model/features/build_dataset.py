"""Build the cell x hour x date training frame for the risk model.

Usage (from the repo root, after algorithm.data.ingest_crime):
    python -m algorithm.features.build_dataset     # writes models/artifacts/dataset.parquet

Each row is one (H3 cell, date, hour). Target = 1 if at least one incident was
reported in that cell during that clock hour on that date.

History features are per-incident weighted sums (see common/severity.py):
each past crime counts  severity x 1 / (1 + age_days / tau) x
exp(-hour_gap**2 / (2 sigma**2)), so recent, serious crimes at a similar time
of day dominate. The recency curve is logarithmic (a power law); tau, the
power and sigma live in common/config.py.

Leakage rule: every history feature for a row on date D uses ONLY incidents
from before D (decayed sums and daily windows end the day before; hour-of-day
windows end at the start of D's month). Nothing from the row's own day leaks in.

Scale: the 5-mile area has ~1,450 covered cells x 1,827 days x 24 hours
(~150M cell-hours), so nothing here materialises the full grid. Positives come
straight from the incident list; negatives are sampled uniformly from the grid.
Training keeps NEG_PER_POS negatives per positive; evaluation frames keep more
and carry a `w` column (1 for positives, 1/sample-rate for negatives) so
metrics computed with those weights equal full-grid metrics in expectation.

Covered cells: a cell inside the circle is modelled only if it had at least
MIN_CELL_INCIDENTS incidents in the training years. Cells below that are mostly
outside LAPD's jurisdiction (Inglewood, Culver City, Beverly Hills, ...), where
"no records" means "no data", not "safe".
"""
import datetime as dt
import sqlite3

import h3
import numpy as np
import pandas as pd
from scipy import sparse

from algorithm.common import config, geo
from algorithm.common.severity import incident_severity

FEATURES = [
    # when
    "hour", "weekday", "month", "is_weekend", "is_dark",
    # where
    "cell_lat", "cell_lon", "dist_bluelight_m",
    # per-incident weighted history: severity x log-scale recency (x time of day)
    "harm_recent", "harm_recent_tod", "nb1_harm_recent", "nb2_harm_recent",
    # plain long-run counts (the place's baseline)
    "cell_hourband_365d", "cell_rate_all", "cell_hourband_rate_all",
    "area_7d",
]


def recency_weight(age_days, tau=config.RECENCY_TAU_DAYS, power=config.RECENCY_POWER):
    """Target recency curve: 1 / (1 + age/tau) ** power (log-scale decay)."""
    return (1.0 + np.asarray(age_days, dtype=float) / tau) ** -power


def recency_mixture(basis=config.RECENCY_BASIS):
    """Weights a_k so that sum_k a_k * 0.5**(age/basis_k) ~= recency_weight(age)
    for ages 1 day .. 4 years (non-negative least squares on relative error).
    Exponentials can be updated day by day, which keeps the feature pass O(days)."""
    from scipy.optimize import nnls
    hl = np.asarray(basis, dtype=float)
    age = np.unique(np.round(np.logspace(0, np.log10(1500), 200)))
    target = recency_weight(age)
    A = 0.5 ** (age[:, None] / hl[None, :])
    a, _ = nnls(A / target[:, None], np.ones(len(age)))
    return dict(zip(hl.tolist(), a.astype(np.float32)))


def tod_kernel(sigma=config.TOD_SIGMA_HOURS) -> np.ndarray:
    """K[h, h'] = weight a crime at hour h' gets for a row at hour h."""
    h = np.arange(24)
    gap = np.abs(h[:, None] - h[None, :])
    gap = np.minimum(gap, 24 - gap)
    return np.exp(-gap ** 2 / (2 * sigma ** 2)).astype(np.float32)


class History:
    """Per-cell incident history with prefix sums for fast, leakage-free
    window lookups. Day index 0 = config.DATA_START."""

    def __init__(self, db_path=config.DB_PATH):
        con = sqlite3.connect(db_path)
        inc = pd.read_sql("SELECT date_occ, hour, cell, tier, crm_cd, crm_cd_desc, weapon_used_cd "
                          "FROM incidents", con)
        con.close()

        self.start = dt.date.fromisoformat(config.DATA_START)
        self.end = dt.date.fromisoformat(config.DATA_END)
        self.n_days = (self.end - self.start).days + 1
        inc["d"] = (pd.to_datetime(inc["date_occ"]) - pd.Timestamp(self.start)).dt.days
        inc = inc[(inc["d"] >= 0) & (inc["d"] < self.n_days)]

        # --- which cells get modelled ---------------------------------------
        area = set(geo.area_cells())
        train_end = self.day_index(dt.date.fromisoformat(config.TRAIN_END))
        n_train = inc[inc["d"] <= train_end].groupby("cell").size()
        self.cells = sorted(c for c, n in n_train.items()
                            if c in area and n >= config.MIN_CELL_INCIDENTS)
        self.no_data_cells = sorted(area - set(self.cells))
        # Counting universe: covered cells first (row index = cell index), then
        # every other cell with incidents, so neighbour sums near gaps are fuller.
        others = sorted(set(inc["cell"]) - set(self.cells))
        self.universe = self.cells + others
        self.cidx = {c: i for i, c in enumerate(self.universe)}
        nu = len(self.universe)

        ci = inc["cell"].map(self.cidx).to_numpy()
        di = inc["d"].to_numpy()
        hi = inc["hour"].to_numpy()
        keys = inc[["crm_cd", "crm_cd_desc", "weapon_used_cd"]].drop_duplicates()
        keys["sev"] = [incident_severity(c, d, w) for c, d, w in keys.itertuples(index=False)]
        sev = inc.merge(keys, on=["crm_cd", "crm_cd_desc", "weapon_used_cd"], how="left")["sev"].to_numpy(np.float32)
        viol = (inc["tier"] == "violent").to_numpy()

        def daily_of(weights=None, mask=None):
            a = np.zeros((nu, self.n_days), dtype=np.float32 if weights is not None else np.int32)
            m = slice(None) if mask is None else mask
            np.add.at(a, (ci[m], di[m]), 1 if weights is None else weights[m])
            return a

        def prefix(a):  # P[:, d] = sum of a over days < d
            z = np.zeros(a.shape[:1] + (1,) + a.shape[2:], dtype=a.dtype)
            return np.concatenate([z, np.cumsum(a, axis=1, dtype=a.dtype)], axis=1)

        daily = daily_of()
        self.P_daily = prefix(daily)
        self.P_area = np.concatenate([[0], np.cumsum(daily[: len(self.cells)].sum(axis=0))])

        # Hour-of-day history at monthly resolution: M[c, m, h] = incidents in
        # months < m at hour h. Month index 0 = DATA_START's month.
        self.month_of_day = self._month_index(np.arange(self.n_days + 1))
        n_months = int(self.month_of_day[-1]) + 1
        mo = self.month_of_day[di]
        cube = np.zeros((nu, n_months, 24), dtype=np.int32)
        np.add.at(cube, (ci, mo, hi), 1)
        self.P_month_hour = prefix(cube)

        # Neighbour rings as sparse matrices: ring 1 (6 cells), ring 2 (12 cells).
        r1r, r1c, r2r, r2c = [], [], [], []
        for c, i in self.cidx.items():
            d1 = set(h3.grid_disk(c, 1)) - {c}
            d2 = set(h3.grid_disk(c, 2)) - d1 - {c}
            for n in d1:
                if n in self.cidx:
                    r1r.append(i); r1c.append(self.cidx[n])
            for n in d2:
                if n in self.cidx:
                    r2r.append(i); r2c.append(self.cidx[n])
        R1 = sparse.csr_matrix((np.ones(len(r1r), np.float32), (r1r, r1c)), shape=(nu, nu))
        R2 = sparse.csr_matrix((np.ones(len(r2r), np.float32), (r2r, r2c)), shape=(nu, nu))
        self.R1, self.R2 = R1, R2

        # Per-incident weighted history. Incidents sorted by day so the decay
        # pass can add one day at a time: S[c, h] = sum of severity x decay over
        # crimes in cell c at hour h on days before the current day.
        order = np.argsort(di, kind="stable")
        self._inc_c, self._inc_h, self._inc_s = ci[order], hi[order], sev[order]
        self._day_start = np.searchsorted(di[order], np.arange(self.n_days + 1))
        self._mix = {hl: a for hl, a in recency_mixture().items() if a > 1e-6}
        self._decay = {hl: np.float32(0.5 ** (1.0 / hl)) for hl in self._mix}
        self.K = tod_kernel()
        self.n_universe = nu
        self._state_cache = {}  # day -> decayed states, for repeated live queries

        cent = np.array([h3.cell_to_latlng(c) for c in self.universe])
        self.cell_lat, self.cell_lon = cent[:, 0], cent[:, 1]
        self.dist_bl = np.array([geo.dist_to_bluelight_m(a, b) for a, b in cent])

        # Positive cell-hours as sorted int64 keys for fast membership tests.
        cov = ci < len(self.cells)
        self.pos_keys = np.unique(self._key(ci[cov], di[cov], hi[cov]))
        self.pos_keys_violent = np.unique(self._key(ci[cov & viol], di[cov & viol], hi[cov & viol]))

    # -- helpers -------------------------------------------------------------
    def day_index(self, date: dt.date) -> int:
        return (date - self.start).days

    def _month_index(self, day_idx):
        dates = pd.Timestamp(self.start) + pd.to_timedelta(np.asarray(day_idx), unit="D")
        return ((dates.year - self.start.year) * 12 + dates.month - self.start.month).to_numpy()

    def _key(self, ci, di, hi):
        return (np.asarray(ci, np.int64) * (self.n_days + 1) + np.asarray(di, np.int64)) * 24 + np.asarray(hi, np.int64)

    def _unkey(self, k):
        hi = k % 24
        rest = k // 24
        return rest // (self.n_days + 1), rest % (self.n_days + 1), hi

    @staticmethod
    def _win(P, ci, d, w):
        return P[ci, d] - P[ci, np.clip(d - w, 0, None)]

    def harm_states(self, days):
        """Yield (day, {half-life: S}) for each requested day (ascending), where
        S[c, h] is the exponentially decayed severity sum of crimes before `day`.
        One forward pass over the calendar; cost ~ n_days, not n_rows.
        recent_state() mixes them into the log-scale recency curve."""
        days = np.unique(np.clip(np.asarray(days), 0, self.n_days))
        S = {k: np.zeros((self.n_universe, 24), np.float32) for k in self._decay}
        cur = 0
        for d in days:
            while cur < d:
                a, b = self._day_start[cur], self._day_start[cur + 1]
                for k, lam in self._decay.items():
                    if b > a:
                        np.add.at(S[k], (self._inc_c[a:b], self._inc_h[a:b]), self._inc_s[a:b])
                    S[k] *= lam  # crimes on day `cur` are 1 day old at `cur + 1`
                cur += 1
            yield d, S

    def recent_state(self, S):
        """Sum of severity x recency_weight(age) per (cell, hour)."""
        return sum(a * S[hl] for hl, a in self._mix.items())

    def _harm_features(self, ci, d, hi):
        out = {f: np.zeros(len(ci), np.float32) for f in
               ("harm_recent", "harm_recent_tod", "nb1_harm_recent", "nb2_harm_recent")}
        rows_by_day = pd.Series(np.arange(len(ci))).groupby(d).indices
        if len(rows_by_day) == 1:  # live scoring: one day, reuse its state
            day = next(iter(rows_by_day))
            if day not in self._state_cache:
                if len(self._state_cache) >= 4:
                    self._state_cache.pop(next(iter(self._state_cache)))
                self._state_cache[day] = {"recent": self.recent_state(next(self.harm_states([day]))[1])}
            states = [(day, self._state_cache[day])]
        else:
            states = ((day, {"recent": self.recent_state(S)}) for day, S in self.harm_states(list(rows_by_day)))
        for day, S in states:
            r = rows_by_day[day]
            c, h = ci[r], hi[r]
            R_ = S["recent"]                                  # (cells, 24) log-decayed severity
            allday = R_.sum(axis=1)
            out["harm_recent"][r] = allday[c]
            out["harm_recent_tod"][r] = (R_[c] * self.K[h]).sum(axis=1)   # x time-of-day weights
            out["nb1_harm_recent"][r] = (self.R1 @ allday)[c]
            out["nb2_harm_recent"][r] = (self.R2 @ allday)[c]
        return out

    def features(self, ci, di, hi, dates=None) -> pd.DataFrame:
        """Feature frame for index arrays (cell idx, day idx, hour).

        `di` may point past the end of the data (live scoring); history windows
        are clipped to what exists. `dates` (array of datetime.date) overrides
        the calendar features when the history day differs from the real day.
        """
        ci, di, hi = np.asarray(ci), np.asarray(di), np.asarray(hi)
        d = np.clip(di, 0, self.n_days)
        if dates is None:
            dates = pd.Timestamp(self.start) + pd.to_timedelta(di, unit="D")
        dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates)))
        weekday = dates.day_of_week.to_numpy()
        month = dates.month.to_numpy()

        # Darkness: civil dawn/dusk per unique date (astral is the slow bit).
        uniq, inv = np.unique(dates.normalize().to_numpy(), return_inverse=True)
        tw = np.array([geo.twilight_hours(pd.Timestamp(u).date()) for u in uniq]).reshape(-1, 2)  # (0, 2) when empty
        mid = hi + 0.5
        is_dark = ((mid < tw[inv, 0]) | (mid > tw[inv, 1])).astype(np.int8)

        # Hour-of-day windows: same hour +/-1, months strictly before the row's month.
        m = self.month_of_day[d]
        band12 = np.zeros(len(ci)); band_all = np.zeros(len(ci))
        for off in (-1, 0, 1):
            h = (hi + off) % 24
            band12 += self.P_month_hour[ci, m, h] - self.P_month_hour[ci, np.clip(m - 12, 0, None), h]
            band_all += self.P_month_hour[ci, m, h]
        yrs = np.maximum(d, 1) / 365.0
        months_yrs = np.maximum(m, 1) / 12.0

        harm = self._harm_features(ci, d, hi)
        X = pd.DataFrame({
            "hour": hi,
            "weekday": weekday,
            "month": month,
            "is_weekend": (weekday >= 5).astype(np.int8),
            "is_dark": is_dark,
            "cell_lat": self.cell_lat[ci],
            "cell_lon": self.cell_lon[ci],
            "dist_bluelight_m": self.dist_bl[ci],
            **harm,
            "cell_hourband_365d": band12,
            "cell_rate_all": self.P_daily[ci, d] / yrs,
            "cell_hourband_rate_all": band_all / months_yrs,
            "area_7d": self.P_area[d] - self.P_area[np.clip(d - 7, 0, None)],
        })
        return X[FEATURES].astype("float32")

    def target(self, ci, di, hi, kind="any"):
        keys = self.pos_keys_violent if kind == "violent" else self.pos_keys
        q = self._key(ci, di, hi)
        pos = np.searchsorted(keys, q)
        return (keys[np.clip(pos, 0, len(keys) - 1)] == q).astype(np.int8)

    # -- frames --------------------------------------------------------------
    def sampled_frame(self, start: str, end: str, neg_per_pos: int, kind="any",
                      seed=config.SEED):
        """All positive cell-hours in [start, end] + uniformly sampled negatives.

        Returns (frame, neg_rate, base_rate). frame has FEATURES + y, w, day,
        cell_idx. w = 1 for positives and 1/neg_rate for negatives, so weighted
        metrics match the full grid. base_rate = true share of positive cell-hours.
        """
        d0 = max(self.day_index(dt.date.fromisoformat(start)), config.WARMUP_DAYS)
        d1 = self.day_index(dt.date.fromisoformat(end))
        nc = len(self.cells)
        keys = self.pos_keys_violent if kind == "violent" else self.pos_keys
        pc, pd_, ph = self._unkey(keys)
        sel = (pd_ >= d0) & (pd_ <= d1)
        pc, pd_, ph = pc[sel], pd_[sel], ph[sel]
        n_grid = nc * (d1 - d0 + 1) * 24
        n_pos = len(pc)
        n_neg_total = n_grid - n_pos

        rng = np.random.default_rng(seed)
        want = min(n_neg_total, neg_per_pos * n_pos)
        draw = int(want * 1.02) + 1000  # positives get rejected below, so overdraw a little
        nc_ = rng.integers(0, nc, draw)
        nd_ = rng.integers(d0, d1 + 1, draw)
        nh_ = rng.integers(0, 24, draw)
        keep = self.target(nc_, nd_, nh_, kind) == 0
        nc_, nd_, nh_ = nc_[keep][:want], nd_[keep][:want], nh_[keep][:want]
        neg_rate = len(nc_) / n_neg_total

        ci = np.concatenate([pc, nc_]); di = np.concatenate([pd_, nd_]); hi = np.concatenate([ph, nh_])
        order = np.lexsort((hi, ci, di))  # time-ordered, deterministic
        ci, di, hi = ci[order], di[order], hi[order]
        X = self.features(ci, di, hi)
        X["y"] = self.target(ci, di, hi, kind)
        X["w"] = np.where(X["y"] == 1, 1.0, 1.0 / neg_rate).astype("float32")
        X["day"] = di
        X["cell_idx"] = ci
        return X, neg_rate, n_pos / n_grid

    def training_frame(self, kind="any", neg_per_pos=config.NEG_PER_POS, seed=config.SEED):
        """Training years (after warm-up) .. TRAIN_END."""
        return self.sampled_frame(config.DATA_START, config.TRAIN_END, neg_per_pos, kind, seed)

    def eval_frame(self, start, end, kind="any", neg_per_pos=config.EVAL_NEG_PER_POS):
        """Out-of-time evaluation frame; use column `w` as sample weights."""
        return self.sampled_frame(start, end, neg_per_pos, kind, seed=config.SEED + 1)


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["any", "violent"], default="any")
    args = ap.parse_args()

    hist = History()
    X, neg_rate, base_rate = hist.training_frame(kind=args.target)
    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    X.to_parquet(config.DATASET_PATH, index=False)
    info = {"rows": len(X), "positives": int(X["y"].sum()), "neg_rate": neg_rate,
            "true_base_rate": base_rate, "covered_cells": len(hist.cells),
            "no_data_cells": len(hist.no_data_cells), "target": args.target}
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
