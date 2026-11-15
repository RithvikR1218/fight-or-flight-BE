"""Serving hook for the crime-risk model.

The Flask API calls `score_cell` for ad-hoc requests and serves the precomputed
cache for the heatmap.

    from scoring.score import score_cell, explain_cell
    score_cell("8929a1d4b4bffff", 22, datetime.date(2026, 11, 14))   # -> 0.0..1.0

Build the cache (every modelled cell x 24 hours for one date):
    python -m scoring.score --build-cache --date 2026-11-14
Explain one score from the command line:
    python -m scoring.score --cell 8929a1d4b4bffff --hour 22 --date 2026-11-14

Two numbers come out of the model:
  prob  calibrated probability that >=1 incident is reported in that cell
        during that hour on that date (usually 0.1%-3%), multiplied by the
        blue-light factor (up to 25% lower right next to a USC emergency
        phone; see config.BLUELIGHT_*).
  risk  0..1 relative score = where `prob` falls among all modelled
        cell-hours in the 2023 test year (0.9 = riskier than 90% of them).
        This is what the heatmap and routing should use.

History features (recent counts) use incidents strictly before `date`. For
dates after 2023 they are anchored at config.LIVE_HISTORY_END (end of 2023,
the last full year before LAPD's NIBRS reporting drop); calendar and darkness
features always use the real date.
"""
import argparse
import datetime as dt
import json
from functools import lru_cache

import h3
import numpy as np
import pandas as pd
import xgboost as xgb

from algorithm.common import config, geo
from algorithm.features.build_dataset import History

LABELS = {
    "hour": "time of day",
    "weekday": "day of week",
    "month": "month",
    "is_weekend": "weekend",
    "is_dark": "after dark",
    "cell_lat": "location (north/south)",
    "cell_lon": "location (east/west)",
    "dist_bluelight_m": "distance to blue-light phone",
    "harm_recent": "recent serious crime here (log-scale recency)",
    "harm_recent_tod": "recent serious crime here around this time of day",
    "nb1_harm_recent": "recent serious crime in adjacent blocks",
    "nb2_harm_recent": "recent serious crime 2 blocks away",
    "cell_hourband_365d": "incidents here at this hour, past year",
    "cell_rate_all": "long-run incident rate here",
    "cell_hourband_rate_all": "long-run rate here at this hour",
    "area_7d": "area-wide incidents, last 7 days",
}


@lru_cache(maxsize=1)
def _load():
    meta = json.loads(config.META_PATH.read_text())
    booster = xgb.Booster()
    booster.load_model(config.MODEL_PATH)
    return booster, meta, History()


def _prob(raw, neg_rate):
    raw = np.clip(raw, 1e-7, 1 - 1e-7)
    return raw * neg_rate / (raw * neg_rate + (1 - raw))


def _risk(prob, meta):
    qs = np.asarray(meta["risk_quantiles"])
    return np.interp(prob, qs, np.linspace(0, 1, len(qs)))


def cell_status(h3_id: str) -> str:
    """'ok' = modelled; 'no_data' = inside the radius but without enough LAPD
    records (mostly other police jurisdictions); 'outside' = beyond the radius."""
    _, _, hist = _load()
    i = hist.cidx.get(h3_id)
    if i is not None and i < len(hist.cells):
        return "ok"
    return "no_data" if h3_id in _no_data_set() else "outside"


@lru_cache(maxsize=1)
def _no_data_set():
    return frozenset(_load()[2].no_data_cells)


def _frame(cells, hours, date, context_features=None):
    booster, meta, hist = _load()
    bad = [c for c in cells if cell_status(c) != "ok"]
    if bad:
        raise KeyError(f"cell(s) not modelled ({cell_status(bad[0])}): {bad[:3]}")
    ci = np.array([hist.cidx[c] for c in cells])
    di = np.full(len(ci), hist.day_index(_history_day(date)))
    X = hist.features(ci, di, np.asarray(hours), dates=[date] * len(ci))
    for k, v in (context_features or {}).items():
        if k not in X.columns:
            raise KeyError(f"unknown feature override: {k}")
        X[k] = v
    return X[meta["features"]]


def _predict(X, with_contribs=False):
    """prob = model probability x blue-light factor (see config.BLUELIGHT_*),
    so a nearby phone always lowers the served score."""
    booster, meta, _ = _load()
    dm = xgb.DMatrix(X)
    prob = _prob(booster.predict(dm), meta["neg_rate"]) * geo.bluelight_factor(X["dist_bluelight_m"].to_numpy())
    risk = _risk(prob, meta)
    if not with_contribs:
        return prob, risk, None
    contribs = booster.predict(dm, pred_contribs=True)[:, :-1]
    return prob, risk, contribs


def _history_day(date: dt.date) -> dt.date:
    """Day the history windows end at (exclusive). Past dates use their own
    history; dates after the reliable data use LIVE_HISTORY_END."""
    cutoff = dt.date.fromisoformat(config.LIVE_HISTORY_END) + dt.timedelta(days=1)
    return min(date, cutoff)


def _as_date(date):
    if date is None:
        return dt.date.today()
    if isinstance(date, dt.datetime):
        return date.date()
    if isinstance(date, str):
        return dt.date.fromisoformat(date[:10])
    return date


def score_cell(h3_id: str, hour: int, date=None, context_features: dict | None = None) -> float | None:
    """0..1 relative crime risk for one H3 cell (res 9) at a clock hour on a date.

    Returns None for a cell inside the radius that has no LAPD data (another
    police jurisdiction): unknown, not safe. Raises KeyError beyond the radius.
    context_features: optional overrides, e.g. {"is_dark": 1}. Keys must be
    model feature names (see LABELS)."""
    status = cell_status(h3_id)
    if status == "no_data":
        return None
    if status == "outside":
        raise KeyError(f"{h3_id} is outside the {config.RADIUS_MILES:g}-mile model area")
    X = _frame([h3_id], [int(hour) % 24], _as_date(date), context_features)
    _, risk, _ = _predict(X)
    return float(risk[0])


def explain_cell(h3_id: str, hour: int, date=None, context_features: dict | None = None,
                 top_k: int = 5) -> dict:
    """Score plus the SHAP factors behind it, for the UI and the pitch."""
    date = _as_date(date)
    status = cell_status(h3_id)
    if status != "ok":
        return {"cell": h3_id, "hour": int(hour), "date": date.isoformat(),
                "risk": None, "prob": None, "status": status, "factors": [],
                "note": "No LAPD records here (likely another police jurisdiction); risk unknown."
                        if status == "no_data" else "Outside the model area."}
    X = _frame([h3_id], [int(hour) % 24], date, context_features)
    prob, risk, contribs = _predict(X, with_contribs=True)
    order = np.argsort(-np.abs(contribs[0]))[:top_k]
    _, meta, _ = _load()
    feats = meta["features"]
    dist = float(X["dist_bluelight_m"].iloc[0])
    return {
        "cell": h3_id, "hour": int(hour), "date": date.isoformat(),
        "risk": round(float(risk[0]), 4), "prob": round(float(prob[0]), 5), "status": "ok",
        # Per-incident weighted history (severity x recency x time-of-day match):
        "harm_score": round(float(X["harm_recent_tod"].iloc[0]), 3),
        "bluelight": {"dist_m": round(dist), "protection": round(geo.bluelight_protection(dist), 3),
                      "factor": round(float(geo.bluelight_factor(dist)), 3)},
        "history_as_of": (_history_day(date) - dt.timedelta(days=1)).isoformat(),
        "factors": [{
            "feature": feats[j], "label": LABELS.get(feats[j], feats[j]),
            "value": round(float(X.iloc[0, j]), 3),
            "effect": round(float(contribs[0, j]), 4),   # + raises risk, - lowers it (log-odds)
        } for j in order],
    }


def build_cache(date=None, top_k: int = 3) -> dict:
    """Risk for every modelled cell x 24 hours on `date`, with top factors.

    top_factors[h] is a list of [feature_index, shap_effect] pairs; look the
    index up in the cache's `features` list (keeps the file small)."""
    date = _as_date(date)
    _, meta, hist = _load()
    cells = list(hist.cells)
    cc = [c for c in cells for _ in range(24)]
    hh = [h for _ in cells for h in range(24)]
    feats = meta["features"]
    if cells:
        X = _frame(cc, hh, date)
        prob, risk, contribs = _predict(X, with_contribs=True)
    else:  # empty history (e.g. crime.db exported before the LAPD backfill): every cell is no-data
        X, prob, risk, contribs = None, np.zeros(0), np.zeros(0), np.zeros((0, len(feats)))
    out = {}
    for i, c in enumerate(cells):
        sl = slice(i * 24, (i + 1) * 24)
        tops = []
        for row in contribs[sl]:
            idx = np.argsort(-np.abs(row))[:top_k]
            tops.append([[int(j), round(float(row[j]), 3)] for j in idx])
        lat, lon = h3.cell_to_latlng(c)
        dist = float(X["dist_bluelight_m"].iloc[i * 24])
        out[c] = {"lat": round(lat, 6), "lon": round(lon, 6),
                  "bluelight_dist_m": round(dist),
                  "bluelight_protection": round(geo.bluelight_protection(dist), 3),
                  "risk": np.round(risk[sl], 4).tolist(),
                  "prob": np.round(prob[sl], 5).tolist(),
                  "top_factors": tops}
    return {
        "date": date.isoformat(),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "h3_res": meta["h3_res"], "target": meta["target"],
        "history_as_of": (_history_day(date) - dt.timedelta(days=1)).isoformat(),
        "center": meta.get("center"), "radius_miles": meta.get("radius_miles"),
        "risk_definition": "percentile of calibrated incident probability (with the "
                           "blue-light factor) among modelled cell-hours in 2023 (0..1)",
        "features": feats,
        "feature_labels": LABELS,
        "cells": out,
        # Inside the radius but no LAPD data (other police jurisdictions): show
        # as "no data" on the map, never as safe.
        "no_data_cells": list(hist.no_data_cells),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--cell")
    ap.add_argument("--hour", type=int, default=22)
    args = ap.parse_args()
    if args.build_cache:
        cache = build_cache(args.date)
        config.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.CACHE_PATH.write_text(json.dumps(cache, separators=(",", ":")))
        risks = np.array([r for v in cache["cells"].values() for r in v["risk"]])
        print(f"Wrote {len(cache['cells'])} cells x 24 h (+{len(cache['no_data_cells'])} no-data cells) "
              f"to {config.CACHE_PATH} (median risk {np.median(risks):.2f}, max {risks.max():.2f})")
    if args.cell:
        print(json.dumps(explain_cell(args.cell, args.hour, args.date), indent=2))


if __name__ == "__main__":
    main()
