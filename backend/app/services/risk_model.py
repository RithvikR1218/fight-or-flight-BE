"""Bridge between the backend and the crime-risk model in algorithm/ (see algorithm/README.md).

The model package is used as-is: `algorithm.scoring.score` (score_cell / explain_cell /
build_cache). This module adds what the backend knows:

* model inputs from backend data: any key in `context.cell_context()` whose name
  matches a model feature is passed as `context_features` (today that's the live
  `is_dark` from the sun ingestor; a retrained model that adds e.g. `temp_c` or
  `lit_road_length_m` picks them up with no backend change),
* the city-wide heatmap (all cells x 24 h) cached per date,
* route costs: risk mapped onto the OSM walk graph's edges.
"""
import datetime as dt
import json
import logging
import threading
from collections import OrderedDict
from zoneinfo import ZoneInfo

import numpy as np

from ..db_utils import fetch_all

log = logging.getLogger(__name__)
LA = ZoneInfo("America/Los_Angeles")
NO_DATA_RISK = 0.5        # neutral risk for cells without LAPD data (per algorithm/README.md)
_CACHE_DATES = 3          # heatmaps kept in memory (~5 MB each)

_lock = threading.Lock()
_heatmaps: "OrderedDict[str, dict]" = OrderedDict()
_edges = None             # static: road edges + their cells, loaded once per process


class ModelUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------- availability
def _modules():
    from algorithm.common import config
    from algorithm.scoring import score
    return config, score


def status() -> dict:
    try:
        config, _ = _modules()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"algorithm package not importable: {exc}"}
    missing = [str(p.name) for p in (config.MODEL_PATH, config.META_PATH, config.DB_PATH) if not p.exists()]
    if missing:
        hint = ("run `./scripts/fetch_data.sh` (or `python -m scripts.export_crime_db`), then restart the api"
                if "crime.db" in missing
                else "train the model (algorithm/README.md)")
        return {"available": False, "reason": f"missing {', '.join(missing)}: {hint}"}
    return {"available": True}


def _require():
    st = status()
    if not st["available"]:
        raise ModelUnavailable(st["reason"])
    return _modules()


def warm():
    """Load model + history (~4 s, ~600 MB) and today's heatmap (~20 s)."""
    if not status()["available"]:
        log.warning("risk model not available: %s", status()["reason"])
        return
    _, score = _modules()
    score._load()
    heatmap(local_now().date())
    log.info("risk model warm")


def meta() -> dict:
    config, _ = _require()
    m = json.loads(config.META_PATH.read_text())
    m.pop("risk_quantiles", None)
    metrics_path = config.ARTIFACT_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        m["test_2023"] = metrics.get("test_2023", {}).get("xgb", metrics.get("test_2023"))
    return m


# ---------------------------------------------------------------- time helpers
def local_now() -> dt.datetime:
    return dt.datetime.now(LA)


def resolve_time(at: dt.datetime | None, date: dt.date | None, hour: int | None):
    """-> (local date, local hour, aware UTC instant, is_now). The model works in LA clock hours."""
    now = local_now()
    if at is not None:
        local = at.astimezone(LA)
    else:
        d = date or now.date()
        h = now.hour if hour is None else int(hour) % 24
        local = dt.datetime(d.year, d.month, d.day, h, 30, tzinfo=LA)
    is_now = local.date() == now.date() and local.hour == now.hour
    return local.date(), local.hour, local.astimezone(dt.timezone.utc), is_now


# ---------------------------------------------------------------- scoring
def model_inputs(context: dict, features: list[str]) -> dict:
    """Backend values whose names match model features (None values are skipped)."""
    out = {}
    for f in features:
        v = context.get(f)
        if v is not None and not isinstance(v, (dict, list, str)):
            out[f] = int(v) if f.startswith("is_") else float(v)
    return out


def score_cell(cell: str, at=None, date=None, hour=None, use_live=True, top_k=5) -> dict:
    """Model score + SHAP factors for one cell, plus the backend context for it.

    Live backend values are passed to the model only for the current hour (they
    describe now, not some other hour)."""
    from .context import cell_context

    config, score = _require()
    d, h, instant, is_now = resolve_time(at, date, hour)
    ctx = cell_context(cell, instant)
    features = json.loads(config.META_PATH.read_text())["features"]
    inputs = model_inputs(ctx, features) if (use_live and is_now) else {}
    result = score.explain_cell(cell, h, d, context_features=inputs or None, top_k=top_k)
    result["model_inputs_from_backend"] = inputs
    result["context"] = ctx
    return result


def heatmap(date: dt.date) -> dict:
    """algorithm build_cache() output for `date`, memoised (and reusing the
    committed risk_cache.json when it's for the same date)."""
    config, score = _require()
    key = date.isoformat()
    with _lock:
        if key in _heatmaps:
            _heatmaps.move_to_end(key)
            return _heatmaps[key]
        cache = None
        if config.CACHE_PATH.exists():
            on_disk = json.loads(config.CACHE_PATH.read_text())
            if on_disk.get("date") == key:
                cache = on_disk
        if cache is None:
            log.info("building risk heatmap for %s (~20 s)", key)
            cache = score.build_cache(date)
            if not cache["cells"]:
                log.warning("risk heatmap for %s has no modelled cells: crime.db has no usable LAPD history; "
                            "every cell gets the neutral no-data risk. Run `./scripts/fetch_data.sh` then restart the api "
                            "(or `python -m scripts.backfill_crime --source lapd --start 2020-01-01` then "
                            "`python -m scripts.export_crime_db`)", key)
        cache["_risk_index"] = {c: np.asarray(v["risk"], dtype=np.float32) for c, v in cache["cells"].items()}
        _heatmaps[key] = cache
        while len(_heatmaps) > _CACHE_DATES:
            _heatmaps.popitem(last=False)
        return cache


def hotzones(date: dt.date, hour: int, cells: set[str] | None = None, min_risk: float = 0.0) -> dict:
    cache = heatmap(date)
    _, score = _modules()
    feats = cache["features"]
    out = []
    for c, v in cache["cells"].items():
        if cells is not None and c not in cells:
            continue
        r = v["risk"][hour]
        if r < min_risk:
            continue
        out.append({"cell": c, "risk": r, "prob": v["prob"][hour],
                    "factors": [{"feature": feats[i], "label": score.LABELS.get(feats[i], feats[i]),
                                 "effect": e} for i, e in v["top_factors"][hour]]})
    no_data = cache["no_data_cells"] if cells is None else [c for c in cache["no_data_cells"] if c in cells]
    return {"date": cache["date"], "hour": hour, "history_as_of": cache["history_as_of"],
            "risk_definition": cache["risk_definition"], "cells": out, "no_data_cells": no_data}


# ---------------------------------------------------------------- routing costs
def _load_edges():
    global _edges
    if _edges is None:
        rows = fetch_all("SELECT u, v, key, length_m, h3_cells FROM road_edges ORDER BY id")
        if not rows:
            raise ModelUnavailable("road_edges is empty: run `python -m scripts.load_roads`")
        cell_ids, edge_of, cell_of = {}, [], []
        for i, r in enumerate(rows):
            for c in r["h3_cells"] or []:
                edge_of.append(i)
                cell_of.append(cell_ids.setdefault(c, len(cell_ids)))
        _edges = {
            "u": np.array([r["u"] for r in rows], dtype=np.int64),
            "v": np.array([r["v"] for r in rows], dtype=np.int64),
            "key": np.array([r["key"] for r in rows], dtype=np.int16),
            "length_m": np.array([r["length_m"] for r in rows], dtype=np.float64),
            "edge_of": np.array(edge_of, dtype=np.int64),
            "cell_of": np.array(cell_of, dtype=np.int64),
            "cells": list(cell_ids),
        }
    return _edges


def edge_costs(date: dt.date, hour: int, lam: float = 4.0, no_data_risk: float = NO_DATA_RISK):
    """Per-edge risk for one hour, as a pandas DataFrame:
    risk_mean / risk_max over the cells the edge crosses, no_data_share, and
    cost = length_m * (1 + lam * risk_mean)  (the cost model from algorithm/README.md)."""
    import pandas as pd

    cache = heatmap(date)
    e = _load_edges()
    idx = cache["_risk_index"]
    cell_risk = np.array([idx[c][hour] if c in idx else np.nan for c in e["cells"]], dtype=np.float64)
    no_data = np.isnan(cell_risk)
    cell_risk = np.where(no_data, no_data_risk, cell_risk)

    n = len(e["u"])
    r = cell_risk[e["cell_of"]]
    sums, counts, nd = np.zeros(n), np.zeros(n), np.zeros(n)
    maxes = np.full(n, -np.inf)
    np.add.at(sums, e["edge_of"], r)
    np.add.at(counts, e["edge_of"], 1)
    np.add.at(nd, e["edge_of"], no_data[e["cell_of"]])
    np.maximum.at(maxes, e["edge_of"], r)
    counts[counts == 0] = 1
    risk_mean = sums / counts
    return pd.DataFrame({
        "u": e["u"], "v": e["v"], "key": e["key"], "length_m": e["length_m"],
        "risk_mean": risk_mean.round(4), "risk_max": np.where(np.isinf(maxes), no_data_risk, maxes).round(4),
        "no_data_share": (nd / counts).round(3),
        "cost": (e["length_m"] * (1 + lam * risk_mean)).round(2),
    })
