"""Simulated crimes for demo recordings.

With DEMO_MODE=true, the frontend's Demo page can drop a fake incident at a point
(POST /api/v1/demo/incidents). It is stored in crime_incidents with source 'demo',
so it shows up wherever incidents are listed, and for DEMO_INCIDENT_TTL_MIN minutes
routes treat everything within DEMO_INCIDENT_RADIUS_M of it as the riskiest kind of
street (overlay(), used by routing.plan), so a walk in progress reroutes around it.

Demo rows never reach the risk model: export_crime_db.py only exports source 'lapd'.
"""
import copy
import datetime as dt
import json
import math
import uuid

import h3
from flask import current_app
from sqlalchemy import text

from ..db_utils import fetch_all
from ..extensions import db

SOURCE = "demo"
# Routing cost inside the radius, in the planner's units (1.0 = a 2023 90th-percentile cell-hour):
# high enough that any detour within the planner's budget beats walking past a fresh incident.
AVOID_COST = 50.0
CATEGORIES = ("Robbery", "Assault", "Theft from person", "Vehicle break-in", "Suspicious person")


def enabled() -> bool:
    return bool(current_app.config.get("DEMO_MODE"))


def _now():
    return dt.datetime.now(dt.timezone.utc)


def _ttl():
    return dt.timedelta(minutes=current_app.config["DEMO_INCIDENT_TTL_MIN"])


def active() -> list[dict]:
    """Unexpired simulated incidents, newest first."""
    return fetch_all("""
        SELECT external_id AS id, ts, ST_Y(geom) AS lat, ST_X(geom) AS lng, h3_cell, category, severity
        FROM crime_incidents WHERE source = :src AND ts >= :since ORDER BY ts DESC
    """, src=SOURCE, since=_now() - _ttl())


def add(lat: float, lng: float, category: str | None = None, severity: int = 1) -> dict:
    """Store one simulated incident at (lat, lng) now. Expired ones are removed first."""
    now = _now()
    db.session.execute(text("DELETE FROM crime_incidents WHERE source = :src AND ts < :since"),
                       {"src": SOURCE, "since": now - _ttl()})
    row = {"src": SOURCE, "id": uuid.uuid4().hex[:12], "ts": now, "lat": lat, "lng": lng,
           "cell": h3.latlng_to_cell(lat, lng, current_app.config["H3_RES"]),
           "cat": category or CATEGORIES[0], "sev": severity,
           "raw": json.dumps({"demo": True})}
    db.session.execute(text("""
        INSERT INTO crime_incidents (source, external_id, ts, geom, h3_cell, category, severity, raw)
        VALUES (:src, :id, :ts, ST_SetSRID(ST_MakePoint(:lng, :lat), 4326), :cell, :cat, :sev, CAST(:raw AS jsonb))
    """), row)
    db.session.commit()
    return {"id": row["id"], "ts": now.isoformat(), "lat": lat, "lng": lng, "h3_cell": row["cell"],
            "category": row["cat"], "severity": severity}


def clear() -> int:
    n = db.session.execute(text("DELETE FROM crime_incidents WHERE source = :src"), {"src": SOURCE}).rowcount
    db.session.commit()
    return n


def _metres(lat1, lon1, lat2, lon2) -> float:
    """Equirectangular distance: exact enough at a few hundred metres."""
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return 6_371_000 * math.hypot(x, y)


def overlay(grid, incidents: list[tuple[float, float]], radius_m: float):
    """A copy of a pathfinding RiskGrid where every point within radius_m of an incident
    is risk 1.0 at the avoid cost, including on campus (a fresh incident beats the campus
    zero-risk zone). The planner samples edges through point(), so routes steer around them."""
    if not incidents:
        return grid
    pad_lat = radius_m / 111_000
    pad_lon = pad_lat / max(0.2, math.cos(math.radians(incidents[0][0])))
    box = (min(a for a, _ in incidents) - pad_lat, max(a for a, _ in incidents) + pad_lat,
           min(o for _, o in incidents) - pad_lon, max(o for _, o in incidents) + pad_lon)

    def near(lat, lon):
        if not (box[0] <= lat <= box[1] and box[2] <= lon <= box[3]):
            return False
        return any(_metres(lat, lon, a, o) <= radius_m for a, o in incidents)

    base = type(grid)

    class DemoOverlayGrid(base):
        def point(self, lat, lon, hour):
            if near(lat, lon):
                return 1.0, AVOID_COST, True
            return super().point(lat, lon, hour)

        def risk_at(self, lat, lon, hour):
            if near(lat, lon):
                return 1.0, True
            return super().risk_at(lat, lon, hour)

    out = copy.copy(grid)          # shares the base grid's (large) dicts; only the class changes
    out.__class__ = DemoOverlayGrid
    return out
