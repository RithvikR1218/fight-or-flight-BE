"""Shared helpers for the USC data scrapers: HTTP session, paths, GeoJSON output, places."""
import functools
import json
import os
import time
from datetime import datetime, timezone

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
POIS = os.path.join(DATA, "pois")          # loaded into the `pois` table by backend/scripts/load_pois.py
USC = os.path.join(DATA, "usc")            # tables that aren't map points (room lists, availability)
PLACES_FILE = os.path.join(USC, "usc_places.json")

# Some sites reject non-browser user agents.
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"


def session(cookie: str | None = None) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json, text/html;q=0.9, */*;q=0.8"})
    if cookie:
        s.headers["Cookie"] = cookie
    return s


def get(s: requests.Session, url: str, retries: int = 3, **kw) -> requests.Response:
    for attempt in range(retries):
        try:
            r = s.get(url, timeout=60, **kw)
            if r.status_code < 500:
                r.raise_for_status()
                return r
        except requests.ConnectionError:
            if attempt == retries - 1:
                raise
        time.sleep(1.5 * (attempt + 1))
    r.raise_for_status()
    return r


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: str, obj, compact: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:   # not the Windows ANSI code page
        if compact:   # big files: skip indentation
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def point(lng: float, lat: float) -> dict:
    return {"type": "Point", "coordinates": [round(lng, 7), round(lat, 7)]}


def feature(geometry: dict, **props) -> dict:
    return {"type": "Feature", "geometry": geometry,
            "properties": {k: v for k, v in props.items() if v is not None and v != [] and v != ""}}


def write_geojson(path: str, features: list[dict], **meta) -> None:
    write_json(path, {"type": "FeatureCollection", **meta, "features": features})


@functools.lru_cache(maxsize=1)
def load_places() -> dict:
    """USC building codes and named outdoor places -> coordinates (see build_places.py)."""
    with open(PLACES_FILE, encoding="utf-8") as f:
        return json.load(f)


def building(code: str) -> dict | None:
    return load_places()["buildings"].get(code.upper())
