"""Room Finder classrooms -> building -> floor: data/usc/usc_rooms.json.

    python -m scripts.usc_data room-floors [--offline]

Inputs:
  * data/usc/roomfinder_rooms.json (the `rooms` step): the classrooms and their campus.
  * data/places/usc_buildings.geojson: UPC building codes, names, centre points.
  * USC FPM GIS `Building` layer (https://fpmgis.usc.edu/server/rest/services/USC_FPM/Building/FeatureServer/0):
    site/campus, above-ground floor count (BldgFlrInt), HasBasement, and the centre point for buildings
    that aren't in usc_buildings.geojson (other campuses). One request, attributes only. --offline skips it.
Floors come from backend/app/services/rooms.py:parse_room_number (rules + evidence documented there).

Indoor pathing only supports UPC, so a room gets `pathing: true` only if it is on UPC and its floor is known.
Room Finder lists only centrally scheduled classrooms: most rooms on campus (offices, labs, department
rooms such as RTH's) are not in this file; search falls back to parse_room_number for those.
"""
import argparse
import importlib.util
import json
import os

import requests

from .common import DATA, ROOT, USC, now_iso, write_json

FPM_BUILDING = "https://fpmgis.usc.edu/server/rest/services/USC_FPM/Building/FeatureServer/0/query"
BUILDINGS_FILE = os.path.join(DATA, "places", "usc_buildings.geojson")
ROOMFINDER_FILE = os.path.join(USC, "roomfinder_rooms.json")
SOC_FILE = os.path.join(USC, "soc_rooms.json")   # optional; {"rooms": [{"code","room","section_count","departments","terms"}]}
OUT = os.path.join(USC, "usc_rooms.json")
UA = "usc-campus-data-script/1.0"
# FPM SITE -> campus. UPCOFF = USC-owned, next to UPC (same street grid; pathable like UPC).
CAMPUS_BY_SITE = {"UPC": "UPC", "UPCOFF": "UPC", "HSC": "HSC", "CAT": "Catalina", "OFFCMP": "Off-campus",
                  "LEASED": "Leased", "USED": "Used", "TENANT": "Tenant", "NONUSC": "Non-USC"}


def _load_rooms_module():
    """backend/app/services/rooms.py by path (importing the `app` package would pull in its settings)."""
    path = os.path.join(ROOT, "backend", "app", "services", "rooms.py")
    spec = importlib.util.spec_from_file_location("usc_rooms_rules", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch_fpm() -> dict[str, dict]:
    r = requests.get(FPM_BUILDING, headers={"User-Agent": UA}, timeout=120, params={
        "where": "1=1", "returnGeometry": "false", "f": "json", "resultRecordCount": 2000,
        "outFields": "BLD_CODE,BLDG_NAME,SITE,BldgFlrInt,HasBasement,LAT,LON,YEAR_DEMO,YEAR_SOLD"})
    r.raise_for_status()
    out = {}
    for f in r.json().get("features", []):
        a = f["attributes"]
        code = (a.get("BLD_CODE") or "").strip().upper()
        if not code or a.get("YEAR_DEMO") or a.get("YEAR_SOLD") or a.get("SITE") in ("DEMO", "SOLD"):
            continue
        out.setdefault(code, a)
    return out


def _title(name: str | None) -> str | None:
    if not name:
        return None
    small = {"of", "and", "for", "the", "at", "in", "&"}
    words = name.split()
    return " ".join(w.lower() if i and w.lower() in small else w[:1] + w[1:].lower() for i, w in enumerate(words))


def build(rf: dict, geo: dict, fpm: dict[str, dict], rules, soc: dict | None = None) -> tuple[dict, list[str]]:
    """Union of Room Finder rooms and (optional) Schedule-of-Classes rooms, deduped on (code, room).
    Room Finder attributes win; SoC adds `section_count`, `departments` and `soc_terms`. SoC codes that
    are neither in usc_buildings.geojson nor in FPM GIS are left out and listed in the report."""
    by_code = {f["properties"]["code"]: f for f in geo["features"]}
    buildings, rooms, report = {}, [], []
    merged: dict[tuple[str, str], dict] = {}
    for r in rf["rooms"]:
        key = (r["building"], rules.normalize_room(r["room"]))
        merged[key] = {"rf": r, "soc": None}
    unresolved: dict[str, int] = {}
    for s in (soc or {}).get("rooms", []):
        code, room = s["code"].upper(), rules.normalize_room(s["room"])
        if code not in by_code and code not in fpm and not any(k[0] == code for k in merged):
            unresolved[code] = unresolved.get(code, 0) + 1
            continue
        merged.setdefault((code, room), {"rf": None, "soc": None})["soc"] = s
    for code, n in sorted(unresolved.items()):
        report.append(f"{code}: Schedule-of-Classes code not in usc_buildings.geojson or FPM GIS; {n} room(s) left out")
    for code in sorted({k[0] for k in merged}):
        g, a = by_code.get(code), fpm.get(code) or {}
        rf_campus = next((r.get("campus") for r in rf["rooms"] if r["building"] == code and r.get("campus")), None)
        campus = rf_campus or (("UPC" if g else None) or CAMPUS_BY_SITE.get(a.get("SITE")))
        if g and campus == "UPC":
            lng, lat = g["geometry"]["coordinates"]
            name, src = g["properties"]["name"], "usc_buildings.geojson"
        elif a.get("LAT") and a.get("LON"):
            lat, lng = round(float(a["LAT"]), 6), round(float(a["LON"]), 6)
            name, src = _title(a.get("BLDG_NAME")), "fpm-gis"
        else:
            lat = lng = name = src = None
        if g is None and campus == "UPC":
            report.append(f"{code}: Room Finder says UPC but the code is not in usc_buildings.geojson")
        if src is None:
            report.append(f"{code}: no coordinates in usc_buildings.geojson or FPM GIS")
        fpm_campus = CAMPUS_BY_SITE.get(a.get("SITE"))
        if a and rf_campus and fpm_campus and fpm_campus != rf_campus:
            report.append(f"{code}: Room Finder campus {rf_campus} but FPM site {a.get('SITE')}")
        n, bsmt = a.get("BldgFlrInt"), a.get("HasBasement")
        floors = None
        if isinstance(n, int) and n > 0:
            floors = ([-1] if bsmt == "YES" else []) + list(range(1, n + 1))
        buildings[code] = {"name": name, "building_id": f"bldg:{code}", "campus": campus,
                           "lat": lat, "lng": lng, "location_source": src,
                           "floor_count": n if isinstance(n, int) else None,
                           "has_basement": {"YES": True, "NO": False}.get(bsmt),
                           "floors": floors, "room_count": 0}
    for (code, nroom), src in sorted(merged.items()):
        r, s, b = src["rf"] or {}, src["soc"] or {}, buildings[code]
        room_no = r.get("room") or nroom
        p = rules.parse_room_number(room_no, code, floor_count=b["floor_count"])
        note = None
        if p["floor"] is not None and p["floor"] < 0 and b["has_basement"] is False:
            note = "FPM says this building has no basement"
        b["room_count"] += 1
        upc = b["campus"] == "UPC"
        room = {
            "id": f"room:{code}-{room_no}", "code": code, "room": room_no,
            "building_id": b["building_id"], "building_name": b["name"], "campus": b["campus"],
            "floor": p["floor"], "floor_label": p["floor_label"], "floor_confidence": p["confidence"],
            "pathing": bool(upc and p["floor"] is not None),
            "lat": b["lat"], "lng": b["lng"],
            "capacity": r.get("capacity"), "classroom_type": r.get("classroom_type"),
            "furniture": r.get("furniture"), "equipment": r.get("equipment") or [], "url": r.get("url"),
            "sources": [x for x, have in (("roomfinder", src["rf"]), ("soc", src["soc"])) if have],
            "section_count": s.get("section_count", 0),
        }
        if s:
            room["departments"] = s.get("departments") or []
            room["soc_terms"] = s.get("terms") or []
        if note:
            room["floor_note"] = note
        if r.get("in_index") is False:
            room["in_roomfinder_index"] = False
        rooms.append(room)
        if p["confidence"] == "unknown":
            report.append(f"{code} {room_no}: floor unknown (room number fits no rule)")
        if note:
            report.append(f"{code} {room_no}: {note}")
    return {"buildings": buildings, "rooms": rooms}, report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="usc_data room-floors")
    ap.add_argument("--offline", action="store_true", help="don't query FPM GIS (no floor counts, no HSC points)")
    args = ap.parse_args(argv)
    with open(ROOMFINDER_FILE, encoding="utf-8") as f:
        rf = json.load(f)
    with open(BUILDINGS_FILE, encoding="utf-8") as f:
        geo = json.load(f)
    soc = None
    if os.path.exists(SOC_FILE):                 # optional: rooms seen in class schedules
        with open(SOC_FILE, encoding="utf-8") as f:
            soc = json.load(f)
    fpm = {} if args.offline else fetch_fpm()
    rules = _load_rooms_module()
    data, report = build(rf, geo, fpm, rules, soc)
    out = {
        "generated_at": now_iso(),
        "sources": {
            "rooms": f"Room Finder {rf.get('source')} (fetched {rf.get('fetched_at')})",
            "schedule_rooms": (f"{soc.get('source')} terms {soc.get('terms')} (fetched {soc.get('fetched_at')})"
                               if soc else None),
            "buildings": "data/places/usc_buildings.geojson (USC FPM GIS + OSM)",
            "floors": None if args.offline else FPM_BUILDING.rsplit("/query", 1)[0] + " (BldgFlrInt, HasBasement, SITE, LAT/LON)",
            "floor_rules": "backend/app/services/rooms.py (FPM LactationRoom / MenstrualProductDispenser room-floor pairs; "
                           "USC Libraries Leavey tour guide for LVL)",
        },
        "floor_convention": "floor is an ordinal level: 1, 2, 3... = numbered floors (1 = first floor); 0 = a G/GDN "
                            "level below floor 1; -1 = B/LL/LG (basement, lower level); -2 = SB. floor_label is the "
                            "building's own label. floor_confidence: exact (documented per-room fact), rule (USC "
                            "numbering: NXX/NXXX -> floor N, letter prefix -> level), unknown (floor null). "
                            "building.floors = [-1 if FPM HasBasement] + 1..FPM BldgFlrInt.",
        "pathing": "true only for UPC rooms with a known floor (indoor pathing supports UPC only)",
        **data,
        "report": report,
    }
    write_json(OUT, out)
    from collections import Counter
    conf = Counter(r["floor_confidence"] for r in data["rooms"])
    print(f"usc_rooms.json: {len(data['rooms'])} rooms in {len(data['buildings'])} buildings; "
          f"floor confidence {dict(conf)}; pathing {sum(r['pathing'] for r in data['rooms'])}")
    for line in report:
        print("  " + line)
