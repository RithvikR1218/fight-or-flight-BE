import json
import os

import pytest

from app.services.rooms import parse_room_number, parse_room_query, split_location

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
BUILDINGS = os.path.join(DATA, "places", "usc_buildings.geojson")
ROOMS = os.path.join(DATA, "usc", "usc_rooms.json")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def codes():
    return {f["properties"]["code"] for f in _load(BUILDINGS)["features"]}


@pytest.fixture(scope="module")
def dataset():
    return _load(ROOMS)


# --- parse_room_number -----------------------------------------------------------------------------

@pytest.mark.parametrize("room, code, floor, label, conf", [
    ("108", "RTH", 1, "1", "rule"),
    ("320", None, 3, "3", "rule"),
    ("201A", "THH", 2, "2", "rule"),
    ("108a", None, 1, "1", "rule"),
    ("601", "SGM", 6, "6", "rule"),
    ("4104A", "DEN", 4, "4", "rule"),          # FPM: DEN 4104A on floor 4
    ("1202", None, 1, "1", "rule"),            # FPM: CRC 1202 on floor 1
    ("B09", "THH", -1, "B", "rule"),
    ("B7B", "MHP", -1, "B", "rule"),
    ("b-2", "SOS", -1, "B", "rule"),
    ("LL101", "VPD", -1, "LL", "rule"),
    ("G430F", None, 0, "G", "rule"),
    ("SB80A", None, -2, "SB", "rule"),
    ("17", "LVL", -1, "LL", "exact"),          # Leavey classrooms are on the Lower Level
    ("3B", "LVL", -1, "LL", "exact"),
    ("188", "ADM", -1, "B", "exact"),          # FPM: Bovard 188 is on B
    ("06", "GFS", None, None, "unknown"),
    ("3B", None, None, None, "unknown"),
    ("13", None, None, None, "unknown"),
    ("12345", None, None, None, "unknown"),
    ("", None, None, None, "unknown"),
    ("lobby", None, None, None, "unknown"),
    ("011", None, None, None, "unknown"),
])
def test_parse_room_number(room, code, floor, label, conf):
    assert parse_room_number(room, code) == {"floor": floor, "floor_label": label, "confidence": conf}


def test_floor_count_limits_rule():
    assert parse_room_number("1001", "WPH", floor_count=12)["confidence"] == "unknown"   # 10 or 1?
    assert parse_room_number("1001", None, floor_count=4)["floor"] == 1
    assert parse_room_number("601", None, floor_count=3)["floor"] is None                 # above the roof


# --- parse_room_query ------------------------------------------------------------------------------

@pytest.mark.parametrize("q, expected", [
    ("rth 108", {"code": "RTH", "room": "108"}),
    ("RTH108", {"code": "RTH", "room": "108"}),
    ("rth-108", {"code": "RTH", "room": "108"}),
    ("  RTH   108 ", {"code": "RTH", "room": "108"}),
    ("rth room 108", {"code": "RTH", "room": "108"}),
    ("thh 201a", {"code": "THH", "room": "201A"}),
    ("108 rth", {"code": "RTH", "room": "108"}),
    ("sos b2", {"code": "SOS", "room": "B2"}),
    ("sosb2", {"code": "SOS", "room": "B2"}),
    ("thh b09", {"code": "THH", "room": "B09"}),
    ("vpd ll101", {"code": "VPD", "room": "LL101"}),
    ("lvl 3b", {"code": "LVL", "room": "3B"}),
    ("lvl 17", {"code": "LVL", "room": "17"}),
    ("gfs 06", {"code": "GFS", "room": "06"}),
    ("law 101", {"code": "LAW", "room": "101"}),
])
def test_parse_room_query(q, expected, codes):
    assert parse_room_query(q, codes) == expected


@pytest.mark.parametrize("q", [
    "7 eleven", "route 66", "studio 54", "cvs 24", "cvs 2401", "rth", "108", "rth 1", "rth 108 parking",
    "tutor hall 108", "car 54", "gap 1969", "sun 24", "xyz 108", "", "pizza 101",
])
def test_parse_room_query_negative(q, codes):
    assert "CVS" not in codes
    assert parse_room_query(q, codes) is None


# --- split_location (schedule "Room" values) -------------------------------------------------------

@pytest.mark.parametrize("loc, expected", [
    ("THH101", ("THH", "101")),
    ("THH 101", ("THH", "101")),
    ("RTH-105", ("RTH", "105")),
    ("SOSB2", ("SOS", "B2")),
    ("VPDLL101", ("VPD", "LL101")),
    ("MHPB7B", ("MHP", "B7B")),
    ("GFS106", ("GFS", "106")),
    ("thh 201a", ("THH", "201A")),
    ("ONLINE", None), ("TBA", None), ("OFFICE", None), ("DEN@Viterbi", None), ("OFF CAMPUS", None),
    ("", None), ("THH", None), ("XYZ101", None),
])
def test_split_location(loc, expected, codes):
    assert split_location(loc, codes) == expected


def test_split_location_without_codes():
    assert split_location("THH101") == ("THH", "101")
    assert split_location("ONLINE") is None


# --- data/usc/usc_rooms.json -----------------------------------------------------------------------

def test_dataset_rooms_have_code_and_room(dataset):
    assert len(dataset["rooms"]) >= 200
    ids = [r["id"] for r in dataset["rooms"]]
    assert len(ids) == len(set(ids))
    for r in dataset["rooms"]:
        assert r["code"] and r["room"]
        assert r["id"] == f"room:{r['code']}-{r['room']}"
        assert r["building_id"] == f"bldg:{r['code']}"
        assert r["floor_confidence"] in ("exact", "rule", "unknown")
        assert (r["floor"] is None) == (r["floor_confidence"] == "unknown")
        assert r["sources"] and set(r["sources"]) <= {"roomfinder", "soc"}
        assert isinstance(r["section_count"], int) and r["section_count"] >= 0
        assert ("soc" in r["sources"]) == (r["section_count"] > 0 or "soc_terms" in r)


def test_dataset_upc_rooms_resolve_to_a_building(dataset, codes):
    for r in dataset["rooms"]:
        if r["campus"] == "UPC":
            assert r["code"] in codes, r["id"]
            assert r["lat"] is not None and r["lng"] is not None
            assert r["pathing"] == (r["floor"] is not None)
        else:
            assert r["pathing"] is False


def test_dataset_room_floor_matches_parser(dataset):
    for r in dataset["rooms"]:
        b = dataset["buildings"][r["code"]]
        p = parse_room_number(r["room"], r["code"], floor_count=b["floor_count"])
        assert (r["floor"], r["floor_label"], r["floor_confidence"]) == (p["floor"], p["floor_label"], p["confidence"])


def test_dataset_known_rooms(dataset):
    by_id = {r["id"]: r for r in dataset["rooms"]}
    rth = by_id.get("room:RTH-108")
    if rth is not None:                       # RTH isn't centrally scheduled, so Room Finder may not list it
        assert rth["floor"] == 1
    assert by_id["room:THH-101"]["floor"] == 1
    assert by_id["room:VPD-LL101"]["floor_label"] == "LL"
    assert by_id["room:LVL-17"]["floor"] == -1
    assert by_id["room:CHP-219"]["campus"] == "HSC" and by_id["room:CHP-219"]["pathing"] is False
    assert by_id["room:DOH-172"]["campus"] == "HSC"     # Doheny Eye building, not Doheny Library (DML)
