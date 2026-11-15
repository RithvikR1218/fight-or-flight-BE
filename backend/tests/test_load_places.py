import json

from scripts.load_places import (
    COLS, annotate_buildings, building_row, google_place_row, name_aliases, poi_row, popularity, room_row,
    search_text,
)

NNBSP, EN = " ", "–"


def feat(lng, lat, **props):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lng, lat]}, "properties": props}


def test_name_aliases():
    assert name_aliases("Starbucks - Figueroa & Jefferson") == ["Starbucks - Figueroa and Jefferson", "Starbucks"]
    assert "Coffee Bean" in name_aliases("The Coffee Bean & Tea Leaf")
    assert "Coffee Bean and Tea Leaf" in name_aliases("The Coffee Bean & Tea Leaf")
    assert "Barnes & Noble" in name_aliases("Barnes and Noble")
    assert name_aliases("Chick-fil-A") == ["ChickfilA"]
    assert "7-11" in name_aliases("7-Eleven")
    assert name_aliases("ATM (Bank of America)") == ["ATM"]
    assert name_aliases("Trojan Grounds") == []
    assert name_aliases("Trojan's Pizza") == []  # apostrophe-only differences normalize away


def test_search_text():
    assert search_text("starbucks figueroa", ["Starbucks", "STARBUCKS FIGUEROA"]) == "starbucks figueroa starbucks"


def test_popularity_shrinks_small_samples():
    assert popularity(5.0, 3) < popularity(4.5, 800)
    assert popularity(4.5, 800) < popularity(4.5, 8000)
    assert 0 <= popularity(None, 0) <= 1
    assert popularity(5.0, 10 ** 7) <= 1.0
    assert popularity(1.0, 1000) < popularity(4.0, 1000)


def test_google_place_row():
    f = feat(-118.2851, 34.0224, id="usc-gm-02365", name="Dulce - USC Village", detail="full",
             categories=["Coffee shop", "Cafe"], address="3096 McClintock Ave, Los Angeles, CA 90007",
             phone="(213) 555-0100", rating=4.6, review_count=512, place_id="ChIJabc",
             open_hours={"Monday": [f"7{NNBSP}AM{EN}9{NNBSP}PM"], "Tuesday": ["Closed"]},
             reviews=["Great"], plus_code="2PJ7+XX")
    r = google_place_row(f, "/data/usc_google_maps_places.geojson")
    assert r["id"] == "gmaps:ChIJabc"
    assert (r["kind"], r["category"], r["subcategory"]) == ("place", "cafe", "coffee_shop")
    assert r["name_norm"] == "dulce usc village"
    assert r["aliases"] == ["Dulce"]
    assert r["search_text"] == "dulce usc village dulce"
    assert r["categories"] == ["Coffee shop", "Cafe"]
    assert (r["lat"], r["lng"]) == (34.0224, -118.2851)
    assert r["hours"] == {"0": [[420, 1260]], "1": []}
    assert r["detail"]["hours_text"]["Tuesday"] == ["Closed"]
    assert r["detail"]["tier"] == "full" and r["detail"]["phone"] == "(213) 555-0100"
    assert "website" not in r["detail"]  # empty values are dropped
    assert r["source_file"] == "usc_google_maps_places.geojson"
    assert set(COLS) <= set(r)
    json.dumps(r)  # everything is JSON-serializable


def test_google_place_row_basic_tier_and_unknown_hours():
    r = google_place_row(feat(-118.3, 34.0, id="usc-gm-00001", name="#1 Buffet", detail="basic",
                              categories=["Buffet restaurant"], rating=0, review_count=0))
    assert r["id"] == "gmaps:usc-gm-00001"
    assert r["rating"] is None and r["review_count"] == 0 and r["hours"] is None
    assert r["category"] == "food"


def test_building_row():
    f = feat(-118.2837, 34.0201, code="DML", name="Edward L. Doheny Jr. Memorial Library",
             aliases=["Doheny Library", "Doheny", "DML"], category="study", subcategory="library",
             entrances=[[-118.2838, 34.0199]], source="usc-fpm-gis+osm")
    r = building_row(f)
    assert r["id"] == "bldg:DML" and r["kind"] == "building"
    assert r["aliases"] == ["DML", "Doheny Library", "Doheny"]
    assert r["search_text"] == "edward l doheny jr memorial library dml doheny library doheny"
    assert (r["category"], r["subcategory"]) == ("study", "library")
    assert r["popularity"] == 0.8
    assert r["detail"]["code"] == "DML" and r["detail"]["entrances"] == [[-118.2838, 34.0199]]


def test_room_row_and_building_floor_facts():
    r = room_row({"id": "room:THH-201", "code": "THH", "room": "201", "building_id": "bldg:THH",
                  "building_name": "Mark Taper Hall of Humanities", "campus": "UPC", "floor": 2, "floor_label": "2",
                  "floor_confidence": "rule", "pathing": True, "lat": 34.0222, "lng": -118.2845, "capacity": 40,
                  "classroom_type": "General Use Classroom", "equipment": ["Projector"], "url": "https://x"})
    assert (r["id"], r["kind"], r["name"], r["category"], r["subcategory"]) == (
        "room:THH-201", "room", "THH 201", "campus", "classroom")
    assert "THH201" in r["aliases"] and "Mark Taper Hall of Humanities 201" in r["aliases"]
    assert r["detail"]["floor"] == 2 and r["detail"]["pathing"] is True and r["detail"]["building_id"] == "bldg:THH"
    assert set(COLS) <= set(r)
    b = building_row(feat(-118.2845, 34.0222, code="THH", name="Mark Taper Hall of Humanities"))
    annotate_buildings([b], {"buildings": {"THH": {"floor_count": 4, "has_basement": True, "campus": "UPC"}}})
    assert b["detail"]["floor_count"] == 4 and b["detail"]["campus"] == "UPC"


def test_poi_row():
    parsed = {"id": "usc-tbl-001", "type": "seating", "name": None, "source": "manual",
              "source_file": "usc_outdoor_table_groups.geojson", "props": json.dumps({"table_count_est": 4}),
              "geom": json.dumps({"type": "Point", "coordinates": [-118.2827514, 34.018763]}), "h3_cell": "x"}
    r = poi_row(parsed)
    assert r["id"] == "poi:usc-tbl-001" and r["kind"] == "poi"
    assert (r["category"], r["subcategory"], r["name"]) == ("outdoors", "outdoor_seating", "Outdoor seating")
    assert r["detail"] == {"poi_type": "seating", "table_count_est": 4}
    assert (r["lat"], r["lng"]) == (34.018763, -118.2827514)
    study = poi_row({**parsed, "id": "s1", "type": "study", "name": "Quiet room"})
    assert (study["category"], study["name"]) == ("study", "Quiet room")
