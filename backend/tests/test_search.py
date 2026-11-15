"""Search ranking behaviour on small dict fixtures (services/search_index.py, no DB) and the search
endpoints with the index injected. The judged eval on the real data is in tests/search_eval/."""
import threading
import time
from datetime import datetime

import numpy as np
import pytest

from app.services import search as svc
from app.services.search_index import SearchIndex, osa_distance, typo_budget, typo_cost
from app.services.text_norm import normalize

USC = (34.0224, -118.2851)
NOW = datetime(2026, 11, 14, 12, 0, tzinfo=svc.LA)
DEG_M = 111195.0          # metres per degree of latitude


def place(name, m_north=0.0, kind="place", category="food", labels=(), aliases=(), address=None,
          popularity=0.3, hours=None, pid=None, detail=None, m_east=0.0):
    return {"id": pid or f"p:{normalize(name).replace(' ', '-')}:{m_north}:{m_east}", "kind": kind, "name": name,
            "aliases": list(aliases), "category": category,
            "subcategory": normalize(labels[0]).replace(" ", "_") if labels else None, "categories": list(labels),
            "address": address, "lat": USC[0] + m_north / DEG_M, "lng": USC[1] + m_east / (DEG_M * 0.829),
            "rating": 4.2, "review_count": 100, "popularity": popularity, "hours": hours, "detail": detail or {}}


def building(code, name, m_north=0.0, aliases=(), **kw):
    r = place(name, m_north, kind="building", category=kw.pop("category", "campus"), pid=f"bldg:{code}",
              aliases=[code, *aliases], popularity=0.8, detail={"code": code, **kw.pop("detail", {})}, **kw)
    r["subcategory"] = "university_building"
    return r


FILLER = [place(f"Filler Spot {i}", 5000 + 37 * i, category="other", labels=["Warehouse"]) for i in range(30)]
BASE = FILLER + [
    building("DMC", "John Medicine Crow Center", 170, aliases=["VKC", "Von KleinSmid Center"]),
    building("DML", "Edward L. Doheny Jr. Memorial Library", 280, aliases=["Doheny Library", "Doheny"],
             category="study"),
    building("RTH", "Ronald Tutor Hall of Engineering", 500, detail={"floor_count": 5}),
    building("THH", "Mark Taper Hall of Humanities", 60, detail={"floor_count": 4}),
    place("Cafe Dulce", 340, category="cafe", labels=["Cafe"], address="3096 McClintock Ave, Los Angeles, CA 90007"),
    place("Dulcinea Salon", 9000, category="services", labels=["Beauty salon"]),
    place("Starbucks", 250, category="cafe", labels=["Coffee shop"], popularity=0.5,
          address="3335 S Figueroa St, Los Angeles, CA 90007"),
    place("Starbucks", 2500, category="cafe", labels=["Coffee shop"], popularity=0.9,
          address="800 W Olympic Blvd, Los Angeles, CA 90015"),
    place("Blue Bottle", 120, category="cafe", labels=["Coffee shop"], address="1000 S Vermont Ave, Los Angeles, CA", m_east=-1000),
    place("Vermont Coffee", 900, category="cafe", labels=["Coffee shop"], address="1200 S Vermont Ave, Los Angeles, CA", m_east=-1000),
    place("Corner Cafe", 1500, category="cafe", labels=["Cafe"], address="1300 S Vermont Ave, Los Angeles, CA", m_east=-1000),
    place("Pizza Palace", 100, labels=["Pizza restaurant"], address="3300 S Figueroa St, Los Angeles, CA"),
    place("Taco Stand", 200, labels=["Taco restaurant"], address="3400 S Figueroa St, Los Angeles, CA"),
    place("Hospital of Emotions", 300, category="entertainment", labels=["Art museum"]),
    place("California Hospital Medical Center", 3000, category="health", labels=["Hospital"]),
    place("Keck Hospital", 4000, category="health", labels=["Hospital", "Medical Center"]),
    place("Hospital", 3100, category="health", labels=["Medical clinic"]),
    place("Ice Cream Sound Recording Studios", 700, category="entertainment", labels=["Recording studio"]),
    place("Cold Stone", 600, labels=["Ice cream shop", "Dessert shop"]),
    place("Frosty", 2600, labels=["Ice cream shop"]),
    place("Scoops", 3600, labels=["Ice cream shop"]),
    place("Mc Donald's", 14000, labels=["Fast food restaurant"]),
    place("McDonald's", 857, labels=["Fast food restaurant"]),
    place("Print Hub", 800, category="services", labels=["Print shop"]),
    place("Quiet Corner", 400, category="cafe", labels=["Cafe"],
          detail={"reviews": [{"text": "Great place to fix a bike, they do bike repair while you wait"}]}),
]


@pytest.fixture(scope="module")
def idx():
    return SearchIndex(BASE)


def run(idx, q, near=USC, **kw):
    kw.setdefault("now", NOW)
    return idx.search(q, near=near, **kw)


def names(res):
    return [r["name"] for r in res]


# ---------------------------------------------------------------- primitives

def test_osa_and_typo_budget():
    assert osa_distance("doheney", "doheny") == 1
    assert osa_distance("cafe", "caef") == 1                    # transposition counts once
    assert osa_distance("abc", "xyz", max_d=1) == 2
    assert [typo_budget(n) for n in (3, 4, 5, 8, 9, 12)] == [0, 0, 1, 1, 2, 2]
    assert typo_cost("starbuks", "starbucks", 1) == 1 and typo_cost("xtarbucks", "starbucks", 1) == 2


# ---------------------------------------------------------------- names, codes, typos

def test_building_code_is_pinned_first(idx):
    out = run(idx, "vkc")
    assert out[0]["id"] == "bldg:DMC" and out[0]["match"] == "name"


def test_prefix_while_typing(idx):
    assert run(idx, "dulc")[0]["name"] == "Cafe Dulce"
    assert run(idx, "starb")[0]["name"] == "Starbucks"


def test_typo_on_rare_name(idx):
    assert run(idx, "doheney")[0]["id"] == "bldg:DML"
    assert run(idx, "starbuks")[0]["name"] == "Starbucks"


def test_spacing_does_not_matter_and_distance_decides(idx):
    """"mc donalds" == "McDonald's": the near one wins over an exact-named far one."""
    out = run(idx, "mc donalds")
    assert out[0]["name"] == "McDonald's" and out[0]["distance_m"] < 1000
    assert "Mc Donald's" in names(out)


def test_distinctive_name_still_found_far_away(idx):
    out = run(idx, "dulcinea")
    assert out[0]["name"] == "Dulcinea Salon" and out[0]["distance_m"] > 8000


def test_same_relevance_nearer_first(idx):
    out = [r for r in run(idx, "starbucks") if r["name"] == "Starbucks"]
    assert [r["distance_m"] < 1000 for r in out] == [True, False]


# ---------------------------------------------------------------- types beat literal names

def test_type_beats_literal_name_for_generic_word(idx):
    out = run(idx, "hospital")
    assert names(out)[:2] == ["California Hospital Medical Center", "Keck Hospital"]
    assert "Hospital of Emotions" not in names(out)               # a museum named "Hospital": gated out


def test_phrase_type_beats_name(idx):
    out = run(idx, "ice cream")
    assert out[0]["name"] == "Cold Stone" and "Ice Cream Sound Recording Studios" not in names(out)


def test_type_word_typo(idx):
    assert run(idx, "coffe")[0]["category"] == "cafe"
    assert run(idx, "pizzza")[0]["name"] == "Pizza Palace"
    assert run(idx, "piza") == []                 # 4 letters: no typo budget (Meilisearch's rule)


def test_ing_form_and_synonyms(idx):
    assert run(idx, "printing")[0]["name"] == "Print Hub"
    assert run(idx, "bike repair")[0]["name"] == "Quiet Corner"   # found through its reviews
    assert run(idx, "bike repair")[0]["match"] == "content"


def test_location_in_query_moves_the_anchor(idx):
    out = run(idx, "coffee near vermont")
    assert all("Vermont" in (r["address"] or "") for r in out[:3])
    assert run(idx, "starbucks figueroa")[0]["address"].startswith("3335 S Figueroa")


def test_gibberish_returns_nothing(idx):
    assert run(idx, "xqzvkj") == [] and run(idx, "zzxxq vvbnm") == []


def test_unknown_extra_word_keeps_results(idx):
    assert run(idx, "pizza xqzvkj")[0]["name"] == "Pizza Palace"


# ---------------------------------------------------------------- short queries, chips, filters

def test_one_char_prefix(idx):
    out = run(idx, "d")
    assert out and all(any(w.startswith("d") for w in normalize(r["name"]).split()) or r["kind"] == "building"
                       for r in out)


def test_empty_query_category_is_nearest(idx):
    out = run(idx, "", categories=["cafe"])
    assert {r["category"] for r in out} == {"cafe"}
    assert [r["distance_m"] for r in out] == sorted(r["distance_m"] for r in out)


def test_no_near_omits_distance_and_uses_popularity(idx):
    out = [r for r in idx.search("starbucks", near=None, now=NOW) if r["name"] == "Starbucks"]
    assert "distance_m" not in out[0] and out[0]["address"].startswith("800 W Olympic")


def test_bbox_and_open_now(idx):
    lat = USC[0] + 300 / DEG_M
    bbox = (USC[1] - 0.01, USC[0], USC[1] + 0.01, lat)
    assert all(r["lat"] <= lat for r in run(idx, "cafe", bbox=bbox))
    open_fn = lambda hours, now: bool(hours.get("5"))
    rows = BASE + [place("Starbucks", 300, category="cafe", labels=["Coffee shop"], hours={"5": [[0, 1440]]},
                         pid="open"),
                   place("Starbucks", 310, category="cafe", labels=["Coffee shop"], hours={"5": []}, pid="closed")]
    out = SearchIndex(rows).search("starbucks", near=USC, now=NOW, open_now=True, is_open_fn=open_fn)
    assert [r["id"] for r in out] == ["open"] and out[0]["open_now"] is True


# ---------------------------------------------------------------- semantic fusion

def _vec(*hot, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    v[list(hot)] = 1.0
    return v / np.linalg.norm(v)


def test_semantic_adds_meaning_but_not_for_names():
    rows = BASE + [place("The Lantern Room", 500, category="other", labels=["Speakeasy"], pid="lantern"),
                   place("Copy Corner", 300, category="services", labels=["Copy shop"], pid="copy",
                         detail={"reviews": [{"text": "Not romantic, but fast copies"}]})]
    vectors = {r["id"]: _vec(1) for r in rows}
    vectors["lantern"] = _vec(0)                     # the only place "about" the query meaning
    idx = SearchIndex(rows, vectors=vectors)
    assert idx.parse("romantic").semantic_text == "romantic"     # a meaning query: embed it
    lexical = idx.search("romantic", near=USC, now=NOW)
    assert [r["id"] for r in lexical] == ["copy"]                # words alone: a review mention
    out = idx.search("romantic", near=USC, now=NOW, qvec=_vec(0))
    assert out[0]["id"] == "lantern" and out[0]["match"] == "semantic"
    assert idx.parse("starbucks").semantic_text is None    # names / codes / prefixes stay lexical
    assert idx.parse("vkc").semantic_text is None and idx.parse("coff").semantic_text is None


def test_query_encoder_degrades_gracefully():
    from app.services.search_semantic import QueryEncoder

    assert QueryEncoder(mode="off").encode("coffee") is None

    class Broken:
        model_name = "x"

        def embed_query(self, text):
            raise RuntimeError("model download blocked")

    enc = QueryEncoder(embedder=Broken(), mode="on")
    assert enc.encode("coffee") is None and "blocked" in enc.disabled_reason


# ---------------------------------------------------------------- rooms

ROOMS = [
    {**place("THH 201", 60, kind="room", category="campus", pid="room:THH-201",
             aliases=["THH201", "Mark Taper Hall of Humanities 201"]),
     "subcategory": "classroom",
     "detail": {"code": "THH", "room": "201", "floor": 2, "floor_label": "2", "floor_confidence": "rule",
                "building_id": "bldg:THH", "building_name": "Mark Taper Hall of Humanities", "campus": "UPC",
                "pathing": True}},
    {**place("THH 105", 60, kind="room", category="campus", pid="room:THH-105"), "subcategory": "classroom",
     "detail": {"code": "THH", "room": "105", "floor": 1, "floor_label": "1", "building_id": "bldg:THH",
                "building_name": "Mark Taper Hall of Humanities", "campus": "UPC", "pathing": True}},
]


@pytest.fixture(scope="module")
def ridx():
    return SearchIndex(BASE + ROOMS)


def test_room_in_dataset(ridx):
    out = ridx.search("thh 201", near=USC, now=NOW)
    assert [r["id"] for r in out[:2]] == ["room:THH-201", "bldg:THH"]
    assert out[0]["floor"] == 2 and out[0]["verified"] is True and out[0]["pathing"] is True


def test_room_synthesized_at_its_building(ridx):
    out = ridx.search("rth 108", near=USC, now=NOW)
    assert out[0]["id"] == "room:RTH-108" and out[0]["verified"] is False
    assert (out[0]["floor"], out[0]["floor_label"], out[0]["pathing"]) == (1, "1", True)
    assert out[1]["id"] == "bldg:RTH" and out[0]["lat"] == out[1]["lat"]
    assert ridx.search("RTH108")[0]["id"] == "room:RTH-108"


def test_room_prefix_listing_and_name_form(ridx):
    assert [r["id"] for r in ridx.search("thh 1")] == ["bldg:THH", "room:THH-105"]
    assert [r["id"] for r in ridx.search("thh")] == ["bldg:THH", "room:THH-105", "room:THH-201"]
    assert ridx.search("taper hall 201")[0]["id"] == "room:THH-201"


def test_rooms_stay_out_of_ordinary_search(ridx):
    assert all(r["kind"] != "room" for r in ridx.search("taper hall", near=USC))
    assert all(r["kind"] != "room" for r in ridx.search("studio 54", near=USC))


# ---------------------------------------------------------------- endpoints (index injected, DB mocked)

@pytest.fixture
def api_index(monkeypatch, ridx):
    monkeypatch.setattr(svc, "get_index", lambda: ridx)
    monkeypatch.setattr(svc, "encoder", lambda: None)
    monkeypatch.setattr(svc, "_is_open", lambda hours, now: None)
    return ridx


def test_search_endpoint_shape(client, api_index):
    resp = client.get("/api/v1/search?q=vkc&near=34.0224,-118.2851&limit=5")
    assert resp.status_code == 200 and "max-age" in resp.headers["Cache-Control"]
    res = resp.get_json()["results"]
    assert res[0]["name"] == "John Medicine Crow Center"
    assert set(res[0]) == {"id", "kind", "name", "category", "subcategory", "address", "lat", "lng",
                           "distance_m", "rating", "review_count", "match"}      # open_now omitted when unknown
    assert isinstance(res[0]["distance_m"], int)


def test_search_room_result_fields(client, api_index):
    res = client.get("/api/v1/search?q=rth%20108").get_json()["results"]
    assert res[0]["kind"] == "room" and res[0]["floor"] == 1 and res[0]["verified"] is False
    assert {"floor", "floor_label", "building_id", "building_name", "code", "room", "pathing"} <= set(res[0])
    assert "floor" not in res[1]                                                   # the building: unchanged shape


def test_search_without_near_has_no_distance(client, api_index):
    res = client.get("/api/v1/search?q=starbucks").get_json()["results"]
    assert res and all("distance_m" not in r for r in res)


def test_search_category_only(client, api_index):
    resp = client.get("/api/v1/search?category=Food,cafe&near=34.0224,-118.2851&limit=3")
    res = resp.get_json()["results"]
    assert resp.status_code == 200 and len(res) == 3 and {r["category"] for r in res} <= {"food", "cafe"}


@pytest.mark.parametrize("qs, status", [
    ("", 400), ("q=%20%20", 400), ("q=!!!", 400),
    ("q=a&limit=0", 422), ("q=a&limit=51", 422),
    ("q=a&near=34.02", 422), ("q=a&near=91,0", 422), ("q=a&near=abc,def", 422),
    ("q=a&bbox=1,2,3", 422), ("q=a&bbox=3,2,1,4", 422),
])
def test_search_param_validation(client, api_index, qs, status):
    resp = client.get("/api/v1/search?" + qs)
    assert resp.status_code == status
    body = resp.get_json()
    assert body["code"] == status and ("message" in body if status == 400 else "query" in body["errors"])


def test_categories_endpoint(client, monkeypatch):
    rows = [{"category": "cafe", "subcategory": "coffee_shop", "count": 5, "label": "Coffee shop"},
            {"category": "food", "subcategory": "chicken_restaurant", "count": 3, "label": "Chicken restaurant"},
            {"category": "food", "subcategory": "thai_restaurant", "count": 7, "label": "Thai cuisine"}]
    monkeypatch.setattr(svc, "fetch_all", lambda sql, **p: rows)
    body = client.get("/api/v1/search/categories").get_json()
    assert [c["id"] for c in body] == ["food", "cafe"]                           # taxonomy order, no zeros
    assert body[0]["count"] == 10 and body[0]["label"] == "Food"
    assert body[0]["types"] == [{"id": "thai_restaurant", "label": "Thai restaurant", "count": 7},
                                {"id": "chicken_restaurant", "label": "Chicken restaurant", "count": 3}]


def test_place_detail(client, monkeypatch):
    row = {**place("Cafe Dulce", 340, category="cafe"), "id": "gmaps:ChIJabc/def", "categories": ["Cafe", "Bakery"],
           "source": "google_maps", "hours": {"0": [[420, 1260]]},
           "detail": {"phone": "(213) 555-0100", "hours_text": {"Monday": ["7 AM-9 PM"]},
                      "reviews": [{"author": "a", "rating": 5, "text": "good"}]}}
    monkeypatch.setattr(svc, "fetch_all", lambda sql, **p: [dict(row)] if p["id"] == row["id"] else [])
    monkeypatch.setattr(svc, "_is_open", lambda hours, now: True)
    body = client.get("/api/v1/places/gmaps:ChIJabc/def?near=34.0224,-118.2851").get_json()
    assert body["hours"]["monday"] == [{"open": "07:00", "close": "21:00"}] and body["hours"]["sunday"] == []
    assert body["phone"] == "(213) 555-0100" and body["open_now"] is True and body["distance_m"] > 0
    assert body["categories"] == ["Cafe", "Bakery"] and "detail" not in body and "floor" not in body
    assert client.get("/api/v1/places/gmaps:nope").status_code == 404


def test_room_detail_stored_and_synthesized(client, monkeypatch, api_index):
    stored = {**ROOMS[0], "source": "usc-roomfinder"}
    monkeypatch.setattr(svc, "fetch_all", lambda sql, **p: [dict(stored)] if p["id"] == stored["id"] else [])
    body = client.get("/api/v1/places/room:THH-201").get_json()
    assert body["kind"] == "room" and body["floor"] == 2 and body["verified"] is True
    body = client.get("/api/v1/places/room:RTH-108").get_json()
    assert body["id"] == "room:RTH-108" and body["floor"] == 1 and body["verified"] is False
    assert body["building_id"] == "bldg:RTH"
    assert client.get("/api/v1/places/room:ZZZ-108").status_code == 404           # unknown building
    assert client.get("/api/v1/places/room:RTH-hello").status_code == 404


def test_reverse_geocode(client, monkeypatch):
    answers = []
    monkeypatch.setattr(svc, "fetch_all", lambda sql, **p: answers.pop(0))
    answers[:] = [[{"id": "bldg:VKC", "kind": "building", "name": "Von KleinSmid Center", "address": None,
                    "distance_m": 12.4}]]
    body = client.get("/api/v1/geocode/reverse?at=34.0219,-118.2840").get_json()
    assert body == {"name": "Von KleinSmid Center", "address": None, "id": "bldg:VKC", "kind": "building",
                    "distance_m": 12}
    answers[:] = [[], [{"name": "['South Figueroa Street', 'Figueroa St']", "distance_m": 30.2}]]
    body = client.get("/api/v1/geocode/reverse?at=34.0219,-118.2840").get_json()
    assert body["name"] == "South Figueroa Street" and body["kind"] == "road"
    answers[:] = [[], []]
    assert client.get("/api/v1/geocode/reverse?at=34.0219,-118.2840").status_code == 404
    assert client.get("/api/v1/geocode/reverse").status_code == 422


def test_openapi_lists_search_routes(client):
    paths = set(client.get("/api/openapi.json").get_json()["paths"])
    assert {"/api/v1/search", "/api/v1/search/categories", "/api/v1/places/{place_id}",
            "/api/v1/geocode/reverse"} <= paths


# ---------------------------------------------------------------- per-worker index lifecycle

def test_index_built_once_and_refreshed_when_version_changes(app, monkeypatch):
    builds, version = [], {"v": "1"}

    def build():
        builds.append(version["v"])
        return SearchIndex(BASE), version["v"], {}

    monkeypatch.setattr(svc, "build_index_from_db", build)
    monkeypatch.setattr(svc, "current_version", lambda meta=None: version["v"])
    monkeypatch.setitem(svc._state, "index", None)
    app.config["SEARCH_INDEX_TTL"] = 0.0
    first = svc.get_index()
    assert svc.get_index() is first and builds == ["1"]          # same version: no rebuild
    version["v"] = "2"
    assert svc.get_index() is first                               # stale check: keep serving, rebuild async
    for _ in range(100):
        if svc._state["version"] == "2":
            break
        time.sleep(0.05)
    assert builds == ["1", "2"] and svc.get_index() is not first
    svc._state.update(index=None, version=None)


def test_index_concurrent_first_build(app, monkeypatch):
    builds = []

    def build():
        builds.append(1)
        time.sleep(0.05)
        return SearchIndex(BASE[:40]), "1", {}

    monkeypatch.setattr(svc, "build_index_from_db", build)
    monkeypatch.setitem(svc._state, "index", None)
    got = []
    ts = [threading.Thread(target=lambda: got.append(svc.get_index())) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(builds) == 1 and len({id(x) for x in got}) == 1
    svc._state.update(index=None, version=None)
