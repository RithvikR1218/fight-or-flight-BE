import pytest

from app.services.place_taxonomy import CATEGORIES, CATEGORY_IDS, QUERY_SYNONYMS, categorize, category_intent, slug
from app.services.text_norm import normalize


@pytest.mark.parametrize("labels, expected", [
    (["Chicken restaurant"], ("food", "chicken_restaurant")),
    (["Coffee shop", "Cafe"], ("cafe", "coffee_shop")),
    (["Bubble tea store"], ("cafe", "bubble_tea_store")),
    (["Bar & grill"], ("food", "bar_grill")),
    (["Cocktail bar"], ("bar", "cocktail_bar")),
    (["Barber shop"], ("services", "barber_shop")),
    (["Convenience store"], ("grocery", "convenience_store")),
    (["Liquor store"], ("grocery", "liquor_store")),
    (["Pharmacy"], ("health", "pharmacy")),
    (["Drug store"], ("health", "drug_store")),
    (["Parking garage"], ("parking", "parking_garage")),
    (["Park & ride"], ("transit", "park_ride")),
    (["Park"], ("outdoors", "park")),
    (["Bus stop"], ("transit", "bus_stop")),
    (["University library"], ("study", "university_library")),
    (["Coworking space"], ("study", "coworking_space")),
    (["ATM"], ("services", "atm")),
    (["Gas station"], ("services", "gas_station")),
    (["Gym"], ("fitness", "gym")),
    (["Art gallery"], ("entertainment", "art_gallery")),
    (["Hotel"], ("lodging", "hotel")),
    (["Police department"], ("safety", "police_department")),
    (["Catholic church"], ("community", "catholic_church")),
    (["Elementary school"], ("education", "elementary_school")),
    (["Women's clothing store"], ("shopping", "womens_clothing_store")),
    (["Public bathroom"], ("services", "public_bathroom")),
])
def test_categorize_real_labels(labels, expected):
    assert categorize(labels) == expected


def test_categorize_falls_through_unmapped_first_label():
    # category from the first label that maps; subcategory stays the primary label
    assert categorize(["Warehouse", "Coffee shop"]) == ("cafe", "warehouse")
    assert categorize([]) == ("other", None)
    assert categorize(None) == ("other", None)
    assert categorize(["Some unknown thing"]) == ("other", "some_unknown_thing")


def test_categories_shape():
    assert all(set(c) == {"id", "label"} for c in CATEGORIES)
    assert len(set(CATEGORY_IDS)) == len(CATEGORY_IDS)
    assert CATEGORY_IDS[-1] == "other"
    for intents in QUERY_SYNONYMS.values():
        for it in intents:
            assert it["category"] in CATEGORY_IDS
    # synonym keys must already be in normalized form or they can never match
    assert all(normalize(k) == k for k in QUERY_SYNONYMS)


def test_slug():
    assert slug("Chicken restaurant") == "chicken_restaurant"
    assert slug("Açaí shop") == "acai_shop"
    assert slug("") is None


@pytest.mark.parametrize("q, expected", [
    ("coffee", [{"category": "cafe", "subcategory": None}]),
    ("food", [{"category": "food", "subcategory": None}]),
    ("lunch", [{"category": "food", "subcategory": None}]),
    ("pizza", [{"category": "food", "subcategory": "pizza_restaurant"}]),
    ("boba", [{"category": "cafe", "subcategory": "bubble_tea_store"}]),
    ("gym", [{"category": "fitness", "subcategory": None}]),
    ("atm", [{"category": "services", "subcategory": "atm"}]),
    ("library", [{"category": "study", "subcategory": "library"}]),
    ("parking", [{"category": "parking", "subcategory": None}]),
    ("drugstore", [{"category": "health", "subcategory": "drug_store"}]),
    ("hospital", [{"category": "health", "subcategory": "hospital"}]),   # not all of health (dentists, ...)
    ("bathroom", [{"category": "services", "subcategory": "public_bathroom"}]),
])
def test_category_intent_exact(q, expected):
    got = category_intent(normalize(q))
    assert got[:len(expected)] == expected     # the main intent first; synonyms may list related types after it


def test_category_intent_in_longer_query_and_phrases():
    assert {"category": "cafe", "subcategory": None} in category_intent("coffee near vkc")
    assert category_intent("ice cream")[0] == {"category": "food", "subcategory": "ice_cream_shop"}
    assert category_intent("starbucks") == []          # brands are matched by name, not as a type
    assert {"category": "transit", "subcategory": "bus_stop"} in category_intent("bus stop")


def test_category_intent_prefixes():
    assert category_intent("coff") == [{"category": "cafe", "subcategory": None}]
    assert {"category": "study", "subcategory": "library"} in category_intent("libr")
    assert {"category": "health", "subcategory": "pharmacy"} in category_intent("pharm")
    assert {"category": "food", "subcategory": "ice_cream_shop"} in category_intent("ice cr")
    # 1-2 characters never fire
    assert category_intent("c") == []
    assert category_intent("co") == []
    assert category_intent("pi") == []
    # an exact synonym does not also fire everything it prefixes ("bar" is not "barber")
    assert category_intent("bar") == [{"category": "bar", "subcategory": None}]


def test_category_intent_none():
    assert category_intent("") == []
    assert category_intent("tommy trojan") == []
    assert category_intent("the") == []
