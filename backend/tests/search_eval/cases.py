"""Judged queries near USC (34.0224, -118.2851): what a sensible person expects to see.

Written from the user's point of view, not from what the ranker returns. Type patterns match the
place's category | subcategory | Google labels | POI type (see harness._text).
"""
from .harness import (Case, first_fields, first_id, first_name, first_within_km, in_top_id, in_top_name, max_results,
                      median_km, min_results, no_rooms, not_in_top_name, rest_rooms_of, second_id, top_address,
                      top_has_restroom, top_name, top_near_point, top_open_late, top_type, top_type_or_name)

CAFE = r"^cafe\b|coffee|espresso|tea house|bubble tea"
FOOD = r"^food\b|restaurant|cafe|bakery|deli|sandwich|food court|fast food"
DESSERT = r"dessert|ice cream|frozen yogurt|bakery|donut|cake|cookie|pastry|patisserie|cupcake|creperie|chocolate"
STUDY = r"^study\b|library|study|coworking"
HOSPITAL = r"hospital|medical center|emergency room|^health \| (hospital|emergency_room|medical_center)"
DOWNTOWN = (34.0430, -118.2510)

CASES = [
    # ---- typing a name character by character
    Case("s", min_results(5), median_km(5, 1.0)),
    Case("st", min_results(5), median_km(5, 1.0)),
    Case("sta", min_results(5), median_km(5, 1.5)),
    Case("star", in_top_name(3, "starbucks")),
    Case("starb", first_name("starbucks"), top_name(3, "starbucks")),
    Case("starbu", top_name(3, "starbucks")),
    Case("starbucks", top_name(5, "starbucks"), first_within_km(0.6)),
    Case("cof", top_type(3, CAFE)),
    Case("coff", top_type(3, CAFE)),
    Case("coffe", top_type(5, CAFE)),
    Case("coffee", top_type(5, CAFE), median_km(5, 0.8)),
    Case("coffee shop", top_type(5, CAFE), median_km(5, 0.8)),
    Case("d", min_results(5), median_km(5, 1.0)),
    Case("du", min_results(3)),
    Case("dulc", first_name("dulce")),
    Case("dulce", first_name("dulce"), first_within_km(1.0)),
    Case("tr", min_results(3)),
    Case("trad", in_top_name(3, "trader joe")),
    Case("trader", first_name("trader joe")),

    # ---- brands, with typos
    Case("starbcks", top_name(3, "starbucks")),
    Case("starbuks", top_name(3, "starbucks")),
    Case("chipolte", first_name("chipotle")),
    Case("chipotle", first_name("chipotle"), first_within_km(1.5)),
    Case("mcdonlds", first_name("mc ?donald")),
    Case("mc donalds", first_name("mc ?donald"), first_within_km(1.5),
         why="the exact-named 'Mc Donald's' 14 km away must not beat McDonald's 857 m away"),
    Case("mcdonalds", first_name("mc ?donald"), first_within_km(1.5)),
    Case("trader joes", first_name("trader joe"), first_within_km(1.0)),
    Case("chik fil a", first_name("chick-fil-a")),
    Case("chick fil a", first_name("chick-fil-a"), first_within_km(1.5)),
    Case("in n out", first_name("in-n-out")),
    Case("cvs", top_name(3, "cvs")),
    Case("target", first_name("^target"), first_within_km(1.0)),
    Case("ralphs", first_name("ralphs"), first_within_km(2.0)),
    Case("chipotle near me", first_name("chipotle")),

    # ---- campus buildings: codes, names, typos
    Case("vkc", first_id("bldg:DMC")),
    Case("dmc", first_id("bldg:DMC")),
    Case("von kleinsmid", first_id("bldg:DMC")),
    Case("doheny", first_id("bldg:DML")),
    Case("doheney", first_id("bldg:DML")),
    Case("dohny libary", first_id("bldg:DML")),
    Case("leavey", first_id("bldg:LVL")),
    Case("leavey library", first_id("bldg:LVL")),
    Case("tommy trojan", first_name("tommy trojan")),
    Case("tutor campus center", first_id("bldg:TCC")),
    # two buildings carry the name (Henry Salvatori CS Center, Grace Ford Salvatori Hall)
    Case("salvatori", in_top_id(2, "bldg:SAL"), in_top_id(2, "bldg:GFS")),
    Case("sal", first_id("bldg:SAL")),
    Case("taper hall", first_id("bldg:THH")),
    Case("lyon center", first_name("lyon")),
    Case("usc bookstore", first_name("bookstore")),

    # ---- place types (and typos of them)
    Case("restaurant", top_type(5, FOOD), median_km(5, 0.6)),
    Case("resturant", top_type(5, FOOD), median_km(5, 0.6)),
    Case("food", top_type(5, FOOD), median_km(5, 0.6)),
    Case("pharmacy", top_type(3, r"pharmacy|drug store")),
    Case("pharmcy", top_type(3, r"pharmacy|drug store")),
    # (Wilson Dental Library is a real USC library, so it may appear; dental offices may not)
    Case("library", top_type(5, STUDY), not_in_top_name(5, "dentist|dental (office|clinic|group|care)")),
    Case("libary", top_type(5, STUDY), not_in_top_name(5, "dentist|dental (office|clinic|group|care)")),
    Case("gas", top_type(3, r"gas station")),
    Case("gas station", top_type(5, r"gas station")),
    Case("groceries", top_type(3, r"grocery|supermarket|market")),
    Case("grocery store", top_type(3, r"grocery|supermarket")),
    Case("gym", top_type(3, r"^fitness|gym|fitness")),
    Case("atm", top_type(3, r"atm|bank")),
    Case("parking", top_type(5, r"^parking"), median_km(5, 0.8)),
    Case("hospital", top_type(3, HOSPITAL), not_in_top_name(10, "emotions"),
         why="a place that IS a hospital beats a clinic or art museum merely named 'Hospital'"),
    Case("urgent care", top_type(3, r"urgent care|walk-in clinic|emergency|medical clinic")),
    Case("bank", top_type(3, r"bank|credit union|atm|financial")),
    Case("bookstore", top_type(3, r"book store|bookstore|book_store", frac=0.67)),
    Case("bathroom", top_has_restroom(3), median_km(3, 3.0),
         why="no campus restroom data: places with a public bathroom or a restroom attribute"),
    Case("park", top_type(3, r"^outdoors|park"), not_in_top_name(3, "parking")),
    Case("museum", top_type(3, "museum")),
    Case("movie theater", top_type(3, "movie theater|cinema")),
    Case("post office", top_type(3, "post office")),
    Case("laundry", top_type(3, r"laundr|dry clean|cleaners")),
    Case("police", top_type(3, r"^safety|police")),
    Case("bus stop", top_type(3, r"bus stop|bus station")),
    Case("hotel", top_type(3, r"hotel|motel|lodging|inn|hostel")),
    Case("cafe", top_type(5, CAFE), not_in_top_name(5, r"\bcare\b")),
    Case("outdoor seating", top_type(3, "outdoor_seating|seating")),

    # ---- cuisines and dishes
    Case("tacos", top_type(5, r"taco|mexican|burrito"), median_km(5, 1.5)),
    Case("sushi", top_type(3, r"sushi|japanese")),
    Case("boba", top_type(3, r"bubble tea|tea")),
    Case("pizza", top_type(5, "pizza"), median_km(5, 1.0)),
    Case("burgers", top_type(5, r"hamburger|burger|fast food"), median_km(5, 1.0)),
    Case("ramen", top_type(3, r"ramen|japanese|noodle")),
    Case("vegan", top_type(3, r"vegan|vegetarian")),
    Case("breakfast", top_type(5, r"breakfast|brunch|diner|bagel|cafe|coffee"), median_km(5, 1.0),
         why="breakfast places nearby, not a far place NAMED 'Breakfast'"),
    Case("dessert", top_type(5, DESSERT), median_km(5, 1.0)),
    Case("ice cream", top_type(3, r"ice cream|frozen yogurt|dessert"), not_in_top_name(10, "recording|studio")),
    Case("thai food", top_type(3, "thai")),
    Case("chinese food", top_type(3, "chinese")),
    Case("sandwiches", top_type(3, r"sandwich|deli")),
    Case("korean bbq", top_type(3, r"korean")),
    Case("pho", not_in_top_name(5, r"photo|phone"),
         why="the study area has no Vietnamese places: few results beat photo studios"),
    Case("bar", top_type(3, r"\bbar\b|pub|lounge"), not_in_top_name(5, "barber")),

    # ---- what people want to do
    Case("somewhere to eat", min_results(5), top_type(5, FOOD), median_km(5, 0.7)),
    Case("cheap lunch", min_results(5), top_type(5, FOOD), median_km(5, 1.5)),
    Case("cheap eats", min_results(5), top_type(5, FOOD), median_km(5, 1.5)),
    Case("late night food", min_results(5), top_type(5, FOOD + r"|grocery|convenience"), median_km(5, 2.0),
         top_open_late(5, 0.6)),
    Case("place to study", top_type(3, STUDY)),
    Case("quiet study spot", top_type(3, STUDY)),
    Case("study room", top_type(3, STUDY)),
    Case("get a haircut", top_type(3, r"barber|hair|beauty salon")),
    Case("workout", top_type(3, r"^fitness|gym")),
    Case("drinks", top_type(5, r"\bbar\b|pub|lounge|cocktail|brewery|wine"), median_km(5, 1.5)),
    Case("date night", min_results(3), top_type(5, r"^food|^bar|^entertainment|restaurant")),
    Case("print something", top_type(3, r"print|copy|shipping and mailing|office supply|post office")),
    Case("printing", top_type(3, r"print|copy|shipping and mailing|office supply")),
    Case("bike repair", top_name(3, r"bike|bicycle|cycl", frac=0.67)),
    Case("fix my phone", top_type(3, r"phone|electronics|computer repair", frac=0.67)),
    Case("phone repair", top_type(3, r"phone|electronics|computer repair", frac=0.67)),

    # ---- name / type + a location in the query
    Case("starbucks figueroa", first_name("starbucks"), top_address(1, "figueroa")),
    Case("coffee near vermont", top_type(3, CAFE), top_address(3, "vermont", frac=0.67)),
    Case("pizza near figueroa", top_type(3, "pizza"), top_address(3, "figueroa", frac=0.67)),
    Case("tacos downtown", top_type_or_name(3, r"taco|mexican"), top_near_point(3, *DOWNTOWN, 2.5)),

    # ---- garbage
    Case("xqzvkj", max_results(1)),
    Case("asdfghjkl", max_results(1)),
    Case("qqqq", max_results(1)),
    Case("zzxxq vvbnm", max_results(1)),

    # ---- classrooms ("<building code> <room>"): data/usc/usc_rooms.json + rooms.parse_room_number
    # RTH is not centrally scheduled, so RTH 108 is synthesized at the building (verified false)
    Case("rth 108", first_id("room:RTH-108"), first_fields(floor=1, verified=False, pathing=True, code="RTH"),
         second_id("bldg:RTH")),
    Case("RTH108", first_id("room:RTH-108"), first_fields(floor=1)),
    Case("rth-108", first_id("room:RTH-108"), first_fields(floor=1)),
    Case("108 rth", first_id("room:RTH-108")),
    Case("thh 201", first_id("room:THH-201"), first_fields(floor=2, verified=True, pathing=True),
         second_id("bldg:THH"), rest_rooms_of("THH", "2", start=2)),
    Case("thh b09", first_id("room:THH-B09"), first_fields(floor=-1, floor_label="B")),
    Case("vpd ll101", first_id("room:VPD-LL101"), first_fields(floor=-1, floor_label="LL")),
    Case("lvl 3b", first_id("room:LVL-3B"), first_fields(floor=-1, floor_label="LL"),
         why="Leavey's classrooms are on the lower level, not the 3rd floor"),
    Case("gfs 06", first_id("room:GFS-06"), first_fields(floor=None, pathing=False),
         why="no rule for a 2-digit room: floor unknown, never guessed"),
    Case("chp 219", first_id("room:CHP-219"), first_fields(floor=2, pathing=False),
         why="Health Sciences campus: no indoor pathing"),
    Case("thh 1", first_id("bldg:THH"), rest_rooms_of("THH", "1")),
    Case("thh", first_id("bldg:THH"), rest_rooms_of("THH", "")),
    Case("tutor hall 108", first_id("room:RTH-108"), first_fields(floor=1)),
    Case("taper 201", first_id("room:THH-201")),
    Case("taper hall 201", first_id("room:THH-201")),
    Case("7 eleven", first_name("7-eleven"), no_rooms()),
    Case("studio 54", no_rooms()),
    Case("cvs 24", no_rooms()),
    Case("route 66", no_rooms()),

    # ---- held out: paraphrases written after the lexical ranker was tuned, never tuned on.
    # They measure what the semantic signal adds; failures here are reported, not fixed by hand.
    Case("romantic dinner", min_results(3), top_type(5, r"^food|^bar|restaurant"), held_out=True),
    Case("watch the game", top_type(3, r"sports bar|\bbar\b|stadium|arena"), held_out=True),
    Case("vintage clothes", top_type(3, r"vintage|thrift|used clothing|clothing"), held_out=True),
    Case("get my nails done", top_type(3, r"nail|beauty salon|spa"), held_out=True),
    Case("caffeine", top_type(3, CAFE), held_out=True),
    Case("sweet treat", top_type(3, DESSERT), held_out=True),
    Case("spicy food", min_results(5), top_type(5, FOOD), held_out=True),
    Case("healthy lunch", min_results(5), top_type(5, FOOD + r"|salad|health food"), held_out=True),
    Case("hangover food", min_results(5), top_type(5, FOOD), held_out=True),
    Case("somewhere to sit outside", top_type(3, r"outdoor_seating|^outdoors|park|plaza"), held_out=True),
    Case("buy textbooks", top_type(3, r"book store|bookstore"), held_out=True),
    Case("cold brew", top_type(3, CAFE), held_out=True),

    # ---- category chips (no text)
    Case("", top_type(5, r"^food"), median_km(5, 0.5), categories=["food"]),
]
