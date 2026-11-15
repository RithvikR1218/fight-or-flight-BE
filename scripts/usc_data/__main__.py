"""USC campus data scrapers. Run from the repo root:

    python -m scripts.usc_data <step> [options]      (python -m scripts.usc_data <step> -h for options)

Steps:
  places      building codes + outdoor places -> coordinates, from OSM (needs `pip install osmium`)
  libraries   LibCal study rooms + availability            -> data/pois/usc_library_rooms.geojson
  bike        USC Bicycle Coalition map                     -> data/pois/usc_bike_resources.geojson
  venues      Trojan Event Services outdoor venues          -> data/pois/usc_outdoor_venues.geojson
  rooms       Room Finder classroom inventory               -> data/usc/roomfinder_rooms.json
  room-floors classrooms -> building -> floor (+ FPM GIS)   -> data/usc/usc_rooms.json  (after `rooms`)
  all         libraries, bike, venues, rooms, room-floors
"""
import sys

from . import bike, build_places, libraries, outdoor_venues, roomfinder, rooms

STEPS = {
    "places": build_places.main,
    "libraries": libraries.main,
    "bike": bike.main,
    "venues": outdoor_venues.main,
    "rooms": roomfinder.main,
    "room-floors": rooms.main,
}
ALL = ["libraries", "bike", "venues", "rooms", "room-floors"]


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] not in (*STEPS, "all"):
        print(__doc__)
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 2)
    step, rest = sys.argv[1], sys.argv[2:]
    if step == "all":
        for name in ALL:
            print(f"== {name}")
            STEPS[name]([])
    else:
        STEPS[step](rest)


if __name__ == "__main__":
    main()
