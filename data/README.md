# data/

| Path | In git? | What |
|---|---|---|
| `pois/*.geojson` | yes | points of interest, loaded by `python -m scripts.load_pois` (and into search by `load_places`) |
| `places/usc_buildings.geojson` | yes | USC campus buildings with codes, for search (`python -m scripts.load_places`) |
| `usc_google_maps_places.geojson` | yes (23 MB) | Google Maps places for search (`python -m scripts.load_places`); schema in `GOOGLE_MAPS_PLACES.md` |
| `raw/usc_roads.osm.pbf` | yes | OSM roads clipped to the study circle + 1.5 km (`scripts/fetch_osm.sh`) |
| `raw/LARIAC6_Buildings_2020.gdb.zip` | no (Release) | LA County building outlines (`./scripts/fetch_data.sh`) |
| `tiles/buildings.pmtiles` | no (Release) | 3D building tiles |
| `checksums.sha256` | yes | checksums of the Release files |
| `work/` | no | downloads + intermediates, safe to delete |

## POIs (`pois/`)

Each file is a GeoJSON FeatureCollection in WGS84 `[lng, lat]`. Each feature needs a `type` property that matches a frontend POI type (`seating`, `study`, ...). `id`, `name` and `source` are optional; every other property is kept as-is and returned by `GET /api/v1/pois`. Drop a new file in and re-run the loader. Rows removed from a file are deleted on the next load.

### USC outdoor table groups

`pois/usc_outdoor_table_groups.geojson` has one Point per group of outdoor tables on the USC University Park campus (15 groups). Each feature has `type: "seating"` (matches the FE `seating` POI type), `table_count_est`, and `source`.

Method: Esri World Imagery (zoom 19) scanned for patio umbrellas and table clusters, then every point was reviewed by hand on a satellite map; 8 groups were added manually where imagery could not show them (shade, no umbrellas). Counts are estimates. Coordinates are WGS84 [lng, lat].

Imagery © Esri, Maxar, Earthstar Geographics; only derived coordinates are stored here.

## Search places (`places` table)

`python -m scripts.load_places` fills the `places` table that place search runs on, from three sources:

- `usc_google_maps_places.geojson` (kind `place`), 15,343 places scraped 2026-11-14 (see `GOOGLE_MAPS_PLACES.md`).
  Places outside the 5-mile study area are skipped (the file reaches 10 miles); `--all-areas` keeps them.
  Read as UTF-8: hours use U+202F before AM/PM and an en dash between times.
- `places/usc_buildings.geojson` (kind `building`): see below.
- `pois/*.geojson` (kind `poi`).

### USC campus buildings

`places/usc_buildings.geojson` has one Point per USC building on and around the University Park campus (194). Each feature has `code` (USC's three-letter building code, e.g. `DML`), `name`, `aliases` (everyday names), `category` (`campus`, `study` for libraries, `parking` for structures), `subcategory`, `address`, `usc_type`, `entrances` (`[lng, lat]` of each mapped entrance) and `source`.

Method: built by `python -m scripts.build_usc_buildings`. Codes, names, addresses, centre points (the `LAT`/`LON` fields) and entrances come from USC Facilities Planning & Management's public GIS (`https://fpmgis.usc.edu/server/rest/services/USC_FPM/`, the `Building` and `BuildingEntrance` layers). It keeps sites `UPC` and `UPCOFF` and drops demolished or sold buildings, gates, vaults and trailers. Names are title-cased from FPM's upper case. Aliases add OpenStreetMap names for the same buildings, matched by code or by the OSM building's centre falling inside the FPM footprint, plus a short hand-kept list of common names (for example `VKC` for the renamed Von KleinSmid Center, now DMC).

OSM data © OpenStreetMap contributors, ODbL.

## USC campus data (`python -m scripts.usc_data`)

Scrapers in `scripts/usc_data/` (only need `requests`; `osmium` to rebuild places). Run from the repo root, e.g. `python -m scripts.usc_data all`, or one step at a time (`-h` lists them). New POI files land in `pois/` and load with `load_pois` (map layer) and `load_places` (search); tables that aren't map points go to `usc/`.

| Output | Type(s) | Source |
|---|---|---|
| `pois/usc_library_rooms.geojson` | `study` | LibCal (`libcal.usc.edu/spaces?lid=2893,2895,13141`): 37 bookable rooms at Doheny, Leavey, Science & Engineering, with capacity, booking link, and `free_slots` per date for the next 7 days |
| `usc/library_availability.json` | — | same availability by room; re-run `libraries` (e.g. every 15 min) to refresh |
| `pois/usc_bike_resources.geojson` | `bike_parking` (`rack_status`: current/outdated/informal/locker), `bike_pump`, `bike_service`, `bike_lane`, `bike_transit`, `bike_zone` | USC Bicycle Coalition Google My Map (KML export). **Dated Oct 2015.** |
| `pois/usc_outdoor_venues.geojson` | `outdoor_venue` | trojanevents.usc.edu outdoor diagrams, with diagram links. Placed via OSM; `location_approx: true` = placed on the adjacent building/plaza. Argue Plaza and Crocker Plaza aren't placed. |
| `usc/roomfinder_rooms.json` | — | sites.usc.edu/roomfinder (WP REST API, 30 s crawl delay per its robots.txt): 220 centrally scheduled classrooms in 22 buildings, with campus (from the front-page index), type, capacity, furniture, boards, equipment, and per-building room counts |
| `usc/usc_rooms.json` | — | the same classrooms with building, campus, coordinates and floor (`room-floors` step; see below) |
| `usc/usc_places.json` | — | building code / venue → coordinates from the OSM study-area extract (`places`) |

Room Finder only covers centrally scheduled classrooms. In `roomfinder_rooms.json`, `HED` and `KOH` have no coordinates because the OSM extract lacks them, and `CHP` and `DOH` are on the Health Sciences campus. `usc_rooms.json` resolves all four: HED and KOH from `usc_buildings.geojson`, CHP and DOH from FPM GIS. Room Finder's `DOH` is the Doheny Eye building at HSC, not Doheny Library, which is `DML`.

### Rooms and floors (`usc/usc_rooms.json`)

`python -m scripts.usc_data room-floors` (run after `rooms`) writes `{generated_at, sources, floor_convention, buildings, rooms, report}`. Each room has `id` (`room:THH-101`), `code`, `room`, `building_id` (`bldg:THH`), `building_name`, `campus`, `floor`, `floor_label`, `floor_confidence`, `pathing`, `lat`/`lng` (the building centre), `capacity`, `classroom_type`, `furniture`, `equipment` and `url`. Each building has `name`, `campus`, `lat`/`lng`, `floor_count`, `has_basement` and `floors`.

**Floor convention.** `floor` is an ordinal level. 1, 2, 3 … are the numbered floors, with 1 as the first floor. 0 is a `G`/`GDN` level below floor 1. -1 is `B`/`LL`/`LG` (basement or lower level), and -2 is `SB`. `floor_label` keeps the building's own label.

`floor_confidence` takes one of three values:
- `exact`: a documented per-room fact.
- `rule`: USC's numbering. NXX and NXXX rooms are on floor N, and a letter prefix names the level.
- `unknown`: `floor` is null. Nothing is guessed.

`pathing` is true only for UPC rooms with a known floor, because indoor pathing supports UPC only. The rules and the evidence for each exception are in `backend/app/services/rooms.py`. `parse_room_number` also serves rooms that Room Finder doesn't list (for example `RTH 108`).

**Provenance.**
- Rooms: Room Finder.
- Building points and names: `places/usc_buildings.geojson` for UPC; the FPM `Building` layer for other campuses.
- Floor counts (`BldgFlrInt`) and basements (`HasBasement`): the FPM `Building` layer.
- Numbering rules: about 170 room-and-floor pairs in FPM's `LactationRoom` and `MenstrualProductDispenser` layers.
- Leavey Library: its seven classrooms (`LVL 13/16/17/3B/3C/3V/3Y`) are on the Lower Level, per the USC Libraries Leavey tour guide (2023) and libanswers.usc.edu/faq/24137.

**Limits.**
- FPM GIS publishes no room or floor-plan layer, so per-room floors are inferred from the numbering, not read from a floor plan.
- Room Finder lists only centrally scheduled classrooms (220), not every room on campus. Offices, labs and department-scheduled rooms (most of RTH, for example) are missing.
- `GFS 06` has a post but isn't in Room Finder's front-page index, and its floor is unknown.
- Every room carries `sources` (`roomfinder`, `soc`) and `section_count`. If `usc/soc_rooms.json` exists (rooms seen in class schedules), it is merged in, deduped on code and room, with Room Finder attributes winning. That file isn't built yet: the current Schedule of Classes (classes.usc.edu) shows room locations only to signed-in USC users.
