# fight-or-flight-BE

Backend for the USC safe-route project. It collects **space-time data** (light, weather, crime) for a 5-mile circle around USC (the same area as the risk model), stores it in **PostgreSQL + PostGIS + TimescaleDB**, and serves it to:

- the **crime-risk model** in `risk-model/` and the **safe-route planner** in `pathfinding/` (by Desh), which the API runs in-process on backend data, and
- the **frontend**, which gets 3D buildings as PMTiles, a risk heatmap, and JSON data layers.

```
LIVE (automatic, ingestor container)
  sun 1m · Open-Meteo 5m · NWS 10m · LAPD crime 60m ──upsert──┐
                                                               ▼
STATIC (one-time scripts, run by you)            PostgreSQL 16 + PostGIS + TimescaleDB
  County buildings (.gdb.zip) ─ build_buildings.sh ─▶  buildings, h3_cells features
        └─▶ data/tiles/buildings.pmtiles                       │
  OSM extract (.osm.pbf) ─ fetch_osm.sh + load_roads ▶ road_nodes, road_edges
                                                               ▼
       crime_incidents ─ export_crime_db ─▶ risk-model/data/crime.db
                                                               ▼
   Flask API (/api/v1) + risk model (risk-model/) + planner (pathfinding/) ──▶ frontend
```

## Quick start

```bash
./scripts/fetch_data.sh           # large data files from the GitHub Release (see "Data files")
cp .env.example .env              # optional; every value has a default
docker compose up -d --build      # db + api (runs migrations) + ingestor
docker compose exec api python -m scripts.seed_h3_cells    # 1,727 res-9 cells, 11 res-6 area cells
open http://localhost:5000/api/docs                        # Swagger UI
```

If ports 5432 or 5000 are already in use, set `DB_PORT` / `API_PORT` in `.env`.

That gives you the **live data**. On first start the crime ingestors backfill from 2024-01-01, which takes a few minutes and pulls about 350k incidents. Watch progress with `curl localhost:5000/api/v1/sources`.

**Buildings and roads are not loaded automatically.** They are one-time loads you run after `seed_h3_cells`; see [Static map layers](#static-map-layers). Until then, `/buildings*` and `/roads*` return empty results or 404, and the building/road columns in `/cells` and `/features` are 0.

```bash
docker compose run --rm tools bash scripts/build_buildings.sh   # ~3 min: buildings -> PostGIS (+ rebuilds PMTiles)
docker compose run --rm tools python -m scripts.load_roads      # ~3 min: OSM walk network -> PostGIS
docker compose exec api python -m scripts.load_pois              # seconds: data/pois/*.geojson -> PostGIS
# (only if data/raw/usc_roads.osm.pbf is missing: bash scripts/fetch_osm.sh first)
```

## Spatial model: H3

Every data point is keyed by an [H3](https://h3geo.org) hexagon, stored as a hex string, so all sources line up on `(h3_cell, time_bucket)`.

| Resolution | Cell size | Used for |
|---|---|---|
| **9** (fine) | ~0.12 km², 1,727 cells | crime, streetlights, buildings, features |
| **6** (area) | ~36 km², 11 cells | sun position and weather (area-wide signals) |

`h3_cells.parent_r6` links the two levels. The `ml_features` view broadcasts area signals down to their res-9 children.

## Database

| Object | Kind | Purpose |
|---|---|---|
| `observations` | hypertable (1-day chunks) | narrow `(h3_cell, metric, ts, source, value)`. New metrics need no migration. Upserts make polling idempotent. |
| `crime_incidents` | hypertable (30-day chunks) | point events with geometry + H3 cell |
| `obs_10min`, `obs_hourly` | continuous aggregates | avg/min/max/last per cell+metric, kept fresh by policies, real-time enabled |
| `crime_hourly` | continuous aggregate | incident counts per cell, hour, category |
| `ml_features` | view | one row per `(h3_cell, 10-min bucket)`: sun, darkness, weather, outages, static cell features (buildings, roads, POI counts), trailing-30-day crime count |
| `h3_cells` | PostGIS table | res-9 grid + static features (buildings, roads, streetlights) |
| `buildings`, `road_nodes`, `road_edges`, `pois`, `streetlights` | PostGIS tables | static layers, loaded by scripts |
| `ingest_runs` | table | ingestion audit log, shown in `/health` and `/sources` |

Policies: raw `observations` compress after 3 days and are dropped after 90 days. Aggregates are kept indefinitely. The schema lives in `backend/migrations/versions/` (`0001_initial` ... `0005_search_index`).

## Ingestion sources

| Source | Data | Interval | Notes |
|---|---|---|---|
| `sun` | `sun_altitude_deg`, `is_dark` | 1 min | computed locally with astral; civil twilight (−6°) = dark |
| `openmeteo` | temp, humidity, precip, cloud cover, wind | 5 min | free, no key; one batched call for all 30 area cells; upstream refreshes every ~15 min |
| `nws` | temp, humidity, visibility, wind, precip | 10 min | stations `KCQT,KLAX`; offline stations are skipped |
| `lapd` | crime 2024 | 60 min | legacy dataset `2nrs-mtv8` (frozen at end of 2024) |
| `lapd_nibrs` | crime 2025+ | 60 min | NIBRS dataset `k7nn-b2ep`, updated ~weekly; coordinates rounded to the hundred-block |
| `la311` | `streetlight_outages_open` | 30 min | disabled until `LA311_DATASET_ID` is set (MyLA311 publishes one dataset per year) |

Polling faster than a source publishes is harmless: writes are upserts. Published crime data lags by days, so treat it as history and training data. The minute-level signals are light and weather.

Add a source by subclassing `Ingestor` (`backend/app/ingestion/base.py`), implementing `fetch()` and `normalize()`, and registering it in `app/ingestion/__init__.py`.

## API (`/api/v1`, full docs at `/api/docs`)

| Endpoint | Purpose |
|---|---|
| `GET /health` | DB status + last run per ingestor |
| `GET /sources` | sources, intervals, last runs |
| `GET /cells?bbox=&format=json\|geojson` | res-9 cells + static features |
| `GET /observations?bbox=&cells=&metric=&from=&to=&bucket=raw\|1m\|10m\|1h` | time series |
| `GET /observations/latest?bbox=&metric=` | latest value per cell/metric, for live inference |
| `GET /observations/metrics` | which metrics exist |
| `GET /incidents?bbox=&from=&to=&category=` | crime points (default: last 30 days) |
| `GET /incidents/hourly?cells=&from=&to=` | hourly crime counts per cell |
| `GET /features?bbox=&from=&to=&format=json\|csv\|parquet` | **the ML feature matrix** |
| `GET /buildings.pmtiles` | 3D building vector tiles (supports HTTP range requests) |
| `GET /buildings?bbox=` | building GeoJSON for a small area |
| `GET /roads?bbox=` | road edges as GeoJSON for a small area |
| `GET /roads/graph?part=nodes\|edges&format=parquet\|csv\|json` | **full walk graph for routing** |
| `GET /pois?bbox=&type=seating,study` | points of interest as GeoJSON (extra attributes in `properties`) |
| `GET /pois/types` | POI types with counts |
| `GET /search?q=&near=lat,lng&bbox=&category=&open_now=&limit=` | **place search / autocomplete** (see [Search API](#search-api)) |
| `GET /search/categories` | categories and place types with counts (for filter chips) |
| `GET /places/{id}` | one place: hours per weekday, phone, website, photo, ratings, reviews |
| `GET /geocode/reverse?at=lat,lng` | name for a point: nearest place/building within ~60 m, else nearest named road |
| `GET /risk/status` | model loaded?, training dates, 2023 test metrics |
| `GET /risk/score?lat=&lng=\|cell=&hour=&date=\|at=` | **risk + top factors for a point**, plus everything the backend knows about that cell |
| `GET /risk/hotzones?hour=&date=&bbox=&min_risk=` | **heatmap**: risk for every modelled cell at an hour, plus `no_data_cells` |
| `GET /risk/edges?hour=&date=&lambda=&format=parquet\|csv\|json` | route costs for every walk-graph edge: `cost = length_m × (1 + λ·risk)` |
| `GET /route?from=lat,lng&to=lat,lng&hour=&date=&max_detour=&phone_weight=` | **safe vs shortest walking route** with risk and blue-light stats (see [Risk model and route planner](#risk-model-and-route-planner)) |
| `POST /admin/ingest/{source}` | run one ingestor now (`X-Admin-Token` if `ADMIN_TOKEN` is set) |
| `GET /auth/login?next=<FE url>` | full-page redirect to Google sign-in (see [Auth](#auth)) |
| `GET /auth/callback` | Google returns here; sets the session cookie, redirects to `next` |
| `GET /auth/me` | `{id, email, name, picture}` when signed in, `401` otherwise |
| `POST /auth/logout` | `204`, clears the session |

`bbox` is `minLng,minLat,maxLng,maxLat`. Times are ISO-8601, interpreted as UTC if no offset is given.

```bash
# training pull for the ML service
curl -o features.parquet "localhost:5000/api/v1/features?from=2026-11-01T00:00:00Z&format=parquet"
```

## Auth

Sign-in is Google OpenID Connect, limited to `@usc.edu` Workspace accounts. Reads stay public.

1. The FE sends the browser to `/api/v1/auth/login?next=<absolute FE URL>`.
2. Google signs the user in and returns to `/api/v1/auth/callback`. The backend checks the ID token
   (`email_verified`, email domain and Workspace `hd` must all be `usc.edu`), then sets the session
   cookie `fof_session` (Flask's signed cookie, httpOnly, 7 days) and redirects to `next`.
3. Refused or failed sign-in redirects to the FE's `/login?error=domain|cancelled|failed|unavailable`.
4. The FE asks `GET /api/v1/auth/me` who is signed in (`401` = signed out); `POST /api/v1/auth/logout` ends it.

`next` must be on an origin in `FRONTEND_ORIGINS` (otherwise it falls back to that list's first
origin), and CORS allows only those origins, with credentials. The session stores only the user's
public profile, keyed by Google's stable `sub`; there is no users table yet.

Setup (`.env`):

- `SECRET_KEY`: long random string (`python -c "import secrets; print(secrets.token_urlsafe(48))"`).
  The same value on every API process, or sessions break between them.
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`: Google Cloud console > APIs & Services > Credentials >
  OAuth client ID (Web application). Authorized redirect URI: `http://localhost:5000/api/v1/auth/callback`
  locally, your API's `/api/v1/auth/callback` in production (also set `OAUTH_REDIRECT_URI` to it
  when behind a proxy).
- `FRONTEND_ORIGINS`: comma-separated FE origins. Default `http://localhost:5173,http://localhost:4173`.
- FE on a different site than the API (production): `SESSION_COOKIE_SAMESITE=None` and
  `SESSION_COOKIE_SECURE=true` (HTTPS only).
- `AUTH_DEV_LOGIN=true` (local only, never in production) adds `GET /api/v1/auth/dev/login?email=&next=`,
  which signs in any `@usc.edu` address without Google, for testing the FE end to end.

Without `SECRET_KEY` and the Google client, `/auth/me` answers `401` and `/auth/login` redirects back
with `error=unavailable`. Sign-in also stays disabled if `SECRET_KEY` is shorter than 32 characters, contains
spaces, or starts with `#` (an `.env` comment read as the value).

### Deploying auth (nginx + HTTPS, frontend on its own domain)

- **nginx must forward the scheme:** `proxy_set_header X-Forwarded-Proto $scheme;` and `proxy_set_header Host $host;`.
  The app trusts one proxy's `X-Forwarded-*` headers (werkzeug `ProxyFix`), so the callback it sends to Google
  is `https://...`. Without the header, Google refuses the plain-http callback (`redirect_uri_mismatch`).
- **Google console:** OAuth client (Web application) with these Authorized redirect URIs, exact match, no trailing slash:
  `https://<host>/api/v1/auth/callback` (production) and `http://localhost:5000/api/v1/auth/callback` (local).
  Set `OAUTH_REDIRECT_URI` to the production one.
- **Frontend on a different domain:** choose one.
  1. **Recommended: the frontend host proxies `/api`.** E.g. a Vercel rewrite of `/api/:path*` to
     `https://<backend host>/api/:path*`. The browser sees one origin, so the cookie is first-party and works in
     Safari/Firefox, CORS isn't involved, and `SESSION_COOKIE_SAMESITE=Lax` stays.
     Use `OAUTH_REDIRECT_URI=https://<frontend host>/api/v1/auth/callback` and register that URI in Google.
  2. **The frontend calls the API host directly.** Set `FRONTEND_ORIGINS=https://<frontend host>`,
     `SESSION_COOKIE_SAMESITE=None`, `SESSION_COOKIE_SECURE=true`, and `OAUTH_REDIRECT_URI=https://<backend host>/api/v1/auth/callback`.
     The cookie is then third-party: Safari (all iPhones) and Firefox block it by default, so sign-in can
     appear to succeed while `/auth/me` keeps answering `401`.
- In both cases, put the frontend's URL in `FRONTEND_ORIGINS` (it is also the allow-list for `next`). CORS on
  `/api/*` only allows those origins.
- `authlib` is a new dependency: deploy with `docker compose up -d --build`.

## Risk model and route planner

Two packages by Desh live next to the backend, and the API runs both in-process:

- **`risk-model/`**: the XGBoost crime-risk model. Training, evaluation and the model card are in `risk-model/README.md` and `risk-model/MODEL_GUIDE.md`. It covers a **5-mile circle** around USC (1,451 modelled cells).
- **`pathfinding/`**: the safe-route planner (`pathfinding/README.md`).

`risk-model/`'s code imports itself as `algorithm.*`, so the Docker image installs it at `/app/algorithm` (`COPY risk-model/ /app/algorithm/`, and the same mount in `docker-compose.yml`). Neither package is modified by the backend. Each gunicorn worker loads the model plus its crime history (~4 s, ~600 MB) and the route graph at startup, which is why `gunicorn.conf.py` runs 2 workers × 4 threads.

**Data the backend gives the model:**

| Model input | Source |
|---|---|
| crime history: per-incident harm (severity × recency × time of day), neighbours, long-run rates, area-wide | `crime_incidents` (LAPD `2nrs-mtv8`, 2020–2024, including `weapon_used_cd`, which raises the severity of armed crimes), exported to `risk-model/data/crime.db` by `scripts/export_crime_db.py`. The backend is the only crime downloader. Its output reproduces the model author's own heatmap (99.8% of cell-hours within 0.01, rank correlation 0.9997). |
| `is_dark` | live sun ingestor, for current-hour requests; other hours use the model's own astral calculation |
| time, location, distance to the 83 USC blue-light phones | computed inside the model (`risk-model/data/bluelight_phones.csv`) |

`/risk/score` builds a context dict for the cell from the backend database: weather, darkness, lit-road share, streetlights, buildings, roads, POIs, and live crime counts including NIBRS 2025+. Every key whose name matches a model feature is passed to the model as an input. The response lists them under `model_inputs_from_backend`; everything else comes back under `context`.

**Safe routes (`GET /api/v1/route?from=lat,lng&to=lat,lng&hour=`):** the planner in `pathfinding/` runs on backend data instead of its standalone inputs:
- **Graph:** `road_edges` from PostGIS (the OSM walk network from `data/raw/usc_roads.osm.pbf`) is split into straight segments, which is the shape `pathfinding.graph.WalkGraph` expects. It covers the model circle plus 1.5 km. There's no Overpass download.
- **Risk:** the model's heatmap for the requested date (`/risk/hotzones`, memoised per date), not a fixed cache file.
- **Response:** `safe` and `shortest` routes, each with length, detour %, mean/max risk, metres in high-risk cells, % without data, blue-light phones passed, `coords` (`[[lat, lng], ...]`) and a GeoJSON `geometry`.
- **Knobs:** `max_detour` (default 0.2, i.e. the safe route is at most 20% longer) and `phone_weight` (0.05).
- **Limits:** trips are capped at 12 km straight-line, and points must be within 300 m of a walkable street.

`/risk/edges` still gives a per-edge cost for anyone running their own router.

Known limits:
- The model anchors its history at the end of 2023, because LAPD's 2024 NIBRS switch halved the old dataset's counts. NIBRS 2025+ data is therefore context only until the model is retrained on it.
- Cells outside the 5-mile model circle, or in other police jurisdictions, have no score. Routes treat them as a neutral 0.5 and report `no_data_pct`.

## Demo mode (simulated crimes)

For recording the app without walking into risky areas. Off unless `DEMO_MODE=true` (then `docker compose up -d`).

| Endpoint | What it does |
|---|---|
| `GET /api/v1/demo/status` | `{enabled, ttl_minutes, avoid_radius_m, categories}` |
| `GET /api/v1/demo/incidents` | active simulated incidents (`{incidents: [{id, ts, lat, lng, category, severity}]}`) |
| `POST /api/v1/demo/incidents` | `{lat, lng, category?, severity?}`: simulate one now (signed-in users only) |
| `DELETE /api/v1/demo/incidents` | remove them all (signed-in users only) |

A simulated incident is stored in `crime_incidents` with `source = 'demo'` (so `/incidents` lists it) and expires after `DEMO_INCIDENT_TTL_MIN` minutes. While it is active, `/route` treats everything within `DEMO_INCIDENT_RADIUS_M` of it as the riskiest street, even on campus, so routes go around it when the detour budget allows; route responses then include `demo_incidents` (how many were active). Demo rows never reach the risk model (`export_crime_db.py` exports `source = 'lapd'` only). The frontend's Demo page (`?demo`) drives these endpoints.

## Data files

GitHub rejects files over 100 MB, so the large files live in a **GitHub Release** (tag `data-v2`; `data-v1` has the earlier 10-mile tiles). Release assets can be up to 2 GB each and don't use Git LFS quota. Everything else is in git.

| File | Size | Where | Purpose |
|---|---|---|---|
| `data/raw/LARIAC6_Buildings_2020.gdb.zip` | 412 MB | Release | building source for `build_buildings.sh` |
| `data/tiles/buildings.pmtiles` | 30 MB | Release | prebuilt 3D tiles; serve as-is or rebuild |
| `data/raw/usc_roads.osm.pbf` | 4.6 MB | git | OSM roads clipped to the circle + 1.5 km |
| `data/checksums.sha256` | — | git | SHA-256 of the Release files |
| `data/work/` | ~1 GB | ignored | downloads + intermediates; safe to delete |
| `risk-model/data/crime.db` | 78 MB | Release | the risk model's LAPD 2020–2024 history (395k incidents), made by `scripts/export_crime_db.py`. Without it the model has no history and every cell gets the neutral no-data risk. |

`./scripts/fetch_data.sh` downloads the Release files into place and verifies their checksums. It resumes partial downloads and skips files already present (a file whose checksum differs, such as a `crime.db` exported locally, is replaced). Pass `--no-tiles` to skip the tiles and build them yourself, or `--no-crime` to keep your own `crime.db` (build it with `scripts.backfill_crime --source lapd --start 2020-01-01` then `scripts.export_crime_db`). After fetching a new `crime.db`, run `docker compose restart api`.

To update the data, regenerate the files, then run `./scripts/publish_data_release.sh` (needs `gh auth login`; use `DATA_RELEASE_TAG=data-v3` for a new version). Name files to upload only those, e.g. `./scripts/publish_data_release.sh risk-model/data/crime.db`. It uploads the files and updates their lines in `data/checksums.sha256`, which you then commit.

### Changing the study radius

The radius (`RADIUS_KM`, default 8.047 km = 5 miles) must match `RADIUS_MILES` in `risk-model/common/config.py`; a test checks this. After changing it:

```bash
docker compose exec api python -m scripts.seed_h3_cells                   # grid: adds/removes cells
docker compose exec api python -m scripts.prune_study_area --yes          # drop rows outside (dry run without --yes)
docker compose run --rm tools bash scripts/fetch_osm.sh                   # roads file for the new circle
docker compose run --rm tools python -m scripts.load_roads
docker compose run --rm tools bash scripts/build_buildings.sh             # buildings + tiles (then publish a new data Release)
docker compose exec api python -m scripts.export_crime_db && docker compose restart api
```

## Static map layers

These change rarely, so they are loaded by scripts instead of the ingestor. Re-running a script replaces that layer (TRUNCATE + load) and recomputes the per-cell features in `h3_cells`. Use the `tools` container: it has tippecanoe and runs as your user.

| Layer | Source | Script | Result |
|---|---|---|---|
| H3 grid | computed | `docker compose exec api python -m scripts.seed_h3_cells` | `h3_cells` (run this first) |
| Buildings | LA County LARIAC6 Building Outlines (file in repo) | `docker compose run --rm tools bash scripts/build_buildings.sh` | `buildings` table, `data/tiles/buildings.pmtiles` |
| Roads | OpenStreetMap extract (file in repo) | `docker compose run --rm tools python -m scripts.load_roads` | `road_nodes`, `road_edges` |
| POIs | `data/pois/*.geojson` (in git) | `docker compose exec api python -m scripts.load_pois` | `pois` |
| Streetlights | LA GeoHub export (optional, bring your own file) | `docker compose exec api python -m scripts.load_streetlights /data/raw/<file>` | `streetlights` |

### Buildings

The source file is `data/raw/LARIAC6_Buildings_2020.gdb.zip` (412 MB, from the data Release; see [Data files](#data-files)). It came from the [Countywide Building Outlines](https://www.arcgis.com/home/item.html?id=e6a1e375e12a4fe6849d896a26ec028a) item: 3.3M footprints countywide, CRS EPSG:6424 (CA State Plane, US feet).

`build_buildings.sh` does the following:
1. Opens the zipped geodatabase in place (no unzip), using layer `LARIAC6_BUILDINGS_2020` (skips the `*_DELETED_*` layer).
2. Keeps `CODE = Building`, dropping `Courtyard` and `Free Standing Solar Structure`. Override with `--codes`.
3. Clips to the 8.047 km (5-mile) circle using each footprint's interior point. That leaves **289,038 buildings**.
4. Converts `HEIGHT`/`ELEV` from feet to metres. 97.6% have a height; the max is 310 m (US Bank Tower).
5. Loads PostGIS and tags each building with its res-9 cell, filling `h3_cells.building_count` / `avg_height_m`.
6. Runs tippecanoe to write `data/tiles/buildings.pmtiles` (~30 MB, zoom 10–16, layer `buildings`, property `height_m`). Footprints with no height get 4 m in the tiles only, so they still extrude.

**Map data stays in the backend.** Shipping ~290k footprints to the browser as GeoJSON would mean ~140 MB. The frontend loads only the visible tiles from `GET /api/v1/buildings.pmtiles`, which supports range requests. Render with MapLibre GL plus the [pmtiles](https://github.com/protomaps/PMTiles) protocol: a `fill-extrusion` layer on source-layer `buildings`, with `fill-extrusion-height: ["get", "height_m"]`.

### Roads

The County file contains no roads, so roads come from **OpenStreetMap**: streets plus sidewalks, footpaths and campus paths. That suits pedestrian routing better than the County street centerlines. The road graph is built from a local file, not the public Overpass API, which proved too unreliable (502s and hung requests).

1. `scripts/fetch_osm.sh` (one-time; the output is committed):
   - Downloads Geofabrik's Southern California extract (~640 MB, kept in `data/work/` and ignored by git).
   - Clips it to the circle with `osmium extract` and keeps only `highway=*` ways.
   - Writes `data/raw/usc_roads.osm.pbf` (4.6 MB, committed to git), clipped to the circle plus `ROAD_MARGIN_KM` (1.5 km) so routes near the edge have streets. A download younger than 7 days is reused. Re-run it whenever you want fresher OSM data.
2. `scripts/load_roads.py` builds the network and loads it:
   - Uses OSMnx with an OSMnx-style "walk" filter that drops motorways, private access, `foot=no`, construction and so on.
   - Keeps the largest connected component, simplifies the graph, and makes it undirected.
   - Loads the result into PostGIS.

What gets stored:
- `road_nodes(osmid, geom, street_count, h3_cell)` holds intersections and path ends.
- `road_edges(u, v, key, name, highway, lit, sidewalk, footway, length_m, geom, h3_cell, h3_cells[])`:
  - `h3_cell` is the cell at the edge's midpoint.
  - `h3_cells` lists every res-9 cell the edge passes through, sampled every 50 m. The ML service can weight an edge by `length_m × risk(cells)`.
  - `lit` is OSM's streetlight tag (`yes`/`no`/…). It's often missing, and NULL means unknown.
- Per-cell features go into `h3_cells`: `road_length_m`, `lit_road_length_m`, `intersection_count`. They are also exposed through `/cells` and `/features`.

For the ML service, pull the full graph with `GET /api/v1/roads/graph?part=nodes` and `?part=edges` (Parquet by default), then build e.g. a networkx graph keyed on `osmid`.

### POIs

Campus points of interest, such as outdoor seating and study spots, live as GeoJSON files in `data/pois/` (format in `data/README.md`). `load_pois` does the following:
- Upserts every file into `pois(id, type, name, source, source_file, props, geom, h3_cell)`. `props` keeps any extra attributes, such as `table_count_est`.
- Deletes rows that were removed from a file.
- Recomputes `h3_cells.poi_counts`, e.g. `{"seating": 4}`.

The frontend reads `GET /api/v1/pois?type=seating`. The ML feature matrix gets `seating_count` and `poi_count` per cell.

Current data: `usc_outdoor_table_groups.geojson`, 15 outdoor table groups found from satellite imagery and checked by hand (PR #1).

## Search API

Google-Maps-style place search over the `places` table: Google Maps places, USC buildings with their codes, campus POIs and classrooms. Public, read-only. Load or refresh the table (and the search index every API worker keeps in memory) with:

```bash
docker compose exec api flask db upgrade                  # once: migration 0005 adds places.embedding + search_meta
docker compose exec api python -m scripts.load_places     # data/usc_google_maps_places.geojson, data/places/, pois, data/usc/usc_rooms.json
```

`load_places` also computes a sentence embedding per place (fastembed, `BAAI/bge-small-en-v1.5`, ~2 min for ~10k places on a laptop CPU; unchanged places reuse their stored vector) and finally bumps `search_meta.places_version`. Each API worker checks that marker every `SEARCH_INDEX_TTL` seconds (one tiny query) and rebuilds its index in the background when it changed. `--embed no` skips embeddings (search then runs lexical-only).

`GET /api/v1/search` parameters:

| Param | Meaning |
|---|---|
| `q` | free text, works from 1 character. Optional when `category` is set |
| `near` | `lat,lng`; ranks by distance and adds `distance_m` (integer metres) |
| `bbox` | `minLng,minLat,maxLng,maxLat` hard filter |
| `category` | comma-separated top-level ids from `/search/categories`, e.g. `food,cafe` |
| `open_now` | `true` keeps only places known to be open now (unknown hours are dropped) |
| `limit` | default 20, max 50 |

Response: `{"results": [{id, kind, name, category, subcategory, address, lat, lng, distance_m, rating, review_count, open_now, match}]}`, best first. `open_now` is left out when hours are unknown. `match` says what matched: `name`, `category`, `content` (description / attributes / reviews), `address`, `semantic` (meaning only) or `room`. No `q` and no `category` returns `400`; malformed params return `422` (same error shape as the rest of the API). `q=` empty with `category=food&near=...` gives the nearest places in that category (the Food / Study / Parking chips).

**Rooms** (`kind: "room"`). `rth 108`, `RTH108`, `108 rth`, `thh b09`, `taper hall 201` return that classroom first, then its building, then the building's other rooms on the same level. `thh 1` / `thh` list the building and its rooms. Room results add `floor` (1, 2, ... ; 0 = G level; -1 = B / LL; -2 = SB; `null` when the number fits no rule), `floor_label`, `building_id`, `building_name`, `code`, `room`, `pathing` and `verified`. The ~220 centrally scheduled classrooms come from `data/usc/usc_rooms.json` (`verified: true`); any other valid `<code> <number>` (e.g. `rth 108`) is answered at its building with the floor read from the number (`verified: false`, `services/rooms.py`). `pathing` is true only on the University Park campus (UPC) with a known floor: indoor pathing does not cover HSC. `GET /api/v1/places/room:RTH-108` returns the same fields, for synthesized rooms too. Rooms never appear in ordinary searches.

### How results are ranked

Everything runs in memory (`app/services/search_index.py`, numpy, no SQL per query), on the same principles as Google local search: **relevance decides whether a place is shown, distance mostly decides the order, prominence breaks ties**.

1. **Query understanding** (cached per query). `q` is normalized (`services/text_norm.py`), stop words dropped, plurals folded against the corpus (`tacos` → taco, `groceries` → grocery). A street or neighbourhood in the query (`coffee near vermont`, `starbucks figueroa`, `tacos downtown`) becomes the distance anchor instead of the user's position. Each word is expanded into index terms: exact, typo (Damerau distance with Meilisearch's budget: 0 typos up to 4 letters, 1 for 5-8, 2 for 9+, a wrong first letter counts as two; `resturant`, `pharmcy`, `doheney`, `chipolte`), prefix (only the last word, the one being typed; not when it is already a complete type word, so `bar` is not `barber` and `park` is not `parking`), `-ing` forms (`printing` → print), word synonyms (`place_taxonomy.TERM_SYNONYMS`: fix → repair, bike → bicycle) and type intents (`place_taxonomy.QUERY_SYNONYMS`: `dessert` → ice cream / bakery / ...).
2. **Relevance** in [0, 1]. Per query word, the best evidence over four fields, weighted like BM25F in the attribute order Meilisearch uses (name > type > description/reviews): name 1.0, type 1.0 (Google labels, subcategory, synonyms, tags derived from opening hours and attributes such as "late night" and "restroom"), content 0.5 (BM25 saturation over description, attributes and reviews), address 0.45. A name match on a *type word* counts only 0.55 (whether a word is a type word comes from the data: it labels places at least as often as it names them), so for `hospital` a place that IS a hospital beats a clinic or an art museum merely named "Hospital". Words are combined by IDF, so a rare word (`dulce`, `doheny`) dominates and a generic one (`center`, `cafe`) does not. The whole query is also matched run-together (`mc donalds` = McDonald's, `chik fil a` ≈ Chick-fil-A). A word that matches nothing costs 25 %; a query of only such words returns nothing (gibberish gives an empty list).
3. **Meaning** (optional). For "what kind of place" queries (complete words, at least one type or non-name word; never names, codes or prefixes, where dense retrieval is weak), the query is embedded (~6 ms, LRU-cached) and compared with every place's vector. The cosine is fused with the lexical relevance by probabilistic OR, `1 - (1 - lex)(1 - β·sem)`, with β = 0.55 shrinking towards 0.16 as the lexical match gets stronger: meaning adds recall when the words find little (`get my nails done`, `somewhere to sit outside`) and only re-orders when they already found real matches. Reciprocal Rank Fusion (k = 60) was evaluated and rejected: it failed 25 of the 138 judged queries vs 1 for the weighted OR (it ignores score gaps: many lexical ties, and far semantic-only places), for one more held-out paraphrase (9/12 vs 8/12).
4. **Gate**: keep places with relevance ≥ max(0.18, 0.6 × best).
5. **Order**: `score = relevance² × proximity × (1 + 0.3 · popularity)` (0.6 for 1-2 character queries, 1.0 without `near`). Proximity is flat within 150 m and then decays as `1 / (1 + (d - 150 m) / S)` (Pelias-style offset + decay), where `S` adapts to the query: the distance of the 5th nearest strong match, clamped to 400 m - 4 km. So `coffee` behaves like a walking-distance search while `hospital` ranks real hospitals kilometres away by distance. Popularity is capped (Pelias' lesson) so it never overrides relevance or proximity. An exact building code or a distinctive full name (≤ 3 places with that name) is pinned first; campus buildings get a 5 % bump.

Quality is measured by a judged query set (`backend/tests/search_eval/`, 150 queries near USC: typing prefixes, brand and type typos, building codes, cuisines, intents like `cheap lunch` / `late night food` / `fix my phone`, mixed name + street, rooms, gibberish, and 12 held-out paraphrases). `pytest` runs it lexical-only on the real data files; with `SEARCH_EVAL_VECTORS=<file>.npz` (from `python -m scripts.load_places --offline-embed <file>.npz`) it also runs with embeddings. `python -m tests.search_eval --latency [-v] [--semantic <file>.npz]` prints pass rates, failures and a latency table.

Settings: `SEARCH_SEMANTIC` (`auto` = on when fastembed and stored vectors are available, `on`, `off`), `SEARCH_EMBED_MODEL` (the loader and the API must use the same model; the image pre-downloads the default), `SEARCH_EMBED_THREADS` (default 1), `SEARCH_INDEX_TTL` (seconds, default 60), `FASTEMBED_CACHE_PATH`. Memory per gunicorn worker: about 90 MB for the index (incl. 15 MB of vectors) plus about 150 MB once the embedding model is loaded; workers build both lazily, warmed in the background by `post_worker_init`.

Search responses are cached for 30 s (`Cache-Control: public`); categories and place details for 5 minutes.

## Development

```bash
docker compose run --rm --no-deps api pytest -q    # unit tests (no DB needed)
docker compose logs -f ingestor
docker compose exec db psql -U safe -d saferoute
```
