"""Load the searchable `places` table from four sources (bulk upsert by id):

  * Google Maps places  data/usc_google_maps_places.geojson  -> kind=place     id gmaps:<place_id|id>
  * campus buildings    data/places/usc_buildings.geojson    -> kind=building  id bldg:<CODE>
  * campus POIs         data/pois/*.geojson                  -> kind=poi       id poi:<pois.id>
  * classrooms          data/usc/usc_rooms.json              -> kind=room      id room:<CODE>-<ROOM>

    docker compose exec api python -m scripts.load_places [--only place,building,poi] [--all-areas] [--embed auto|yes|no]
    python -m scripts.load_places --offline-embed vectors.npz     # no DB: vectors for the search eval

Re-running is safe: rows are upserted, and rows of a source file that are no longer in it are
deleted. Google places outside the study area (5 mi of USC; the file reaches 10 mi) are skipped
unless --all-areas is given, since routes can't reach them.

Each row also gets a sentence embedding (services/search_semantic.py, fastembed) for semantic
search; unchanged rows reuse their stored vector (embed_key = hash of model + text). The load ends
by bumping search_meta.places_version, which tells every API worker to rebuild its in-memory
search index (services/search_index.py) within SEARCH_INDEX_TTL seconds.
"""
import argparse
import glob
import json
import math
import os
import re
import time

from app.services.opening_hours import parse_hours
from app.services.place_taxonomy import categorize, slug
from app.services.text_norm import normalize

PLACES_FILE = "usc_google_maps_places.geojson"
BUILDINGS_FILE = os.path.join("places", "usc_buildings.geojson")
ROOMS_FILE = os.path.join("usc", "usc_rooms.json")

# Popularity: 0.6 * reach (log-scaled review count) + 0.4 * quality (Bayesian-shrunk rating).
PRIOR_REVIEWS = 20          # a rating needs ~20 reviews before it counts as much as the prior
DEFAULT_MEAN_RATING = 4.3   # used when the file gives nothing to average
REVIEW_REF = 10000          # review count that maps to full reach (log scale)
W_REACH, W_QUALITY = 0.6, 0.4
BUILDING_POPULARITY = 0.8

# POI type (data/pois) -> (category, subcategory, display name when the feature has none)
POI_TYPES = {
    "seating": ("outdoors", "outdoor_seating", "Outdoor seating"),
    "study": ("study", "study_spot", "Study spot"),
}

# Short forms people type for chains, keyed by the normalized base name.
BRAND_ALIASES = {
    "the coffee bean tea leaf": ["Coffee Bean"],
    "in n out burger": ["In-N-Out", "Innout"],
    "7 eleven": ["7-11", "Seven Eleven"],
    "cvs": ["CVS Pharmacy"],
    "cvs pharmacy": ["CVS"],
    "walgreens pharmacy": ["Walgreens"],
    "trader joes": ["TJs", "Trader Joe"],
    "mcdonalds": ["McDonald"],
    "starbucks coffee company": ["Starbucks"],
    "chick fil a": ["Chickfila"],
    "the habit burger grill": ["The Habit", "Habit Burger"],
    "wells fargo bank": ["Wells Fargo"],
    "chase bank": ["Chase"],
    "bank of america financial center": ["Bank of America", "BofA"],
}

_BRANCH = re.compile(r"\s+[-–—|@·]\s+.*$|\s*\([^()]*\)\s*$|\s*:\s+.*$")
_HYPHEN_IN_WORD = re.compile(r"(?<=\w)-(?=\w)")


# ---- pure transforms (unit-tested; no DB, no app context) -------------------------------------

def name_aliases(name: str) -> list[str]:
    """Search aliases derived from a place name: the name without a trailing branch suffix
    ('Starbucks - Figueroa' -> 'Starbucks'), '&' <-> 'and', without a leading 'The', hyphens
    closed up ('Chick-fil-A' -> 'Chickfila'), plus known brand short forms. Excludes the name itself."""
    name = (name or "").strip()
    cands = []
    base = name
    for _ in range(2):  # "X - Y (Z)" -> "X"
        stripped = _BRANCH.sub("", base).strip()
        if len(stripped) >= 2 and stripped != base:
            base = stripped
    for v in dict.fromkeys((name, base)):  # ordered: aliases must be deterministic
        cands.append(v)
        if "&" in v:
            cands.append(re.sub(r"\s*&\s*", " and ", v))
        if re.search(r"\band\b", v, re.I):
            cands.append(re.sub(r"\s+and\s+", " & ", v, flags=re.I))
    for v in list(cands):
        if re.match(r"the\s+", v, re.I) and len(v) > 8:
            cands.append(re.sub(r"^the\s+", "", v, flags=re.I))
        if _HYPHEN_IN_WORD.search(v):
            cands.append(_HYPHEN_IN_WORD.sub("", v))
    for v in list(cands):
        cands += BRAND_ALIASES.get(normalize(v), [])
    return dedupe_aliases(name, cands)


def dedupe_aliases(name: str, aliases) -> list[str]:
    """Keep aliases whose normalized form is new (not the name's, not an earlier alias's)."""
    seen, out = {normalize(name)}, []
    for a in aliases:
        a = (a or "").strip()
        n = normalize(a)
        if n and n not in seen:
            seen.add(n)
            out.append(a)
    return out


def search_text(name_norm: str, aliases) -> str:
    """name_norm + normalized aliases, single-space separated, each distinct form once."""
    parts = [name_norm]
    for a in aliases:
        n = normalize(a)
        if n and n not in parts:
            parts.append(n)
    return " ".join(p for p in parts if p)


def popularity(rating, review_count, mean_rating=DEFAULT_MEAN_RATING) -> float:
    """0..1 prominence. The rating is shrunk toward `mean_rating` with a prior of PRIOR_REVIEWS
    reviews, so a 5.0 from 3 reviews ranks below a 4.5 from 800."""
    n = max(int(review_count or 0), 0)
    r = float(rating) if rating else None
    reach = min(math.log1p(n) / math.log1p(REVIEW_REF), 1.0)
    shrunk = mean_rating if r is None else (PRIOR_REVIEWS * mean_rating + n * r) / (PRIOR_REVIEWS + n)
    quality = min(max((shrunk - 1.0) / 4.0, 0.0), 1.0)
    return round(W_REACH * reach + W_QUALITY * quality, 4)


def mean_rating(features) -> float:
    rs = [float(f["properties"]["rating"]) for f in features
          if (f.get("properties") or {}).get("rating") and (f["properties"].get("review_count") or 0) > 0]
    return round(sum(rs) / len(rs), 3) if rs else DEFAULT_MEAN_RATING


def _point(feat):
    g = feat.get("geometry") or {}
    if g.get("type") != "Point" or not g.get("coordinates"):
        raise ValueError(f"feature {(feat.get('properties') or {}).get('id')}: not a Point")
    lng, lat = g["coordinates"][:2]
    return float(lat), float(lng)


def _row(**kw):
    kw["name_norm"] = normalize(kw["name"])
    kw["search_text"] = search_text(kw["name_norm"], kw["aliases"])
    kw.setdefault("hours", None)
    kw.setdefault("embedding", None)     # filled by attach_embeddings (float32 bytes)
    kw.setdefault("embed_key", None)
    kw["detail"] = {k: v for k, v in (kw.get("detail") or {}).items() if v not in (None, "", [], {})}
    return kw


def google_place_row(feat, source_file=PLACES_FILE, mean=DEFAULT_MEAN_RATING) -> dict:
    p = feat.get("properties") or {}
    lat, lng = _point(feat)
    name = (p.get("name") or "").strip()
    if not name:
        raise ValueError(f"place {p.get('id')}: missing name")
    labels = [str(x) for x in (p.get("categories") or []) if x]
    category, sub = categorize(labels)
    n = int(p.get("review_count") or 0)
    rating = float(p["rating"]) if p.get("rating") else None
    return _row(
        id=f"gmaps:{p.get('place_id') or p['id']}",
        kind="place",
        name=name,
        aliases=name_aliases(name),
        category=category,
        subcategory=sub,
        categories=labels,
        address=p.get("address"),
        lat=lat, lng=lng,
        rating=rating,
        review_count=n,
        popularity=popularity(rating, n, mean),
        hours=parse_hours(p.get("open_hours")),
        detail={
            "phone": p.get("phone"), "website": p.get("website"), "photo_url": p.get("photo_url"),
            "rating_breakdown": p.get("rating_breakdown"), "attributes": p.get("attributes"),
            "description": p.get("description"), "reviews": p.get("reviews"), "plus_code": p.get("plus_code"),
            "hours_text": p.get("open_hours"), "tier": p.get("detail"), "source_id": p.get("id"),
            "place_id": p.get("place_id"),
        },
        source="google-maps",
        source_file=os.path.basename(source_file),
    )


def building_row(feat, source_file=BUILDINGS_FILE) -> dict:
    p = feat.get("properties") or {}
    lat, lng = _point(feat)
    code = str(p["code"]).strip().upper()
    aliases = [code] + [a for a in (p.get("aliases") or []) if a]
    return _row(
        id=f"bldg:{code}",
        kind="building",
        name=p["name"],
        aliases=dedupe_aliases(p["name"], aliases),
        category=p.get("category") or "campus",
        subcategory=p.get("subcategory") or "university_building",
        categories=[],
        address=p.get("address"),
        lat=lat, lng=lng,
        rating=None, review_count=0,
        popularity=BUILDING_POPULARITY,
        detail={"code": code, "entrances": p.get("entrances"), "usc_type": p.get("usc_type")},
        source=p.get("source") or "usc-fpm-gis",
        source_file=os.path.basename(source_file),
    )


def poi_row(r: dict) -> dict:
    """A row from services.pois.parse_feature_collection -> a places row."""
    from shapely.geometry import shape

    props = json.loads(r["props"]) if isinstance(r.get("props"), str) else (r.get("props") or {})
    cat, sub, default_name = POI_TYPES.get(r["type"], (None, slug(r["type"]), None))
    if cat is None:
        cat, _ = categorize([str(r["type"]).replace("_", " ")])
    pt = shape(json.loads(r["geom"]) if isinstance(r["geom"], str) else r["geom"]).representative_point()
    name = r.get("name") or default_name or str(r["type"]).replace("_", " ").capitalize()
    aliases = ["Outdoor tables", "Outdoor seating"] if r["type"] == "seating" else []
    return _row(
        id=f"poi:{r['id']}",
        kind="poi",
        name=name,
        aliases=dedupe_aliases(name, aliases),
        category=cat,
        subcategory=sub,
        categories=[],
        address=None,
        lat=pt.y, lng=pt.x,
        rating=None, review_count=0,
        popularity=0.3,
        detail={"poi_type": r["type"], **props},
        source=r.get("source"),
        source_file=r["source_file"],
    )


# ---- DB ---------------------------------------------------------------------------------------

COLS = ["id", "kind", "name", "name_norm", "aliases", "search_text", "category", "subcategory", "categories",
        "address", "lat", "lng", "rating", "review_count", "popularity", "hours", "detail", "source",
        "source_file", "embedding", "embed_key"]
TEMPLATE = ("(%(id)s, %(kind)s, %(name)s, %(name_norm)s, %(aliases)s::text[], %(search_text)s, %(category)s, "
            "%(subcategory)s, %(categories)s::text[], %(address)s, %(lat)s, %(lng)s, %(rating)s, %(review_count)s, "
            "%(popularity)s, %(hours)s::jsonb, %(detail)s::jsonb, %(source)s, %(source_file)s, "
            "%(embedding)s::bytea, %(embed_key)s)")
UPDATE_COLS = [c for c in COLS if c != "id"]
MERGE = f"""
INSERT INTO places ({", ".join(COLS)}, geom)
SELECT {", ".join(COLS)}, ST_SetSRID(ST_MakePoint(lng, lat), 4326) FROM _places_in
ON CONFLICT (id) DO UPDATE SET {", ".join(f"{c} = EXCLUDED.{c}" for c in UPDATE_COLS)}, geom = EXCLUDED.geom
"""


def upsert(conn, rows, kind, source_files):
    """Bulk upsert `rows` (execute_values into a temp table, then one INSERT ... ON CONFLICT), and
    delete rows of `kind` from `source_files` that are no longer present."""
    from psycopg2.extras import Json, execute_values

    with conn.cursor() as cur:
        # Same column types as places, no constraints (geom is filled in by MERGE from lat/lng).
        cur.execute("DROP TABLE IF EXISTS _places_in")
        cur.execute(f"CREATE TEMP TABLE _places_in AS SELECT {', '.join(COLS)} FROM places WITH NO DATA")
        data = [{**r, "hours": Json(r["hours"]) if r["hours"] is not None else None, "detail": Json(r["detail"]),
                 "embedding": r.get("embedding"), "embed_key": r.get("embed_key")}
                for r in rows]
        execute_values(cur, f"INSERT INTO _places_in ({', '.join(COLS)}) VALUES %s", data,
                       template=TEMPLATE, page_size=2000)
        cur.execute(MERGE)
        cur.execute("DELETE FROM places p WHERE p.kind = %s AND p.source_file = ANY(%s) "
                    "AND NOT EXISTS (SELECT 1 FROM _places_in i WHERE i.id = p.id)", (kind, list(source_files)))
        deleted = cur.rowcount
        cur.execute("DROP TABLE _places_in")
    return deleted


def _read(path):
    with open(path, encoding="utf-8") as f:   # explicit: the Windows default (cp1252) garbles names
        return json.load(f)


def _dedupe(rows):
    out = {}
    for r in rows:
        out.setdefault(r["id"], r)
    return list(out.values())


ROOM_DETAIL_KEYS = ("code", "room", "floor", "floor_label", "floor_confidence", "building_id", "building_name",
                    "campus", "pathing", "capacity", "classroom_type", "furniture", "equipment", "url")


def room_row(r, source_file=ROOMS_FILE) -> dict:
    """A classroom from data/usc/usc_rooms.json -> a places row (kind=room, at its building)."""
    code, room = str(r["code"]).upper(), str(r["room"]).upper()
    bname = r.get("building_name") or code
    aliases = [f"{code}{room}", f"{code}-{room}", f"{bname} {room}", f"Room {room}"]
    return _row(
        id=r.get("id") or f"room:{code}-{room}",
        kind="room",
        name=f"{code} {room}",
        aliases=dedupe_aliases(f"{code} {room}", aliases),
        category="campus",
        subcategory="classroom",
        categories=[],
        address=None,
        lat=float(r["lat"]), lng=float(r["lng"]),
        rating=None, review_count=0,
        popularity=0.3,
        detail={k: r.get(k) for k in ROOM_DETAIL_KEYS},
        source="usc-roomfinder",
        source_file=os.path.basename(source_file),
    )


def annotate_buildings(building_rows, rooms_doc):
    """Floor facts for the buildings Room Finder covers (floor_count, has_basement, campus): used by
    search to place synthesized rooms ("rth 108") on a floor."""
    info = (rooms_doc or {}).get("buildings") or {}
    for r in building_rows:
        b = info.get(r["detail"]["code"])
        if b:
            r["detail"].update({k: b.get(k) for k in ("floor_count", "has_basement", "campus") if b.get(k) is not None})


def collect_rows(data_dir, kinds=("place", "building", "poi", "room"), all_areas=False, places_path=None,
                 buildings_path=None, log=print):
    """-> [(kind, rows, source_files)] from the data files. Needs an app context (study area, H3).
    Used by the DB load below and by the offline search eval (tests/search_eval), so both index
    exactly the same rows."""
    from app.services.h3_utils import in_study_area
    from app.services.pois import parse_feature_collection

    batches = []
    rooms_path = os.path.join(data_dir, ROOMS_FILE)
    rooms_doc = _read(rooms_path) if os.path.exists(rooms_path) else None
    if "place" in kinds:
        path = places_path or os.path.join(data_dir, PLACES_FILE)
        if os.path.exists(path):
            feats = _read(path)["features"]
            mean = mean_rating(feats)
            rows = [google_place_row(f, path, mean) for f in feats]
            kept = rows if all_areas else [r for r in rows if in_study_area(r["lat"], r["lng"])]
            batches.append(("place", _dedupe(kept), [os.path.basename(path)]))
            log(f"{os.path.basename(path)}: {len(kept)} places ({len(rows) - len(kept)} outside the study "
                f"area skipped; mean rating {mean})")
        else:
            log(f"no Google places file at {path} (see data/README.md); skipping")
    if "building" in kinds:
        path = buildings_path or os.path.join(data_dir, BUILDINGS_FILE)
        if os.path.exists(path):
            rows = [building_row(f, path) for f in _read(path)["features"]]
            annotate_buildings(rows, rooms_doc)
            batches.append(("building", _dedupe(rows), [os.path.basename(path)]))
            log(f"{os.path.basename(path)}: {len(rows)} buildings")
        else:
            log(f"no buildings file at {path}; skipping")
    if "room" in kinds:
        if rooms_doc:
            src = rooms_doc.get("rooms") or []
            rows = [room_row(r, rooms_path) for r in src if r.get("lat") is not None and r.get("lng") is not None]
            skipped = [r.get("id") for r in src if r.get("lat") is None or r.get("lng") is None]
            batches.append(("room", _dedupe(rows), [os.path.basename(rooms_path)]))
            log(f"{ROOMS_FILE}: {len(rows)} rooms" + (f" ({len(skipped)} without coordinates skipped: "
                                                      f"{', '.join(map(str, skipped))})" if skipped else ""))
        else:
            log(f"no rooms file at {rooms_path}; skipping")
    if "poi" in kinds:
        files = sorted(glob.glob(os.path.join(data_dir, "pois", "*.geojson")))
        rows = []
        for path in files:
            rows += [poi_row(r) for r in parse_feature_collection(_read(path), path)]
        batches.append(("poi", _dedupe(rows), [os.path.basename(p) for p in files]))
        log(f"pois/: {len(rows)} POIs from {len(files)} file(s)")
    return batches


# ---- embeddings (optional; semantic search) ---------------------------------------------------

def attach_embeddings(rows, existing=None, embedder=None, log=print, batch_size=64):
    """Set row["embedding"] (float32 bytes, L2-normalized) and row["embed_key"] for every row.
    Rows whose embed_key (model + text hash) matches `existing` {id: (embed_key, embedding)} reuse
    the stored vector, so a reload only embeds new or changed places. Returns how many were embedded."""
    from app.services import search_semantic as sem

    embedder = embedder or sem.get_embedder()
    existing = existing or {}
    todo = []
    for r in rows:
        txt = sem.embedding_text(r)
        r["embed_key"] = sem.embed_key(embedder.model_name, txt)
        old = existing.get(r["id"])
        if old and old[0] == r["embed_key"] and old[1]:
            r["embedding"] = bytes(old[1])
        else:
            todo.append((r, txt))
    t0 = time.time()
    for i in range(0, len(todo), 1024):
        chunk = todo[i:i + 1024]
        vecs = embedder.embed_documents([t for _, t in chunk], batch_size=batch_size)
        for (r, _), v in zip(chunk, vecs):
            r["embedding"] = sem.to_bytes(v)
        log(f"  embedded {min(i + 1024, len(todo))}/{len(todo)} ({time.time() - t0:.0f}s)")
    return len(todo)


def offline_embed(out_path, rows, log=print):
    """No DB: embed rows and save {ids, vectors, keys, model} to an .npz (used by the search eval).
    Reuses vectors from an existing file at out_path when their keys match."""
    import numpy as np

    from app.services import search_semantic as sem

    existing = {}
    if os.path.exists(out_path):
        old = np.load(out_path, allow_pickle=False)
        existing = {i: (k, sem.to_bytes(v)) for i, k, v in zip(old["ids"], old["keys"], old["vectors"])}
    n = attach_embeddings(rows, existing, log=log)
    ids = np.array([r["id"] for r in rows])
    vecs = np.stack([sem.from_bytes(r["embedding"]) for r in rows])
    np.savez(out_path, ids=ids, keys=np.array([r["embed_key"] for r in rows]), vectors=vecs,
             model=np.array(sem.get_embedder().model_name))
    log(f"wrote {out_path}: {len(rows)} vectors ({n} newly embedded)")


def write_meta(conn, **values):
    with conn.cursor() as cur:
        for k, v in values.items():
            cur.execute("INSERT INTO search_meta (key, value, updated_at) VALUES (%s, %s, now()) "
                        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()", (k, str(v)))


def existing_embeddings(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, embed_key, embedding FROM places WHERE embedding IS NOT NULL")
        return {i: (k, e) for i, k, e in cur.fetchall()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--places", help=f"Google places GeoJSON (default: $DATA_DIR/{PLACES_FILE})")
    ap.add_argument("--buildings", help=f"campus buildings GeoJSON (default: $DATA_DIR/{BUILDINGS_FILE})")
    ap.add_argument("--only", default="place,building,poi,room", help="comma list of kinds to load")
    ap.add_argument("--all-areas", action="store_true", help="keep Google places outside the study area")
    ap.add_argument("--embed", choices=("auto", "yes", "no"), default="auto",
                    help="compute sentence embeddings for semantic search (auto: when fastembed is installed)")
    ap.add_argument("--offline-embed", metavar="OUT.npz",
                    help="no DB: embed the rows from the data files into an .npz (for the search eval)")
    args = ap.parse_args()
    kinds = {k.strip() for k in args.only.split(",") if k.strip()}

    from sqlalchemy import text

    from app import create_app
    from app.extensions import db
    from app.services import search_semantic as sem

    app = create_app()
    with app.app_context():
        data_dir = app.config["DATA_DIR"]
        t0 = time.time()
        batches = collect_rows(data_dir, kinds, args.all_areas, args.places, args.buildings)
        if args.offline_embed:
            offline_embed(args.offline_embed, [r for _, rows, _ in batches for r in rows])
            return

        conn = db.session.connection().connection  # raw psycopg2 connection, same transaction
        embed = args.embed == "yes" or (args.embed == "auto" and sem.package_available())
        if args.embed == "yes" and not sem.package_available():
            raise SystemExit("--embed yes: fastembed is not installed (pip install fastembed)")
        if embed:
            old = existing_embeddings(conn)
            for kind, rows, _ in batches:
                n = attach_embeddings(rows, old)
                print(f"  {kind}: {n} embedded, {len(rows) - n} reused")
        else:
            print("embeddings skipped (fastembed not installed or --embed no): search runs lexical-only")
            for _, rows, _ in batches:
                for r in rows:
                    r.setdefault("embedding", None)
                    r.setdefault("embed_key", None)
        for kind, rows, files in batches:
            deleted = upsert(conn, rows, kind, files)
            print(f"  {kind}: upserted {len(rows)}, deleted {deleted} stale")
        meta = {"places_version": f"{time.time():.3f}"}   # API workers rebuild their index when this changes
        if embed:
            meta.update(embedding_model=sem.get_embedder().model_name, embedding_dim=sem.get_embedder().dim)
        write_meta(conn, **meta)
        db.session.commit()
        counts = db.session.execute(text(
            "SELECT kind, category, count(*) FROM places GROUP BY 1, 2 ORDER BY 1, 3 DESC")).all()
        print(f"done in {time.time() - t0:.1f}s; places by kind/category:")
        for kind, cat, n in counts:
            print(f"  {kind:9s} {cat:14s} {n}")


if __name__ == "__main__":
    main()
