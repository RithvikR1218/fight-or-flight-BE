"""USC Learning Environments' Room Finder (sites.usc.edu/roomfinder) -> classroom inventory.

    python -m scripts.usc_data rooms

The site is WordPress; every classroom is a post titled "<BLDG> <room>" whose body is a table
(Classroom Type, Capacity, Furniture Type, Whiteboards, Blackboards, TV Monitors, Equipment,
Special Technology, Wallbox Type). We read them all through the public WP REST API
(/wp-json/wp/v2/posts), plus the front page, whose index table is the only place that gives each
room's campus (UPC / HSC). Posts carry no floor or location text, no categories or tags (all are
"Uncategorized") and no custom fields; there is no building-name table on the site.

robots.txt asks for `Crawl-delay: 30`, so we wait 30 s between requests (5 requests, ~2 minutes).

Writes data/usc/roomfinder_rooms.json:
  {"rooms": [{"room_id": "THH 101", "building": "THH", "room": "101", "campus": "UPC", "capacity": 94, ...}],
   "buildings": {"THH": {"campus": "UPC", "room_count": 37, "total_seats": ..., "name": ..., "lat": ..., "lng": ...}}}

Room Finder only lists centrally-scheduled general-use classrooms (220 of them), not every room
in a building (offices, labs, department-owned rooms such as most of RTH aren't there).
`in_index: false` marks a room that has a post but isn't in the front-page table (GFS 06 in Nov 2026).
"""
import argparse
import html
import re
import time
from collections import defaultdict

import requests

from .common import USC, building, get, now_iso, write_json

SITE = "https://sites.usc.edu/roomfinder"
API = f"{SITE}/wp-json/wp/v2"
CRAWL_DELAY = 30            # seconds, from sites.usc.edu/roomfinder/robots.txt
UA = "usc-campus-data-script/1.0 (+research; classroom inventory)"
ROW_RX = re.compile(r"<tr>\s*<td>(.*?)</td>\s*<td>(.*?)</td>\s*</tr>", re.S)
# "THH 101", "THH B09", "MHP B7B", "LVL 3B", "VPD LL101", "GFS 06"
TITLE_RX = re.compile(r"^([A-Z]{2,4})[\s-]+((?:LL|LG|SB|[BG])?\d+[A-Z]?)\b")


def text(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment)
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _int(s: str | None) -> int | None:
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else None


def _na(s: str | None) -> str | None:
    return None if not s or s.strip().upper() in ("N/A", "NA", "NONE") else s.strip()


def parse_room(post: dict) -> dict | None:
    title = text(post["title"]["rendered"]).upper()
    m = TITLE_RX.match(title)
    if not m:
        return None
    table = {text(k).rstrip(":"): text(v) for k, v in ROW_RX.findall(post["content"]["rendered"])}
    return {
        "room_id": f"{m.group(1)} {m.group(2)}",
        "building": m.group(1),
        "room": m.group(2),
        "campus": None,                      # filled from the index table
        "classroom_type": table.get("Classroom Type"),
        "capacity": _int(table.get("Capacity")),
        "furniture": table.get("Furniture Type"),
        "whiteboards": _int(table.get("Whiteboards")),
        "blackboards": _int(table.get("Blackboards")),
        "tv_monitors": _int(table.get("TV Monitors")),
        "equipment": [e.strip() for e in re.split(r"\n+", table.get("Equipment", "")) if e.strip()],
        "special_technology": _na(table.get("Special Technology")),
        "wallbox": _na(table.get("Wallbox Type")),
        "url": post["link"],
        "in_index": False,
    }


def parse_index(page_html: str) -> dict[str, str]:
    """Front-page table: Room | Campus | Room Type | Capacity | Furniture | View Details(link) -> {url: campus}."""
    out = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page_html, re.S):
        cells = [text(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)]
        link = re.search(r'href="([^"]+)"', row)
        if link and len(cells) >= 2:
            out[link.group(1)] = cells[1].strip().upper() or None
    return out


def scrape() -> list[dict]:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json"})
    rooms, page = {}, 1
    while True:
        r = get(s, f"{API}/posts", params={"per_page": 100, "page": page, "_fields": "title,content,link"})
        for post in r.json():
            room = parse_room(post)
            if room:
                rooms.setdefault(room["room_id"], room)   # in case a room gets a duplicate post
            else:
                print(f"  skipped post title {text(post['title']['rendered'])!r}")
        if page >= int(r.headers.get("X-WP-TotalPages", 1)):
            break
        page += 1
        time.sleep(CRAWL_DELAY)
    time.sleep(CRAWL_DELAY)
    pages = get(s, f"{API}/pages", params={"per_page": 100, "_fields": "link,content"}).json()
    front = next((p for p in pages if p["link"].rstrip("/") == SITE), None)
    campus_by_url = parse_index(front["content"]["rendered"]) if front else {}
    for room in rooms.values():
        if room["url"] in campus_by_url:
            room["campus"], room["in_index"] = campus_by_url[room["url"]], True
    # rooms missing from the index inherit their building's campus
    bcampus = {r["building"]: r["campus"] for r in rooms.values() if r["campus"]}
    for room in rooms.values():
        room["campus"] = room["campus"] or bcampus.get(room["building"])
    return sorted(rooms.values(), key=lambda x: (x["building"], x["room"]))


def summarize(rooms: list[dict]) -> dict:
    by_bldg = defaultdict(list)
    for r in rooms:
        by_bldg[r["building"]].append(r)
    out = {}
    for code, rs in sorted(by_bldg.items()):
        campus = rs[0]["campus"]
        # usc_places.json is a UPC-area OSM extract: don't take its match for a code on another campus
        # (Room Finder's DOH is the Doheny Eye building at HSC, not Doheny Library, which is DML).
        b = (building(code) or {}) if campus in (None, "UPC") else {}
        out[code] = {"name": b.get("name"), "campus": campus, "lat": b.get("lat"), "lng": b.get("lng"),
                     "room_count": len(rs), "total_seats": sum(r["capacity"] or 0 for r in rs),
                     "rooms": [r["room"] for r in rs]}
    return out


def main(argv=None):
    argparse.ArgumentParser(prog="usc_data rooms").parse_args(argv)
    rooms = scrape()
    buildings = summarize(rooms)
    write_json(f"{USC}/roomfinder_rooms.json", {"fetched_at": now_iso(), "source": API + "/posts",
                                                 "buildings": buildings, "rooms": rooms})
    no_loc = [c for c, b in buildings.items() if b["lat"] is None]
    print(f"roomfinder_rooms.json: {len(rooms)} rooms in {len(buildings)} buildings"
          + (f"; no coordinates for {no_loc} (usc_rooms.json resolves them from FPM GIS)" if no_loc else ""))
