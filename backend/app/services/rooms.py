"""USC room numbers: which floor a room is on, and spotting "<building code> <room>" in a search query.

Pure Python, no DB. Used by search ("rth 108" -> room 108, Ronald Tutor Hall, floor 1) and by
scripts/usc_data/rooms.py (`python -m scripts.usc_data room-floors`), which builds data/usc/usc_rooms.json.

Floor convention (`floor`, an ordinal level, for sorting and for indoor pathing):
    1, 2, 3 ...  numbered floors as the building labels them (US style: 1 = the street-level first floor)
    0            a level labelled G / GDN (ground / garden) that sits below floor 1
    -1           the first level below that: B (basement), LL (lower level), LG (lower ground), L
    -2           SB (sub-basement)
`floor_label` is what the building calls the level ("1", "B", "LL", ...). A building has either a
G level or a B/LL level below floor 1 in every FPM record we have, so the gap at 0 or -1 is harmless.

Rules, from the evidence in USC FPM GIS (fpmgis.usc.edu, LactationRoom and MenstrualProductDispenser
layers record FLOOR + ROOM for ~170 rooms in ~110 buildings, Nov 2026):
  * NXX (3 digits, optional letter suffix: 108, 201A, 405Z) -> floor N. Every 3-digit UPC/HSC sample
    agrees (RTH 182 -> 1, LVL 303 -> 3, ESH 402 -> 4, VPD 405Z -> 4), except ADM 188 (basement), below.
  * NXXX (4 digits) -> floor N, the first digit: DEN 4104A -> 4, UNH 2331A -> 2, UNH 3137Q -> 3,
    CIC 1111 / CRC 1202 / MHC 1383 / NBC 1180 / SST 1415 -> 1. Not NN: no sample puts a 4-digit room on
    floor 10+. The only counter-example is HSA (a leased Alhambra office building: 6201 on floor 2).
    For a building of 10+ floors, 4 digits could be NN + XX, so the answer is "unknown" there.
  * Prefix letters B / LL / LG / L / G / GDN / SB name the level (THH B09, SOS B2, MHP B7B, VPD LL101,
    ESH LL181A, NRT LG581, GCS SB80A, NTT G430F); the digits after them are not a floor.
  * Anything else (1-2 digit rooms, "3B", 5+ digits) has no general rule -> "unknown", unless the
    building is in ROOM_OVERRIDES.
Confidence: "exact" = a documented per-room fact (ROOM_OVERRIDES); "rule" = the pattern above;
"unknown" = floor None.
"""
from __future__ import annotations

import re

LEVEL_PREFIX = {        # label -> ordinal floor
    "SB": -2,
    "B": -1, "LL": -1, "LG": -1, "L": -1,
    "G": 0, "GDN": 0,
}

# Rooms whose floor the standard pattern gets wrong or can't read, with the evidence.
# code -> {room: floor_label}
ROOM_OVERRIDES: dict[str, dict[str, str]] = {
    # Leavey Library: Room Finder lists exactly one auditorium (LVL 17) and six classrooms (13, 16, 3B,
    # 3C, 3V, 3Y). USC Libraries' Leavey tour guide (2023) puts "the Auditorium ... along with the six
    # fully-equipped Classrooms ... reserved through Classroom Scheduling" on the Lower Level (basement
    # floor plan), and libanswers.usc.edu/faq/24137 says the 6 Classroom Scheduling classrooms are on
    # the Lower Level. Leavey's numbered floors use NXX (LibCal: 1st Floor-113B, 2nd Floor-201A,
    # 3rd Floor-301C; FPM: LVL 111 on 1, LVL 303 on 3), so these are not 1st/3rd-floor rooms.
    "LVL": {r: "LL" for r in ("13", "16", "17", "3B", "3C", "3V", "3Y")},
    # Bovard (ADM): FPM's LactationRoom layer has room 188 on floor B.
    "ADM": {"188": "B"},
    # Law school (LAW): FPM's MenstrualProductDispenser layer has room 13A on floor B.
    "LAW": {"13A": "B"},
}

_ROOM_RX = re.compile(r"^(SB|GDN|LL|LG|L|B|G)?(\d+)([A-Z]{0,2})$")


def _floor_of(label: str) -> int:
    return LEVEL_PREFIX[label] if label in LEVEL_PREFIX else int(label)


def _result(floor_label: str | None, confidence: str) -> dict:
    return {"floor": None if floor_label is None else _floor_of(floor_label),
            "floor_label": floor_label, "confidence": confidence}


def normalize_room(room: str) -> str:
    """'b-09' -> 'B09', ' 201a ' -> '201A'."""
    return re.sub(r"[\s\-.]", "", (room or "").upper())


def parse_room_number(room: str, code: str | None = None, floor_count: int | None = None) -> dict:
    """Floor of a room number. Returns {"floor": int|None, "floor_label": str|None,
    "confidence": "exact"|"rule"|"unknown"} (convention in the module docstring).
    `floor_count` (above-ground floors, e.g. FPM BldgFlrInt) is optional; with it, 4-digit rooms in
    10+-floor buildings, and rule floors above the top floor, come back "unknown"."""
    r = normalize_room(room)
    override = ROOM_OVERRIDES.get((code or "").upper(), {}).get(r)
    if override:
        return _result(override, "exact")
    m = _ROOM_RX.match(r)
    if not m:
        return _result(None, "unknown")
    prefix, digits, _suffix = m.groups()
    if prefix:
        return _result(prefix, "rule")
    if len(digits) == 3 and digits[0] != "0":
        label = digits[0]
    elif len(digits) == 4 and digits[0] != "0" and not (floor_count and floor_count >= 10):
        label = digits[0]
    else:
        return _result(None, "unknown")
    if floor_count and int(label) > floor_count:
        return _result(None, "unknown")
    return _result(label, "rule")


# --- schedule location strings ---------------------------------------------------------------------

# Schedule-of-Classes style location values that are not a room.
# (Anything without a room number is dropped anyway; this list catches the ones that look like one.)
_NON_ROOM_RX = re.compile(r"^(ONLINE.*|TBA.*|TBD|OFFICE.*|OFF ?CAMPUS.*|ARR.*|NONE|N/?A|VARIES|.*@.*|.*VITERBI.*)$")


def split_location(loc: str, known_codes: set[str] | None = None) -> tuple[str, str] | None:
    """'THH101' / 'THH 101' / 'SOSB2' / 'VPDLL101' / 'RTH-105' -> ('THH', '101') ...; None for non-rooms
    ('ONLINE', 'TBA', 'OFFICE', 'DEN@Viterbi', ...). Glued strings are ambiguous ('SOSB2' could be
    SOSB + 2), so with `known_codes` the split whose code is known wins, longest code first."""
    s = re.sub(r"\s+", " ", (loc or "").strip().upper())
    if not s or _NON_ROOM_RX.match(s):
        return None
    codes = {c.upper() for c in known_codes} if known_codes else None
    m = re.match(r"^([A-Z]{2,4})[\s\-]+(\S+)$", s)          # explicit separator: trust it
    if m:
        code, room = m.group(1), normalize_room(m.group(2))
        if re.fullmatch(r"(?:SB|GDN|LL|LG|B|G)?\d{1,4}[A-Z]{0,2}", room) and (codes is None or code in codes):
            return code, room
        return None
    glued = s.replace(" ", "")
    splits = []
    for n in (4, 3, 2):
        code, room = glued[:n], glued[n:]
        if code.isalpha() and re.fullmatch(r"(?:SB|GDN|LL|LG|B|G)?\d{1,4}[A-Z]{0,2}", room):
            splits.append((code, room))
    if codes is not None:
        splits = [x for x in splits if x[0] in codes]
    else:   # no code list: prefer a plain 3-letter code (USC's standard), then the shortest code
        splits.sort(key=lambda x: (len(x[0]) != 3, len(x[0])))
    return splits[0] if splits else None


# --- query parsing ---------------------------------------------------------------------------------

_FILLER = r"(?:room|rm|no|#)?"
# room token: optional level prefix, digits, optional 1-2 letter suffix; or digit(s) + letter ("3B")
_ROOM_TOKEN = r"(?:sb|gdn|ll|lg|b|g)?-?\d{1,5}[a-z]{0,2}"
_SPLIT_RX = re.compile(rf"^([a-z]{{2,4}})\s*[-#.]?\s*{_FILLER}\s*[-#.]?\s*({_ROOM_TOKEN})$")
_REV_RX = re.compile(rf"^{_FILLER}\s*({_ROOM_TOKEN})\s*[-,]?\s*(?:in|at)?\s*([a-z]{{2,4}})$")


# USC building codes that are also English words; "car 54", "gap 1969", "sun 24" are not rooms, so
# for these only a 3-digit room (with optional suffix) or a level-prefixed room counts ("law 101", "lab b12").
WORD_CODES = {"ONE", "SUN", "TAP", "GAP", "CAR", "LAB", "LAW", "WIN", "EGG", "CAP", "COL", "HIL", "MUS",
              "PED", "RAN", "FIG", "BIT", "CEN", "HER", "JEF", "TYL", "VIS", "STU", "TST", "ELB"}


def _valid_room(tok: str, code: str) -> str | None:
    t = normalize_room(tok)
    # at least 2 characters with a digit; a bare single digit ("cvs 2") is too likely to be something else
    if len(t) < 2 or not re.fullmatch(r"(SB|GDN|LL|LG|B|G)?\d{1,4}[A-Z]{0,2}", t):
        return None
    if code in WORD_CODES and not re.fullmatch(r"(?:(SB|GDN|LL|LG|B|G)\d{1,4}|[1-9]\d\d)[A-Z]?", t):
        return None
    return t


def parse_room_query(q: str, known_codes: set[str]) -> dict | None:
    """'rth 108' / 'RTH108' / 'rth-108' / 'thh 201a' / '108 rth' / 'sos b2' / 'lvl 3b' ->
    {"code": "RTH", "room": "108"}; None unless the letters are a known building code and the rest
    is a room number (so '7 eleven', 'route 66', 'studio 54', 'cvs 24' stay ordinary queries)."""
    s = re.sub(r"\s+", " ", (q or "").strip().lower())
    if not s or len(s) > 24:
        return None
    codes = {c.upper() for c in known_codes}
    cands: list[tuple[str, str]] = []
    m = _SPLIT_RX.match(s)
    if m:
        cands.append((m.group(1), m.group(2)))
    m = _REV_RX.match(s)
    if m:
        cands.append((m.group(2), m.group(1)))
    # glued with a level prefix, e.g. "sosb2", "vpdll101": try each code length
    glued = s.replace(" ", "")
    if re.fullmatch(r"[a-z]{3,7}\d{1,5}[a-z]{0,2}", glued):
        for n in (4, 3, 2):
            cands.append((glued[:n], glued[n:]))
    for code, room in cands:
        code = code.upper()
        room = _valid_room(room, code)
        if code in codes and room:
            return {"code": code, "room": room}
    return None
