"""In-memory place search: retrieval and ranking over every searchable place, in numpy.

The corpus is small (~10k places), so each API worker loads the searchable fields once
(services/search.py: lazily, refreshed when the loader bumps search_meta.places_version) and
answers every query here without touching the database. Tests and the offline eval build the
same index straight from the data files (scripts/load_places.collect_rows).

Ranking = relevance gate, then distance-major ordering (Google local search: relevance, distance,
prominence):

1. Query understanding (cached per query): normalize, drop stop words, fold plurals, find street /
   neighbourhood anchors ("coffee near vermont"), expand every word into index terms: exact,
   typo (Damerau/OSA distance, budget 0 for <=4 chars, 1 for 5-8, 2 for 9+, a wrong first letter
   counts as two; Meilisearch's rules), prefix (only the last word, the one being typed),
   -ing forms and term synonyms (place_taxonomy.TERM_SYNONYMS), and type intents
   (place_taxonomy.QUERY_SYNONYMS).
2. Evidence per query word and place, from four fields (BM25F-style field weights):
     name     1.0  (x0.55 when the word is a type word such as "hospital" or "cafe": a place merely
                    NAMED "Hospital of Emotions" is weak evidence for the query "hospital")
     type     1.0  (Google labels, subcategory, synonyms, derived tags such as "late night")
     content  0.5  (BM25 saturation over description, attributes and reviews)
     address  0.45
   Word evidence = best field (+ a little for a second field). Relevance = IDF-weighted mean over
   the query words, so rare words ("dulce", "doheny") decide multi-word queries and a word that
   matches nothing costs coverage. Whole-query compaction ("mc donalds" = "McDonald's",
   "chik fil a" ~ "Chick-fil-A") and an optional semantic score (cosine of sentence embeddings,
   fused by probabilistic OR, weight shrinking as the lexical match gets stronger) can raise it.
   Classrooms ("rth 108") are answered by room_search() before any of this.
3. Gate: keep places with relevance >= max(0.18, 0.6 x best relevance), so a nearby place that is
   only loosely related never shows up when real matches exist.
4. Order: score = relevance^2 x proximity x (1 + capped popularity). Proximity is flat within 150 m
   and then decays as 1 / (1 + (d - 150 m) / S) (Pelias-style offset + decay) where S adapts to the
   query: the distance of the ~5th nearest strong match, clamped to 400 m..4 km. "coffee" (dozens
   within 500 m) feels like a walking-distance search, "hospital" (nearest ones km away) ranks real
   hospitals by distance without letting a nearby literal-name match win. Popularity is capped
   (Pelias' lesson) so it only breaks near-ties when a location is given.
   An exact match on a building code or a distinctive full name (<= 3 places) is pinned first.
"""
from __future__ import annotations

import bisect
import logging
import math
import re
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .text_norm import normalize

log = logging.getLogger(__name__)

try:
    from rapidfuzz import process as _rf_process
    from rapidfuzz.distance import OSA as _OSA
except ImportError:  # pragma: no cover - optional speed-up, pure-Python fallback below
    _rf_process = _OSA = None

EARTH_R = 6371008.8

# Query words with no place meaning. Removed unless the query has nothing else.
STOP_WORDS = frozenset("""
a an and at by for in into of on or the to with from i im me my we our us you your some any
somewhere something someplace anywhere anything where wheres what whats which is are can could get
go find want need looking look lets let like please there here then near nearby around closest
""".split())
# Prepositions that introduce a location ("coffee near vermont").
LOCATION_PREPS = frozenset("near nearby around by on off at in along close closest".split())
# Generic words that only refine ("pizza place", "coffee shop", "good tacos"): low weight.
FILLER = frozenset("place places spot spots shop shops store stores thing things area good best nice great "
                   "open now new top local nearby close closest around near".split())
FILLER_WEIGHT = 0.3
LOCATION_WEIGHT = 0.3
# Very common review words: not indexed as content.
CONTENT_STOP = frozenset("""
a an the and or but if of to in on at for with from by as is are was were be been being am it its it s
this that these those they them their there here we our us you your i me my he she his her him
so very really just also too not no yes do does did done have has had will would can could should
get got go went come came one two all any some more most much many lot lots always never ever
good great nice best love loved amazing awesome excellent delicious friendly staff place places
recommend recommended highly definitely back time times again would like even well only still
what when where which who why how than then out up about over after before because while
""".split())
STREET_TYPES = frozenset("st street ave av avenue blvd boulevard dr drive way pl place rd road pkwy parkway walk "
                         "ln lane ct court ter terrace hwy highway cir circle sq square mall row".split())
DIRECTIONS = frozenset("n s e w north south east west ne nw se sw".split())
ADDRESS_NOISE = frozenset("los angeles ca california usa united states suite ste unit fl floor".split())
# A few neighbourhoods people type ("tacos downtown"): centre and radius in metres.
NEIGHBOURHOODS = {
    "downtown": (34.0430, -118.2510, 1500), "dtla": (34.0430, -118.2510, 1500),
    "koreatown": (34.0610, -118.3005, 1200), "ktown": (34.0610, -118.3005, 1200),
    "little tokyo": (34.0500, -118.2405, 500), "chinatown": (34.0623, -118.2383, 700),
    "arts district": (34.0410, -118.2335, 800), "exposition park": (34.0165, -118.2870, 600),
    "expo park": (34.0165, -118.2870, 600), "university park": (34.0250, -118.2840, 900),
    "west adams": (34.0330, -118.3120, 1200), "leimert park": (34.0090, -118.3320, 900),
    "jefferson park": (34.0270, -118.3320, 700), "pico union": (34.0470, -118.2830, 900),
    "westlake": (34.0580, -118.2790, 1000), "echo park": (34.0782, -118.2606, 1000),
    "fashion district": (34.0380, -118.2520, 700), "south park": (34.0390, -118.2650, 600),
    "historic south central": (34.0200, -118.2560, 1200),
}

# Field weights and match qualities.
W_NAME, W_TYPE, W_CONTENT, W_ADDR = 1.0, 1.0, 0.5, 0.45
TYPE_WORD_NAME_FACTOR = 0.55   # < GATE_REL: a name-only match on a type word never passes the gate
                               # when places of that type exist
Q_FUZZY_STEP = 0.18          # quality lost per typo
Q_SYNONYM = 0.85             # term synonym / -ing form
Q_INTENT, Q_INTENT_PREFIX = 0.9, 0.75
Q_LABEL_PRIMARY, Q_LABEL_OTHER, Q_CATEGORY_WORD, Q_TAG = 1.0, 0.85, 0.6, 0.75
Q_FOLDED_NAME = 0.85         # name matched only after plural folding ("ralphs" vs "Ralph Hall")
Q_BIGRAM = 1.0               # two query words joined match one index term ("late night", "book store")
SECOND_FIELD_BONUS = 0.1
UNKNOWN_WORD_FACTOR = 0.75
BM25_K1, BM25_B = 1.2, 0.75

GATE_ABS, GATE_REL = 0.18, 0.6
OFFSET_M = 150.0
SCALE_MIN_M, SCALE_MAX_M, SCALE_K = 400.0, 4000.0, 5
POP_CAP_NEAR, POP_CAP_SHORT, POP_CAP_NO_NEAR = 0.3, 0.6, 1.0
BUILDING_BUMP = 1.05
PIN_MAX = 3

SEM_BETA = 0.55              # a pure-semantic match tops out at this relevance
SEM_MIN_SHARE = 0.3          # share of SEM_BETA kept when the lexical match is already perfect
SEM_TOP = 200                # semantic evidence for the best-matching places only
QUERY_CACHE = 2048


# ---------------------------------------------------------------------------------- text helpers

def typo_budget(n):
    return 0 if n <= 4 else 1 if n <= 8 else 2


def osa_distance(a, b, max_d=None):
    """Damerau-Levenshtein (optimal string alignment). Stops early past max_d."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if max_d is not None and abs(la - lb) > max_d:
        return max_d + 1
    prev2, prev = None, list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if max_d is not None and min(cur) > max_d:
            return max_d + 1
        prev2, prev = prev, cur
    return prev[lb]


def typo_cost(t, w, d):
    """Meilisearch: a typo on the first letter counts as two."""
    return d + (1 if t[:1] != w[:1] else 0)


def haversine_m(lat1, lng1, lat2, lng2):
    """Metres; lat2/lng2 may be numpy arrays."""
    p1 = math.radians(lat1)
    p2 = np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lng2) - lng1)
    a = np.sin(dp / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def _plural_candidates(t):
    if len(t) <= 3 or not t.isalpha() or not t.endswith("s") or t.endswith(("ss", "us", "is")):
        return []
    out = [t[:-1]]
    if t.endswith("es"):
        out.append(t[:-2])
    if t.endswith("ies") and len(t) > 4:
        out += [t[:-3] + "y", t[:-3] + "ie"]
    return out


class Folder:
    """Plural folding checked against the corpus ("tacos" -> taco, "groceries" -> grocery,
    "cookies" -> cookie, "sandwiches" -> sandwich): of the singular forms a rule allows, the one
    seen most often in the corpus wins. Same function for documents and queries."""

    def __init__(self, counts):
        self.counts = counts
        self._memo = {}

    def __call__(self, t):
        f = self._memo.get(t)
        if f is None:
            cands = _plural_candidates(t)
            if not cands:
                f = t
            else:
                f = max(cands, key=lambda c: (self.counts.get(c, 0), -len(c)))
                if not self.counts.get(f):
                    f = cands[2] if t.endswith("ies") and len(cands) > 2 else cands[0]
            if len(self._memo) < 500000:
                self._memo[t] = f
        return f


def _tokens(text):
    return normalize(text).split()


def _street_key(address):
    """'3335 S Figueroa St, Los Angeles, CA 90007' -> 'figueroa'; None if there is no street."""
    if not address:
        return None
    words = _tokens(str(address).split(",")[0])
    while words and (words[0][:1].isdigit() or words[0] in DIRECTIONS):
        words = words[1:]
    while words and (words[-1] in STREET_TYPES or words[-1] in DIRECTIONS):
        words = words[:-1]
    if not words or len(words) > 4 or all(w.isdigit() for w in words):
        return None
    return " ".join(words)


def hours_tags(hours):
    """Derived type tags from opening hours: open past 23:30 (or overnight) on 3+ days -> late night;
    round the clock on 5+ days -> 24 hours."""
    if not hours:
        return []
    late = allday = 0
    for spans in hours.values():
        spans = spans or []
        if any(c >= 1410 or c - o >= 1440 for o, c in spans):
            late += 1
        if any(o <= 0 and c - o >= 1440 for o, c in spans):
            allday += 1
    tags = []
    if late >= 3:
        tags += ["latenight", "overnight"]     # matched by the query bigram "late night"
    if allday >= 5:
        tags += ["24h", "24hour", "24hours", "allday", "allnight"]
    return tags


def attribute_tags(detail):
    """Type tags from Google attributes that answer common needs ("bathroom", "wifi")."""
    tags = []
    for o in _attr_options(detail):
        low = o.lower()
        if "restroom" in low:
            tags += ["restroom", "bathroom"]
        if low in ("wi-fi", "free wi-fi"):
            tags += ["wifi"]
    return list(dict.fromkeys(tags))


def _attr_options(detail):
    out = []
    for g in (detail or {}).get("attributes") or []:
        if isinstance(g, dict):
            out += [o for o in g.get("options") or [] if isinstance(o, str)]
        elif isinstance(g, str):
            out.append(g)
    return out


def content_text(row):
    """Description, attribute options, review texts and free-text POI properties."""
    d = row.get("detail") or {}
    parts = []
    for k in ("description", "layer", "group", "library", "room", "notes"):
        if isinstance(d.get(k), str):
            parts.append(d[k])
    parts += _attr_options(d)
    for rv in d.get("reviews") or []:
        t = rv.get("text") if isinstance(rv, dict) else rv
        if isinstance(t, str):
            parts.append(t)
    return " ".join(parts)


# ---------------------------------------------------------------------------------- query model

@dataclass
class Term:
    raw: str
    folded: str
    weight: float = 1.0
    last: bool = False
    filler: bool = False
    location: bool = False
    known: bool = False
    complete: bool = False           # an exact or corrected whole word (not only a prefix)
    expansions: list = field(default_factory=list)   # (term, quality, kind) kind: exact|fold|fuzzy|prefix|syn
    bigrams: list = field(default_factory=list)      # this word + a neighbour = one index term
    best: str = ""


@dataclass
class ParsedQuery:
    norm: str
    tokens: list
    terms: list                       # non-stop Terms (location terms included, flagged)
    intents: list                     # (term indices, intents, quality)
    compact: str
    generic: bool                     # every word is a type word ("hospital", "ice cream")
    pinned: np.ndarray | None
    anchor: tuple | None              # ("street", key) | ("hood", (lat, lng, r))
    short: bool
    semantic_text: str | None


class LRU:
    def __init__(self, n):
        self.n, self.d, self.lock = n, OrderedDict(), threading.Lock()

    def get(self, k):
        with self.lock:
            v = self.d.get(k)
            if v is not None:
                self.d.move_to_end(k)
            return v

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            if len(self.d) > self.n:
                self.d.popitem(last=False)


def _to_postings(d):
    """{term: {doc: quality}} -> {term: (int32 ids, float32 qualities)}"""
    out = {}
    for t, m in d.items():
        ids = np.fromiter(m.keys(), dtype=np.int32, count=len(m))
        q = np.fromiter(m.values(), dtype=np.float32, count=len(m))
        out[t] = (ids, q)
    return out


def _put(d, term, doc, q):
    m = d[term]
    if m.get(doc, 0.0) < q:
        m[doc] = q


RESULT_FIELDS = ("id", "kind", "name", "category", "subcategory", "address", "lat", "lng",
                 "rating", "review_count")


# ---------------------------------------------------------------------------------- the index

class SearchIndex:
    """Immutable after construction; safe to share between threads."""

    def __init__(self, rows, vectors=None, semantic_model=None):
        """rows: dicts shaped like scripts/load_places rows (id, kind, name, aliases, category,
        subcategory, categories, address, lat, lng, rating, review_count, popularity, hours, detail
        and optionally embedding bytes). vectors: optional {id: float32 vector} overriding embeddings."""
        from .place_taxonomy import CATEGORIES, QUERY_SYNONYMS, TERM_SYNONYMS, slug

        rows = [r for r in rows if r.get("name") and r.get("lat") is not None and r.get("lng") is not None]
        self.N = N = len(rows)
        self.docs = [{k: r.get(k) for k in RESULT_FIELDS} for r in rows]
        self.hours = [r.get("hours") for r in rows]
        self.labels = [list(r.get("categories") or []) for r in rows]
        self.lat = np.array([float(r["lat"]) for r in rows], dtype=np.float64)
        self.lng = np.array([float(r["lng"]) for r in rows], dtype=np.float64)
        self.pop = np.array([float(r.get("popularity") or 0.0) for r in rows], dtype=np.float32)
        self.is_building = np.array([r.get("kind") == "building" for r in rows])
        cats = sorted({r.get("category") or "other" for r in rows})
        self.cat_code = {c: i for i, c in enumerate(cats)}
        self.cat = np.array([self.cat_code[r.get("category") or "other"] for r in rows], dtype=np.int16)
        self.id_pos = {r["id"]: i for i, r in enumerate(rows)}
        cat_label = {c["id"]: c["label"] for c in CATEGORIES}

        # ---- pass 1: raw word counts (drive plural folding) + per-doc texts
        counts = defaultdict(int)
        name_strings, contents = [], []
        for r in rows:
            ns = [n for n in dict.fromkeys(normalize(x) for x in [r["name"]] + list(r.get("aliases") or [])) if n]
            name_strings.append(ns)
            c = normalize(content_text(r))
            contents.append(c)
            for s in ns:
                for w in s.split():
                    counts[w] += 1
            for lab in r.get("categories") or []:
                for w in _tokens(lab):
                    counts[w] += 1
            for w in c.split():
                counts[w] += 1
        self.fold = fold = Folder(dict(counts))

        name_p, type_p, addr_p = defaultdict(dict), defaultdict(dict), defaultdict(dict)
        prefix_p = defaultdict(dict)                 # 1..3-char prefixes of name words
        content_tf = defaultdict(dict)
        dl = np.zeros(N, dtype=np.float32)
        exact_full, compact = defaultdict(set), defaultdict(set)
        slug_docs = defaultdict(set)
        streets = defaultdict(list)

        for i, r in enumerate(rows):
            # name
            for s in name_strings[i]:
                exact_full[s].add(i)
                compact[s.replace(" ", "")].add(i)
                words = s.split()
                for j, w in enumerate(words):
                    _put(name_p, w, i, 1.0)
                    _put(name_p, fold(w), i, Q_FOLDED_NAME)      # "Ralphs" is not quite "Ralph"
                    for k in range(1, 4):
                        if len(w) >= k:
                            _put(prefix_p, w[:k], i, (1.0 if j == 0 else 0.9) * (0.7 + 0.3 * k / len(w)))
                    if j + 1 < len(words) and min(len(w), len(words[j + 1])) <= 3:
                        joined = w + words[j + 1]
                        _put(name_p, joined, i, 1.0)
                        _put(name_p, fold(joined), i, Q_FOLDED_NAME)
            # type: labels, subcategory, category words, POI type, derived tags
            d = r.get("detail") or {}
            labels = [x for x in (r.get("categories") or []) if x]
            for j, lab in enumerate(labels):
                q = Q_LABEL_PRIMARY if j == 0 else Q_LABEL_OTHER
                slug_docs[slug(lab)].add(i)
                words = [fold(w) for w in _tokens(lab)]
                for w in words:
                    _put(type_p, w, i, q)
                if len(words) > 1:
                    raw = _tokens(lab)
                    _put(type_p, "".join(raw), i, q)                   # "Book store" -> bookstore
                    for a, b in zip(raw, raw[1:]):                     # "Ice cream shop" -> icecream
                        _put(type_p, a + b, i, q)
            extra = [r.get("subcategory"), d.get("poi_type")]
            for sl in extra:
                if sl:
                    slug_docs[sl].add(i)
                    q = Q_LABEL_PRIMARY if not labels else Q_LABEL_OTHER
                    for w in _tokens(sl.replace("_", " ")):
                        _put(type_p, fold(w), i, q if r.get("kind") != "building" else Q_CATEGORY_WORD)
            cat = r.get("category") or "other"
            for w in _tokens(cat) + _tokens(cat_label.get(cat, "")):
                _put(type_p, fold(w), i, Q_CATEGORY_WORD)
            if r.get("kind") == "building":
                for w in ("campus", "usc", "building"):
                    _put(type_p, w, i, Q_CATEGORY_WORD)
            for w in hours_tags(r.get("hours")) + attribute_tags(d):
                _put(type_p, w, i, Q_TAG)
            # address + street
            for w in _tokens(r.get("address")):
                if w not in ADDRESS_NOISE and not w.isdigit():
                    _put(addr_p, fold(w), i, 1.0)
            sk = _street_key(r.get("address"))
            if sk:
                streets[sk].append(i)
            # content
            toks = [fold(w) for w in contents[i].split() if w not in CONTENT_STOP and len(w) > 1]
            dl[i] = len(toks)
            tf = defaultdict(int)
            for w in toks:
                tf[w] += 1
            for w, n in tf.items():
                content_tf[w][i] = n

        # BM25 saturation in [0, 1): tf (k1 + 1) / (tf + k1 (1 - b + b dl / avgdl)) / (k1 + 1)
        avgdl = float(dl[dl > 0].mean()) if (dl > 0).any() else 1.0
        norm_dl = BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
        content_p = {}
        for w, m in content_tf.items():
            ids = np.fromiter(m.keys(), dtype=np.int32, count=len(m))
            tf = np.fromiter(m.values(), dtype=np.float32, count=len(m))
            content_p[w] = (ids, (tf / (tf + norm_dl[ids])).astype(np.float32))

        self.name_p = _to_postings(name_p)
        self.type_p = _to_postings(type_p)
        self.addr_p = _to_postings(addr_p)
        self.prefix_p = _to_postings(prefix_p)
        self.content_p = content_p
        self.exact_full = {k: np.array(sorted(v), dtype=np.int32) for k, v in exact_full.items()}
        self.compact = {k: np.array(sorted(v), dtype=np.int32) for k, v in compact.items() if len(k) >= 3}
        self.compact_keys = sorted(self.compact)
        self.name_keys = sorted(self.name_p)
        self.type_keys = sorted(self.type_p)
        self.addr_keys = sorted(self.addr_p)
        self.slug_docs = {k: np.array(sorted(v), dtype=np.int32) for k, v in slug_docs.items()}
        self.streets = {k: np.array(v, dtype=np.int32) for k, v in streets.items() if len(v) >= 3}

        # document frequencies (any field) -> IDF query weights
        df = defaultdict(int)
        for p in (self.name_p, self.type_p, self.addr_p, self.content_p):
            for t, (ids, _) in p.items():
                df[t] = max(df[t], len(ids))
        self.df = dict(df)
        self.name_df = {t: len(v[0]) for t, v in self.name_p.items()}
        self.type_df = {t: len(v[0]) for t, v in self.type_p.items()}

        # synonyms, folded to index terms
        self.term_syn = {fold(k): [fold(s) for s in v] for k, v in TERM_SYNONYMS.items()}
        self.intents = {}
        for key, intents in QUERY_SYNONYMS.items():
            self.intents.setdefault(tuple(fold(w) for w in key.split()), []).extend(intents)
        self.intent_keys = sorted(" ".join(k) for k in self.intents)
        single_syn = {k[0] for k in self.intents if len(k) == 1}
        # "type words": mostly used as a type, not as a distinctive name ("cafe", "hospital", "ice")
        self.type_words = {t for t, n in self.type_df.items()
                           if n >= 3 and n >= 0.4 * self.name_df.get(t, 0)} | single_syn
        self._intent_cache = {}

        # fuzzy vocabulary: names, types, addresses, and content words seen in 4+ places
        vocab = set(self.name_p) | set(self.type_p) | {t for t, (ids, _) in self.addr_p.items() if len(ids) >= 3}
        vocab |= {t for t, (ids, _) in self.content_p.items() if len(ids) >= 4}
        vocab |= set(self.term_syn) | {w for k in self.intents for w in k}
        self.vocab = sorted(t for t in vocab if len(t) >= 4 and t.isalpha() or (len(t) >= 4 and t.isalnum()))
        self.complete_vocab = set(self.name_p) | set(self.type_p) | set(self.addr_p) | set(self.content_p)

        # semantic vectors
        self.vectors = None
        self.semantic_model = semantic_model
        vecs = []
        for r in rows:
            v = (vectors or {}).get(r["id"]) if vectors is not None else None
            if v is None and r.get("embedding"):
                v = np.frombuffer(bytes(r["embedding"]), dtype="<f4")
            vecs.append(v)
        dims = {len(v) for v in vecs if v is not None}
        if len(dims) == 1:
            dim = dims.pop()
            m = np.zeros((N, dim), dtype=np.float32)
            has = np.zeros(N, dtype=bool)
            for i, v in enumerate(vecs):
                if v is not None:
                    m[i] = v
                    has[i] = True
            if has.mean() >= 0.5:
                n = np.linalg.norm(m, axis=1, keepdims=True)
                self.vectors = m / np.maximum(n, 1e-12)
                self.has_vec = has
        self._qcache = LRU(QUERY_CACHE)
        self._fcache = LRU(QUERY_CACHE * 4)
        _ = self.content_keys, self.intents_words      # build lazily-derived lists now, not on a request

        # classrooms: only reachable through room queries ("rth 108", "thh 1", "taper 201")
        self.is_room = np.array([r.get("kind") == "room" for r in rows])
        self.building_by_code, self.rooms_by_code, self.room_info = {}, defaultdict(list), {}
        self.building_info = {}
        for i, r in enumerate(rows):
            d = r.get("detail") or {}
            if r.get("kind") == "building" and d.get("code"):
                code = str(d["code"]).upper()
                self.building_by_code[code] = i
                self.building_info[code] = {"building_id": r["id"], "building_name": r["name"], "lat": r["lat"],
                                            "lng": r["lng"], "address": r.get("address"),
                                            "campus": d.get("campus") or "UPC", "floor_count": d.get("floor_count")}
            elif r.get("kind") == "room" and d.get("code"):
                code = str(d["code"]).upper()
                self.rooms_by_code[code].append((str(d.get("room") or "").upper(), i))
                self.room_info[i] = d
        for code, rooms in self.rooms_by_code.items():
            rooms.sort(key=lambda x: _room_sort_key(x[0]))
            if code not in self.building_info:      # e.g. HSC buildings, not in the campus buildings file
                d = self.room_info[rooms[0][1]]
                self.building_info[code] = {"building_id": d.get("building_id") or f"bldg:{code}",
                                            "building_name": d.get("building_name") or code,
                                            "lat": self.lat[rooms[0][1]], "lng": self.lng[rooms[0][1]],
                                            "address": None, "campus": d.get("campus"), "floor_count": None}
        self.known_codes = set(self.building_info)

    # ------------------------------------------------------------------ memory estimate
    def nbytes(self):
        tot = self.lat.nbytes * 2 + self.pop.nbytes + self.cat.nbytes
        for p in (self.name_p, self.type_p, self.addr_p, self.content_p, self.prefix_p):
            tot += sum(a.nbytes + b.nbytes + 200 for a, b in p.values())
        tot += sum(v.nbytes + 100 for v in self.exact_full.values())
        tot += sum(v.nbytes + 100 for v in self.compact.values())
        if self.vectors is not None:
            tot += self.vectors.nbytes
        return tot

    # ------------------------------------------------------------------ query understanding
    def _fuzzy(self, t):
        """[(vocab term, typo cost)] within t's budget (first-letter typo counts double)."""
        budget = typo_budget(len(t))
        if not budget:
            return []
        hit = self._fcache.get(t)
        if hit is not None:
            return hit
        out = []
        if _rf_process is not None:
            for w, d, _ in _rf_process.extract(t, self.vocab, scorer=_OSA.distance, score_cutoff=budget, limit=None):
                c = typo_cost(t, w, int(d))
                if 0 < c <= budget:
                    out.append((w, c))
        else:
            for w in self.vocab:
                if abs(len(w) - len(t)) <= budget:
                    d = osa_distance(t, w, budget)
                    c = typo_cost(t, w, d)
                    if 0 < c <= budget:
                        out.append((w, c))
        # the most common spellings first (a typo usually means the common word)
        out.sort(key=lambda x: (x[1], -self.df.get(x[0], 0)))
        out = out[:12]
        self._fcache.put(t, out)
        return out

    def _prefix_terms(self, keys, t, cap=60):
        i = bisect.bisect_left(keys, t)
        out = []
        while i < len(keys) and keys[i].startswith(t):
            if keys[i] != t:
                out.append(keys[i])
            i += 1
        if len(out) > cap:
            out.sort(key=lambda w: -self.df.get(w, 0))
            out = out[:cap]
        return out

    def _expand(self, term: Term):
        t, f = term.raw, term.folded
        exp = {}

        def add(w, q, kind):
            if q > exp.get(w, (0, ""))[0]:
                exp[w] = (q, kind)

        if t in self.complete_vocab:
            add(t, 1.0, "exact")
        if f != t and f in self.complete_vocab:
            add(f, 1.0, "fold")          # full quality for types/content, Q_FOLDED_NAME for names
        exact_strong = any(w in self.name_p or w in self.type_p for w in (t, f))
        # -ing forms: "printing" -> print, "studying" -> study
        if t.endswith("ing") and len(t) >= 6 and t not in self.type_p:
            base = t[:-3]
            for w in (base, base + "e", base[:-1] if len(base) > 2 and base[-1] == base[-2] else None,
                      base[:-1] + "y" if base.endswith("i") else None):
                if w and (w in self.type_p or w in self.name_p or w in self.intents_words):
                    add(w, Q_SYNONYM + 0.05, "syn")
        # prefix: only the word being typed. Discounted when that word is already a complete
        # type word ("pho" -> Vietnamese places, not "Photo studio").
        has_prefix = False
        if term.last and not self._typed_type_word(term):
            pre_q = 1.0
            for w in self._prefix_terms(self.name_keys, t, cap=200 if len(t) >= 4 else 0):
                add(w, pre_q * (0.75 + 0.25 * len(t) / len(w)), "prefix")
                has_prefix = True
            if len(t) >= 3:
                for keys in (self.type_keys, self.addr_keys):
                    for w in self._prefix_terms(keys, t, cap=40):
                        add(w, pre_q * 0.85 * (0.5 + 0.5 * len(t) / len(w)), "prefix")
                        has_prefix = True
                # complete the word to a likely content term too ("late night foo")
                for w in self._prefix_terms(self.content_keys, t, cap=10) if len(t) >= 4 else ():
                    add(w, pre_q * 0.8 * (0.5 + 0.5 * len(t) / len(w)), "prefix")
            if len(t) <= 3 and self._prefix_terms(self.name_keys, t, cap=1):
                has_prefix = True
        # typos: only when the word is not already a name or type word; a word still being typed
        # that is a prefix of real words is more likely unfinished than misspelt
        if not exact_strong and len(t) >= 5:
            fq = 0.8 if has_prefix else 1.0
            for w, c in self._fuzzy(f if len(f) >= 5 else t):
                add(w, fq * (1.0 - Q_FUZZY_STEP * c), "fuzzy")
        # synonyms of the word and of its best correction
        base_words = [w for w, (q, k) in exp.items() if k in ("exact", "fold", "syn") or (k == "fuzzy" and q >= 0.6)] or [f]
        for bw in base_words[:3]:
            for s in self.term_syn.get(bw, ()):
                if s in self.complete_vocab:
                    add(s, Q_SYNONYM * exp.get(bw, (1.0, ""))[0], "syn")
        term.expansions = [(w, q, k) for w, (q, k) in exp.items()]
        term.known = bool(term.expansions) or (term.last and len(t) <= 3 and bool(self._prefix_terms(self.name_keys, t, cap=1)))
        term.complete = any(k in ("exact", "fold", "fuzzy", "syn") for _, _, k in term.expansions)
        if any(k in ("exact", "fold") for _, _, k in term.expansions):
            term.best = f                       # folded form: what synonym keys are stored as
        else:
            best = max((x for x in term.expansions if x[2] != "prefix"), key=lambda x: x[1], default=None)
            term.best = best[0] if best else f
        dfs = [self.df.get(w, 0) for w, q, k in term.expansions if k != "prefix" and q >= 0.6]
        dfx = max(dfs) if dfs else max([self.df.get(w, 0) for w, _, _ in term.expansions] or [0])
        term.weight = math.log(1.0 + self.N / (dfx + 1.0))

    def _names_all(self, terms):
        """Some place's name contains every one of these words (exact, plural or typo)."""
        docs = None
        for t in terms:
            parts = [self.name_p[w][0] for w, q, k in t.expansions + t.bigrams if k != "prefix" and w in self.name_p]
            if not parts:
                return False
            ids = np.unique(np.concatenate(parts))
            docs = ids if docs is None else np.intersect1d(docs, ids, assume_unique=True)
            if not len(docs):
                return False
        return docs is not None and len(docs) > 0

    def _typed_type_word(self, term):
        """The word being typed is already a complete type word ("bar", "park", "pho", "cafe"):
        don't also complete it to longer words ("barber", "parking", "photo", "cafeteria")."""
        return len(term.raw) >= 3 and (term.raw in self.type_words or term.folded in self.type_words)

    @property
    def intents_words(self):
        s = getattr(self, "_iw", None)
        if s is None:
            s = self._iw = {w for k in self.intents for w in k}
        return s

    @property
    def content_keys(self):
        s = getattr(self, "_ck", None)
        if s is None:
            s = self._ck = sorted(t for t, (ids, _) in self.content_p.items() if len(ids) >= 3)
        return s

    def _find_anchor(self, tokens):
        """-> (start, end, anchor) for a street / neighbourhood named in the query, or None."""
        n = len(tokens)
        for i in range(n):
            for m in (3, 2, 1):
                j = i + m
                if j > n:
                    continue
                phrase = " ".join(tokens[i:j])
                end = j + 1 if j < n and tokens[j] in STREET_TYPES else j
                rest = [t for k, t in enumerate(tokens) if (k < i or k >= end) and t not in STOP_WORDS
                        and t not in LOCATION_PREPS]
                if not rest:
                    continue
                after_prep = i > 0 and tokens[i - 1] in LOCATION_PREPS
                at_end = end == n
                if phrase in NEIGHBOURHOODS and (after_prep or at_end):
                    return i, end, ("hood", NEIGHBOURHOODS[phrase])
                if phrase in self.streets:
                    on_street = len(self.streets[phrase])
                    named = self.name_df.get(self.fold(phrase.replace(" ", "")), 0) if m > 1 else \
                        self.name_df.get(self.fold(phrase), 0)
                    if after_prep or (at_end and i > 0 and named < 0.25 * on_street):
                        return i, end, ("street", phrase)
        return None

    def parse(self, q) -> ParsedQuery:
        norm = normalize(q)
        hit = self._qcache.get(norm)
        if hit is not None:
            return hit
        tokens = norm.split()
        anchor_span = self._find_anchor(tokens) if len(tokens) >= 2 else None
        loc_idx = set(range(anchor_span[0], anchor_span[1])) if anchor_span else set()
        sig_idx = [k for k, t in enumerate(tokens) if t not in STOP_WORDS and not (t in LOCATION_PREPS and anchor_span)]
        if not sig_idx:
            sig_idx = list(range(len(tokens)))
        filler_ok = any(tokens[k] not in FILLER and k not in loc_idx for k in sig_idx)
        terms = []
        for k in sig_idx:
            t = tokens[k]
            term = Term(raw=t, folded=self.fold(t), last=(k == len(tokens) - 1),
                        filler=filler_ok and t in FILLER, location=k in loc_idx)
            self._expand(term)
            if term.filler:
                term.weight *= FILLER_WEIGHT
            if term.location:
                term.weight *= LOCATION_WEIGHT
            terms.append(term)
        # adjacent words that form one index term: "late night" (hours tag), "book store", "ice cream"
        for a, b in zip(terms, terms[1:]):
            for j in dict.fromkeys((a.raw + b.raw, self.fold(a.raw + b.raw), a.raw + b.folded)):
                if j in self.type_p or j in self.name_p:
                    for t in (a, b):
                        t.bigrams.append((j, Q_BIGRAM, "bigram"))
                        t.known = t.complete = True

        # type intents from synonym phrases (folded, typo-corrected words), longest first
        intents, used = [], set()
        words = [t.best for t in terms]
        for m in (3, 2, 1):
            for i in range(len(terms) - m + 1):
                span = tuple(range(i, i + m))
                if used & set(span) or any(terms[k].location for k in span):
                    continue
                key = tuple(words[i:i + m])
                if key in self.intents:
                    intents.append((span, self.intents[key], Q_INTENT))
                    used |= set(span)
        last = terms[-1] if terms else None
        if last and last.last and len(last.raw) >= 3 and (len(terms) - 1) not in used and not last.complete:
            for m in (1, 2):
                if m > len(terms):
                    break
                pre = " ".join(words[len(terms) - m:-1] + [last.raw])
                for key in self._prefix_terms(self.intent_keys, pre, cap=20):
                    span = tuple(range(len(terms) - m, len(terms)))
                    intents.append((span, self.intents[tuple(key.split())], Q_INTENT_PREFIX))

        content_terms = [t for t in terms if not t.location]
        generic = bool(content_terms) and all(
            any(w in self.type_words for w, q, k in t.expansions if k != "prefix" and q >= 0.8) or t.filler
            for t in content_terms)
        comp = "".join(tokens)
        pinned = None
        if not generic:
            # spacing-insensitive: "Mc Donald's" and "McDonald's" are the same name, so a chain is
            # never pinned just because one branch spells it like the query
            ids = self.compact.get(comp) if len(comp) >= 3 else None
            if ids is None:
                ids = self.exact_full.get(norm)
            if ids is not None and 0 < len(ids) <= PIN_MAX:
                pinned = ids
        short = 0 < len(norm) <= 2
        known = [t for t in content_terms if t.known]
        sem_text = None
        # Embed only "what kind of place" queries: complete words (not a prefix being typed), and at
        # least one type word or a word no place is named after. Names, brands and codes stay lexical
        # (dense retrieval is weak on identifiers: "ralphs" would drift to "Ralph Hall").
        typed = any(w in self.type_words for t in known for w, q, k in t.expansions if k != "prefix")
        if (len(norm) >= 4 and pinned is None and known and (typed or not self._names_all(known))
                and all(t.complete for t in content_terms if t.known)):
            sem_text = " ".join(t.best if t.complete else t.raw for t in content_terms if t.known)
        anchor = anchor_span[2] if anchor_span else None
        pq = ParsedQuery(norm=norm, tokens=tokens, terms=terms, intents=intents, compact=comp, generic=generic,
                         pinned=pinned, anchor=anchor, short=short, semantic_text=sem_text)
        self._qcache.put(norm, pq)
        return pq

    def _intent_docs(self, intents):
        key = tuple((it["category"], it.get("subcategory")) for it in intents)
        hit = self._intent_cache.get(key)
        if hit is not None:
            return hit
        parts = []
        for cat, sub in key:
            code = self.cat_code.get(cat)
            if sub:
                ids = self.slug_docs.get(sub)
                if ids is not None:
                    parts.append(ids)
            elif code is not None:
                parts.append(np.nonzero(self.cat == code)[0].astype(np.int32))
        ids = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int32)
        if len(self._intent_cache) < 4096:
            self._intent_cache[key] = ids
        return ids

    # ------------------------------------------------------------------ relevance
    def relevance(self, pq: ParsedQuery, qvec=None, fusion="or"):
        """-> (rel[N] float32, field[N] int8 (0 name, 1 type, 2 content, 3 address, 4 semantic))."""
        N = self.N
        rel = np.zeros(N, dtype=np.float32)
        field = np.zeros(N, dtype=np.int8)
        terms = pq.terms
        if not terms:
            return rel, field
        if pq.short:
            ids, q = self.prefix_p.get(pq.tokens[0], (None, None)) if len(pq.tokens) == 1 else (None, None)
            if ids is None and len(pq.tokens) > 1:
                return rel, field
            if ids is not None:
                rel[ids] = q
            return rel, field

        intent_by_term = defaultdict(list)
        for span, intents, quality in pq.intents:
            for k in span:
                intent_by_term[k].append((intents, quality))

        wsum = 0.0
        fsum = np.zeros((4, N), dtype=np.float32)
        n_unknown = 0
        for k, t in enumerate(terms):
            if not t.known:
                n_unknown += 1
                continue
            fe = np.zeros((4, N), dtype=np.float32)
            for w, q, kind in t.expansions + t.bigrams:
                p = self.name_p.get(w)
                if p is not None:
                    a = q * W_NAME * (TYPE_WORD_NAME_FACTOR if w in self.type_words else 1.0)
                    if kind == "fold":
                        a *= Q_FOLDED_NAME
                    _maxset(fe[0], p[0], a * p[1])
                p = self.type_p.get(w)
                if p is not None:
                    _maxset(fe[1], p[0], q * W_TYPE * p[1])
                p = self.content_p.get(w)
                if p is not None:
                    _maxset(fe[2], p[0], q * W_CONTENT * p[1])
                p = self.addr_p.get(w)
                if p is not None:
                    _maxset(fe[3], p[0], q * W_ADDR * p[1])
            if t.last and len(t.raw) <= 3 and not self._typed_type_word(t):
                p = self.prefix_p.get(t.raw)
                if p is not None:
                    _maxset(fe[0], p[0], W_NAME * p[1])
            for intents, quality in intent_by_term.get(k, ()):
                ids = self._intent_docs(intents)
                if len(ids):
                    _maxset(fe[1], ids, np.float32(quality * W_TYPE))
            best = fe.max(axis=0)
            second = np.minimum(fe.sum(axis=0) - best, 1.0)
            e = np.minimum(best + SECOND_FIELD_BONUS * second, 1.0)
            rel += t.weight * e
            fsum += t.weight * fe
            wsum += t.weight
        if wsum <= 0:
            return rel, field
        rel /= wsum
        field = fsum.argmax(axis=0).astype(np.int8)
        if n_unknown:
            rel *= UNKNOWN_WORD_FACTOR ** n_unknown

        # whole query as one run-together name: "mc donalds", "chik fil a", "vonklein"
        comp = pq.compact
        if len(comp) >= 4 and not pq.generic:
            boost = np.zeros(N, dtype=np.float32)
            ids = self.compact.get(comp)
            if ids is not None:
                boost[ids] = 1.0
            for key in self._prefix_terms(self.compact_keys, comp, cap=200):
                _maxset(boost, self.compact[key], np.float32(0.8 + 0.2 * len(comp) / len(key)))
            if len(comp) >= 5 and len(pq.tokens) >= 2:
                for key, c in self._fuzzy_compact(comp):
                    _maxset(boost, self.compact[key], np.float32(1.0 - Q_FUZZY_STEP * c))
            up = boost > rel
            field[up] = 0
            rel = np.maximum(rel, boost)

        if qvec is not None and self.vectors is not None:
            sem = self.semantic_scores(qvec)
            if fusion == "rrf":
                rel = self._rrf(rel, sem)
            else:
                # Probabilistic OR, with the semantic weight shrinking as the lexical match gets
                # stronger: meaning adds recall when the words alone find little ("date night"),
                # and only re-orders when they already found real matches ("coffee").
                top = float(rel.max())
                beta = SEM_BETA * min(max((1.0 - top) / 0.5, SEM_MIN_SHARE), 1.0)
                fused = 1.0 - (1.0 - rel) * (1.0 - beta * sem)
                field[(beta * sem > rel) & (sem > 0)] = 4
                rel = fused.astype(np.float32)
        return rel, field

    def _fuzzy_compact(self, comp):
        budget = typo_budget(len(comp))
        if not budget or _rf_process is None:
            return []
        out = []
        for w, d, _ in _rf_process.extract(comp, self.compact_keys, scorer=_OSA.distance, score_cutoff=budget,
                                           limit=50):
            c = typo_cost(comp, w, int(d))
            if 0 < c <= budget:
                out.append((w, c))
        return out

    def semantic_scores(self, qvec):
        """Cosine similarity rescaled to [0, 1] over the SEM_TOP best places (0 for the rest)."""
        cos = self.vectors @ np.asarray(qvec, dtype=np.float32).reshape(-1)
        cos[~self.has_vec] = -1.0
        k = min(SEM_TOP, self.N - 1)
        if k <= 0:
            return np.zeros(self.N, dtype=np.float32)
        floor = np.partition(cos, -k)[-k]
        top = float(cos.max())
        s = np.clip((cos - floor) / max(top - floor, 1e-6), 0.0, 1.0)
        return s.astype(np.float32)

    def _rrf(self, rel, sem, k=60):
        """Reciprocal rank fusion of the lexical and semantic rankings (ties share a rank),
        rescaled so the best fused place has the best lexical relevance."""
        def ranks(x):
            order = np.argsort(-x, kind="stable")
            r = np.empty(len(x), dtype=np.float32)
            # dense ranks: equal scores share a rank
            xs = x[order]
            dense = np.concatenate([[0], np.cumsum(xs[1:] < xs[:-1])])
            r[order] = dense
            return r
        f = np.zeros_like(rel)
        lex_on = rel > 0
        sem_on = sem > 0
        f[lex_on] += 1.0 / (k + ranks(rel)[lex_on])
        f[sem_on] += 1.0 / (k + ranks(sem)[sem_on])
        if f.max() <= 0:
            return rel
        return (f / f.max() * max(float(rel.max()), SEM_BETA)).astype(np.float32)

    # ------------------------------------------------------------------ search
    def search(self, q=None, near=None, bbox=None, categories=None, open_now=False, limit=20, now=None,
               qvec=None, is_open_fn=None, fusion="or", explain=False):
        """Best places first, as API result dicts (+ distance_m, open_now, match)."""
        mask = np.ones(self.N, dtype=bool)
        if bbox:
            min_lng, min_lat, max_lng, max_lat = bbox
            mask &= (self.lng >= min_lng) & (self.lng <= max_lng) & (self.lat >= min_lat) & (self.lat <= max_lat)
        if categories:
            codes = [self.cat_code[c] for c in categories if c in self.cat_code]
            mask &= np.isin(self.cat, codes)
        if q and (not categories or "campus" in categories) and not bbox:
            rooms = self.room_search(q, near=near, limit=limit)
            if rooms is not None:
                return rooms
        mask &= ~self.is_room            # classrooms only come back for room queries
        dist = haversine_m(near[0], near[1], self.lat, self.lng) if near else None

        pq = self.parse(q) if q and normalize(q) else None
        if pq is None:
            # category chips: nearest places in the category, small popularity tiebreak
            cand = np.nonzero(mask)[0]
            if dist is not None:
                score = 1.0 / (1.0 + np.maximum(dist[cand] - OFFSET_M, 0) / SCALE_MIN_M) * (1 + 0.15 * self.pop[cand])
            else:
                score = self.pop[cand].astype(np.float64)
            rel = np.ones(self.N, dtype=np.float32)
            return self._finish(cand, score, rel, None, dist, open_now, limit, now, is_open_fn, explain, "category")

        rel, fld = self.relevance(pq, qvec if pq.semantic_text else None, fusion=fusion)
        rel = np.where(mask, rel, 0.0)
        top = float(rel.max()) if self.N else 0.0
        if top <= 0:
            return []
        gate = max(GATE_ABS, GATE_REL * top)
        cand = np.nonzero(rel >= gate)[0]
        if pq.pinned is not None:
            cand = np.union1d(cand, pq.pinned[mask[pq.pinned]])
        r = rel[cand].astype(np.float64)

        # effective distance: from the user, or from the street / neighbourhood named in the query
        d_eff = None
        if pq.anchor is not None:
            d_eff = self._anchor_distance(pq.anchor, cand, near)
            if dist is not None:
                d_eff = d_eff + 0.2 * dist[cand]
        elif dist is not None:
            d_eff = dist[cand]
        if d_eff is not None:
            strong = d_eff[r >= 0.85 * top]
            if len(strong) >= SCALE_K:
                scale = float(np.partition(strong, SCALE_K - 1)[SCALE_K - 1])
            else:
                scale = float(strong.max()) if len(strong) else SCALE_MIN_M
            scale = min(max(scale, SCALE_MIN_M), SCALE_MAX_M)
            prox = 1.0 / (1.0 + np.maximum(d_eff - OFFSET_M, 0.0) / scale)
            pop_cap = POP_CAP_SHORT if pq.short else POP_CAP_NEAR
        else:
            prox = 1.0
            pop_cap = POP_CAP_NO_NEAR
        score = (r ** 2) * prox * (1.0 + pop_cap * self.pop[cand])
        score = np.where(self.is_building[cand], score * BUILDING_BUMP, score)
        if pq.pinned is not None:
            score = score + np.isin(cand, pq.pinned) * 10.0
        return self._finish(cand, score, rel, fld, dist, open_now, limit, now, is_open_fn, explain, None)

    def _anchor_distance(self, anchor, cand, near):
        kind, val = anchor
        if kind == "hood":
            lat, lng, radius = val
            return np.maximum(haversine_m(lat, lng, self.lat[cand], self.lng[cand]) - radius, 0.0)
        pts = self.streets[val]
        if near is not None and len(pts) > 1:
            dp = haversine_m(near[0], near[1], self.lat[pts], self.lng[pts])
            close = pts[dp <= max(3000.0, float(np.min(dp)) + 1500.0)]
            pts = close if len(close) else pts
        if len(pts) > 300:
            pts = pts[np.linspace(0, len(pts) - 1, 300).astype(int)]
        # equirectangular is plenty at this scale
        k = math.cos(math.radians(float(self.lat[pts].mean())))
        dy = (self.lat[cand][:, None] - self.lat[pts][None, :]) * 111195.0
        dx = (self.lng[cand][:, None] - self.lng[pts][None, :]) * 111195.0 * k
        return np.sqrt(dx * dx + dy * dy).min(axis=1)

    def _finish(self, cand, score, rel, fld, dist, open_now, limit, now, is_open_fn, explain, match):
        order = np.argsort(-score, kind="stable")
        names = ("name", "category", "content", "address", "semantic")
        out = []
        for j in order:
            i = int(cand[j])
            hours = self.hours[i]
            o = is_open_fn(hours, now) if (hours and is_open_fn) else None
            if open_now and o is not True:
                continue
            r = dict(self.docs[i])
            if dist is not None:
                r["distance_m"] = int(round(float(dist[i])))
            if o is not None:
                r["open_now"] = o
            r["match"] = match if match else names[int(fld[i])]
            if explain:
                r["_score"] = float(score[j])
                r["_rel"] = float(rel[i])
                r["_labels"] = self.labels[i]
            out.append(r)
            if len(out) >= limit:
                break
        return out


    # ------------------------------------------------------------------ classrooms
    def room_search(self, q, near=None, limit=20):
        """Room queries -> [room, its building, sibling rooms], or None for an ordinary query.
          "rth 108" / "RTH108" / "108 rth"   exact room (synthesized with verified=false when the room
                                             is not in the dataset, so any valid number gets a floor)
          "thh 1", "thh b"                   the building, then its rooms starting with that
          "thh"                              the building, then its rooms
          "taper hall 201", "201 taper"      building name + room number, when the name clearly
                                             means one building"""
        from .rooms import WORD_CODES, parse_room_query

        rq = parse_room_query(q, self.known_codes)
        if rq:
            return self._room_results(rq["code"], rq["room"], near, limit)
        tokens = normalize(q).split()
        if not tokens:
            return None
        code = tokens[0].upper()
        if code in self.known_codes and code not in WORD_CODES:
            if len(tokens) == 1 and self.rooms_by_code.get(code):
                return self._room_listing(code, "", near, limit)
            if len(tokens) == 2 and re.fullmatch(r"(?:sb|gdn|ll|lg|b|g)?\d{0,3}[a-z]?", tokens[1]):
                return self._room_listing(code, tokens[1].upper(), near, limit)
        # building name + room number
        num_rx = r"(?:sb|gdn|ll|lg|b|g)?\d{3,4}[a-z]?"
        for num, rest in ((tokens[-1], tokens[:-1]), (tokens[0], tokens[1:])):
            if rest and re.fullmatch(num_rx, num):
                code = self._resolve_building(" ".join(rest))
                if code:
                    return self._room_results(code, num.upper(), near, limit)
        return None

    def _resolve_building(self, text):
        """The one building a name clearly means ("tutor hall" -> RTH), else None."""
        pq = self.parse(text)
        if not pq.terms or pq.generic or pq.short:
            return None
        if pq.pinned is not None:
            b = [int(i) for i in pq.pinned if self.docs[int(i)]["kind"] == "building"]
            if len(b) == 1:
                return next(c for c, i in self.building_by_code.items() if i == b[0])
        rel, _ = self.relevance(pq)
        cand = [(float(rel[i]), c) for c, i in self.building_by_code.items() if rel[i] > 0]
        cand.sort(reverse=True)
        if not cand or cand[0][0] < 0.9 or (len(cand) > 1 and cand[1][0] >= 0.85 * cand[0][0]):
            return None
        return cand[0][1]

    def room_result(self, code, room, near=None):
        """One room as an API result: from the dataset (verified) or synthesized at its building."""
        from .rooms import parse_room_number

        code, room = code.upper(), room.upper()
        b = self.building_info.get(code)
        if b is None:
            return None
        i = self.id_pos.get(f"room:{code}-{room}")
        if i is not None and self.is_room[i]:
            d = self.room_info.get(i, {})
            r = dict(self.docs[i])
            r.update(address=b.get("address"), floor=d.get("floor"), floor_label=d.get("floor_label"),
                     floor_confidence=d.get("floor_confidence"), building_id=d.get("building_id") or b["building_id"],
                     building_name=d.get("building_name") or b["building_name"], code=code, room=room,
                     pathing=bool(d.get("pathing")), verified=True, capacity=d.get("capacity"),
                     classroom_type=d.get("classroom_type"))
        else:
            f = parse_room_number(room, code, b.get("floor_count"))
            upc = code in self.building_by_code and (b.get("campus") or "UPC") == "UPC"
            r = {"id": f"room:{code}-{room}", "kind": "room", "name": f"{code} {room}", "category": "campus",
                 "subcategory": "classroom", "address": b.get("address"), "lat": float(b["lat"]),
                 "lng": float(b["lng"]), "rating": None, "review_count": 0, "floor": f["floor"],
                 "floor_label": f["floor_label"], "floor_confidence": f["confidence"],
                 "building_id": b["building_id"], "building_name": b["building_name"], "code": code, "room": room,
                 "pathing": bool(upc and f["floor"] is not None), "verified": False}
        if near:
            r["distance_m"] = int(round(float(haversine_m(near[0], near[1], r["lat"], r["lng"]))))
        r["match"] = "room"
        return r

    def _building_result(self, code, near):
        i = self.building_by_code.get(code)
        if i is None:
            return None
        r = dict(self.docs[i])
        if near:
            r["distance_m"] = int(round(float(haversine_m(near[0], near[1], r["lat"], r["lng"]))))
        r["match"] = "name"
        return r

    def _room_results(self, code, room, near, limit):
        out = [x for x in (self.room_result(code, room, near), self._building_result(code, near)) if x]
        m = re.match(r"^([A-Z]*)(\d?)", room)
        same_level = m.group(1) or m.group(2)        # "108" -> "1", "B09" -> "B", "LL101" -> "LL"
        for rm, _ in self.rooms_by_code.get(code, ()):
            if len(out) >= limit:
                break
            if rm != room and rm.startswith(same_level):
                out.append(self.room_result(code, rm, near))
        return out[:limit]

    def _room_listing(self, code, prefix, near, limit):
        out = [x for x in [self._building_result(code, near)] if x]
        for rm, _ in self.rooms_by_code.get(code, ()):
            if len(out) >= limit:
                break
            if rm.startswith(prefix):
                out.append(self.room_result(code, rm, near))
        return out or None


def _room_sort_key(room):
    m = re.match(r"^([A-Z]*)(\d*)(.*)$", room)
    pre, num, rest = m.groups() if m else ("", "", room)
    return (pre != "", pre, int(num) if num else -1, rest)


def _maxset(arr, ids, vals):
    """arr[ids] = max(arr[ids], vals); ids are unique within one posting list."""
    cur = arr[ids]
    arr[ids] = np.maximum(cur, vals)
