"""Place search (autocomplete), place details and reverse geocoding. Public, read-only.
Algorithm: app/services/search.py."""
import marshmallow as ma
from flask_smorest import Blueprint, abort

from ..services import search as svc
from .params import BaseArgs, BBox, CsvList, LatLng

blp = Blueprint("search", __name__, url_prefix="/api/v1",
                description="Place search / autocomplete, place details, reverse geocoding")

SEARCH_CACHE = {"Cache-Control": "public, max-age=30"}        # open_now changes by the minute
STATIC_CACHE = {"Cache-Control": "public, max-age=300"}


class SearchArgs(BaseArgs):
    q = ma.fields.String(validate=ma.validate.Length(max=100),
                         metadata={"description": "free text; works from 1 character. Optional when `category` is set"})
    near = LatLng(metadata={"description": "lat,lng: rank by distance from here and return `distance_m`"})
    bbox = BBox(metadata={"description": "minLng,minLat,maxLng,maxLat hard filter"})
    category = CsvList(metadata={"description": "comma-separated top-level category ids (see /search/categories)"})
    open_now = ma.fields.Boolean(load_default=False, metadata={"description": "only places known to be open now"})
    limit = ma.fields.Integer(load_default=20, validate=ma.validate.Range(1, 50))


class NearArgs(BaseArgs):
    near = LatLng(metadata={"description": "lat,lng: adds `distance_m`"})


class ReverseArgs(BaseArgs):
    at = LatLng(required=True, metadata={"description": "lat,lng"})


@blp.route("/search")
@blp.arguments(SearchArgs, location="query")
@blp.response(200)
def search(args):
    """Places best first: only places relevant to `q` (name, type, description/reviews, meaning;
    typo-tolerant), ordered mostly by distance from `near`, then popularity. An exact building code
    or distinctive name comes first. "<building code> <room>" ("rth 108") returns that classroom
    (kind "room", with its floor) and its building. Empty `q` with `category` = nearest places in
    those categories. Each result has `match`: name | category | content | address | semantic | room."""
    q = (args.get("q") or "").strip()
    categories = [c.lower() for c in args.get("category") or []]
    if not svc.normalize(q) and not categories:
        abort(400, message="pass q or category")
    results = svc.search(q, near=args.get("near"), bbox=args.get("bbox"), categories=categories,
                         open_now=args["open_now"], limit=args["limit"])
    return {"results": results}, 200, SEARCH_CACHE


@blp.route("/search/categories")
@blp.response(200)
def categories():
    """Categories with counts and their place types (subcategories), in display order."""
    return svc.category_tree(), 200, STATIC_CACHE


@blp.route("/places/<path:place_id>")
@blp.arguments(NearArgs, location="query")
@blp.response(200)
def place(args, place_id):
    """One place: search fields plus weekly hours, phone, website, photo, ratings, reviews, aliases.
    Rooms (room:<CODE>-<ROOM>) add floor, floor_label, building_id/name, pathing, capacity, ...;
    a valid room that is not in the dataset is answered from its building (verified: false)."""
    out = svc.place_detail(place_id, near=args.get("near"))
    if out is None:
        abort(404, message=f"unknown place '{place_id}'")
    return out, 200, STATIC_CACHE


@blp.route("/geocode/reverse")
@blp.arguments(ReverseArgs, location="query")
@blp.response(200)
def reverse(args):
    """Name for a point: nearest place within ~60 m (buildings first), else the nearest named road."""
    out = svc.reverse_geocode(*args["at"])
    if out is None:
        abort(404, message="nothing named near this point")
    return out
