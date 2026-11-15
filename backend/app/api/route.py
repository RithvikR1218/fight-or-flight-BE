"""Safe walking routes (planner: pathfinding/, see pathfinding/README.md)."""
from datetime import timezone

import marshmallow as ma
from flask_smorest import Blueprint, abort

from ..services import risk_model, routing
from .params import BaseArgs, LatLng

blp = Blueprint("route", __name__, url_prefix="/api/v1/route",
                description="Safe walking routes: low crime risk, past blue-light phones, bounded detour")


class RouteArgs(BaseArgs):
    start = LatLng(data_key="from", required=True, metadata={"description": "lat,lng"})
    end = LatLng(data_key="to", required=True, metadata={"description": "lat,lng"})
    at = ma.fields.AwareDateTime(default_timezone=timezone.utc,
                                 metadata={"description": "ISO-8601 instant (overrides date/hour)"})
    date = ma.fields.Date(metadata={"description": "LA local date, default today"})
    hour = ma.fields.Integer(validate=ma.validate.Range(0, 23),
                             metadata={"description": "LA local hour 0-23, default now"})
    max_detour = ma.fields.Float(load_default=0.20, validate=ma.validate.Range(0, 1),
                                 metadata={"description": "safe route may be at most this much longer (0.2 = 20%)"})
    phone_weight = ma.fields.Float(load_default=0.01, validate=ma.validate.Range(0, 1),
                                   metadata={"description": "exposure points one blue-light phone passed off campus "
                                                            "is worth (a tie-breaker; pathfinding/planner.py)"})


@blp.route("")
@blp.arguments(RouteArgs, location="query")
@blp.response(200)
def route(args):
    """`safe` and `shortest` routes with stats (length, detour %, mean/max risk, metres in
    high-risk cells, % without data, blue-light phones passed). Each route has `coords`
    ([[lat, lng], ...]) and a GeoJSON `geometry` (LineString, [lng, lat])."""
    try:
        return routing.plan(args["start"], args["end"], args.get("at"), args.get("date"), args.get("hour"),
                            max_detour=args["max_detour"], phone_weight=args["phone_weight"])
    except routing.RouteError as exc:
        abort(400, message=str(exc))
    except risk_model.ModelUnavailable as exc:
        abort(503, message=f"route planner unavailable: {exc}")
