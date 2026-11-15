import marshmallow as ma
from flask import current_app
from flask.views import MethodView
from flask_smorest import Blueprint, abort

from ..auth import current_user
from ..services import demo_incidents
from ..services.h3_utils import in_study_area

blp = Blueprint("demo", __name__, url_prefix="/api/v1/demo",
                description="Simulated crimes for demo recordings (only with DEMO_MODE=true)")


class DemoIncidentBody(ma.Schema):
    lat = ma.fields.Float(required=True, validate=ma.validate.Range(-90, 90))
    lng = ma.fields.Float(required=True, validate=ma.validate.Range(-180, 180))
    category = ma.fields.String(validate=ma.validate.Length(1, 60))
    severity = ma.fields.Integer(load_default=1, validate=ma.validate.OneOf([1, 2]))


def _require_demo(write=False):
    if not demo_incidents.enabled():
        abort(404, message="demo mode is off on this server (set DEMO_MODE=true)")
    if write and not current_user() and not current_app.config.get("TESTING"):
        abort(401, message="sign in to simulate incidents")


@blp.route("/status")
@blp.response(200)
def status():
    """Whether simulated incidents are available here, and their settings."""
    c = current_app.config
    return {"enabled": demo_incidents.enabled(), "ttl_minutes": c["DEMO_INCIDENT_TTL_MIN"],
            "avoid_radius_m": c["DEMO_INCIDENT_RADIUS_M"], "categories": list(demo_incidents.CATEGORIES)}


@blp.route("/incidents")
class DemoIncidents(MethodView):
    @blp.response(200)
    def get(self):
        """Unexpired simulated incidents, newest first."""
        _require_demo()
        return {"incidents": demo_incidents.active()}

    @blp.arguments(DemoIncidentBody)
    @blp.response(201)
    def post(self, body):
        """Simulate an incident at a point. Routes planned in the next
        DEMO_INCIDENT_TTL_MIN minutes avoid DEMO_INCIDENT_RADIUS_M around it."""
        _require_demo(write=True)
        if not in_study_area(body["lat"], body["lng"]):
            abort(422, message="point is outside the study area")
        return demo_incidents.add(body["lat"], body["lng"], body.get("category"), body["severity"])

    @blp.response(200)
    def delete(self):
        """Remove every simulated incident."""
        _require_demo(write=True)
        return {"removed": demo_incidents.clear()}
