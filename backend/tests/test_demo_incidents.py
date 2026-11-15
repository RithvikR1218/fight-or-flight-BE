import pytest

from app import create_app
from app.services import demo_incidents
from tests.test_routing import _ladder_graph


def _flat_grid():
    """A RiskGrid with the same low risk and cost everywhere (no campus zones in play)."""
    pytest.importorskip("algorithm.common.geo")
    from pathfinding.risk import RiskGrid

    class FlatGrid(RiskGrid):
        def __init__(self):   # skip reading the cache file
            self.date, self.res, self.step_m = "2026-11-15", 10, 20.0
            self.no_data_risk, self.no_data_cost = 0.5, 0.5
            self.cells, self.cost, self.no_data = {}, {}, frozenset()

        def point(self, lat, lon, hour):
            return 0.1, 0.1, True

        def risk_at(self, lat, lon, hour):
            return 0.1, True

    return FlatGrid()


def test_overlay_marks_only_points_near_an_incident():
    grid = _flat_grid()
    inc = (34.0200, -118.2850)
    ov = demo_incidents.overlay(grid, [inc], radius_m=100)
    assert ov.point(*inc, 22) == (1.0, demo_incidents.AVOID_COST, True)
    assert ov.point(34.0200, -118.2840, 22) == (1.0, demo_incidents.AVOID_COST, True)   # ~92 m east
    assert ov.point(34.0200, -118.2830, 22) == (0.1, 0.1, True)                          # ~185 m east
    assert ov.risk_at(*inc, 22) == (1.0, True)
    assert grid.point(*inc, 22) == (0.1, 0.1, True)          # the shared base grid is untouched
    assert demo_incidents.overlay(grid, [], radius_m=100) is grid


def test_overlay_beats_the_campus_zero_risk_zone():
    pytest.importorskip("algorithm.common.geo")
    from pathfinding.risk import RiskGrid

    grid = _flat_grid()
    tommy = (34.0206, -118.2856)
    assert RiskGrid.point(grid, *tommy, 22)[0] == 0.0         # campus is risk 0 normally
    assert demo_incidents.overlay(grid, [tommy], 100).point(*tommy, 22)[0] == 1.0


def test_route_steers_around_a_simulated_incident():
    pytest.importorskip("algorithm.common.geo")
    from pathfinding.planner import plan_route

    g = _ladder_graph()
    start, end = (34.0200, -118.2900), (34.0200, -118.2800)   # both ends on the south street
    before = plan_route(start, end, hour=22, max_detour=0.3, graph=g, risk=_flat_grid())
    assert all(abs(lat - 34.0200) < 1e-6 for lat, _ in before["safe"].coords)   # straight along the south street

    incident = [(34.0200, -118.2850)]                                           # middle of the south street
    after = plan_route(start, end, hour=22, max_detour=0.3, graph=g,
                       risk=demo_incidents.overlay(_flat_grid(), incident, radius_m=50))   # streets are ~67 m apart
    lats = [lat for lat, _ in after["safe"].coords]
    assert any(abs(lat - 34.0206) < 1e-6 for lat in lats)                       # switched to the north street
    assert after["safe"].max_risk < 1.0                                         # never inside the radius


def test_demo_endpoints_are_off_by_default(client):
    assert client.get("/api/v1/demo/status").get_json()["enabled"] is False
    assert client.get("/api/v1/demo/incidents").status_code == 404
    assert client.post("/api/v1/demo/incidents", json={"lat": 34.02, "lng": -118.285}).status_code == 404
    assert client.delete("/api/v1/demo/incidents").status_code == 404


def test_demo_writes_need_sign_in_when_enabled():
    app = create_app(DEMO_MODE=True)     # not TESTING: the sign-in check applies
    with app.app_context():
        c = app.test_client()
        r = c.post("/api/v1/demo/incidents", json={"lat": 34.02, "lng": -118.285})
        assert r.status_code == 401
        assert c.delete("/api/v1/demo/incidents").status_code == 401
        bad = c.post("/api/v1/demo/incidents", json={"lat": 34.02})
        assert bad.status_code == 422
