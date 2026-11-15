from . import (admin, auth, buildings, cells, demo, features, health, incidents, observations, pois, risk, roads,
               route, search)


def register_blueprints(api):
    for module in (health, auth, cells, observations, incidents, features, buildings, roads, pois, search, risk,
                   route, admin, demo):
        api.register_blueprint(module.blp)
