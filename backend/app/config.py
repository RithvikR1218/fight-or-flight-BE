from datetime import timedelta
import os


def _env(name, default, cast=str):
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return cast(val)


def _bool(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


class Config:
    SQLALCHEMY_DATABASE_URI = _env(
        "DATABASE_URL", "postgresql+psycopg2://safe:safe@localhost:5432/saferoute"
    )
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}
    DATA_DIR = _env("DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")))

    # Study area
    USC_LAT = _env("USC_LAT", 34.0224, float)
    USC_LNG = _env("USC_LNG", -118.2851, float)
    RADIUS_KM = _env("RADIUS_KM", 8.047, float)          # 5 miles, same as risk-model RADIUS_MILES
    ROAD_MARGIN_KM = _env("ROAD_MARGIN_KM", 1.5, float)  # walk graph extends this far past the circle
    H3_RES = _env("H3_RES", 9, int)
    H3_AREA_RES = _env("H3_AREA_RES", 6, int)

    # Ingestion intervals (minutes)
    POLL_SUN_MINUTES = _env("POLL_SUN_MINUTES", 1, float)
    POLL_OPENMETEO_MINUTES = _env("POLL_OPENMETEO_MINUTES", 5, float)
    POLL_NWS_MINUTES = _env("POLL_NWS_MINUTES", 10, float)
    POLL_LAPD_MINUTES = _env("POLL_LAPD_MINUTES", 60, float)
    POLL_LA311_MINUTES = _env("POLL_LA311_MINUTES", 30, float)

    # External sources
    SOCRATA_APP_TOKEN = _env("SOCRATA_APP_TOKEN", "")
    LAPD_DATASET_ID = _env("LAPD_DATASET_ID", "2nrs-mtv8")
    CRIME_BACKFILL_START = _env("CRIME_BACKFILL_START", "2024-01-01")
    LAPD_NIBRS_DATASET_ID = _env("LAPD_NIBRS_DATASET_ID", "k7nn-b2ep")
    NIBRS_BACKFILL_START = _env("NIBRS_BACKFILL_START", "2025-01-01")
    LA311_DATASET_ID = _env("LA311_DATASET_ID", "")
    NWS_STATIONS = _env("NWS_STATIONS", "KCQT,KLAX")
    HTTP_USER_AGENT = _env("HTTP_USER_AGENT", "usc-safe-route-hackathon")
    HTTP_TIMEOUT = _env("HTTP_TIMEOUT", 15, int)

    ADMIN_TOKEN = _env("ADMIN_TOKEN", "")
    # Simulated crimes for demo recordings (/api/v1/demo/incidents). Off in production; see README "Demo mode".
    DEMO_MODE = _env("DEMO_MODE", False, _bool)
    DEMO_INCIDENT_TTL_MIN = _env("DEMO_INCIDENT_TTL_MIN", 60, int)     # a simulated crime expires after this
    DEMO_INCIDENT_RADIUS_M = _env("DEMO_INCIDENT_RADIUS_M", 120, float)  # routes avoid this radius around one

    # Place search (services/search.py). The embedding model id is read by services/search_semantic.py
    # from SEARCH_EMBED_MODEL directly, so the loader script and the API agree.
    SEARCH_INDEX_TTL = _env("SEARCH_INDEX_TTL", 60.0, float)   # seconds between "did places change?" checks
    SEARCH_SEMANTIC = _env("SEARCH_SEMANTIC", "auto")          # auto | on | off

    # Sign-in (Google OAuth, USC accounts only) and the session cookie. See README "Auth".
    SECRET_KEY = _env("SECRET_KEY", "")                   # signs the session cookie; required for sign-in
    GOOGLE_CLIENT_ID = _env("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = _env("GOOGLE_CLIENT_SECRET", "")
    OAUTH_REDIRECT_URI = _env("OAUTH_REDIRECT_URI", "")   # empty = this API's /api/v1/auth/callback
    AUTH_ALLOWED_DOMAIN = _env("AUTH_ALLOWED_DOMAIN", "usc.edu")
    # FE origins: CORS allow-list (with credentials) and the only places `next` may point to
    FRONTEND_ORIGINS = _env("FRONTEND_ORIGINS", "http://localhost:5173,http://localhost:4173")
    AUTH_DEV_LOGIN = _env("AUTH_DEV_LOGIN", False, _bool)  # local testing without Google; never in production
    SESSION_COOKIE_NAME = "fof_session"
    SESSION_COOKIE_HTTPONLY = True
    # FE on another site (production): SESSION_COOKIE_SAMESITE=None and SESSION_COOKIE_SECURE=true
    SESSION_COOKIE_SAMESITE = _env("SESSION_COOKIE_SAMESITE", "Lax")
    SESSION_COOKIE_SECURE = _env("SESSION_COOKIE_SECURE", False, _bool)
    PERMANENT_SESSION_LIFETIME = timedelta(days=_env("SESSION_DAYS", 7, int))

    # flask-smorest / OpenAPI
    API_TITLE = "USC Safe-Route Data API"
    API_VERSION = "v1"
    OPENAPI_VERSION = "3.0.3"
    OPENAPI_URL_PREFIX = "/api"
    OPENAPI_JSON_PATH = "openapi.json"
    OPENAPI_SWAGGER_UI_PATH = "/docs"
    OPENAPI_SWAGGER_UI_URL = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.11.0/"
    JSON_SORT_KEYS = False
