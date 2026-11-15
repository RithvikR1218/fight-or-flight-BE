from geoalchemy2 import Geometry
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from ..extensions import db


class Place(db.Model):
    """Searchable place (Google Maps place, campus building or campus POI). Schema: migrations 0004_places."""
    __tablename__ = "places"
    id = db.Column(db.Text, primary_key=True)
    kind = db.Column(db.Text, nullable=False)
    name = db.Column(db.Text, nullable=False)
    name_norm = db.Column(db.Text, nullable=False)
    aliases = db.Column(ARRAY(db.Text), nullable=False, server_default="{}")
    search_text = db.Column(db.Text, nullable=False)
    category = db.Column(db.Text, nullable=False)
    subcategory = db.Column(db.Text)
    categories = db.Column(ARRAY(db.Text), nullable=False, server_default="{}")
    address = db.Column(db.Text)
    lat = db.Column(db.Float(precision=53), nullable=False)
    lng = db.Column(db.Float(precision=53), nullable=False)
    geom = db.Column(Geometry("POINT", srid=4326, spatial_index=False), nullable=False)
    rating = db.Column(db.REAL)
    review_count = db.Column(db.Integer, nullable=False, server_default="0")
    popularity = db.Column(db.REAL, nullable=False, server_default="0")
    hours = db.Column(JSONB)
    detail = db.Column(JSONB, nullable=False, server_default="{}")
    source = db.Column(db.Text)
    source_file = db.Column(db.Text)
    embedding = db.Column(db.LargeBinary)      # 0005: float32 vector for semantic search
    embed_key = db.Column(db.Text)


class SearchMeta(db.Model):
    """Key/value markers for the in-memory search index (migration 0005_search_index)."""
    __tablename__ = "search_meta"
    key = db.Column(db.Text, primary_key=True)
    value = db.Column(db.Text, nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, server_default=db.func.now())
