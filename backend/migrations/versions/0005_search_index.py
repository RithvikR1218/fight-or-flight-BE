"""Search: per-place sentence embeddings + a version marker for the in-memory search index.

places.embedding  float32 little-endian, L2-normalized (services/search_semantic.py); NULL = none
places.embed_key  hash of model + embedded text, so a reload re-embeds only changed places
search_meta       key/value: places_version (bumped by scripts/load_places; API workers rebuild
                  their index when it changes), embedding_model, embedding_dim

Revision ID: 0005_search_index
Revises: 0004_places
Create Date: 2026-11-15
"""
from alembic import op

revision = "0005_search_index"
down_revision = "0004_places"
branch_labels = None
depends_on = None

UP = """
ALTER TABLE places ADD COLUMN IF NOT EXISTS embedding BYTEA;
ALTER TABLE places ADD COLUMN IF NOT EXISTS embed_key TEXT;
CREATE TABLE IF NOT EXISTS search_meta (
  key         TEXT PRIMARY KEY,
  value       TEXT NOT NULL,
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO search_meta (key, value) VALUES ('places_version', '0') ON CONFLICT (key) DO NOTHING;
"""


def upgrade():
    op.execute(UP)


def downgrade():
    op.execute("DROP TABLE IF EXISTS search_meta")
    op.execute("ALTER TABLE places DROP COLUMN IF EXISTS embed_key")
    op.execute("ALTER TABLE places DROP COLUMN IF EXISTS embedding")
