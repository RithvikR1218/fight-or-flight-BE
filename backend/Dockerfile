FROM python:3.12-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FLASK_APP=wsgi:app
WORKDIR /app
# Build context = repo root, so the risk model and route planner ship in the image.
# risk-model/ is installed as the `algorithm` package: its code imports itself as algorithm.*
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Place search embedding model, baked into the image so the first search doesn't download it.
# Must match SEARCH_EMBED_MODEL (default in app/services/search_semantic.py).
ENV FASTEMBED_CACHE_PATH=/opt/fastembed SEARCH_EMBED_MODEL=BAAI/bge-small-en-v1.5 HF_HUB_DISABLE_TELEMETRY=1
RUN python -c "import os; from fastembed import TextEmbedding; TextEmbedding(os.environ['SEARCH_EMBED_MODEL'], cache_dir=os.environ['FASTEMBED_CACHE_PATH'])"
COPY backend/ /app/
COPY risk-model/ /app/algorithm/
COPY pathfinding/ /app/pathfinding/
EXPOSE 5000
CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:app"]

# ---- tippecanoe build (only used by the `tools` target) ----
FROM debian:bookworm-slim AS tippecanoe-build
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates libsqlite3-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/felt/tippecanoe.git /src \
    && make -C /src -j"$(nproc)" && make -C /src install

FROM app AS tools
RUN apt-get update && apt-get install -y --no-install-recommends libsqlite3-0 zlib1g osmium-tool curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=tippecanoe-build /usr/local/bin/tippecanoe* /usr/local/bin/
COPY --from=tippecanoe-build /usr/local/bin/tile-join /usr/local/bin/
