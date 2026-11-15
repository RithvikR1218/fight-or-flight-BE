"""Optional semantic (meaning-based) signal for place search: sentence embeddings via fastembed.

* The loader (scripts/load_places.py) embeds one short text per place (`embedding_text`) and stores
  the L2-normalized float32 vector in places.embedding; the API holds all vectors as one matrix
  (services/search_index.py) and scores a query by a brute-force dot product.
* Only the query is embedded at request time (~5-10 ms on one CPU core, LRU-cached).
* Everything here degrades gracefully: without the `fastembed` package, the model files or the
  stored vectors, search runs on its lexical signals and this module logs why, once.

Settings (environment):
  SEARCH_SEMANTIC       auto (default: on when fastembed + vectors are available) | on | off
  SEARCH_EMBED_MODEL    fastembed model id (default below); the loader and the API must agree
  SEARCH_EMBED_THREADS  onnxruntime threads for query embedding (default 1: requests are concurrent)
  FASTEMBED_CACHE_PATH  model cache directory (the Docker image pre-downloads the model there)
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
TEXT_MAX_CHARS = 700
_QUERY_CACHE = 4096


def setting(name, default=None):
    v = os.getenv(name)
    return default if v is None or v == "" else v


def model_name():
    return setting("SEARCH_EMBED_MODEL", DEFAULT_MODEL)


def semantic_mode():
    return str(setting("SEARCH_SEMANTIC", "auto")).strip().lower()


def package_available() -> bool:
    try:
        import fastembed  # noqa: F401
    except Exception:
        return False
    return True


# ---- what gets embedded ---------------------------------------------------------------------------

def _words(slug):
    return (slug or "").replace("_", " ")


def embedding_text(row) -> str:
    """One short description per place: name, type labels, description, notable attributes and the
    start of a few reviews. Shared by the loader and the eval so stored and cached vectors agree."""
    d = row.get("detail") or {}
    parts = [row.get("name") or ""]
    labels = list(row.get("categories") or [])
    if not labels:
        labels = [_words(row.get("subcategory")), _words(d.get("poi_type"))]
        if row.get("kind") == "building":
            labels.append("USC campus building")
    parts.append(", ".join(x for x in dict.fromkeys(labels) if x))
    aliases = [a for a in row.get("aliases") or [] if len(a) > 3][:3]
    if aliases:
        parts.append("also called " + ", ".join(aliases))
    for k in ("description", "layer", "group", "library"):
        if isinstance(d.get(k), str) and d[k].strip():
            parts.append(d[k].strip())
    opts = [o for g in d.get("attributes") or [] if isinstance(g, dict) for o in g.get("options") or []
            if isinstance(o, str) and not o.startswith("Wheelchair")]
    if opts:
        parts.append(", ".join(opts[:8]))
    for rv in (d.get("reviews") or [])[:3]:
        t = rv.get("text") if isinstance(rv, dict) else rv
        if isinstance(t, str) and t.strip():
            parts.append(t.strip()[:220])
    text = ". ".join(p.strip().rstrip(".") for p in parts if p and p.strip())
    return text[:TEXT_MAX_CHARS]


def embed_key(model, text) -> str:
    return hashlib.sha1(f"{model}\n{text}".encode("utf-8")).hexdigest()[:20]


def to_bytes(vec) -> bytes:
    return np.asarray(vec, dtype="<f4").tobytes()


def from_bytes(b) -> np.ndarray:
    return np.frombuffer(bytes(b), dtype="<f4")


def normalize_rows(m):
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(n, 1e-12)


# ---- the model ------------------------------------------------------------------------------------

class Embedder:
    """Lazy, thread-safe fastembed wrapper. The ONNX model (~70-130 MB RAM) loads on first use."""

    def __init__(self, name=None, threads=None, cache_dir=None):
        self.model_name = name or model_name()
        self.threads = int(threads or setting("SEARCH_EMBED_THREADS", 1))
        self.cache_dir = cache_dir or setting("FASTEMBED_CACHE_PATH")
        self._model = None
        self._doc_model = None
        self._lock = threading.Lock()
        self._cache = OrderedDict()
        self.dim = None

    def _load(self, threads):
        from fastembed import TextEmbedding

        kw = {"cache_dir": self.cache_dir} if self.cache_dir else {}
        return TextEmbedding(self.model_name, threads=threads, **kw)

    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load(self.threads)
        return self._model

    def embed_query(self, text) -> np.ndarray:
        with self._lock:
            hit = self._cache.get(text)
            if hit is not None:
                self._cache.move_to_end(text)
                return hit
        m = self.model()
        with self._lock:   # onnxruntime sessions are thread-safe, but one at a time keeps CPU use flat
            v = normalize_rows(next(iter(m.query_embed([text]))))
            self.dim = v.shape[-1]
            self._cache[text] = v
            if len(self._cache) > _QUERY_CACHE:
                self._cache.popitem(last=False)
        return v

    def embed_documents(self, texts, batch_size=64) -> np.ndarray:
        if self._doc_model is None:
            self._doc_model = self._load(None)     # loader: use every core
        out = normalize_rows(np.stack(list(self._doc_model.embed(list(texts), batch_size=batch_size))))
        self.dim = out.shape[-1]
        return out


_EMBEDDER = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                _EMBEDDER = Embedder()
    return _EMBEDDER


class QueryEncoder:
    """What the API uses: embed a query, or None when semantic search is unavailable/disabled.
    The first failure (no package, model download failed, ...) is logged once and disables it."""

    def __init__(self, embedder=None, mode=None):
        self.mode = mode or semantic_mode()
        self._embedder = embedder
        self.disabled_reason = None
        if self.mode in ("off", "0", "false", "no"):
            self.disabled_reason = "SEARCH_SEMANTIC=off"
        elif embedder is None and not package_available():
            self.disabled_reason = "fastembed is not installed"
            log.info("semantic search disabled: %s (lexical signals only)", self.disabled_reason)

    @property
    def model_name(self):
        return (self._embedder or get_embedder()).model_name if not self.disabled_reason else None

    def encode(self, text):
        if self.disabled_reason:
            return None
        try:
            return (self._embedder or get_embedder()).embed_query(text)
        except Exception as e:   # model files missing / download blocked / onnxruntime error
            self.disabled_reason = f"{type(e).__name__}: {e}"
            log.warning("semantic search disabled: %s (lexical signals only)", self.disabled_reason)
            return None
