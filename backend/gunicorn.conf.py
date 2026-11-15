"""Gunicorn settings for the API.

The risk model keeps ~600 MB of crime history in memory per process, so we run
few processes with threads (see risk-model/README.md) and warm the model and the
route planner's walk graph in the background when each worker starts, so the first
/risk or /route request doesn't wait.
"""
import os
import threading

bind = "0.0.0.0:5000"
workers = int(os.getenv("GUNICORN_WORKERS", "2"))
threads = int(os.getenv("GUNICORN_THREADS", "4"))
timeout = 300
accesslog = "-"


def post_worker_init(worker):
    def warm():
        with worker.wsgi.app_context():  # worker.wsgi is the loaded Flask app
            from app.services import risk_model, routing, search
            search.warm()       # place-search index (~60 MB) + embedding model (~100 MB) per worker
            risk_model.warm()
            routing.warm()

    threading.Thread(target=warm, name="risk-model-warmup", daemon=True).start()
