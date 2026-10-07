"""
app.py
--------
Flask application factory. Preloads the bundled sample dataset under a
fixed id ("sample") so the frontend can show something immediately, and
enables CORS so the standalone frontend (served from a different
origin/port, or opened as a local file) can call this API.
"""

import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS

from agent.data_loader import load_csv
from api.routes import api_bp
from api.dataset_store import store

SAMPLE_DATASET_ID = "sample"


def create_app() -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB uploads

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"error": "That file is too large (limit 25 MB)."}), 413

    # Demo-scope CORS: allow any origin. Restrict this to your frontend's
    # actual origin before deploying this anywhere public.
    CORS(app)

    _preload_sample_dataset()

    app.register_blueprint(api_bp, url_prefix="/api")
    return app


def _preload_sample_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_path = os.path.join(base_dir, "data", "sample_sales.csv")
    if os.path.exists(sample_path):
        df = load_csv(sample_path)
        store.add_with_id(SAMPLE_DATASET_ID, df, "sample_sales.csv (bundled example)")
