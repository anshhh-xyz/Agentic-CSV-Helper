"""
routes.py -- all HTTP endpoints the frontend talks to.

    GET  /api/health                         liveness check
    GET  /api/tools                          registered tools (name, category, description)
    GET  /api/datasets/<dataset_id>/schema   column info for a dataset
    POST /api/upload                         upload a CSV -> dataset_id
    POST /api/ask                            ask a question -> answer + analysis trace + chart URLs
    GET  /api/plots/<filename>               serve a generated chart image

The GROQ_API_KEY lives only on this server; the browser never sees it.
Errors returned to the client are short and friendly -- no tracebacks.
"""

import logging
import os

from flask import Blueprint, jsonify, request, send_from_directory

from agent import tool_registry
from agent.config import PLOTS_DIR
from agent.data_loader import load_csv
from agent.llm_client import LLMError, LLMRateLimitError
from agent.orchestrator import run_agent
from agent.tools._base import ToolContext
from api.dataset_store import store

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)

MAX_QUESTION_CHARS = 2000


def _schema_for(df):
    return tool_registry.dispatch("dataset_profile", {}, ToolContext(df=df, plots_dir=PLOTS_DIR))


def _frontend_schema(df):
    """Shape the UI expects: n_rows, n_columns, columns[{name, dtype, ...}]."""
    return _schema_for(df)


@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@api_bp.route("/tools", methods=["GET"])
def tools():
    return jsonify({"count": len(tool_registry.list_tools()), "tools": tool_registry.list_tools()})


@api_bp.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part named 'file' in the request."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are supported."}), 400

    try:
        df = load_csv(file)
    except Exception:
        logger.exception("CSV upload failed to parse")
        return jsonify({"error": "Could not read that file as a CSV. Check it is a valid comma-separated file."}), 400
    if df.empty or df.shape[1] == 0:
        return jsonify({"error": "The uploaded CSV has no data."}), 400

    dataset_id = store.add(df, file.filename)
    return jsonify({"dataset_id": dataset_id, "name": file.filename, "schema": _frontend_schema(df)})


@api_bp.route("/datasets/<dataset_id>/schema", methods=["GET"])
def get_schema(dataset_id):
    df = store.get(dataset_id)
    if df is None:
        return jsonify({"error": "Unknown dataset. Upload a dataset first."}), 404
    return jsonify({"dataset_id": dataset_id, "name": store.get_name(dataset_id), "schema": _frontend_schema(df)})


@api_bp.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True) or {}
    dataset_id = body.get("dataset_id")
    question = str(body.get("question") or "").strip()

    if not dataset_id or not question:
        return jsonify({"error": "Both 'dataset_id' and 'question' are required."}), 400
    if len(question) > MAX_QUESTION_CHARS:
        return jsonify({"error": f"Question is too long (max {MAX_QUESTION_CHARS} characters)."}), 400

    df = store.get(dataset_id)
    if df is None:
        return jsonify({"error": "Unknown dataset. Upload a dataset first."}), 404

    try:
        result = run_agent(df, question, verbose=False)
    except RuntimeError as e:  # missing GROQ_API_KEY: a setup message, safe to show
        return jsonify({"error": str(e)}), 500
    except LLMRateLimitError as e:
        return jsonify({"error": str(e)}), 429
    except LLMError as e:
        return jsonify({"error": str(e)}), 502
    except Exception:
        logger.exception("Unexpected agent failure")
        return jsonify({"error": "The agent hit an unexpected problem. Please try again."}), 500

    plot_urls = [f"/api/plots/{os.path.basename(p)}" for p in result["plot_files"]]
    return jsonify({
        "answer": result["answer"],
        "tool_trace": result["tool_trace"],
        "plot_urls": plot_urls,
        "rounds": result["rounds"],
    })


@api_bp.route("/plots/<path:filename>", methods=["GET"])
def get_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)
