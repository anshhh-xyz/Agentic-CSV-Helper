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
import hashlib
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
from agent.memory import store as memory_store

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)

MAX_QUESTION_CHARS = 2000
HASH_LENGTH = 24

def _schema_for(df):
    return tool_registry.dispatch("dataset_profile", {}, ToolContext(df=df, plots_dir=PLOTS_DIR))

def _frontend_schema(df):
    """Shape the UI expects: n_rows, n_columns, columns[{name, dtype, ...}]."""
    return _schema_for(df)

def _read_upload():
    """Validates the multipart upload and returns (raw_bytes, filename) or (None, error_response)."""
    if "file" not in request.files:
        return None, (jsonify({"error": "No file part named 'file' in the request."}), 400)
    file = request.files["file"]
    if file.filename == "":
        return None, (jsonify({"error": "No file selected."}), 400)
    if not file.filename.lower().endswith(".csv"):
        return None, (jsonify({"error": "Only .csv files are supported."}), 400)

    raw = file.read()
    if not raw:
        return None, (jsonify({"error": "The uploaded CSV has no data."}), 400)
    return (raw, file.filename), None


def _load_df_from_bytes(raw: bytes, filename: str):
    import io
    try:
        df = load_csv(io.BytesIO(raw))
    except Exception:
        logger.exception("CSV upload failed to parse: %s", filename)
        return None, (jsonify({"error": "Could not read that file as a CSV. Check it is a valid comma-separated file."}), 400)
    if df.empty or df.shape[1] == 0:
        return None, (jsonify({"error": "The uploaded CSV has no data."}), 400)
    return df, None

@api_bp.route("/commit/<dataset_id>", methods=["POST"])
def commit(dataset_id):
    if not store.exists(dataset_id):
        return jsonify({"error": "Unknown dataset."}), 404
    committed = store.commit(dataset_id)
    if not committed:
        return jsonify({"error": "There are no pending changes to make permanent for this dataset."}), 400
    return jsonify({"committed": True, "schema": _schema_for(store.get(dataset_id))})

@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@api_bp.route("/tools", methods=["GET"])
def tools():
    return jsonify({"count": len(tool_registry.list_tools()), "tools": tool_registry.list_tools()})


@api_bp.route("/upload", methods=["POST"])



@api_bp.route("/upload", methods=["POST"])
def upload():
    upload_data, err = _read_upload()
    if err:
        return err
    raw, filename = upload_data

    content_hash = hashlib.sha256(raw).hexdigest()[:HASH_LENGTH]

    # Exact content already known -> same dataset, memory just works (same scope).
    if store.exists(content_hash):
        df = store.get(content_hash)
        store.record_name(filename, content_hash)
        return jsonify({
            "dataset_id": content_hash, "name": filename, "status": "existing",
            "schema": _schema_for(df),
        })

    df, err = _load_df_from_bytes(raw, filename)
    if err:
        return err

    previous_id = store.find_by_name(filename)
    if previous_id and previous_id != content_hash:
        # Same filename, different content: don't guess -- ask the caller.
        store.stash(content_hash, df, filename)
        return jsonify({
            "status": "needs_resolution",
            "filename": filename,
            "new_dataset_id": content_hash,
            "previous_dataset_id": previous_id,
            "message": (
                f"A dataset named '{filename}' was already analysed. Is this an update to the "
                "same dataset (keep its remembered facts) or a different dataset (start fresh)?"
            ),
        })

    store.add_with_id(content_hash, df, filename)
    store.record_name(filename, content_hash)
    return jsonify({"dataset_id": content_hash, "name": filename, "status": "new", "schema": _schema_for(df)})

@api_bp.route("/upload/resolve", methods=["POST"])
def upload_resolve():
    body = request.get_json(silent=True) or {}
    new_id = body.get("new_dataset_id")
    previous_id = body.get("previous_dataset_id")
    choice = body.get("choice")  # "continue" | "new"

    if not new_id or choice not in ("continue", "new"):
        return jsonify({"error": "'new_dataset_id' and choice ('continue' or 'new') are required."}), 400

    stashed = store.pop_stash(new_id)
    if stashed is None:
        return jsonify({"error": "That upload has expired or was already resolved. Please re-upload."}), 404

    if choice == "continue":
        if not previous_id or not store.exists(previous_id):
            return jsonify({"error": "The previous dataset is no longer available. Choose 'new' instead."}), 404
        store.add_with_id(previous_id, stashed["df"], stashed["name"])
        store.record_name(stashed["name"], previous_id)
        dataset_id = previous_id
    else:
        store.add_with_id(new_id, stashed["df"], stashed["name"])
        store.record_name(stashed["name"], new_id)
        dataset_id = new_id

    return jsonify({
        "dataset_id": dataset_id, "name": stashed["name"], "status": "existing" if choice == "continue" else "new",
        "schema": _schema_for(store.get(dataset_id)),
    })


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
        facts = memory_store.get_all(dataset_id)
        extra_context = None
        if facts:
            lines = [f"- ({f['kind']}) {f['key']}: {f['value']}" for f in facts]
            extra_context = "Remembered facts for this dataset:\n" + "\n".join(lines)
        result = run_agent(df, question, verbose=False, dataset_id=dataset_id, extra_context=extra_context)
    except RuntimeError as e:  # missing GROQ_API_KEY: a setup message, safe to show
        return jsonify({"error": str(e)}), 500
    except LLMRateLimitError as e:
        return jsonify({"error": str(e)}), 429
    except LLMError as e:
        return jsonify({"error": str(e)}), 502
    except Exception:
        logger.exception("Unexpected agent failure")
        return jsonify({"error": "The agent hit an unexpected problem. Please try again."}), 500
    if result["data_changed"]:
        store.set_pending(dataset_id, result["final_df"])

    plot_urls = [f"/api/plots/{os.path.basename(p)}" for p in result["plot_files"]]
    return jsonify({
        "answer": result["answer"],
        "tool_trace": result["tool_trace"],
        "plot_urls": plot_urls,
        "rounds": result["rounds"],
        "data_changed": result["data_changed"],
    })


@api_bp.route("/plots/<path:filename>", methods=["GET"])
def get_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

@api_bp.route("/memory/<dataset_id>", methods=["GET"])
def get_memory(dataset_id):
    return jsonify({"memory": memory_store.get_all(dataset_id)})

@api_bp.route("/memory/<dataset_id>/<key>", methods=["DELETE"])
def delete_memory(dataset_id, key):
    removed = memory_store.delete(dataset_id, key)
    if not removed:
        return jsonify({"error": f"Nothing is remembered under '{key}'."}), 404
    return jsonify({"deleted": key})
