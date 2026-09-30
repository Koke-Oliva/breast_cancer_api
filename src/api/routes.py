from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from src.api import model as model_service
from src.api.model import InputValidationError

logger = logging.getLogger(__name__)
bp = Blueprint("api", __name__)


@bp.get("/")
def index():
    return jsonify({
        "service": "breast-cancer-api",
        "status": "ok",
        "health": "/api/health",
        "schema": "/api/schema",
        "predict": "/api/predict",
    })


@bp.get("/api/health")
def health():
    try:
        bundle = model_service.get_bundle()
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "model_version": bundle["model_version"],
        })
    except Exception:
        logger.exception("Health check failed while loading model.")
        return jsonify({"status": "degraded", "model_loaded": False}), 503


@bp.get("/api/schema")
def schema():
    try:
        return jsonify(model_service.get_contract())
    except Exception:
        logger.exception("Could not load prediction schema.")
        return jsonify({"error": "service_unavailable"}), 503


@bp.post("/api/predict")
def predict():
    if not request.is_json:
        return jsonify({
            "error": "unsupported_media_type",
            "message": "Content-Type must be application/json.",
        }), 415

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({
            "error": "invalid_json",
            "message": "Request body must be a JSON object.",
        }), 400

    if "instances" not in payload:
        return jsonify({
            "error": "validation_error",
            "message": "Missing required field 'instances'.",
        }), 422

    try:
        results, model_version = model_service.predict_batch(payload["instances"])
        return jsonify({
            "model_version": model_version,
            "results": results,
        })
    except InputValidationError as exc:
        logger.info("Prediction validation failed: %s", exc)
        return jsonify({
            "error": "validation_error",
            "message": str(exc),
        }), 422
    except FileNotFoundError:
        logger.exception("Model artifact unavailable.")
        return jsonify({"error": "service_unavailable"}), 503
    except Exception:
        logger.exception("Unexpected prediction error.")
        return jsonify({"error": "internal_server_error"}), 500
