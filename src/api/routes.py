# -*- coding: utf-8 -*-
from flask import Blueprint, request, jsonify
from src.api import model as model_mod

bp = Blueprint("api", __name__)

@bp.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@bp.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        if "instances" not in payload:
            return jsonify({"error": "Missing 'instances'"}), 400

        rows = payload["instances"]
        results = model_mod.predict_batch(rows)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
