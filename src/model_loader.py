# ============================================================
# MODEL LOADER + CHALLENGER SYSTEM — Resume Screener AI
# ============================================================

import os
import json
import joblib
import logging
import time
import numpy as np

from src.config import (
    MODEL_DIR, MIN_F1_IMPROVEMENT,
    MIN_ACCURACY_THRESHOLD, MAX_GENERALIZATION_GAP
)

logger = logging.getLogger(__name__)

CHALLENGER_LOG = os.path.join(MODEL_DIR, "challenger_log.json")


# ============================================================
# JSON SERIALIZER — converts numpy types to native Python
# ============================================================

def _json_safe(obj):
    """
    Recursively converts numpy scalar types (bool_, int64, float64)
    to native Python types so json.dump never crashes.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_safe(i) for i in obj]
    elif isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    else:
        # numpy bool_, int64, float32, float64, etc.
        try:
            return obj.item()
        except AttributeError:
            return str(obj)


# ============================================================
# LOAD LATEST (CHAMPION) MODEL
# ============================================================

def load_latest_model():
    """Loads champion model from registry."""
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        raise FileNotFoundError(
            f"Model registry not found at {registry_path}. "
            "Run train_model.py first."
        )

    with open(registry_path) as f:
        registry = json.load(f)

    model_path = os.path.join(MODEL_DIR, registry["model_name"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    payload       = joblib.load(model_path)
    pipeline      = payload["pipeline"]
    label_encoder = payload.get("label_encoder", None)

    logger.info("Champion model loaded: %s", model_path)
    return pipeline, label_encoder


# ============================================================
# LOAD CHAMPION METRICS
# ============================================================

def _load_champion_metrics() -> dict:
    """Reads current champion metrics from model card."""
    registry_path = os.path.join(MODEL_DIR, "latest_model.json")

    if not os.path.exists(registry_path):
        return {}

    with open(registry_path) as f:
        registry = json.load(f)

    card_path = registry.get("model_card_path", "")

    if not card_path or not os.path.exists(card_path):
        logger.warning("Champion model card not found: %s", card_path)
        return {}

    with open(card_path) as f:
        card = json.load(f)

    metrics = card.get("metrics", {})

    return {
        "model_name":     card.get("model_name", "unknown"),
        "f1_macro":       float(metrics.get("test_f1_macro",  0) or 0),
        "accuracy":       float(metrics.get("test_accuracy",   0) or 0),
        "train_test_gap": float(metrics.get("train_test_gap",  0) or 0),
    }


# ============================================================
# CHALLENGER COMPARISON — 3-GATE SYSTEM
# ============================================================

def run_challenger_comparison(
    challenger_name:       str,
    challenger_f1:         float,
    challenger_accuracy:   float,
    challenger_gap:        float,
    challenger_model_path: str,
    challenger_card_path:  str = "",
) -> dict:

    os.makedirs(MODEL_DIR, exist_ok=True)

    champion = _load_champion_metrics()

    # ── No existing champion → auto-promote ───────────────────
    if not champion:
        logger.info("No champion found — challenger auto-promoted as first model")
        _update_registry(challenger_model_path, challenger_card_path)
        result = {
            "decision":            "PROMOTED",
            "reason":              "No existing champion — first model auto-promoted",
            "challenger_name":     challenger_name,
            "challenger_f1":       round(float(challenger_f1),       4),
            "challenger_accuracy": round(float(challenger_accuracy), 4),
            "champion_name":       None,
            "champion_f1":         None,
            "evaluated_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_challenger_log(result)
        return result

    champion_f1   = champion.get("f1_macro",  0.0)
    champion_acc  = champion.get("accuracy",   0.0)
    champion_name = champion.get("model_name", "unknown")

    logger.info("=" * 55)
    logger.info("CHAMPION vs CHALLENGER")
    logger.info(
        "  Champion  : %-22s  F1_macro=%.4f  Acc=%.4f",
        champion_name, champion_f1, champion_acc
    )
    logger.info(
        "  Challenger: %-22s  F1_macro=%.4f  Acc=%.4f",
        challenger_name, challenger_f1, challenger_accuracy
    )
    logger.info("=" * 55)

    # ── 3 Gates ───────────────────────────────────────────────
    gate1 = bool((challenger_f1 - champion_f1) >= MIN_F1_IMPROVEMENT)
    gate2 = bool(challenger_accuracy >= MIN_ACCURACY_THRESHOLD)
    gate3 = bool(challenger_gap      <= MAX_GENERALIZATION_GAP)

    if gate1 and gate2 and gate3:
        decision = "PROMOTED"
        reason   = (
            f"Challenger beats champion: "
            f"F1 {champion_f1:.4f} → {challenger_f1:.4f} "
            f"(+{challenger_f1 - champion_f1:.4f})"
        )
        logger.info("✅ CHALLENGER PROMOTED → new champion: %s", challenger_name)
        _update_registry(challenger_model_path, challenger_card_path)

    else:
        decision = "REJECTED"
        failed   = []
        if not gate1:
            failed.append(
                f"F1 improvement {challenger_f1 - champion_f1:+.4f} < {MIN_F1_IMPROVEMENT}"
            )
        if not gate2:
            failed.append(
                f"Accuracy {challenger_accuracy:.4f} < {MIN_ACCURACY_THRESHOLD}"
            )
        if not gate3:
            failed.append(
                f"Train-test gap {challenger_gap:.4f} > {MAX_GENERALIZATION_GAP}"
            )
        reason = "Gates failed: " + " | ".join(failed)
        logger.info("❌ CHALLENGER REJECTED — champion '%s' retained", champion_name)
        logger.info("   Reason: %s", reason)

    result = {
        "decision":              decision,
        "reason":                reason,
        "evaluated_at":          time.strftime("%Y-%m-%d %H:%M:%S"),
        "challenger_name":       challenger_name,
        "challenger_f1":         round(float(challenger_f1),       4),
        "challenger_accuracy":   round(float(challenger_accuracy), 4),
        "challenger_gap":        round(float(challenger_gap),      4),
        "champion_name":         champion_name,
        "champion_f1":           round(float(champion_f1),         4),
        "champion_accuracy":     round(float(champion_acc),        4),
        "gates": {
            "f1_improvement_passed": gate1,   # native Python bool
            "accuracy_passed":       gate2,
            "gap_passed":            gate3,
        }
    }

    _save_challenger_log(result)
    return result


# ============================================================
# HELPERS
# ============================================================

def _update_registry(model_path: str, model_card_path: str = ""):
    registry = {
        "model_name":      os.path.basename(model_path),
        "model_card_path": model_card_path or "",
    }
    reg_path = os.path.join(MODEL_DIR, "latest_model.json")
    with open(reg_path, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Registry updated → %s", registry["model_name"])


def _save_challenger_log(result: dict):
    """Appends challenger result — converts all numpy types before JSON dump."""
    history = []
    if os.path.exists(CHALLENGER_LOG):
        try:
            with open(CHALLENGER_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(result)

    # Convert ALL numpy types recursively — no more bool_ crash
    with open(CHALLENGER_LOG, "w") as f:
        json.dump(_json_safe(history), f, indent=2)

    logger.info("Challenger log saved → %s", CHALLENGER_LOG)
