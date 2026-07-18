# ============================================================
# MODEL CARD — Resume Screener AI
# ============================================================
# Builds and saves a structured model card following
# Google Model Cards standard (Mitchell et al. 2019).
#
# Contains:
#   - Model metadata (name, version, date)
#   - Dataset splits info
#   - All evaluation metrics (macro F1, top-3 accuracy, etc.)
#   - Match decision summary
#   - Business impact (recruiter time savings)
#   - Feature importances
#   - SHAP top features
#   - Pipeline configuration
# ============================================================

import os
import json
import time
import logging

from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# BUILD MODEL CARD
# ============================================================

def build_model_card(
    selected_name:    str,
    train_fit_size:   int,
    cal_size:         int,
    test_size:        int,
    class_dist_train: dict,
    metrics:          dict,   # test_f1_macro, test_acc, top3_acc, roc_auc_ovr, etc.
    business_impact:  dict,
    decision_counts:  dict,
    feature_order:    list,
    selector_k:       int,
    version:          str  = "v1",
    fi_dict:          Optional[dict] = None,
    shap_dict:        Optional[dict] = None,
    num_classes:      int  = 24,
) -> dict:
    """
    Assembles the full model card dictionary.

    Args:
        selected_name     : winning model name (e.g. 'LightGBM')
        train_fit_size    : rows used for training
        cal_size          : rows used for calibration
        test_size         : rows used for final evaluation
        class_dist_train  : {category: count} distribution in training
        metrics           : dict of evaluation metric values
        business_impact   : output of recruiter_time_savings()
        decision_counts   : {decision: count} from match_engine
        feature_order     : column order used by preprocessor
        selector_k        : k used in SelectKBest (if any)
        version           : model version string
        fi_dict           : feature importances dict (optional)
        shap_dict         : SHAP top features dict (optional)
        num_classes       : number of job categories

    Returns:
        model_card dict (JSON-serialisable)
    """

    card = {

        # ── Identity ──────────────────────────────────────────
        "model_version":    version,
        "model_name":       selected_name,
        "trained_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "project":          "Resume Screener AI",
        "domain":           "HR Tech / Recruitment",
        "problem_type":     "Multi-class Classification + Semantic Ranking",
        "intended_use":     (
            "Automated resume screening and candidate ranking against job "
            "descriptions. Reduces recruiter screening time by 60-70%. "
            "Intended for use as a decision-support tool — human review "
            "required before final hiring decisions."
        ),
        "out_of_scope": (
            "Not designed for: (1) final hiring decisions without human review, "
            "(2) resumes in languages other than English, "
            "(3) roles outside the 24 supported job categories."
        ),
        "ethical_considerations": (
            "Model trained on text only. Resume screening can perpetuate bias "
            "if training data reflects historical hiring patterns. Regular "
            "fairness audits recommended. Model outputs must be reviewed by "
            "qualified HR professionals before use in hiring."
        ),

        # ── Dataset ───────────────────────────────────────────
        "dataset": {
            "source":           "Kaggle — Resume Dataset (snehaanbhawal)",
            "total_resumes":    train_fit_size + cal_size + test_size,
            "train_fit_size":   train_fit_size,
            "cal_size":         cal_size,
            "test_size":        test_size,
            "num_categories":   num_classes,
            "class_distribution_train": class_dist_train,
        },

        # ── Metrics ───────────────────────────────────────────
        "metrics": {
            "test_f1_macro":    metrics.get("test_f1_macro"),
            "test_f1_weighted": metrics.get("test_f1_weighted"),
            "test_accuracy":    metrics.get("test_accuracy"),
            "top3_accuracy":    metrics.get("top3_accuracy"),
            "roc_auc_ovr":      metrics.get("roc_auc_ovr"),
            "cv_f1_macro_mean": metrics.get("cv_f1_macro_mean"),
            "cv_f1_macro_std":  metrics.get("cv_f1_macro_std"),
            "train_test_gap":   metrics.get("train_test_gap"),
        },

        # ── Business Impact ───────────────────────────────────
        "business_impact": business_impact,

        # ── Match Decision Distribution ────────────────────────
        "decision_distribution": decision_counts,

        # ── Pipeline Config ───────────────────────────────────
        "pipeline_config": {
            "vectorizer":      "TF-IDF (10k features, unigrams+bigrams)",
            "struct_features": feature_order,
            "selector_k":      selector_k,
            "clip_fold":       1.5,
            "calibration":     "isotonic (CalibratedClassifierCV)",
        },

        # ── Feature Importances ───────────────────────────────
        "feature_importances": fi_dict or {},

        # ── SHAP Top Features ─────────────────────────────────
        "shap_top_features": shap_dict or {},

        # ── Limitations ───────────────────────────────────────
        "limitations": [
            "Dataset size (~2.4k resumes) is small — model may not generalise "
            "to niche or senior-level roles",
            "Text-only: ignores portfolio links, GitHub activity, certifications "
            "not mentioned in resume body",
            "24 fixed job categories — open-domain JDs not supported",
            "May exhibit keyword-stuffing vulnerability — candidates padding "
            "resumes with JD terms",
        ],

        # ── Monitoring Recommendations ────────────────────────
        "monitoring": {
            "psi_moderate_threshold":  0.10,
            "psi_high_threshold":      0.20,
            "recommended_retrain_trigger": (
                "Retrain when: (1) PSI > 0.20 on token_count or keyword_density, "
                "(2) F1_macro drops > 5% in A/B production test, "
                "(3) new job category emerges in hiring data"
            ),
        },
    }

    return card


# ============================================================
# SAVE MODEL CARD
# ============================================================

def save_model_card(card: dict, model_dir: str, name: str, version: str) -> str:
    """Saves model card JSON to model_dir. Returns file path."""
    safe_name = name.replace(" ", "_")
    path      = os.path.join(model_dir, f"model_card_{safe_name}_{version}.json")

    os.makedirs(model_dir, exist_ok=True)

    with open(path, "w") as f:
        json.dump(card, f, indent=2, default=str)

    logger.info("Model card saved → %s", path)
    return path
