# ============================================================
# EVALUATION — Resume Screener AI
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from datetime import datetime
from typing   import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    top_k_accuracy_score,
)
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.model_selection   import cross_val_score, StratifiedKFold

from src.config  import RANDOM_STATE, N_JOBS, CV_FOLDS, MODEL_DIR, JOB_CATEGORIES
from src.metrics import psi, recruiter_time_savings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================
# SAFE PREDICT PROBA
# ============================================================

def safe_predict_proba(pipe, X):
    """Handles LinearSVC (no predict_proba) via decision_function."""
    try:
        return pipe.predict_proba(X)
    except AttributeError:
        try:
            df = pipe.decision_function(X)
            exp_df = np.exp(df - df.max(axis=1, keepdims=True))
            return exp_df / exp_df.sum(axis=1, keepdims=True)
        except Exception as e:
            logger.error("safe_predict_proba fully failed: %s", e)
            return None


# ============================================================
# GET BASE ESTIMATOR — unwrap CalibratedClassifierCV
# ============================================================

def _get_base_clf(clf):
    """
    Unwraps CalibratedClassifierCV to get the underlying estimator.
    Works for sklearn's CalibratedClassifierCV with cv='prefit'.
    """
    # Direct estimator attribute (sklearn >= 1.2 prefit)
    if hasattr(clf, "estimator"):
        return clf.estimator
    # Older sklearn
    if hasattr(clf, "base_estimator"):
        return clf.base_estimator
    # CalibratedClassifierCV fitted — get from first calibrated classifier
    if hasattr(clf, "calibrated_classifiers_"):
        inner = clf.calibrated_classifiers_[0]
        if hasattr(inner, "estimator"):
            return inner.estimator
        if hasattr(inner, "base_estimator"):
            return inner.base_estimator
    # Already a base model
    return clf


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_models(
    pipelines: Dict,
    X_train:   object,
    y_train:   object,
    X_test:    object,
    y_test:    object,
) -> pd.DataFrame:
    """Evaluates all fitted pipelines, returns summary DataFrame."""

    rows = []

    for name, pipe in pipelines.items():

        logger.info("Evaluating: %s", name)

        try:
            train_acc = accuracy_score(y_train, pipe.predict(X_train))
        except Exception:
            train_acc = None

        y_pred      = pipe.predict(X_test)
        test_acc    = accuracy_score(y_test, y_pred)
        f1_macro    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec        = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec         = recall_score(y_test, y_pred,    average="macro", zero_division=0)

        proba    = safe_predict_proba(pipe, X_test)
        top3_acc = None
        roc_auc  = None
        if proba is not None:
            try:
                top3_acc = top_k_accuracy_score(y_test, proba, k=3)
                roc_auc  = roc_auc_score(
                    y_test, proba, multi_class="ovr", average="macro"
                )
            except Exception:
                pass

        gap = (train_acc - test_acc) if train_acc is not None else None

        try:
            cv_scores = cross_val_score(
                pipe, X_train, y_train,
                scoring = "f1_macro",
                cv      = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                n_jobs  = N_JOBS,
            )
            cv_mean = cv_scores.mean()
            cv_std  = cv_scores.std()
        except Exception:
            cv_mean = cv_std = None

        rows.append({
            "model":       name,
            "train_acc":   round(train_acc,    4) if train_acc   is not None else None,
            "test_acc":    round(test_acc,     4),
            "gap":         round(gap,          4) if gap         is not None else None,
            "f1_macro":    round(f1_macro,     4),
            "f1_weighted": round(f1_weighted,  4),
            "precision":   round(prec,         4),
            "recall":      round(rec,          4),
            "top3_acc":    round(top3_acc,     4) if top3_acc    is not None else None,
            "roc_auc_ovr": round(roc_auc,      4) if roc_auc     is not None else None,
            "cv_f1_mean":  round(cv_mean,      4) if cv_mean     is not None else None,
            "cv_f1_std":   round(cv_std,       4) if cv_std      is not None else None,
        })

        logger.info(
            "  %-22s  acc=%.4f  f1_macro=%.4f  top3=%-6s  roc=%s",
            name, test_acc, f1_macro,
            f"{top3_acc:.4f}" if top3_acc else "N/A",
            f"{roc_auc:.4f}"  if roc_auc  else "N/A",
        )

    return pd.DataFrame(rows).set_index("model")


# ============================================================
# SELECT BEST MODEL
# ============================================================

def select_best_model(
    summary:     pd.DataFrame,
    pipelines:   dict,
    scaled_pl:   dict,
    unscaled_pl: dict,
) -> Tuple[str, object]:

    valid  = summary.dropna(subset=["f1_macro"])
    if valid.empty:
        raise ValueError("No models could be evaluated successfully")

    ranked = valid.sort_values(["f1_macro", "top3_acc"], ascending=False)
    selected_name = ranked.index[0]
    selected_pipe = pipelines[selected_name]

    logger.info(
        "=== SELECTED MODEL: %s  (f1_macro=%.4f) ===",
        selected_name, ranked.loc[selected_name, "f1_macro"]
    )

    return selected_name, selected_pipe


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_with_holdout(
    pipe:  object,
    X_cal: object,
    y_cal: object,
    name:  str,
) -> Optional[object]:

    if not hasattr(pipe, "predict_proba"):
        logger.info("Calibration skipped for %s (no predict_proba)", name)
        return None

    try:
        calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv="prefit")
        calibrated.fit(X_cal, y_cal)
        logger.info("Calibration successful for %s", name)
        return calibrated
    except Exception as e:
        logger.warning("Calibration failed for %s: %s", name, e)
        return None


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def compute_feature_importance(
    pipe:          object,
    X_test:        object,
    y_test:        object,
    feature_names: List[str] = None,
    top_n:         int = 20,
) -> Optional[dict]:

    try:
        clf      = pipe.named_steps.get("classifier", None)
        base_clf = _get_base_clf(clf) if clf is not None else None

        if base_clf is not None and hasattr(base_clf, "feature_importances_"):
            imps  = base_clf.feature_importances_
            names = feature_names or [f"f{i}" for i in range(len(imps))]
            idx   = np.argsort(imps)[::-1][:top_n]
            return {names[i]: round(float(imps[i]), 6) for i in idx}

        if base_clf is not None and hasattr(base_clf, "coef_"):
            coef  = np.abs(base_clf.coef_).mean(axis=0)
            names = feature_names or [f"f{i}" for i in range(len(coef))]
            idx   = np.argsort(coef)[::-1][:top_n]
            return {names[i]: round(float(coef[i]), 6) for i in idx}

    except Exception as e:
        logger.warning("Feature importance extraction failed: %s", e)

    return None


# ============================================================
# SHAP — fully fixed for CalibratedClassifierCV
# ============================================================

def compute_shap(
    pipe:        object,
    X_test:      object,
    top_n:       int = 20,
    max_samples: int = 100,
) -> Optional[dict]:

    if not SHAP_AVAILABLE:
        logger.info("SHAP not available — skipping")
        return None

    try:
        clf = pipe.named_steps.get("classifier", None)
        pre = pipe.named_steps.get("preprocessor", None)

        if clf is None or pre is None:
            logger.info("SHAP skipped — pipeline steps not found")
            return None

        # ── Unwrap CalibratedClassifierCV to get base tree model ──
        base_clf = _get_base_clf(clf)

        # Only tree models supported by TreeExplainer
        if not hasattr(base_clf, "feature_importances_"):
            logger.info("SHAP skipped — %s is not a tree model", type(base_clf).__name__)
            return None

        # ── Transform features ─────────────────────────────────────
        X_arr = pre.transform(X_test)
        if hasattr(X_arr, "toarray"):
            X_arr = X_arr.toarray()

        X_sample = X_arr[:max_samples].astype(np.float64)

        # ── Compute SHAP ───────────────────────────────────────────
        explainer = shap.TreeExplainer(base_clf)
        shap_vals = explainer.shap_values(X_sample)

        # Multi-class: list of arrays → mean absolute across classes
        if isinstance(shap_vals, list):
            mean_shap = np.mean([np.abs(sv) for sv in shap_vals], axis=0).mean(axis=0)
        else:
            mean_shap = np.abs(shap_vals).mean(axis=0)

        try:
            feat_names = pre.get_feature_names_out()
        except Exception:
            feat_names = np.array([f"f{i}" for i in range(len(mean_shap))])

        idx = np.argsort(mean_shap)[::-1][:top_n]
        result = {str(feat_names[i]): round(float(mean_shap[i]), 6) for i in idx}

        logger.info("SHAP computed — top feature: %s", list(result.keys())[0])
        return result

    except Exception as e:
        logger.warning("SHAP computation failed: %s", e)
        return None


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

def plot_confusion_matrix(
    y_true:    object,
    y_pred:    object,
    labels:    List[str],
    save_path: str = None,
) -> None:

    cm  = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, annot_kws={"size": 7}
    )
    ax.set_title("Normalised Confusion Matrix — Resume Screener AI", fontsize=14)
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("True Category")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved → %s", save_path)

    plt.close()


# ============================================================
# SAVE MODEL
# ============================================================

def save_model_and_card(
    pipe:         object,
    name:         str,
    version:      str = "v1",
    label_encoder = None,
) -> Tuple[str, str]:

    os.makedirs(MODEL_DIR, exist_ok=True)
    safe_name  = name.replace(" ", "_")
    model_path = os.path.join(MODEL_DIR, f"resume_model_{safe_name}_{version}.joblib")

    payload = {"pipeline": pipe, "label_encoder": label_encoder}
    joblib.dump(payload, model_path)
    logger.info("Model saved → %s", model_path)

    return model_path, ""


# ============================================================
# MLFLOW
# ============================================================

def mlflow_log_run(
    model_name: str,
    metrics:    dict,
    params:     dict   = None,
    model_path: str    = None,
    pipe:       object = None,
) -> None:

    if not MLFLOW_AVAILABLE:
        return

    try:
        mlflow.set_experiment("resume-screener-ai")
        with mlflow.start_run(run_name=model_name):
            mlflow.log_metrics(metrics)
            if params:
                mlflow.log_params(params)
            if pipe is not None:
                mlflow.sklearn.log_model(pipe, artifact_path="model")
        logger.info("MLflow run logged: %s", model_name)
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)
