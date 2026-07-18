# ============================================================
# TRAINING PIPELINE — Resume Screener AI
# ============================================================
# End-to-end pipeline:
#   1. Load & validate data
#   2. Text cleaning + feature engineering
#   3. Feature type detection
#   4. Train/cal/test split (stratified)
#   5. Leakage detection
#   6. Build dual preprocessors (scaled + unscaled)
#   7. Tune scaled models (Logistic, LinearSVC, SGD, NB)
#   8. Tune unscaled models (RF, ExtraTrees, XGB, LGBM)
#   9. Evaluate all models
#  10. Select best model (f1_macro primary)
#  11. Calibrate probabilities
#  12. Compute feature importance + SHAP
#  13. Build & save model card (Google Model Cards standard)
#  14. Challenger comparison (3-gate system)
#  15. MLflow experiment tracking
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
    top_k_accuracy_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config        import (
    RANDOM_STATE, N_JOBS, MODEL_DIR, JOB_CATEGORIES, NUM_CLASSES
)
from src.data_loader   import (
    validate_input_data, add_engineered_features, detect_feature_types
)
from src.preprocessing import (
    build_preprocessors, encode_labels, safe_k, SELECT_K
)
from src.model_tuning  import scaled_models, unscaled_models, tune_models
from src.evaluation    import (
    evaluate_models, select_best_model, calibrate_with_holdout,
    compute_feature_importance, compute_shap,
    save_model_and_card, mlflow_log_run, safe_predict_proba,
    plot_confusion_matrix,
)
from src.leakage_check  import detect_leakage
from src.model_card     import build_model_card, save_model_card
from src.model_loader   import run_challenger_comparison
from src.metrics        import psi, recruiter_time_savings, simple_drift_report
from src.match_engine   import score_applicant, get_decision_counts

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_training():

    # ─────────────────────────────────────────────────────────
    # 1. LOAD & VALIDATE DATA
    # ─────────────────────────────────────────────────────────
    DATA_PATH = os.getenv(
        "RESUME_DATA_PATH",
        r"D:\Data Science Datasets\resum_screener_dataset\Resume.csv"
    )

    logger.info("Loading data from: %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df = validate_input_data(df)

    # ─────────────────────────────────────────────────────────
    # 2. FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────
    df = add_engineered_features(df)

    # ─────────────────────────────────────────────────────────
    # 3. ENCODE LABELS
    # ─────────────────────────────────────────────────────────
    y_raw = df["Category"]

    # Fit label encoder on ALL known categories (not just present ones)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(sorted(df["Category"].unique()))
    y_enc = le.transform(y_raw)

    X = df.drop(columns=["Category"])

    logger.info(
        "Dataset  |  shape=%s  |  categories=%d",
        df.shape, len(le.classes_)
    )

    # ─────────────────────────────────────────────────────────
    # 4. TRAIN / TEST / CALIBRATION SPLIT (stratified)
    # ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE
    )

    # Reserve 20% of train for calibration
    X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )

    logger.info(
        "Split  |  train_fit=%d  cal=%d  test=%d",
        len(X_train_fit), len(X_cal), len(X_test)
    )

    # ─────────────────────────────────────────────────────────
    # 5. LEAKAGE DETECTION
    # ─────────────────────────────────────────────────────────
    y_train_raw = le.inverse_transform(y_train_fit)
    leak_warnings = detect_leakage(
        X_train_fit, pd.Series(y_train_raw), text_col="cleaned_resume"
    )
    if leak_warnings:
        for w in leak_warnings:
            logger.warning("LEAKAGE: %s", w)
    else:
        logger.info("No obvious leakage detected")

    # ─────────────────────────────────────────────────────────
    # 6. BUILD PREPROCESSORS
    # ─────────────────────────────────────────────────────────
    cont_cols, bin_cols = detect_feature_types(X_train_fit, text_col="cleaned_resume")

    pre_scaled, pre_unscaled, feature_order = build_preprocessors(
        cont_cols  = cont_cols,
        bin_cols   = bin_cols,
        text_col   = "cleaned_resume",
        X_train    = X_train_fit,
    )

    # ─────────────────────────────────────────────────────────
    # 7. TUNE SCALED MODELS (Logistic, LinearSVC, NB, SGD, KNN)
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Tuning scaled models ...")
    scaled_pipelines = tune_models(
        scaled_models, pre_scaled, X_train_fit, y_train_fit
    )

    # ─────────────────────────────────────────────────────────
    # 8. TUNE UNSCALED MODELS (RF, ExtraTrees, LGBM, XGBoost)
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Tuning unscaled models ...")
    unscaled_pipelines = tune_models(
        unscaled_models, pre_unscaled, X_train_fit, y_train_fit
    )

    all_pipelines = {**scaled_pipelines, **unscaled_pipelines}

    # ─────────────────────────────────────────────────────────
    # 9. EVALUATE ALL MODELS
    # ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Evaluating all models ...")
    summary = evaluate_models(
        all_pipelines, X_train_fit, y_train_fit, X_test, y_test
    )

    print("\n" + "=" * 65)
    print("ALL MODELS SUMMARY")
    print("=" * 65)
    print(summary.to_string())

    # Save experiment results CSV
    exp_path = os.path.join(MODEL_DIR, "model_experiment_results.csv")
    summary.to_csv(exp_path)
    logger.info("Experiment results saved → %s", exp_path)

    # ─────────────────────────────────────────────────────────
    # 10. SELECT BEST MODEL
    # ─────────────────────────────────────────────────────────
    selected_name, selected_pipe = select_best_model(
        summary, all_pipelines, scaled_pipelines, unscaled_pipelines
    )

    # ─────────────────────────────────────────────────────────
    # 11. DETAILED EVALUATION — SELECTED MODEL
    # ─────────────────────────────────────────────────────────
    y_pred       = selected_pipe.predict(X_test)
    test_acc     = accuracy_score(y_test, y_pred)
    f1_macro     = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_weighted  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    train_acc    = accuracy_score(y_train_fit, selected_pipe.predict(X_train_fit))
    gap          = train_acc - test_acc

    proba    = safe_predict_proba(selected_pipe, X_test)
    top3_acc = None
    roc_auc  = None
    if proba is not None:
        try:
            top3_acc = top_k_accuracy_score(y_test, proba, k=3)
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(
                y_test, proba, multi_class="ovr", average="macro"
            )
        except Exception as e:
            logger.warning("ROC/Top3 failed: %s", e)

    print("\n" + "=" * 65)
    print(f"SELECTED MODEL: {selected_name}")
    print("=" * 65)
    print(f"  Test Accuracy  : {test_acc:.4f}")
    print(f"  F1 Macro       : {f1_macro:.4f}")
    print(f"  F1 Weighted    : {f1_weighted:.4f}")
    print(f"  Top-3 Accuracy : {top3_acc:.4f}" if top3_acc else "  Top-3 Accuracy: N/A")
    print(f"  Train-Test Gap : {gap:.4f}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=[le.classes_[i] for i in sorted(set(y_test))],
        zero_division=0
    ))

    # ─────────────────────────────────────────────────────────
    # 12. CONFUSION MATRIX PLOT
    # ─────────────────────────────────────────────────────────
    cm_path = os.path.join("docs", "plots", "confusion_matrix.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    label_names = [le.classes_[i] for i in range(len(le.classes_))]
    plot_confusion_matrix(y_test, y_pred, label_names, save_path=cm_path)

    # ─────────────────────────────────────────────────────────
    # 13. PROBABILITY CALIBRATION
    # ─────────────────────────────────────────────────────────
    calibrated_pipe = calibrate_with_holdout(
        selected_pipe, X_cal, y_cal, selected_name
    )
    final_pipe = calibrated_pipe if calibrated_pipe is not None else selected_pipe

    # ─────────────────────────────────────────────────────────
    # 14. FEATURE IMPORTANCE + SHAP
    # ─────────────────────────────────────────────────────────
    try:
        tfidf_vocab = list(
            selected_pipe.named_steps["preprocessor"]
            .named_transformers_["tfidf"]
            .get_feature_names_out()
        )
    except Exception:
        tfidf_vocab = None

    fi_dict   = compute_feature_importance(
        selected_pipe, X_test, y_test, feature_names=tfidf_vocab
    )
    shap_dict = compute_shap(selected_pipe, X_test)

    # ─────────────────────────────────────────────────────────
    # 15. BUSINESS IMPACT METRICS
    # ─────────────────────────────────────────────────────────
    biz_impact = recruiter_time_savings(
        y_true = y_test,
        y_pred = y_pred,
    )

    # ─────────────────────────────────────────────────────────
    # 16. SAVE MODEL + ARTIFACTS
    # ─────────────────────────────────────────────────────────
    VERSION = "v1"
    model_path, _ = save_model_and_card(
        final_pipe, selected_name, VERSION, label_encoder=le
    )

    # Class distribution for model card
    from collections import Counter
    class_dist = {
        le.classes_[k]: int(v)
        for k, v in sorted(Counter(y_train_fit).items())
    }

    metrics_dict = {
        "test_f1_macro":    round(f1_macro,    4),
        "test_f1_weighted": round(f1_weighted, 4),
        "test_accuracy":    round(test_acc,    4),
        "top3_accuracy":    round(top3_acc,    4) if top3_acc else None,
        "roc_auc_ovr":      round(roc_auc,     4) if roc_auc  else None,
        "cv_f1_macro_mean": summary.loc[selected_name, "cv_f1_mean"]
                            if "cv_f1_mean" in summary.columns else None,
        "cv_f1_macro_std":  summary.loc[selected_name, "cv_f1_std"]
                            if "cv_f1_std" in summary.columns else None,
        "train_test_gap":   round(gap,         4),
    }

    card = build_model_card(
        selected_name    = selected_name,
        train_fit_size   = len(X_train_fit),
        cal_size         = len(X_cal),
        test_size        = len(X_test),
        class_dist_train = class_dist,
        metrics          = metrics_dict,
        business_impact  = biz_impact,
        decision_counts  = {"shortlisted": int((y_pred == y_test).sum()),
                            "rejected": int((y_pred != y_test).sum())},
        feature_order    = feature_order,
        selector_k       = SELECT_K,
        version          = VERSION,
        fi_dict          = fi_dict,
        shap_dict        = shap_dict,
        num_classes      = len(le.classes_),
    )

    safe_name = selected_name.replace(" ", "_")
    card_path = save_model_card(card, MODEL_DIR, safe_name, VERSION)

    # ─────────────────────────────────────────────────────────
    # 17. PSI DRIFT REPORT (train vs test score distribution)
    # ─────────────────────────────────────────────────────────
    if proba is not None:
        train_proba = safe_predict_proba(final_pipe, X_train_fit)
        if train_proba is not None:
            # Max class probability as a scalar drift signal
            mid        = len(proba) // 2
            psi_val    = psi(proba[:mid].max(axis=1), proba[mid:].max(axis=1))
            logger.info("PSI (confidence score distribution): %.4f", psi_val)

            drift_df = pd.DataFrame({
                "feature":  ["max_class_confidence"],
                "psi":      [round(psi_val, 4)],
                "status":   ["STABLE" if psi_val < 0.10 else
                             "MODERATE" if psi_val < 0.20 else "HIGH"],
            })
            drift_path = os.path.join(MODEL_DIR, "feature_drift_report.csv")
            drift_df.to_csv(drift_path, index=False)
            logger.info("Drift report saved → %s", drift_path)

    # ── Monitor scores CSV ────────────────────────────────────
    if proba is not None:
        monitor_df = pd.DataFrame({
            "score":  proba.max(axis=1),
            "pred":   le.inverse_transform(y_pred),
            "actual": le.inverse_transform(y_test),
            "correct": (y_pred == y_test).astype(int),
        })
        monitor_path = os.path.join(MODEL_DIR, "monitor_scores.csv")
        monitor_df.to_csv(monitor_path, index=False)
        logger.info("Monitor scores saved → %s", monitor_path)

    # ─────────────────────────────────────────────────────────
    # 18. CHAMPION-CHALLENGER COMPARISON
    # ─────────────────────────────────────────────────────────
    challenger_result = run_challenger_comparison(
        challenger_name       = selected_name,
        challenger_f1         = f1_macro,
        challenger_accuracy   = test_acc,
        challenger_gap        = gap,
        challenger_model_path = model_path,
        challenger_card_path  = card_path,
    )

    print("\n" + "=" * 65)
    print(f"CHALLENGER RESULT: {challenger_result['decision']}")
    print(f"  Reason: {challenger_result['reason']}")
    print("=" * 65)

    # ─────────────────────────────────────────────────────────
    # 19. MLFLOW EXPERIMENT TRACKING
    # ─────────────────────────────────────────────────────────
    mlflow_log_run(
        model_name = selected_name,
        metrics    = {k: v for k, v in metrics_dict.items() if v is not None},
        params     = {"vectorizer": "TF-IDF", "selector_k": SELECT_K},
        pipe       = final_pipe,
    )

    logger.info("=" * 65)
    logger.info("TRAINING COMPLETE")
    logger.info("  Model     : %s", selected_name)
    logger.info("  F1 Macro  : %.4f", f1_macro)
    logger.info("  Accuracy  : %.4f", test_acc)
    logger.info("  Top-3 Acc : %s",  f"{top3_acc:.4f}" if top3_acc else "N/A")
    logger.info("  Saved to  : %s",  model_path)
    logger.info("=" * 65)

    return {
        "model_name":   selected_name,
        "f1_macro":     f1_macro,
        "accuracy":     test_acc,
        "model_path":   model_path,
        "card_path":    card_path,
    }


if __name__ == "__main__":
    run_training()
