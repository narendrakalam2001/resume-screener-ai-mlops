# ============================================================
# METRICS — Resume Screener AI
# ============================================================

import numpy as np
import pandas as pd
import logging

from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    top_k_accuracy_score
)

logger = logging.getLogger(__name__)


# ============================================================
# THRESHOLD TUNING (binary: top-class vs rest)
# ============================================================

def tune_threshold(
    y_true:           np.ndarray,
    y_prob:           np.ndarray,
    target_recall:    float = None,
    target_precision: float = None
) -> float:
    """
    Default: maximize F1.
    Optional: constrain on recall or precision first.
    For multi-class use macro-averaged proba of predicted class.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx  = np.nanargmax(f1_scores)
    best_thr  = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    if target_recall is not None:
        idxs = np.where(recall >= target_recall)[0]
        if idxs.size > 0:
            chosen   = idxs[np.argmax(precision[idxs])]
            best_thr = thresholds[chosen] if chosen < len(thresholds) else 0.5

    if target_precision is not None:
        idxs = np.where(precision >= target_precision)[0]
        if idxs.size > 0:
            chosen   = idxs[np.argmax(recall[idxs])]
            best_thr = thresholds[chosen] if chosen < len(thresholds) else 0.5

    logger.info(
        "Best threshold: %.4f  |  precision=%.4f  recall=%.4f  f1=%.4f",
        best_thr, precision[best_idx], recall[best_idx], f1_scores[best_idx]
    )

    return float(best_thr)


# ============================================================
# PSI — Population Stability Index
# ============================================================

def psi(expected, actual, buckets: int = 10) -> float:
    """
    Population Stability Index — measures distribution shift.

    PSI < 0.1   → no significant shift (stable)
    PSI 0.1–0.2 → moderate shift (monitor closely)
    PSI > 0.2   → major shift (retrain recommended)

    Correct implementation:
      1. Compute quantile bin EDGES from `expected` (reference)
      2. Bin BOTH using those SAME edges
      3. Compare proportions

    Common bug to avoid:
      Ranking both distributions independently → PSI always ~0.
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual   = np.asarray(actual,   dtype=float)

        quantiles  = np.linspace(0, 100, buckets + 1)
        bin_edges  = np.percentile(expected, quantiles)
        bin_edges  = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0]  = min(bin_edges[0],  actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        exp_hist, _ = np.histogram(expected, bins=bin_edges)
        act_hist, _ = np.histogram(actual,   bins=bin_edges)

        exp_pct = exp_hist / (exp_hist.sum() + 1e-9)
        act_pct = act_hist / (act_hist.sum() + 1e-9)

        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)

        psi_value = float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))
        return psi_value

    except Exception as e:
        logger.warning("PSI computation failed: %s", e)
        return float("nan")


# ============================================================
# RANKING METRICS (HR-relevant)
# ============================================================

def precision_at_k(y_true, y_scores, k: int = 10) -> float:
    """
    Precision@K: fraction of top-K predictions that are correct.
    Relevant for resume ranking use-case.
    """
    df = (
        pd.DataFrame({"y": y_true, "s": y_scores})
        .sort_values("s", ascending=False)
    )
    top_k = df.head(k)
    return float((top_k["y"] == top_k["y"].mode()[0]).mean())


def recall_at_k(y_true, y_prob, k: float = 0.05) -> float:
    """
    Recall@K: fraction of positives captured in top k% of ranked list.
    """
    df = (
        pd.DataFrame({"y": y_true, "p": y_prob})
        .sort_values("p", ascending=False)
    )
    top_n = int(len(df) * k)
    return float(df.iloc[:top_n]["y"].sum() / (df["y"].sum() + 1e-9))


def lift_at_k(y_true, y_prob, k: float = 0.05) -> float:
    base = np.mean(y_true)
    return float(recall_at_k(y_true, y_prob, k) / (base + 1e-9))


def ks_statistic(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


# ============================================================
# BUSINESS IMPACT: RECRUITER TIME SAVINGS
# ============================================================

def recruiter_time_savings(
    y_true,
    y_pred,
    minutes_per_resume:        float = 8.0,   # avg manual screening time
    automation_rate_target:    float = 0.70,   # HR industry benchmark (70%)
    hourly_rate_usd:           float = 25.0,   # recruiter cost per hour
) -> dict:
    """
    Estimates business value of automated resume screening.

    Based on Naukri.com / Darwinbox benchmark:
      - AI screener reduces recruiter time by 60–70%
      - Each manual screening: 5–10 minutes per resume
      - For 1000 applicants/month: 133 hours saved

    Returns cost savings and automation coverage metrics.
    """
    n_total       = len(y_true)
    n_correct     = int((np.array(y_true) == np.array(y_pred)).sum())
    accuracy      = n_correct / (n_total + 1e-9)

    # Time saved (only on correctly auto-screened resumes)
    n_auto        = int(n_total * automation_rate_target * accuracy)
    hours_saved   = (n_auto * minutes_per_resume) / 60.0
    cost_saved    = hours_saved * hourly_rate_usd

    # False positives sent to recruiter unnecessarily
    n_fp = 0  # multi-class mein binary FP concept apply nahi hota

    result = {
        "total_resumes":          n_total,
        "auto_screened":          n_auto,
        "automation_coverage":    round(n_auto / (n_total + 1e-9), 4),
        "hours_saved_per_1k":     round(hours_saved * (1000 / n_total), 2),
        "cost_saved_per_1k_usd":  round(cost_saved * (1000 / n_total), 2),
        "accuracy":               round(accuracy, 4),
    }

    logger.info(
        "Business impact  |  coverage=%.1f%%  hours_saved/1k=%.1f  cost_saved/1k=$%.0f",
        result["automation_coverage"] * 100,
        result["hours_saved_per_1k"],
        result["cost_saved_per_1k_usd"],
    )

    return result


# ============================================================
# DRIFT REPORT — feature mean shift
# ============================================================

def simple_drift_report(
    X_ref: pd.DataFrame,
    X_new: pd.DataFrame,
    top_n: int = 10
) -> pd.Series:
    """Computes relative mean drift between reference and new data."""
    numeric_cols = X_ref.select_dtypes(include=np.number).columns
    diffs = (X_ref[numeric_cols].mean() - X_new[numeric_cols].mean()).abs()
    rel   = (diffs / (X_ref[numeric_cols].std().replace(0, 1))).sort_values(ascending=False)
    return rel.head(top_n)
