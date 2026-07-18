# ============================================================
# LEAKAGE CHECK — Resume Screener AI
# ============================================================

import re
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def detect_leakage(
    X_train:        pd.DataFrame,
    y_train:        pd.Series,
    text_col:       str   = "cleaned_resume",
    threshold_corr: float = 0.99
) -> list:
    """
    Runs leakage heuristics on training set.
    Returns list of warning strings. Empty = no leakage.
    """
    warnings = []

    for col in X_train.columns:
        if col == text_col:
            continue

        try:
            # ── Check 1: exact match with target ──────────────
            if X_train[col].astype(str).equals(y_train.astype(str)):
                warnings.append(
                    f"[LEAKAGE] '{col}' is identical to target → remove this feature"
                )
                continue

            # ── Check 2: near-perfect numeric correlation ─────
            if np.issubdtype(X_train[col].dtype, np.number):
                col_vals = X_train[col].fillna(0)

                # ── FIX: skip zero-variance columns ───────────
                # has_education = 99% constant → std ≈ 0 → divide-by-zero in corrcoef
                if col_vals.std() < 1e-9:
                    logger.info(
                        "Leakage check skipped for '%s' — zero variance (constant column)", col
                    )
                    continue

                y_num = pd.factorize(y_train)[0].astype(float)

                # y_num also needs variance check
                if y_num.std() < 1e-9:
                    continue

                corr_matrix = np.corrcoef(col_vals.values, y_num)
                corr = abs(corr_matrix[0, 1])

                # Handle NaN from corrcoef (shouldn't happen after std check)
                if np.isnan(corr):
                    continue

                if corr >= threshold_corr:
                    warnings.append(
                        f"[LEAKAGE] '{col}' has corr={corr:.4f} with target "
                        f"(>= {threshold_corr}) → possible leakage"
                    )

        except Exception as e:
            logger.warning("Leakage check failed for column '%s': %s", col, e)

    # ── Check 3: category label literal in resume text ────────
    if text_col in X_train.columns:
        try:
            sample = X_train[[text_col]].copy()
            sample["__label"] = y_train.values

            def _label_in_text(row):
                label = str(row["__label"]).lower()
                text  = str(row[text_col]).lower()
                return text.count(label) >= 3

            flag_rate = sample.apply(_label_in_text, axis=1).mean()
            if flag_rate > 0.5:
                warnings.append(
                    f"[LEAKAGE] Category label appears 3+ times in >{flag_rate:.0%} "
                    f"of resumes — possible labelling-sourced leakage. "
                    f"Monitor eval carefully."
                )
        except Exception as e:
            logger.warning("Text leakage check failed: %s", e)

    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("Leakage check passed — no obvious leakage detected")

    return warnings
