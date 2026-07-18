# ============================================================
# SCREENING SERVICE — Resume Screener AI
# ============================================================

import numpy as np
import pandas as pd
import logging

from src.data_loader  import add_engineered_features, clean_resume_text
from src.match_engine import score_applicant, rank_candidates

logger = logging.getLogger(__name__)


# ============================================================
# PREPARE FEATURES — for API inference
# ============================================================

def prepare_features(resume_text: str) -> pd.DataFrame:
    """
    Takes raw resume text string → returns engineered feature DataFrame.
    Mirrors the training pipeline feature engineering exactly.
    """
    df = pd.DataFrame({"Resume_str": [resume_text]})
    df = add_engineered_features(df)
    return df


# ============================================================
# SCREEN RESUME — single candidate
# ============================================================

def screen_resume(
    pipeline,
    label_encoder,
    resume_text:    str,
    jd_text:        str = "",
    years_required: int = 0,
) -> dict:
    """
    Full screening pipeline for one resume:
      1. Feature engineering
      2. Model classification (predict_proba)
      3. Match engine scoring (rules + ML)

    Returns structured screening result dict.
    """
    df = prepare_features(resume_text)

    # ── Model inference ────────────────────────────────────────
    try:
        proba = pipeline.predict_proba(df)[0]
    except AttributeError:
        # LinearSVC fallback
        try:
            df_fn = pipeline.decision_function(df)[0]
            exp_df = np.exp(df_fn - df_fn.max())
            proba  = exp_df / exp_df.sum()
        except Exception as e:
            logger.error("Inference failed: %s", e)
            proba  = np.ones(len(label_encoder.classes_)) / len(label_encoder.classes_)

    # ── Top-3 categories ──────────────────────────────────────
    top3_idx        = np.argsort(proba)[::-1][:3]
    predicted_idx   = top3_idx[0]
    predicted_cat   = label_encoder.classes_[predicted_idx]
    ml_score        = float(proba[predicted_idx])

    top3 = [
        {
            "category":   label_encoder.classes_[i],
            "confidence": round(float(proba[i]), 4),
        }
        for i in top3_idx
    ]

    # ── Resume row for match engine ────────────────────────────
    resume_row = df.iloc[0].to_dict()

    # ── Match engine scoring ───────────────────────────────────
    match_result = score_applicant(
        resume_row     = resume_row,
        ml_score       = ml_score,
        jd_text        = jd_text,
        years_required = years_required,
    )

    result = {
        "predicted_category": predicted_cat,
        "top3_categories":    top3,
        **match_result,
    }

    logger.info(
        "Screen  |  cat=%s  ml=%.3f  final=%.3f  rec=%s",
        predicted_cat,
        ml_score,
        match_result["final_score"],
        match_result["recommendation"],
    )

    return result


# ============================================================
# BATCH RANK — for ranking a pool of candidates
# ============================================================

def batch_rank(
    pipeline,
    label_encoder,
    resumes:        list,
    jd_text:        str = "",
    years_required: int = 0,
    top_k:          int = 10,
) -> list:
    """
    Screens all resumes, returns top_k ranked candidates.
    """
    resume_rows  = []
    ml_scores    = []
    predictions  = []

    for resume_text in resumes:
        df = prepare_features(resume_text)

        try:
            proba = pipeline.predict_proba(df)[0]
        except AttributeError:
            try:
                df_fn = pipeline.decision_function(df)[0]
                exp_df = np.exp(df_fn - df_fn.max())
                proba  = exp_df / exp_df.sum()
            except Exception:
                proba  = np.ones(len(label_encoder.classes_)) / len(label_encoder.classes_)

        top1_idx = np.argmax(proba)
        ml_scores.append(float(proba[top1_idx]))
        predictions.append(label_encoder.classes_[top1_idx])
        resume_rows.append(df.iloc[0].to_dict())

    ranked = rank_candidates(
        resumes        = resume_rows,
        ml_scores      = ml_scores,
        jd_text        = jd_text,
        years_required = years_required,
        top_k          = top_k,
    )

    # Attach predicted category to each ranked result
    for r in ranked:
        idx = r["candidate_idx"]
        r["predicted_category"] = predictions[idx]

    return ranked
