# ============================================================
# RESUME SCREENER API — FastAPI Serving
# ============================================================
# Endpoints:
#   GET  /            → home
#   GET  /health      → health check
#   GET  /model_info  → current champion model card
#   POST /predict     → single resume vs JD match
#   POST /rank        → batch ranking of candidates
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import logging
import time
import os
import json

from src.model_loader            import load_latest_model
from services.screening_service  import screen_resume, batch_rank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "Resume Screener AI — API",
    description = (
        "AI-powered resume screening system. Ranks candidates against job "
        "descriptions using TF-IDF + ML classification. "
        "Deployed at: Naukri.com / Darwinbox / Workday equivalent stack."
    ),
    version     = "1.0.0",
)

# ── Load champion model on startup ────────────────────────────
try:
    pipeline, label_encoder = load_latest_model()
    logger.info("Champion model loaded successfully")
except Exception as e:
    logger.error("Model loading failed: %s", e)
    pipeline      = None
    label_encoder = None


# ============================================================
# INPUT SCHEMAS
# ============================================================

class ResumeInput(BaseModel):
    resume_text: str = Field(
        ...,
        description="Raw resume text (plain text, not HTML)",
        min_length=50,
    )
    job_description: Optional[str] = Field(
        default="",
        description="Job description text for keyword overlap scoring",
    )
    years_required: Optional[int] = Field(
        default=0,
        description="Minimum years of experience required by JD",
        ge=0,
    )


class BatchRankInput(BaseModel):
    resumes: List[str] = Field(
        ...,
        description="List of raw resume texts",
        min_items=1,
        max_items=500,
    )
    job_description: str = Field(
        ...,
        description="Job description text to rank candidates against",
        min_length=20,
    )
    years_required: Optional[int] = Field(default=0, ge=0)
    top_k:          Optional[int] = Field(default=10, ge=1, le=100)


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message":     "Resume Screener AI API is live 🚀",
        "docs":        "/docs",
        "health":      "/health",
        "predict":     "POST /predict",
        "rank":        "POST /rank",
    }


@app.get("/health")
def health():
    return {
        "status":       "running",
        "model_loaded": pipeline is not None,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/model_info")
def model_info():
    registry_path = "resume_models/latest_model.json"
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            reg = json.load(f)

        # Also attach model card metrics if available
        card_path = reg.get("model_card_path", "")
        if card_path and os.path.exists(card_path):
            with open(card_path) as f:
                card = json.load(f)
            reg["metrics"]         = card.get("metrics", {})
            reg["business_impact"] = card.get("business_impact", {})
        return reg
    return {"error": "Model registry not found — run train_model.py first"}


# ── Single resume screening ────────────────────────────────────

@app.post("/predict")
def predict(data: ResumeInput):
    """
    Screens a single resume against an optional job description.

    Returns:
        - predicted_category : most likely job category
        - top3_categories    : top-3 predicted categories with confidence
        - final_score        : composite match score [0, 1]
        - match_band         : STRONG / MODERATE / WEAK
        - recommendation     : SHORTLIST / REVIEW / REJECT
        - keyword_overlap    : fraction of JD keywords found in resume
        - rule_triggered     : any business rule that overrode ML score
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — run train_model.py first"
        )

    start  = time.time()
    result = screen_resume(
        pipeline        = pipeline,
        label_encoder   = label_encoder,
        resume_text     = data.resume_text,
        jd_text         = data.job_description or "",
        years_required  = data.years_required or 0,
    )

    # ── Log prediction ─────────────────────────────────────────
    _log_prediction(result, data.resume_text[:100])

    result["latency_seconds"] = round(time.time() - start, 4)
    return result


# ── Batch ranking ──────────────────────────────────────────────

@app.post("/rank")
def rank(data: BatchRankInput):
    """
    Ranks a pool of candidates against a job description.
    Returns top_k candidates sorted by match score descending.

    Typical use: 500 applicants for one JD → returns top 10.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — run train_model.py first"
        )

    if len(data.resumes) == 0:
        raise HTTPException(status_code=400, detail="No resumes provided")

    start  = time.time()
    ranked = batch_rank(
        pipeline        = pipeline,
        label_encoder   = label_encoder,
        resumes         = data.resumes,
        jd_text         = data.job_description,
        years_required  = data.years_required or 0,
        top_k           = data.top_k or 10,
    )

    return {
        "total_candidates":    len(data.resumes),
        "top_k":               data.top_k,
        "processing_seconds":  round(time.time() - start, 4),
        "ranked_candidates":   ranked,
    }


# ============================================================
# PREDICTION LOGGER
# ============================================================

def _log_prediction(result: dict, resume_snippet: str):
    """Appends prediction to logs/prediction_logs.csv."""
    log_record = {
        "timestamp":          time.time(),
        "resume_snippet":     resume_snippet,
        "predicted_category": result.get("predicted_category"),
        "final_score":        result.get("final_score"),
        "match_band":         result.get("match_band"),
        "recommendation":     result.get("recommendation"),
        "rule_triggered":     result.get("rule_triggered"),
    }

    log_path = "logs/prediction_logs.csv"
    os.makedirs("logs", exist_ok=True)

    log_df = pd.DataFrame([log_record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)
