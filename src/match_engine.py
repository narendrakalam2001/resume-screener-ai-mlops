# ============================================================
# MATCH ENGINE — Resume Screener AI
# ============================================================
# The Match Engine is the business-logic layer sitting between
# raw ML probability scores and final hiring decisions.
#
# It mirrors Credit Risk's risk_engine.py:
#   - ML similarity score → Match Band
#   - Business rules (keyword overlap, token count)
#   - Final recommendation: SHORTLIST / REVIEW / REJECT
#
# Used by:
#   - services/screening_service.py
#   - monitoring/monitoring_dashboard.py
# ============================================================

import logging
import numpy as np

from src.config import MATCH_BANDS, MIN_KEYWORD_OVERLAP, MIN_RESUME_TOKENS
from src.data_loader import extract_jd_keywords, clean_resume_text

logger = logging.getLogger(__name__)


# ============================================================
# MATCH BAND LOOKUP
# ============================================================

def get_match_band(score: float) -> str:
    """Maps similarity score [0, 1] → STRONG / MODERATE / WEAK."""
    for band, (lo, hi) in MATCH_BANDS.items():
        if lo <= score < hi:
            return band
    return "WEAK"


# ============================================================
# KEYWORD OVERLAP SCORE
# ============================================================

def keyword_overlap_score(resume_text: str, jd_keywords: list) -> float:
    """
    Computes fraction of JD keywords found in the resume.
    Keyword coverage is a standard ATS signal.
    """
    if not jd_keywords:
        return 0.0

    lower_resume = resume_text.lower()
    hits = sum(1 for kw in jd_keywords if kw in lower_resume)
    return round(hits / len(jd_keywords), 4)


# ============================================================
# EXPERIENCE BOOST
# ============================================================

def experience_boost(resume_text: str, years_required: int = 0) -> float:
    """
    Detects years of experience mentions in resume.
    Gives a small boost if resume mentions >= required years.
    Returns a float in [0.0, 0.10].
    """
    import re
    from src.config import EXPERIENCE_BOOST_WEIGHT

    if years_required <= 0:
        return 0.0

    pattern = re.compile(r"(\d+)\s*(?:year|yr|years|yrs)", re.IGNORECASE)
    mentions = pattern.findall(resume_text)
    if not mentions:
        return 0.0

    max_years = max(int(y) for y in mentions)
    if max_years >= years_required:
        return EXPERIENCE_BOOST_WEIGHT
    elif max_years >= years_required - 1:
        return EXPERIENCE_BOOST_WEIGHT * 0.5
    return 0.0


# ============================================================
# SCORE APPLICANT — CORE MATCH ENGINE
# ============================================================

def score_applicant(
    resume_row:      dict,
    ml_score:        float,
    jd_text:         str   = "",
    years_required:  int   = 0,
) -> dict:
    """
    Full scoring pipeline for one resume against a JD.

    Args:
        resume_row      : dict with at least 'cleaned_resume', 'token_count'
        ml_score        : model similarity / classification confidence [0, 1]
        jd_text         : raw job description text (optional for keyword check)
        years_required  : minimum years of experience required by JD

    Returns:
        Structured match result dict.
    """
    resume_text = resume_row.get("cleaned_resume", "")
    token_count = resume_row.get("token_count", 0)

    # ── Business Rule 1: Resume too short ─────────────────────
    rule_triggered = None
    if token_count < MIN_RESUME_TOKENS:
        rule_triggered = f"RESUME_TOO_SHORT (tokens={token_count} < {MIN_RESUME_TOKENS})"
        final_score    = 0.0
        recommendation = "REJECT"
        kw_overlap     = 0.0

    else:
        # ── Keyword overlap score ──────────────────────────────
        jd_keywords = extract_jd_keywords(jd_text, top_n=30) if jd_text else []
        kw_overlap  = keyword_overlap_score(resume_text, jd_keywords) if jd_keywords else None

        # ── Business Rule 2: Critical keyword miss ────────────
        if kw_overlap is not None and kw_overlap < MIN_KEYWORD_OVERLAP and jd_keywords:
            rule_triggered = (
                f"LOW_KEYWORD_OVERLAP "
                f"(coverage={kw_overlap:.2%} < {MIN_KEYWORD_OVERLAP:.0%})"
            )
            final_score    = ml_score * 0.5  # penalise but don't hard-reject
        else:
            # ── Experience boost ───────────────────────────────
            boost      = experience_boost(resume_text, years_required)
            final_score = min(1.0, ml_score + boost)

        # ── Recommendation logic ───────────────────────────────
        band = get_match_band(final_score)

        if band == "STRONG":
            recommendation = "SHORTLIST"
        elif band == "MODERATE":
            recommendation = "REVIEW"
        else:
            recommendation = "REJECT"

    band = get_match_band(final_score)

    result = {
        "ml_score":        round(float(ml_score),    4),
        "final_score":     round(float(final_score), 4),
        "match_band":      band,
        "recommendation":  recommendation,
        "keyword_overlap": round(float(kw_overlap), 4) if kw_overlap is not None else None,
        "rule_triggered":  rule_triggered,
        "token_count":     int(token_count),
    }

    logger.info(
        "Match  |  ml=%.3f  final=%.3f  band=%s  decision=%s",
        ml_score, final_score, band, recommendation
    )

    return result


# ============================================================
# BATCH MATCH — for ranking a pool of resumes
# ============================================================

def rank_candidates(
    resumes:        list,   # list of resume dicts (with cleaned_resume, token_count, etc.)
    ml_scores:      list,   # model probability for each resume
    jd_text:        str  = "",
    years_required: int  = 0,
    top_k:          int  = 10,
) -> list:
    """
    Scores and ranks all candidates for a given JD.
    Returns top_k ranked candidates with full match details.

    Used for batch screening pipelines (e.g., 500+ applicants per JD).
    """
    results = []
    for i, (row, score) in enumerate(zip(resumes, ml_scores)):
        match = score_applicant(row, score, jd_text, years_required)
        match["candidate_idx"] = i
        results.append(match)

    # Sort by final_score descending
    ranked = sorted(results, key=lambda x: x["final_score"], reverse=True)

    # Append rank
    for rank, r in enumerate(ranked, start=1):
        r["rank"] = rank

    logger.info(
        "Ranking complete  |  total=%d  shortlisted=%d  review=%d  rejected=%d",
        len(ranked),
        sum(1 for r in ranked if r["recommendation"] == "SHORTLIST"),
        sum(1 for r in ranked if r["recommendation"] == "REVIEW"),
        sum(1 for r in ranked if r["recommendation"] == "REJECT"),
    )

    return ranked[:top_k]


# ============================================================
# DECISION COUNTS HELPER (for model card)
# ============================================================

def get_decision_counts(results: list) -> dict:
    """Aggregates recommendation counts from a ranked result list."""
    counts = {"SHORTLIST": 0, "REVIEW": 0, "REJECT": 0}
    for r in results:
        rec = r.get("recommendation", "REJECT")
        counts[rec] = counts.get(rec, 0) + 1
    return counts
