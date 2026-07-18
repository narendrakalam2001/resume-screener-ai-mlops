# ============================================================
# DATA LOADER + TEXT FEATURE ENGINEERING — Resume Screener AI
# ============================================================

import re
import logging
import numpy as np
import pandas as pd

from src.config import (
    JOB_CATEGORIES, MIN_RESUME_TOKENS, MAX_RESUME_TOKENS
)

logger = logging.getLogger(__name__)

# ── Required columns for Kaggle Resume Dataset ───────────────
REQUIRED_COLS = ["Resume_str", "Category"]

# ── Common resume section headers ────────────────────────────
SECTION_HEADERS = [
    "education", "experience", "skills", "projects",
    "certifications", "summary", "objective", "achievements"
]

# ── Tech skill keyword pools (for feature engineering) ───────
TECH_KEYWORDS = {
    "python", "java", "javascript", "sql", "r", "scala", "c++",
    "machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy", "spark", "hadoop", "kafka",
    "docker", "kubernetes", "aws", "azure", "gcp", "git",
    "react", "angular", "node", "django", "flask", "fastapi",
    "tableau", "power bi", "excel", "jira", "agile", "scrum",
}

SOFT_KEYWORDS = {
    "leadership", "communication", "teamwork", "problem solving",
    "analytical", "creative", "collaboration", "adaptable",
    "management", "presentation", "negotiation",
}


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and normalises the Kaggle Resume dataset.
    Handles column naming, missing values, deduplication.
    """
    # ── Normalize column names ─────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.strip("_")
    )

    # ── Drop irrelevant ID columns ─────────────────────────────
    for col in ("ID", "id", "Resume_html"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logger.info("Dropped column: %s", col)

    # ── Check required columns ─────────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Null check ─────────────────────────────────────────────
    nulls = df[REQUIRED_COLS].isnull().sum().sum()
    if nulls > 0:
        before = len(df)
        df.dropna(subset=REQUIRED_COLS, inplace=True)
        logger.warning("Dropped %d rows with null resume/category", before - len(df))

    # ── Minimum size check ─────────────────────────────────────
    if df.shape[0] < 50:
        raise ValueError("Dataset too small for training (< 50 rows)")

    # ── Deduplication ──────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(subset=["Resume_str"], ignore_index=True, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate resumes", dropped)

    # ── Validate categories ────────────────────────────────────
    unknown_cats = set(df["Category"].unique()) - set(JOB_CATEGORIES)
    if unknown_cats:
        logger.warning("Unknown categories found (kept): %s", unknown_cats)

    logger.info(
        "Data validation passed  |  shape=%s  |  categories=%d",
        df.shape, df["Category"].nunique()
    )

    return df


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_resume_text(text: str) -> str:
    """
    Cleans raw resume text:
      - Removes URLs, emails, phone numbers
      - Removes special chars, extra whitespace
      - Lowercases
    Retains meaningful tokens for TF-IDF.
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove phone numbers
    text = re.sub(r"\b[\+]?[\d\s\-\(\)]{7,15}\b", " ", text)
    # Remove special characters (keep alphanumeric + spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    HR-grade feature engineering on top of raw resume text.
    Adds structural and keyword-based signals used by ATS systems.
    """
    df = df.copy()

    # ── Clean text ─────────────────────────────────────────────
    df["cleaned_resume"] = df["Resume_str"].apply(clean_resume_text)

    # ── Token count (proxy for resume length / experience depth) ─
    df["token_count"] = df["cleaned_resume"].apply(lambda t: len(t.split()))

    # Clip to configured bounds
    df["token_count"] = df["token_count"].clip(
        lower=MIN_RESUME_TOKENS, upper=MAX_RESUME_TOKENS
    )

    # ── Tech keyword density ───────────────────────────────────
    def _tech_density(text: str) -> float:
        tokens = set(text.lower().split())
        hits   = len(tokens & TECH_KEYWORDS)
        return round(hits / (len(tokens) + 1e-9), 4)

    df["tech_keyword_density"] = df["cleaned_resume"].apply(_tech_density)

    # ── Soft skill count ───────────────────────────────────────
    def _soft_count(text: str) -> int:
        lower = text.lower()
        return sum(1 for kw in SOFT_KEYWORDS if kw in lower)

    df["soft_skill_count"] = df["cleaned_resume"].apply(_soft_count)

    # ── Section coverage ───────────────────────────────────────
    def _section_count(text: str) -> int:
        lower = text.lower()
        return sum(1 for h in SECTION_HEADERS if h in lower)

    df["section_coverage"] = df["cleaned_resume"].apply(_section_count)

    # ── Unique word ratio (lexical diversity) ──────────────────
    def _lexical_diversity(text: str) -> float:
        tokens = text.split()
        if not tokens:
            return 0.0
        return round(len(set(tokens)) / len(tokens), 4)

    df["lexical_diversity"] = df["cleaned_resume"].apply(_lexical_diversity)

    # ── Has education section flag ─────────────────────────────
    df["has_education"] = df["cleaned_resume"].str.contains(
        r"education|degree|bachelor|master|phd|b\.sc|m\.sc",
        regex=True, na=False
    ).astype(int)

    # ── Has experience section flag ────────────────────────────
    df["has_experience"] = df["cleaned_resume"].str.contains(
        r"experience|worked|employment|internship|company",
        regex=True, na=False
    ).astype(int)

    logger.info(
        "Feature engineering done  |  total columns=%d", df.shape[1]
    )

    return df


# ============================================================
# FEATURE TYPE DETECTION
# ============================================================

def detect_feature_types(df: pd.DataFrame, text_col: str = "cleaned_resume"):
    """
    Returns lists of column types for the structured feature block.
    Excludes text columns and target ('Category').
    """
    EXCLUDE = {"Resume_str", "cleaned_resume", "Category"}

    continuous_cols = []
    binary_cols     = []

    for col in df.columns:
        if col in EXCLUDE:
            continue
        if col == text_col:
            continue

        n_unique = df[col].nunique()

        if n_unique == 2:
            binary_cols.append(col)
        elif np.issubdtype(df[col].dtype, np.number):
            continuous_cols.append(col)

    logger.info(
        "Feature types  |  continuous=%d  binary=%d",
        len(continuous_cols), len(binary_cols)
    )

    return continuous_cols, binary_cols


# ============================================================
# JD KEYWORD EXTRACTOR (for matching)
# ============================================================

def extract_jd_keywords(jd_text: str, top_n: int = 30) -> list:
    """
    Extracts top keyword tokens from a Job Description.
    Used for keyword overlap scoring in the match engine.
    """
    cleaned = clean_resume_text(jd_text)
    tokens  = cleaned.split()

    # Frequency-based selection (simple, no external deps)
    from collections import Counter
    freq    = Counter(tokens)

    # Remove very common stop-words
    STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "are", "you",
        "will", "have", "from", "our", "your", "not", "all", "can",
        "was", "has", "but", "its", "we", "in", "to", "of", "a",
        "an", "is", "be", "as", "on", "at", "by", "or", "it", "if"
    }
    keywords = [
        word for word, _ in freq.most_common(top_n * 3)
        if word not in STOPWORDS and len(word) > 2
    ][:top_n]

    return keywords
