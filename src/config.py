# ============================================================
# CONFIGURATION — Resume Screener AI
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE   = 42
N_JOBS         = -1

# ── Cross-validation ─────────────────────────────────────────
CV_FOLDS             = 5
RANDOM_SEARCH_ITERS  = 8

# ── Text Processing ──────────────────────────────────────────
MAX_FEATURES_TFIDF   = 10000    # TF-IDF vocabulary size
NGRAM_RANGE          = (1, 2)   # unigrams + bigrams
MAX_DF               = 0.95     # ignore terms in > 95% docs
MIN_DF               = 2        # ignore terms in < 2 docs
TFIDF_SUBLINEAR_TF   = True     # log-scale TF

# ── Embedding / Similarity ───────────────────────────────────
SIMILARITY_MODEL     = "tfidf"   # tfidf | sentence_transformer
TOP_K_RESULTS        = 10        # top-K ranked candidates to return

# ── Classification ───────────────────────────────────────────
# 24 job categories in the Kaggle Resume Dataset
JOB_CATEGORIES = [
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL",
    "ARTS", "AUTOMOBILE", "AVIATION", "BANKING",
    "BPO", "BUSINESS-DEVELOPMENT", "CHEF", "CONSTRUCTION",
    "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA", "ENGINEERING",
    "FINANCE", "FITNESS", "HEALTHCARE", "HR",
    "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER",
]
NUM_CLASSES = len(JOB_CATEGORIES)

# ── Match Score Bands ─────────────────────────────────────────
MATCH_BANDS = {
    "STRONG":   (0.70, 1.01),
    "MODERATE": (0.40, 0.70),
    "WEAK":     (0.00, 0.40),
}

# ── Business rule thresholds ─────────────────────────────────
MIN_RESUME_TOKENS        = 20    # reject too-short resumes
MAX_RESUME_TOKENS        = 5000  # truncate very long resumes
MIN_KEYWORD_OVERLAP      = 0.10  # JD keyword coverage floor
EXPERIENCE_BOOST_WEIGHT  = 0.10  # bonus for experience match

# ── PSI drift thresholds ─────────────────────────────────────
PSI_MODERATE   = 0.10   # PSI >= 0.10 → moderate drift, monitor
PSI_HIGH       = 0.20   # PSI >= 0.20 → critical drift, retrain

# ── Score monitoring alert ───────────────────────────────────
SCORE_MEAN_ALERT = 0.50  # alert if avg match score drops below

# ── Challenger promotion gates ───────────────────────────────
MIN_F1_IMPROVEMENT     = 0.005
MIN_ACCURACY_THRESHOLD = 0.80
MAX_GENERALIZATION_GAP = 0.10

# ── Paths ─────────────────────────────────────────────────────
MODEL_DIR   = "resume_models"
METRICS_LOG = "resume_models/metrics_log.csv"
TESTS_DIR   = "tests"
SERVING_DIR = "serving"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs",    exist_ok=True)
