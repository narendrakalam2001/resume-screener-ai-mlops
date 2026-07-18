# ============================================================
# PREPROCESSING — Resume Screener AI
# ============================================================
# Dual pipeline architecture:
#   preprocessor_scaled   → for linear/distance models (LR, SVM, SGD)
#   preprocessor_unscaled → for tree models (RF, XGB, LGBM)
#
# Text branch:  TF-IDF vectoriser (tuned for resume text)
# Struct branch: Clipper → PowerTransformer/StandardScaler
#
# clone() pattern preserved — no shared fitted objects.
# ============================================================

import numpy as np
import pandas as pd
import logging

from typing import List, Tuple

from sklearn.base              import BaseEstimator, TransformerMixin
from sklearn.compose           import ColumnTransformer
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

CLIP_FOLD  = 1.5
SELECT_K   = 15

logger = logging.getLogger(__name__)


# ============================================================
# CLIPPER — IQR-based outlier clipping transformer
# ============================================================

class Clipper(BaseEstimator, TransformerMixin):
    """
    Clips values to [Q1 - fold*IQR, Q3 + fold*IQR].
    Fitted on train set, applied to train + test.
    Prevents outlier leakage through scaling.

    get_feature_names_out() implemented so sklearn ColumnTransformer
    can extract clean feature names (fixes f0, f1... warning).
    """

    def __init__(self, fold: float = CLIP_FOLD):
        self.fold = fold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        q1  = np.quantile(X, 0.25, axis=0)
        q3  = np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1

        self.lower_ = q1 - self.fold * iqr
        self.upper_ = q3 + self.fold * iqr

        eps         = 1e-9
        self.upper_ = np.where(
            self.upper_ == self.lower_, self.upper_ + eps, self.upper_
        )
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        n = getattr(self, "n_features_in_", 1)
        return np.array([f"x{i}" for i in range(n)], dtype=object)


# ============================================================
# LABEL ENCODER WRAPPER
# ============================================================

def encode_labels(y: pd.Series, categories: list) -> Tuple[np.ndarray, LabelEncoder]:
    """Encodes string category labels to integers."""
    le = LabelEncoder()
    le.fit(categories)
    y_enc = le.transform(y)
    return y_enc, le


# ============================================================
# SAFE FEATURE COUNT HELPER
# ============================================================

def safe_k(requested_k: int, preprocessor: ColumnTransformer, X_sample: pd.DataFrame) -> int:
    """Returns min(requested_k, actual_output_features)."""
    fitted = preprocessor.fit(X_sample)
    try:
        feat_names = fitted.get_feature_names_out()
    except Exception:
        arr = fitted.transform(X_sample)
        if hasattr(arr, "toarray"):
            arr = arr.toarray()
        feat_names = [f"f{i}" for i in range(arr.shape[1])]

    k = min(requested_k, max(1, len(feat_names)))
    logger.info(
        "safe_k: requested=%d  available=%d  using=%d",
        requested_k, len(feat_names), k
    )
    return k


# ============================================================
# PREPROCESSOR BUILDER
# ============================================================

def build_preprocessors(
    cont_cols:    List[str],
    bin_cols:     List[str],
    text_col:     str,
    X_train:      pd.DataFrame,
    clip_fold:    float = CLIP_FOLD,
    max_features: int   = None,
) -> Tuple[ColumnTransformer, ColumnTransformer, List[str]]:
    """
    Builds dual ColumnTransformer pipelines:
      preprocessor_scaled   → linear / SVM / Logistic Regression
      preprocessor_unscaled → tree models (RF, XGB, LGBM)

    Text branch uses TF-IDF on text_col.
    Structured features go through Clipper + optional scaling.

    Returns:
        preprocessor_scaled, preprocessor_unscaled, feature_order
    """
    from src.config import (
        MAX_FEATURES_TFIDF, NGRAM_RANGE, MAX_DF,
        MIN_DF, TFIDF_SUBLINEAR_TF
    )

    _max_features = max_features or MAX_FEATURES_TFIDF

    # ── Detect skewed structured columns ──────────────────────
    numeric_cont = [c for c in cont_cols if c in X_train.columns and
                    X_train[c].dtype != object]
    skewed     = [c for c in numeric_cont if abs(X_train[c].skew()) > 0.8]
    non_skewed = [c for c in numeric_cont if c not in skewed]
    valid_bin  = [c for c in bin_cols if c in X_train.columns]

    logger.info("Skewed cols (%d): %s",     len(skewed),     skewed)
    logger.info("Non-skewed cols (%d): %s", len(non_skewed), non_skewed)

    # ── TF-IDF builder (called twice for independence) ────────
    def _tfidf():
        return TfidfVectorizer(
            max_features  = _max_features,
            ngram_range   = NGRAM_RANGE,
            max_df        = MAX_DF,
            min_df        = MIN_DF,
            sublinear_tf  = TFIDF_SUBLINEAR_TF,
            strip_accents = "unicode",
            analyzer      = "word",
            token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z0-9+#\.]{1,}\b",
        )

    # ── Build transformer lists ────────────────────────────────
    scaled_struct   = []
    unscaled_struct = []

    if skewed:
        scaled_struct.append((
            "skewed",
            Pipeline([
                ("clip",  Clipper(fold=clip_fold)),
                ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
                ("scale", StandardScaler()),
            ]),
            skewed
        ))
        unscaled_struct.append((
            "skewed",
            Pipeline([("clip", Clipper(fold=clip_fold))]),
            skewed
        ))

    if non_skewed:
        scaled_struct.append((
            "non_skew",
            Pipeline([
                ("clip",  Clipper(fold=clip_fold)),
                ("scale", StandardScaler()),
            ]),
            non_skewed
        ))
        unscaled_struct.append((
            "non_skew",
            Pipeline([("clip", Clipper(fold=clip_fold))]),
            non_skewed
        ))

    if valid_bin:
        scaled_struct.append(("bin", "passthrough", valid_bin))
        unscaled_struct.append(("bin", "passthrough", valid_bin))

    # ── Assemble ColumnTransformers ───────────────────────────
    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ("tfidf", _tfidf(), text_col),
            *scaled_struct,
        ],
        remainder="drop"
    )

    preprocessor_unscaled = ColumnTransformer(
        transformers=[
            ("tfidf", _tfidf(), text_col),
            *unscaled_struct,
        ],
        remainder="drop"
    )

    feature_order = [text_col] + skewed + non_skewed + valid_bin

    logger.info("Feature order: %s", feature_order)

    return preprocessor_scaled, preprocessor_unscaled, feature_order
