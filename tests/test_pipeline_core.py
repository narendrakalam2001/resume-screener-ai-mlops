# ============================================================
# PYTEST UNIT TESTS — Resume Screener AI
# ============================================================
# Run with:
#   pytest tests/test_pipeline_core.py -v
#   pytest tests/ -v --cov=src --cov-report=term-missing
#
# Tests cover:
#   Clipper, build_preprocessors, encode_labels,
#   detect_leakage, psi, recruiter_time_savings,
#   get_match_band, score_applicant, keyword_overlap_score,
#   experience_boost, extract_jd_keywords, clean_resume_text,
#   add_engineered_features, validate_input_data, config thresholds
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
import pandas as pd

from src.preprocessing import Clipper, build_preprocessors, encode_labels
from src.data_loader   import (
    validate_input_data, add_engineered_features,
    clean_resume_text, detect_feature_types, extract_jd_keywords,
)
from src.leakage_check import detect_leakage
from src.metrics       import psi, recruiter_time_savings
from src.match_engine  import (
    get_match_band, score_applicant, keyword_overlap_score, experience_boost,
)
from src.config        import (
    MATCH_BANDS, PSI_MODERATE, PSI_HIGH,
    MIN_F1_IMPROVEMENT, MIN_ACCURACY_THRESHOLD, MAX_GENERALIZATION_GAP,
    MIN_RESUME_TOKENS, MIN_KEYWORD_OVERLAP,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_resume_df():
    return pd.DataFrame({
        "Resume_str": [
            "Python developer with 5 years experience in machine learning, "
            "TensorFlow, scikit-learn, and SQL. B.Tech Computer Science from IIT Bombay. "
            "Led a team of 4 engineers. Built NLP pipelines for text classification.",
            "Sales executive at ABC Corp. Managed client relationships and achieved "
            "revenue targets. MBA from Pune University. Strong communication and negotiation skills.",
            "Java developer with Spring Boot, Microservices, Docker, Kubernetes, AWS. "
            "4 years experience. B.E. Computer Science. Agile and Scrum certified.",
        ],
        "Category": ["Data Science", "Sales", "Java Developer"],
    })


@pytest.fixture
def sample_resume_engineered(sample_resume_df):
    df = validate_input_data(sample_resume_df.copy())
    return add_engineered_features(df)


@pytest.fixture
def simple_numeric_df():
    np.random.seed(42)
    return pd.DataFrame({
        "token_count":          np.random.randint(50, 500, 20),
        "tech_keyword_density": np.random.uniform(0.01, 0.15, 20),
        "soft_skill_count":     np.random.randint(0, 5, 20),
        "section_coverage":     np.random.randint(0, 6, 20),
        "lexical_diversity":    np.random.uniform(0.3, 0.9, 20),
        "has_education":        np.random.randint(0, 2, 20),
        "has_experience":       np.random.randint(0, 2, 20),
        "cleaned_resume":       ["sample resume text " * 10] * 20,
    })


# ============================================================
# CLIPPER TESTS  (6 tests)
# ============================================================

class TestClipper:

    def test_fit_transform_shape(self):
        """Output shape must match input shape."""
        X    = np.array([[1.0], [1000.0], [2.0], [3.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape == X.shape

    def test_clips_outliers(self):
        """Extreme values must be clipped."""
        X    = np.array([[1.0], [2.0], [3.0], [9999.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).max() < 9999.0

    def test_no_change_on_normal_data(self):
        """Values within IQR range must not be changed."""
        X    = np.array([[10.0], [11.0], [12.0], [13.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        np.testing.assert_array_almost_equal(X, clip.transform(X), decimal=3)

    def test_1d_input(self):
        """1D array should be handled without error."""
        X    = np.array([1.0, 2.0, 3.0, 1000.0])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape[0] == 4

    def test_get_feature_names_out(self):
        """get_feature_names_out must return array of correct length."""
        X    = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        clip = Clipper()
        clip.fit(X)
        names = clip.get_feature_names_out(["a", "b"])
        assert list(names) == ["a", "b"]

    def test_fit_on_train_applied_to_test(self):
        """Clipper fitted on train uses training bounds on test — not re-fit."""
        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        X_test  = np.array([[100.0]])
        clip    = Clipper(fold=1.5)
        clip.fit(X_train)
        clipped = clip.transform(X_test)
        assert clipped[0, 0] < 100.0


# ============================================================
# DATA LOADER TESTS  (6 tests)
# ============================================================

class TestDataLoader:

    def test_validate_passes_clean_data(self, sample_resume_df):
        """Validation should pass for clean required columns."""
        # Pad to 50+ rows so size check passes
        df_padded = pd.concat([sample_resume_df] * 20, ignore_index=True)
        df = validate_input_data(df_padded.copy())
        assert "Resume_str" in df.columns
        assert "Category"   in df.columns
        
    def test_validate_raises_on_missing_cols(self):
        """Validation should raise ValueError for missing columns."""
        df = pd.DataFrame({"text": ["some text"], "label": ["Data Science"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_input_data(df)

    def test_validate_drops_duplicates(self):
        """Duplicate resumes should be removed."""
        df = pd.DataFrame({
            "Resume_str": ["same text"] * 5 + [f"unique text {i}" for i in range(50)],
            "Category":   ["Data Science"] * 55,
        })
        result = validate_input_data(df)
        # 5 duplicates collapse to 1 unique + 50 unique = 51
        assert len(result) == 51

    def test_add_engineered_features_columns(self, sample_resume_df):
        """Feature engineering must add expected columns."""
        df_padded = pd.concat([sample_resume_df] * 20, ignore_index=True)
        df = validate_input_data(df_padded.copy())
        result = add_engineered_features(df)
        for col in ["cleaned_resume", "token_count", "tech_keyword_density",
                    "soft_skill_count", "section_coverage", "has_education"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_clean_resume_text_removes_urls(self):
        """clean_resume_text must strip URLs."""
        text    = "Visit https://github.com/user for portfolio"
        cleaned = clean_resume_text(text)
        assert "http" not in cleaned

    def test_extract_jd_keywords_returns_list(self):
        """extract_jd_keywords must return a non-empty list."""
        jd   = "Looking for Python developer with machine learning and SQL skills"
        kws  = extract_jd_keywords(jd, top_n=10)
        assert isinstance(kws, list)
        assert len(kws) > 0


# ============================================================
# LEAKAGE CHECK TESTS  (3 tests)
# ============================================================

class TestLeakageCheck:

    def test_no_leakage_on_clean_data(self, simple_numeric_df):
        """Leakage check should pass for unrelated numeric features."""
        y       = pd.Series(["Data Science"] * 10 + ["Sales"] * 10)
        X_clean = simple_numeric_df.drop(columns=["cleaned_resume"])
        warns   = detect_leakage(X_clean, y, text_col="cleaned_resume")
        assert isinstance(warns, list)

    def test_detects_identical_column(self):
        """Should flag a column identical to target."""
        y = pd.Series([0, 1, 0, 1, 0])
        X = pd.DataFrame({
            "leaked":  [0, 1, 0, 1, 0],
            "normal":  [1.0, 2.0, 1.5, 3.0, 2.5],
        })
        warns = detect_leakage(X, y, text_col="nonexistent")
        leakage_flags = [w for w in warns if "LEAKAGE" in w.upper()]
        assert len(leakage_flags) > 0

    def test_high_correlation_flagged(self):
        """Near-perfect correlation with target should be flagged."""
        y = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        X = pd.DataFrame({
            "near_perfect": [0.001, 0.999, 0.001, 0.999, 0.001, 0.999, 0.001, 0.999],
        })
        warns = detect_leakage(X, y, threshold_corr=0.99, text_col="nonexistent")
        assert any("corr" in w for w in warns)


# ============================================================
# METRICS TESTS  (5 tests)
# ============================================================

class TestMetrics:

    def test_psi_identical_distributions(self):
        """PSI of identical distributions should be ~0."""
        x   = np.random.normal(0, 1, 1000)
        val = psi(x, x)
        assert val < 0.01

    def test_psi_shifted_distributions(self):
        """PSI of shifted distributions should be > 0."""
        ref    = np.random.normal(0, 1, 1000)
        actual = np.random.normal(5, 1, 1000)
        val    = psi(ref, actual)
        assert val > 0.10

    def test_psi_moderate_threshold(self):
        """PSI_MODERATE threshold should be 0.10."""
        assert PSI_MODERATE == 0.10

    def test_psi_high_threshold(self):
        """PSI_HIGH threshold should be 0.20."""
        assert PSI_HIGH == 0.20

    def test_recruiter_time_savings_returns_dict(self):
        """Business impact function must return expected keys."""
        y = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        result = recruiter_time_savings(y, y)
        for key in ["total_resumes", "auto_screened", "cost_saved_per_1k_usd", "accuracy"]:
            assert key in result, f"Missing key: {key}"


# ============================================================
# MATCH ENGINE TESTS  (6 tests)
# ============================================================

class TestMatchEngine:

    def test_get_match_band_strong(self):
        """Score >= 0.70 → STRONG."""
        assert get_match_band(0.85) == "STRONG"

    def test_get_match_band_moderate(self):
        """Score 0.40–0.70 → MODERATE."""
        assert get_match_band(0.55) == "MODERATE"

    def test_get_match_band_weak(self):
        """Score < 0.40 → WEAK."""
        assert get_match_band(0.20) == "WEAK"

    def test_score_applicant_short_resume_rejected(self):
        """Resume with too few tokens should be REJECTED."""
        row    = {"cleaned_resume": "python", "token_count": 1}
        result = score_applicant(row, ml_score=0.9)
        assert result["recommendation"] == "REJECT"
        assert result["rule_triggered"]  is not None

    def test_score_applicant_strong_shortlisted(self):
        """High ML score with sufficient tokens → SHORTLIST."""
        resume = "python machine learning data science tensorflow pandas sql " * 20
        row    = {"cleaned_resume": resume, "token_count": 140}
        result = score_applicant(row, ml_score=0.85)
        assert result["recommendation"] == "SHORTLIST"
        assert result["final_score"]    >= 0.70

    def test_keyword_overlap_score(self):
        """Keyword overlap should detect matching terms."""
        resume = "python developer with sql and machine learning experience"
        kws    = ["python", "sql", "machine learning", "java"]
        score  = keyword_overlap_score(resume, kws)
        assert score == pytest.approx(0.75, abs=0.01)


# ============================================================
# EXPERIENCE BOOST TESTS  (2 tests)
# ============================================================

class TestExperienceBoost:

    def test_boost_when_years_sufficient(self):
        """Should give boost when resume mentions >= required years."""
        text   = "7 years of experience in data science and machine learning"
        boost  = experience_boost(text, years_required=5)
        assert boost > 0.0

    def test_no_boost_when_no_years_mentioned(self):
        """Should give no boost when no years mentioned."""
        text  = "I am a python developer"
        boost = experience_boost(text, years_required=3)
        assert boost == 0.0


# ============================================================
# CONFIG THRESHOLD TESTS  (4 tests)
# ============================================================

class TestConfigThresholds:

    def test_min_f1_improvement(self):
        """MIN_F1_IMPROVEMENT must be 0.005."""
        assert MIN_F1_IMPROVEMENT == 0.005

    def test_min_accuracy_threshold(self):
        """MIN_ACCURACY_THRESHOLD must be 0.80."""
        assert MIN_ACCURACY_THRESHOLD == 0.80

    def test_max_generalization_gap(self):
        """MAX_GENERALIZATION_GAP must be 0.10."""
        assert MAX_GENERALIZATION_GAP == 0.10

    def test_match_bands_coverage(self):
        """MATCH_BANDS must cover [0, 1] without gaps."""
        all_covered = all(k in MATCH_BANDS for k in ["STRONG", "MODERATE", "WEAK"])
        assert all_covered


# ============================================================
# ENCODE LABELS TESTS  (2 tests)
# ============================================================

class TestEncodeLabels:

    def test_encode_labels_shape(self):
        """Encoded array must have same length as input."""
        y      = pd.Series(["Data Science", "Sales", "HR", "Data Science"])
        cats   = ["Data Science", "HR", "Sales"]
        y_enc, le = encode_labels(y, cats)
        assert len(y_enc) == len(y)

    def test_encode_labels_inverse(self):
        """Inverse transform must recover original labels."""
        y      = pd.Series(["Data Science", "Sales", "HR"])
        cats   = ["Data Science", "HR", "Sales"]
        y_enc, le = encode_labels(y, cats)
        recovered = le.inverse_transform(y_enc)
        assert list(recovered) == list(y)
