# ============================================================
# MODEL TUNING — Resume Screener AI
# ============================================================
# Defines scaled_models (linear/SVM) and unscaled_models (trees)
# for multi-class resume category classification.
# Mirrors credit-risk model_tuning.py conventions.
# ============================================================

import logging
import numpy as np

from typing import Dict, Tuple

from sklearn.base            import BaseEstimator
from sklearn.compose         import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline        import Pipeline

from sklearn.linear_model   import LogisticRegression, SGDClassifier
from sklearn.svm            import LinearSVC
from sklearn.naive_bayes    import MultinomialNB, ComplementNB
from sklearn.ensemble       import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.neighbors      import KNeighborsClassifier

try:
    from xgboost  import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from src.config import RANDOM_STATE, N_JOBS, CV_FOLDS, RANDOM_SEARCH_ITERS

logger = logging.getLogger(__name__)


# ============================================================
# HELPER
# ============================================================

def _compute_n_iter(param_dist: dict, budget: int) -> int:
    if not param_dist:
        return 1
    prod = 1
    for v in param_dist.values():
        try:
            prod *= len(v)
        except TypeError:
            prod *= budget
    return min(budget, max(1, prod))


# ============================================================
# MODEL GRIDS
# ============================================================

# Distance / linear models → need scaled (TF-IDF already normalised)
scaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "LogisticRegression": (
        LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000,
            multi_class="auto", solver="lbfgs", n_jobs=N_JOBS,
        ),
        {
            "classifier__C":            [0.1, 1, 5, 10],
            "classifier__class_weight": [None, "balanced"],
            "classifier__penalty":      ["l2"],
        }
    ),

    "LinearSVC": (
        LinearSVC(
            random_state=RANDOM_STATE, max_iter=3000,
            dual="auto",          # ← FutureWarning fix
        ),
        {
            "classifier__C":            [0.1, 1, 5, 10],
            "classifier__class_weight": [None, "balanced"],
        }
    ),

    "SGDClassifier": (
        SGDClassifier(
            loss="modified_huber", random_state=RANDOM_STATE,
            n_jobs=N_JOBS, max_iter=1000
        ),
        {
            "classifier__alpha":        [1e-5, 1e-4, 1e-3],
            "classifier__class_weight": [None, "balanced"],
        }
    ),

    "KNN": (
        KNeighborsClassifier(n_jobs=N_JOBS),
        {
            "classifier__n_neighbors": [3, 5, 9, 15],
            "classifier__weights":     ["uniform", "distance"],
        }
    ),
}

# Tree / ensemble models → benefit from unscaled sparse TF-IDF
unscaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {
    
    "ComplementNB": (
            ComplementNB(),
            {
                "classifier__alpha": [0.01, 0.1, 0.5, 1.0],
            }
        ),

    "RandomForest": (
        RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=N_JOBS, class_weight="balanced"
        ),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__max_depth":     [None, 20, 40],
            "classifier__min_samples_split": [2, 5],
        }
    ),

    "ExtraTrees": (
        ExtraTreesClassifier(
            random_state=RANDOM_STATE, n_jobs=N_JOBS, class_weight="balanced"
        ),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__max_depth":     [None, 20, 40],
            "classifier__min_samples_split": [2, 5],
        }
    ),
}

if LGBMClassifier is not None:
    unscaled_models["LightGBM"] = (
        LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            class_weight="balanced",
            verbosity=-1,
            # num_class hata diya — LightGBM auto-detect karta hai
        ),
        {
            "classifier__n_estimators":      [100, 200],      # 300 hata diya
            "classifier__max_depth":         [6, -1],          # 10 hata diya
            "classifier__learning_rate":     [0.1, 0.2],       # 0.05 hata diya
            "classifier__num_leaves":        [31],             # sirf 1 value
            "classifier__min_child_samples": [20],             # sirf 1 value
        }
    )

if XGBClassifier is not None:
    unscaled_models["XGBoost"] = (
        XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
            eval_metric="mlogloss", verbosity=0,
            use_label_encoder=False,
        ),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__max_depth":     [4, 6, 8],
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__subsample":     [0.8, 1.0],
        }
    )


# ============================================================
# TUNE MODELS
# ============================================================

def tune_models(
    model_dict:    dict,
    preprocessor:  ColumnTransformer,
    X_train:       object,
    y_train:       object,
    selector_k:    int  = None,
) -> dict:
    """
    Runs RandomizedSearchCV for each model in model_dict.
    Builds: preprocessor → [selector] → classifier pipeline.

    Text classifiers (NB, LinearSVC) work best without feature selection
    — selector is only added when selector_k is explicitly provided and small.

    Returns dict of {name: fitted_pipeline}.
    """
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.pipeline import Pipeline as SkPipeline

    pipelines = {}

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    for name, (clf, param_grid) in model_dict.items():

        logger.info("Tuning: %s", name)

        steps = [("preprocessor", preprocessor)]

        # Skip selector for NB and LinearSVC — sparse TF-IDF is their strength
        skip_selector = name in ("ComplementNB", "MultinomialNB", "LinearSVC", "SGDClassifier")

        if selector_k and not skip_selector:
            steps.append(("selector", SelectKBest(chi2, k=selector_k)))

        steps.append(("classifier", clf))

        # Prefix param_grid keys correctly
        full_grid = {}
        for k, v in param_grid.items():
            full_grid[k] = v

        n_iter = _compute_n_iter(param_grid, RANDOM_SEARCH_ITERS)

        pipe = Pipeline(steps)

        try:
            search = RandomizedSearchCV(
                pipe,
                param_distributions = full_grid,
                n_iter      = n_iter,
                scoring     = "f1_macro",
                cv          = cv,
                refit       = True,
                n_jobs      = N_JOBS,
                random_state= RANDOM_STATE,
                verbose     = 0,
            )
            search.fit(X_train, y_train)
            pipelines[name] = search.best_estimator_

            logger.info(
                "  %-20s  best_cv_f1_macro=%.4f  params=%s",
                name,
                search.best_score_,
                {k: v for k, v in search.best_params_.items()},
            )

        except Exception as e:
            logger.warning("Tuning failed for %s: %s — skipping", name, e)

    return pipelines
