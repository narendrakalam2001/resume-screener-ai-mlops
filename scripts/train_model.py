# ============================================================
# TRAIN MODEL — Resume Screener AI
# ============================================================
# Usage:
#   python scripts/train_model.py
#   RESUME_DATA_PATH=data/UpdatedResumeDataSet.csv python scripts/train_model.py
# ============================================================

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training_pipeline import run_training

if __name__ == "__main__":
    result = run_training()
    print("\nTraining complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")
