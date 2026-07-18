# ============================================================
# RUN DASHBOARD — Resume Screener AI
# ============================================================
# Usage:
#   python scripts/run_dashboard.py
# ============================================================

import sys
import os
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    dashboard_path = os.path.join(
        os.path.dirname(__file__), "..", "monitoring", "monitoring_dashboard.py"
    )
    dashboard_path = os.path.abspath(dashboard_path)

    print(f"Starting Streamlit dashboard: {dashboard_path}")
    print("Dashboard will open at: http://localhost:8501")

    subprocess.run(
        ["streamlit", "run", dashboard_path, "--server.port", "8501"],
        check=True,
    )
