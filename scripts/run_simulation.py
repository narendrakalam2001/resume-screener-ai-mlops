# ============================================================
# RUN SIMULATION — Resume Screener AI
# ============================================================
# Usage:
#   python scripts/run_simulation.py
#   python scripts/run_simulation.py --scenario strong --n 30
#   python scripts/run_simulation.py --scenario weak   --n 15
# ============================================================

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.resume_simulator import simulate_applications

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Application Simulator")
    parser.add_argument(
        "--n",        type=int, default=20,
        help="Number of resumes to simulate (default: 20)"
    )
    parser.add_argument(
        "--scenario", type=str, default="random",
        choices=["random", "strong", "weak"],
        help="Simulation scenario: random | strong | weak (default: random)"
    )
    args = parser.parse_args()

    simulate_applications(n=args.n, scenario=args.scenario)
