# ============================================================
# RUN API — Resume Screener AI
# ============================================================
# Usage:
#   python scripts/run_api.py
#   python scripts/run_api.py --port 8001
# ============================================================

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Resume Screener API")
    parser.add_argument("--host",   default="0.0.0.0",  help="Host address")
    parser.add_argument("--port",   default=8000, type=int, help="Port number")
    parser.add_argument("--reload", action="store_true",   help="Enable auto-reload (dev)")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "serving.resume_screener_api:app",
        host    = args.host,
        port    = args.port,
        reload  = args.reload,
    )
