"""
Root application server launcher script.
Run via: python server.py
"""

import sys
import os
import uvicorn

# Ensure current working directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.config import settings

if __name__ == "__main__":
    print(f"🚀 Starting Local Web Testing AI Agent at http://{settings.HOST}:{settings.PORT}")
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=False)

