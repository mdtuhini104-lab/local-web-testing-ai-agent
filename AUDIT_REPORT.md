# QA Automation & Full-Stack Security Audit Report

## Critical Bugs

1. **Missing Build Dependencies (Build Failure)**
   - `requirements.txt` is missing crucial dependencies: `requests`, `httpx`, `asyncio`, `uvicorn[standard]`, and `python-docx`. The server fails to start out-of-the-box because of missing `uvicorn`, and `test_api.py` crashes on missing `requests`.
   - The docx report generation in `agent_runner.py` crashes if `python-docx` is not installed, which causes the final output stage of the pipeline to crash or fail.

2. **Unhandled Promise / Request Rejections (Runtime Check)**
   - In `test_api.py`, `requests.post()` and `requests.get()` are not wrapped in a `try/except` block to catch connection errors. If the server is not running or takes time to start, it raises an uncaught `ConnectionRefusedError`, blocking the UI/CLI test flow.

3. **Concurrency Memory Leak (Functional Flaw)**
   - In `app/api/routes.py`, `start_test_run` uses `asyncio.create_task` but attaches it to an `active_tasks` dictionary without handling exceptions or cleaning it up on completion. Over time, this creates memory leaks and unhandled hanging tasks on long-running concurrent jobs.

## High/Medium Risks

1. **Broken Access Control & File Traversal (Security Flaw)**
   - In `app/main.py`, the storage directory is mounted publicly: `app.mount("/storage", StaticFiles(directory=settings.STORAGE_DIR), name="storage")`. This exposes sensitive test snapshots, databases (`test_runs.db`), and audit reports to anyone without authentication.

2. **Route Protection Loopholes (CORS)**
   - `app/main.py` uses `allow_origins=["*"]` along with `allow_credentials=True`. This is a serious misconfiguration in FastAPI that exposes the API to CSRF or cross-origin attacks.

3. **Silent Error Swallowing in Pipelines (Logic Flaw)**
   - In `app/api/routes.py` inside the `/runs/{run_id}/download/docx` route, there's a blanket `except Exception as e: pass` for JSON parsing. If the JSON is corrupted, it silently fails and generates a fallback without logging the error, hiding potential data loss.

4. **Exposed Config & Unused Deprecated Keys (Best Practice)**
   - `verify_setup.py` and `.env.example` reference `GEMINI_API_KEY` and dummy variables but don't properly handle secure injection in production.

## Recommended Fixes

1. **Dependency Fixes:** Add missing packages to `requirements.txt`.
```text
requests>=2.31.0
httpx>=0.27.0
uvicorn[standard]>=0.28.0
python-docx>=1.1.0
```

2. **Test Script Fix:** Wrap `test_api.py` requests in a try-except to handle startup lag gracefully.
```python
import requests
from requests.exceptions import ConnectionError

def test():
    try:
        resp = requests.post("http://127.0.0.1:8000/api/runs", json={...})
    except ConnectionError:
        print("Error: Backend server is not running. Start it with `python server.py`.")
```

3. **CORS & Access Control:** Restrict `allow_origins` in `main.py` and secure the static mounts (do not expose DB files).
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

4. **Error Handling:** Add explicit logging instead of `pass` in `routes.py`.
```python
except Exception as e:
    import logging
    logging.error(f"Failed to parse report JSON for docx generation: {e}")
```