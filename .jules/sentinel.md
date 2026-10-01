## 2026-03-30 - Path Traversal Prevention on Report Download API Routes
**Vulnerability:** API endpoints accepting `run_id` and `batch_id` as path parameters directly constructed file paths with `os.path.join(settings.STORAGE_DIR, "runs", run_id, ...)` without input validation, allowing potential path traversal attacks.
**Learning:** FastAPI path parameters representing file/directory identifiers must be strictly validated before constructing file paths on disk.
**Prevention:** Validate all identifier path parameters using a strict regex (`^[a-zA-Z0-9_-]+$`) or verify path boundaries with `os.path.abspath` containment checks before reading or writing files.
