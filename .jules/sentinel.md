## 2026-09-24 - Input Sanitization for File Path Routes
**Vulnerability:** Path traversal risk in API endpoints accepting `run_id` or `batch_id` parameters directly in file path resolution (`os.path.join`).
**Learning:** FastAPI route path parameters passed directly into `FileResponse` or `os.path.join` without regex/whitelist validation can allow unauthorized directory traversal.
**Prevention:** Always validate all path parameters with `re.match(r"^[a-zA-Z0-9_\-]+$", identifier)` or raise HTTP 400 Bad Request before attempting filesystem operations.
