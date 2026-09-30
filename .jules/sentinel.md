## 2026-09-30 - Path Traversal in Report Download Endpoints
**Vulnerability:** Endpoints accepting `run_id` or `batch_id` as path parameters used raw string concatenation (`os.path.join(..., run_id, ...)`) when fetching reports from disk. Without validation, malicious path sequences (e.g. `../`) could lead to path traversal / arbitrary file read vulnerabilities.
**Learning:** In FastAPI, path parameters can contain URL-encoded traversal characters (e.g. `%2F`). If passed directly to `os.path.join`, Python resolves relative path navigation relative to the base directory.
**Prevention:** Strictly validate `run_id` and `batch_id` using an alphanumeric regex pattern (`^[a-zA-Z0-9_\-]+$`) before performing file system path joins.
