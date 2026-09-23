## 2025-01-01 - Path Traversal Prevention in Report Download & Detail Routes
**Vulnerability:** Path traversal risk where unvalidated `run_id` and `batch_id` URL parameters could be used in `os.path.join` to access or download files outside the intended `./storage/runs/` directory.
**Learning:** Even when `os.path.join` is used with a fixed base directory, unvalidated user input containing path separators (`/`, `\`) or relative path tokens (`..`) can alter directory target resolution unless parameters are strictly sanitized or validated before path construction.
**Prevention:** Defensively validate all resource identifiers passed in API paths against an explicit regex whitelist (e.g. `^[a-zA-Z0-9_\-]+$`) and reject non-matching inputs with HTTP 400.
