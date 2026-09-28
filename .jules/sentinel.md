## 2026-03-31 - Path Traversal Prevention in Report Download Routes
**Vulnerability:** Unsanitized path parameters `run_id` and `batch_id` in report download endpoints (`/api/runs/{run_id}/download/*`, `/api/batch/{batch_id}/*`) allowed potential path traversal sequences (e.g. `..`).
**Learning:** When generating file paths dynamically using `os.path.join(base_dir, user_param, filename)`, validating the parameter against a strict regex `^[a-zA-Z0-9_\-]+$` ensures safe file operations without breaking route handlers or leaking server filesystem artifacts.
**Prevention:** Always sanitize or validate path parameters before using them in filesystem path construction or database lookups.
