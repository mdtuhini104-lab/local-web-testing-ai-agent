## 2026-03-01 - Path Traversal Vulnerability in Report Download Endpoints
**Vulnerability:** Download endpoints in `app/api/routes.py` allowed path traversal via unvalidated `run_id` and `batch_id` parameters (e.g. `../..`).
**Learning:** File path construction using `os.path.join` with raw user-supplied inputs does not prevent escaping `base_dir` unless canonical absolute paths are explicitly validated.
**Prevention:** Always use a helper like `safe_join_storage` that computes absolute paths and ensures the target path starts with `os.path.abspath(base_dir) + os.sep`.
