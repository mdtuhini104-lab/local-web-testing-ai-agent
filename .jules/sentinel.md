## 2026-03-30 - Path Traversal Prevention in Report Download Routes
**Vulnerability:** User-controlled path parameters (`run_id`, `batch_id`) in REST API endpoints were passed directly to `os.path.join()` without format validation or sanitization, potentially allowing directory traversal (`../`).
**Learning:** Even when `os.path.join()` is used, relative path sequences like `..` in path segments are resolved relative to the base directory, exposing arbitrary filesystem locations if not restricted.
**Prevention:** Enforce strict format validation using regex (`^[a-zA-Z0-9_\-]+$`) on all identifier parameters before constructing file paths or performing filesystem operations.
