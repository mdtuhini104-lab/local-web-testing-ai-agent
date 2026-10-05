## 2026-03-31 - Path Traversal Prevention in File Downloads
**Vulnerability:** Unsanitized `run_id` and `batch_id` URL path parameters passed directly into `os.path.join` enabled path traversal outside storage bounds.
**Learning:** `os.path.join` does not sanitize `..` or relative directory components when appending path segments.
**Prevention:** Validate input parameters against a strict whitelist pattern (`^[a-zA-Z0-9_\-]+$`) and verify resolved absolute paths using `startswith` against the allowed base directory (`settings.STORAGE_DIR`).
