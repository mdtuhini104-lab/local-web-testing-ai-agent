## 2026-03-30 - Path Traversal Prevention in Report Downloads
**Vulnerability:** Unsanitized `run_id` and `batch_id` route parameters allowed constructing file paths with traversal sequences outside `STORAGE_DIR`.
**Learning:** Router parameters for file access require both regex character validation (`^[a-zA-Z0-9_\-]+$`) and path resolution checks using `os.path.abspath`.
**Prevention:** Apply `_validate_safe_id` and `_get_safe_path` helpers on any file download route accepting user-supplied parameters.
