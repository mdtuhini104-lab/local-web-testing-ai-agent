## 2026-03-30 - Path Traversal Prevention in Dynamic Report Download Endpoints
**Vulnerability:** API path parameters (`run_id` and `batch_id`) were joined directly into local storage directory paths (`os.path.join(settings.STORAGE_DIR, "runs", run_id, ...)`), allowing potential path traversal.
**Learning:** Even when IDs are generated as timestamps by the system, accepting arbitrary strings as URL path parameters without strict identifier validation creates arbitrary file read risks.
**Prevention:** Always sanitize and validate identifier path parameters using a strict character whitelist (`_validate_safe_id`) and check for path traversal sequences (`..`, slashes, leading/trailing dots) before using them in file path construction.
