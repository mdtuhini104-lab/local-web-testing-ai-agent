"""
Security Unit Tests for Sentinel Security Enhancements
Tests path traversal protection, input validation, and secure storage access.
"""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import validate_id, safe_join_storage

client = TestClient(app)


def test_validate_id_valid():
    """Valid IDs should pass validation without error."""
    assert validate_id("run_20260101_120000", "run_id") == "run_20260101_120000"
    assert validate_id("batch_12345", "batch_id") == "batch_12345"
    assert validate_id("valid-id-123_test", "id") == "valid-id-123_test"


def test_validate_id_invalid_path_traversal():
    """IDs containing path traversal characters must raise 400 Bad Request."""
    invalid_ids = [
        "../etc/passwd",
        "../../secret.txt",
        "run/../id",
        "run\\..\\id",
        "run_id;cat /etc/passwd",
        "run_id|ls",
        "run_id<script>",
        " ",
        "",
    ]
    for bad_id in invalid_ids:
        with pytest.raises(HTTPException) as exc_info:
            validate_id(bad_id, "test_param")
        assert exc_info.value.status_code == 400
        assert "contains unsafe characters" in str(exc_info.value.detail) or "Invalid" in str(exc_info.value.detail)


def test_safe_join_storage_valid():
    """Safe paths within storage directory should resolve correctly."""
    path = safe_join_storage("runs", "run_123", "report.json")
    assert "storage" in path
    assert path.endswith("report.json")


def test_safe_join_storage_traversal_blocked():
    """Paths escaping settings.STORAGE_DIR must raise 403 Forbidden."""
    with pytest.raises(HTTPException) as exc_info:
        safe_join_storage("..", "..", "etc", "passwd")
    assert exc_info.value.status_code == 403
    assert "Access denied" in str(exc_info.value.detail)


def test_api_path_traversal_download_json_rejected():
    """API endpoint /api/runs/{run_id}/download/json must reject path traversal attempts."""
    response = client.get("/api/runs/invalid..id/download/json")
    assert response.status_code == 400
    assert "unsafe characters" in response.json().get("detail", "")


def test_api_path_traversal_download_markdown_rejected():
    """API endpoint /api/runs/{run_id}/download/markdown must reject path traversal attempts."""
    response = client.get("/api/runs/invalid..id/download/markdown")
    assert response.status_code == 400
    assert "unsafe characters" in response.json().get("detail", "")


def test_api_path_traversal_download_docx_rejected():
    """API endpoint /api/runs/{run_id}/download/docx must reject path traversal attempts."""
    response = client.get("/api/runs/invalid..id/download/docx")
    assert response.status_code == 400
    assert "unsafe characters" in response.json().get("detail", "")


def test_api_path_traversal_batch_details_rejected():
    """API endpoint /api/batch/{batch_id} must reject path traversal attempts."""
    response = client.get("/api/batch/invalid..id")
    assert response.status_code == 400
    assert "unsafe characters" in response.json().get("detail", "")


def test_api_valid_id_format_response():
    """Legitimate ID formats should pass validation (returning 404 if file not found, rather than 400)."""
    response = client.get("/api/runs/run_20260101_000000/download/json")
    assert response.status_code == 404
    assert response.json().get("detail") == "JSON report file not found"
