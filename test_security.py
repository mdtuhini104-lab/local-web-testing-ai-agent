import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.routes import _validate_safe_id
from app.main import app


def test_validate_safe_id_valid():
    """Test that valid run and batch IDs pass validation."""
    valid_ids = [
        "run_20260330_120000",
        "batch_20260330_120000",
        "batch_1_app1_127.0.0.1_8000",
        "valid-id-123_test",
    ]
    for identifier in valid_ids:
        assert _validate_safe_id(identifier) == identifier


def test_validate_safe_id_path_traversal():
    """Test that path traversal attempts raise HTTP 400."""
    invalid_ids = [
        "../../etc/passwd",
        "../secret.txt",
        "..\\windows\\system32",
        "run_id/subdir",
        "run_id\\subdir",
        "..",
        ".",
        ".hidden",
        "test.",
        "run_id%00",
        "run_id;cat /etc/passwd",
        "run_id|ls",
    ]
    for identifier in invalid_ids:
        with pytest.raises(HTTPException) as exc_info:
            _validate_safe_id(identifier)
        assert exc_info.value.status_code == 400
        assert "Invalid identifier format" in exc_info.value.detail


def test_api_endpoints_path_traversal():
    """Test API endpoints reject path traversal identifier formats with HTTP 400."""
    traversal_paths = [
        "/api/runs/test..id",
        "/api/runs/.hidden",
        "/api/runs/test..id/download/json",
        "/api/runs/.hidden/download/markdown",
        "/api/batch/test..id",
        "/api/batch/.hidden/download/markdown",
    ]
    with TestClient(app) as client:
        for path in traversal_paths:
            response = client.get(path)
            assert response.status_code == 400
            assert "Invalid identifier format" in response.json().get("detail", "")


def test_api_endpoints_valid_id_not_found():
    """Test API endpoints with a valid identifier format return 404 when item is not found."""
    with TestClient(app) as client:
        response = client.get("/api/runs/nonexistent_run_123")
        assert response.status_code == 404
        assert "Test run not found" in response.json().get("detail", "")
