import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.main import app
from app.api.routes import validate_safe_id, safe_storage_path

client = TestClient(app)

def test_validate_safe_id_rejects_traversal():
    with pytest.raises(HTTPException) as exc_info:
        validate_safe_id("../etc/passwd")
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        validate_safe_id("run_123/..")
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        validate_safe_id("run..123")
    assert exc_info.value.status_code == 400

    with pytest.raises(HTTPException) as exc_info:
        validate_safe_id(r"run123\secret")
    assert exc_info.value.status_code == 400

def test_validate_safe_id_accepts_valid_ids():
    assert validate_safe_id("run_20260301_120000") == "run_20260301_120000"
    assert validate_safe_id("batch_20260301_120000") == "batch_20260301_120000"

def test_safe_storage_path_rejects_escaping_path():
    with pytest.raises(HTTPException) as exc_info:
        safe_storage_path("runs", "..", "..", "etc", "passwd")
    assert exc_info.value.status_code == 400

def test_path_traversal_attempts_via_api_client():
    # Test valid ID returns 404 when not found
    resp = client.get("/api/runs/run_20260301_120000/download/json")
    assert resp.status_code == 404

    # Test path traversal with '..' in path segment parameter
    resp = client.get("/api/runs/run..123/download/json")
    assert resp.status_code == 400
    assert "path traversal" in resp.json()["detail"].lower() or "invalid" in resp.json()["detail"].lower()

    resp = client.get("/api/runs/run..123/download/markdown")
    assert resp.status_code == 400

    resp = client.get("/api/runs/run..123/download/docx")
    assert resp.status_code == 400

    resp = client.get("/api/runs/run..123")
    assert resp.status_code == 400

    resp = client.get("/api/batch/batch..123")
    assert resp.status_code == 400

    resp = client.get("/api/batch/batch..123/download/markdown")
    assert resp.status_code == 400
