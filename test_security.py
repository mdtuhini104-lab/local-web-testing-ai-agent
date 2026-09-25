import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.main import app
from app.api.routes import validate_identifier


def test_validate_identifier_unit():
    """Unit test for validate_identifier function against malicious path traversal strings."""
    invalid_identifiers = [
        "../etc/passwd",
        "..",
        "run/../id",
        "run_id/../../secret",
        "run id with spaces",
        "run_id;drop_table",
        "run_id*",
        "../../storage/qa_knowledge_store.json",
        "run_id\0nullbyte",
    ]

    for inv_id in invalid_identifiers:
        with pytest.raises(HTTPException) as exc_info:
            validate_identifier(inv_id)
        assert exc_info.value.status_code == 400
        assert "Invalid run_id or batch_id format." in exc_info.value.detail


def test_validate_identifier_unit_valid():
    """Unit test for validate_identifier with valid identifiers."""
    valid_ids = ["run_20260222_120000", "batch_20260222_120000", "test-run-123"]
    for valid_id in valid_ids:
        assert validate_identifier(valid_id) == valid_id


def test_validate_identifier_api_routes():
    """Integration test checking that API endpoints validate run_id and batch_id."""
    with TestClient(app) as client:
        # Test valid ID format that does not exist in DB returns 404 (not 400)
        valid_id = "run_20260222_120000"
        res = client.get(f"/api/runs/{valid_id}")
        assert res.status_code == 404
        assert res.json().get("detail") == "Test run not found"

        # Test invalid run_id formats return 400 Bad Request
        invalid_ids = [
            "run id with space",
            "run_id;sql",
            "run_id*",
            "run_id$1",
        ]
        for inv_id in invalid_ids:
            res = client.get(f"/api/runs/{inv_id}")
            assert res.status_code == 400
            assert "Invalid run_id or batch_id format." in res.json().get("detail", "")

            res = client.get(f"/api/runs/{inv_id}/download/json")
            assert res.status_code == 400
            assert "Invalid run_id or batch_id format." in res.json().get("detail", "")

            res = client.get(f"/api/batch/{inv_id}")
            assert res.status_code == 400
            assert "Invalid run_id or batch_id format." in res.json().get("detail", "")
