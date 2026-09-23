"""
Security verification test script for Path Traversal vulnerability fix in API routes.
"""

from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.main import app
from app.api.routes import _validate_id

client = TestClient(app)

def test_path_traversal_protection():
    malicious_ids = [
        ".._secret_file",
        "run_123;cat /etc/passwd",
        "run_123*",
        "run 123",
        "run_123<script>",
        "run.123",
        "../etc/passwd",
        "..\\windows\\win.ini",
    ]

    endpoints = [
        "/api/runs/{id}",
        "/api/runs/{id}/download/json",
        "/api/runs/{id}/download/markdown",
        "/api/runs/{id}/download/docx",
        "/api/reports/{id}/docx",
        "/api/batch/{id}",
        "/api/batch/{id}/download/markdown",
    ]

    print("🛡️ Running Path Traversal Security Tests...")

    for ep in endpoints:
        for mal_id in malicious_ids:
            url = ep.format(id=mal_id)
            response = client.get(url)
            if response.status_code != 404:
                assert response.status_code == 400, f"Expected HTTP 400 for {url}, got {response.status_code} ({response.text})"
                assert response.json().get("detail") == "Invalid identifier format"

    print("✅ All path traversal attack payloads reaching handler successfully returned HTTP 400!")

    # Direct unit test of the _validate_id helper function
    valid_test_ids = ["run_123456", "batch-20250101_120000", "test_app1_example_com"]
    for v_id in valid_test_ids:
        assert _validate_id(v_id) == v_id

    invalid_test_ids = ["../secret", "run/123", "run..123", "run_123;cmd", "run_123*", "run 123", ""]
    for inv_id in invalid_test_ids:
        try:
            _validate_id(inv_id)
            assert False, f"Expected HTTPException for invalid id '{inv_id}'"
        except HTTPException as exc:
            assert exc.status_code == 400
            assert exc.detail == "Invalid identifier format"

    print("✅ Unit validation tests for _validate_id passed successfully!")

if __name__ == "__main__":
    test_path_traversal_protection()
