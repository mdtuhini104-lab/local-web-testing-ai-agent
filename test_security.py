"""
Security Unit Tests for Path Traversal Protection
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

def test_get_run_details_path_traversal():
    with TestClient(app) as client:
        # Test valid format (returns 404 since run doesn't exist, but passes identifier validation)
        res_valid = client.get("/api/runs/run_20260101_120000")
        assert res_valid.status_code == 404

        # Test path traversal payloads (using URL encoding for slashes where required by HTTP router)
        payloads = [
            "..%2F..%2Fetc%2Fpasswd",
            "..%5C..%5Cwindows%5Csystem32",
            "run_123..",
            "run_123;cat /etc/passwd",
            "run_123*",
            "invalid_id!@#",
        ]

        for payload in payloads:
            res = client.get(f"/api/runs/{payload}")
            assert res.status_code in (400, 404), f"Payload {payload} did not fail safely: got {res.status_code}"
            if res.status_code == 400:
                assert res.json().get("detail") == "Invalid identifier format"


def test_validate_identifier_unit():
    from fastapi import HTTPException
    from app.api.routes import validate_identifier

    # Valid
    assert validate_identifier("run_20260101_120000") == "run_20260101_120000"
    assert validate_identifier("batch-123_abc") == "batch-123_abc"

    # Invalid payloads
    invalid_payloads = [
        "../etc/passwd",
        "..\\windows",
        "run/123",
        "run..123",
        "run;123",
        "",
        "run*",
    ]
    for inv in invalid_payloads:
        with pytest.raises(HTTPException) as exc_info:
            validate_identifier(inv)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Invalid identifier format"


def test_download_report_json_path_traversal():
    with TestClient(app) as client:
        res = client.get("/api/runs/..%2F..%2Fetc%2Fpasswd/download/json")
        assert res.status_code in (400, 404)
        if res.status_code == 400:
            assert res.json().get("detail") == "Invalid identifier format"


def test_batch_endpoints_path_traversal():
    with TestClient(app) as client:
        payloads = [
            "../../etc/passwd",
            "batch_123/..",
            "batch_123\\..",
        ]

        for payload in payloads:
            res_details = client.get(f"/api/batch/{payload}")
            assert res_details.status_code in (400, 404)
            if res_details.status_code == 400:
                assert res_details.json().get("detail") == "Invalid identifier format"

            res_md = client.get(f"/api/batch/{payload}/download/markdown")
            assert res_md.status_code in (400, 404)
            if res_md.status_code == 400:
                assert res_md.json().get("detail") == "Invalid identifier format"
