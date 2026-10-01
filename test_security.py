"""
Security Unit Tests for Local Web Testing AI Agent.
Tests path traversal vulnerability protections in API endpoints.
"""

import sys
from unittest.mock import MagicMock

# Mock heavy ML dependencies if not present in the lightweight test environment
for mod in [
    'torch',
    'torch.nn',
    'transformers',
    'sentence_transformers',
    'chromadb',
    'duckduckgo_search',
    'einops',
    'timm',
]:
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.routes import validate_identifier
from app.main import app

client = TestClient(app)


def test_validate_identifier_valid():
    """Test that valid run and batch IDs pass validation."""
    assert validate_identifier("run_20260330_120000") == "run_20260330_120000"
    assert validate_identifier("batch_20260330_120000_app1") == "batch_20260330_120000_app1"
    assert validate_identifier("valid-id-123_456") == "valid-id-123_456"


def test_validate_identifier_invalid_path_traversal():
    """Test that path traversal patterns trigger HTTP 400."""
    invalid_ids = [
        "../etc/passwd",
        "../../storage/runs",
        "run_id/subdir",
        "run_id\\subdir",
        "..%2F..%2F",
        "run;cat /etc/passwd",
        "run..id",
        "",
        "   ",
    ]
    for invalid_id in invalid_ids:
        with pytest.raises(HTTPException) as exc_info:
            validate_identifier(invalid_id)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Invalid identifier format."


def test_api_path_traversal_prevention_endpoints():
    """Test that API endpoints reject path traversal / invalid identifiers with HTTP 400."""
    path_traversal_payloads = [
        "run..id",
        "run_id;dir",
        "..invalid",
    ]

    for payload in path_traversal_payloads:
        # GET /api/runs/{run_id}
        res = client.get(f"/api/runs/{payload}")
        assert res.status_code == 400, f"Expected 400 for {payload}, got {res.status_code}"
        assert res.json().get("detail") == "Invalid identifier format."

        # GET /api/runs/{run_id}/download/json
        res = client.get(f"/api/runs/{payload}/download/json")
        assert res.status_code == 400, f"Expected 400 for {payload}, got {res.status_code}"

        # GET /api/runs/{run_id}/download/markdown
        res = client.get(f"/api/runs/{payload}/download/markdown")
        assert res.status_code == 400, f"Expected 400 for {payload}, got {res.status_code}"

        # GET /api/runs/{run_id}/download/docx
        res = client.get(f"/api/runs/{payload}/download/docx")
        assert res.status_code == 400, f"Expected 400 for {payload}, got {res.status_code}"

        # GET /api/batch/{batch_id}
        res = client.get(f"/api/batch/{payload}")
        assert res.status_code == 400, f"Expected 400 for {payload}, got {res.status_code}"

        # GET /api/batch/{batch_id}/download/markdown
        res = client.get(f"/api/batch/{payload}/download/markdown")
        assert res.status_code == 400, f"Expected 400 for {payload}, got {res.status_code}"
