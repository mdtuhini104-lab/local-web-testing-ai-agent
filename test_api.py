import requests
from requests.exceptions import ConnectionError, Timeout
import time
import json
from fastapi.testclient import TestClient
from app.main import app

def test_path_traversal_protection():
    """Verify that path traversal payloads in run_id or batch_id return HTTP 400 Bad Request."""
    bad_identifiers = [
        "invalid..id",
        "invalid_id.json",
        "run_123;SELECT",
        "run_123<script>",
        "run_123%00",
    ]
    with TestClient(app) as client:
        for bad_id in bad_identifiers:
            res = client.get(f"/api/runs/{bad_id}")
            assert res.status_code == 400, f"Expected 400 for run_id '{bad_id}', got {res.status_code}"
            assert res.json()["detail"] == "Invalid identifier format"

            res_json = client.get(f"/api/runs/{bad_id}/download/json")
            assert res_json.status_code == 400

            res_batch = client.get(f"/api/batch/{bad_id}")
            assert res_batch.status_code == 400


def test_valid_identifier_sanitization():
    """Verify that valid identifiers pass sanitization (returns 404 when not found, rather than 400)."""
    valid_id = "run_20260101_120000"
    with TestClient(app) as client:
        res = client.get(f"/api/runs/{valid_id}")
        assert res.status_code == 404


def test_live_api_if_running():
    try:
        resp = requests.get("http://127.0.0.1:8000/api/models", timeout=2)
        if resp.status_code == 200:
            print("Live server running on 8000")
    except Exception:
        pass


if __name__ == "__main__":
    test_path_traversal_protection()
    test_valid_identifier_sanitization()
    print("✅ Security tests passed!")

