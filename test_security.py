import sys
from unittest.mock import MagicMock

# Mock non-essential heavy/optional dependencies if missing
for mod in [
    "torch", "torchvision", "transformers", "accelerate", "einops", "timm",
    "sentence_transformers", "chromadb", "docx", "playwright", "playwright.async_api",
    "PIL", "Pillow", "duckduckgo_search", "docx.shared", "docx.enum.text", "docx.enum.table"
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_download_report_json_path_traversal():
    response = client.get("/api/runs/..%2F..%2Frequirements.txt/download/json")
    assert response.status_code in (400, 404)

def test_download_report_markdown_path_traversal():
    response = client.get("/api/runs/..%2F..%2Frequirements.txt/download/markdown")
    assert response.status_code in (400, 404)

def test_download_report_docx_path_traversal():
    response = client.get("/api/runs/..%2F..%2Frequirements.txt/download/docx")
    assert response.status_code in (400, 404)

def test_get_batch_details_path_traversal():
    response = client.get("/api/batch/..%2F..%2Frequirements.txt")
    assert response.status_code in (400, 404)

def test_download_batch_markdown_path_traversal():
    response = client.get("/api/batch/..%2F..%2Frequirements.txt/download/markdown")
    assert response.status_code in (400, 404)

def test_invalid_identifier_chars():
    response = client.get("/api/runs/run_123;invalid/download/json")
    assert response.status_code == 400

def test_valid_identifier_format():
    response = client.get("/api/runs/run_valid_123/download/json")
    # Valid format should proceed to file check and return 404 since run_valid_123 doesn't exist
    assert response.status_code == 404
    assert response.json()["detail"] == "JSON report file not found"
