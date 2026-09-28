"""
Security unit tests for API endpoints.
Tests path traversal vulnerability protection on run_id and batch_id parameters.
"""

import sys
import pytest
import asyncio
from unittest.mock import MagicMock

# Mock heavyweight ML/browser dependencies if not installed in test environment
for mod in [
    'torch',
    'PIL',
    'transformers',
    'torchvision',
    'accelerate',
    'einops',
    'timm',
    'sentence_transformers',
    'chromadb',
    'docx',
    'playwright',
    'playwright.async_api',
    'duckduckgo_search',
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from fastapi.testclient import TestClient
from app.main import app
from app.db import database


@pytest.fixture(autouse=True)
def setup_db():
    asyncio.run(database.init_db())


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_get_run_details_path_traversal(client):
    # Attempting path traversal in run_id parameter
    response = client.get("/api/runs/../etc/passwd")
    assert response.status_code in [400, 404]

    response = client.get("/api/runs/run_123;SELECT")
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid identifier parameter."}


def test_download_report_json_path_traversal(client):
    response = client.get("/api/runs/../../etc/passwd/download/json")
    assert response.status_code in [400, 404]

    response = client.get("/api/runs/invalid..id/download/json")
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid identifier parameter."}


def test_download_report_markdown_path_traversal(client):
    response = client.get("/api/runs/invalid/../id/download/markdown")
    assert response.status_code in [400, 404]

    response = client.get("/api/runs/invalid@id/download/markdown")
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid identifier parameter."}


def test_download_report_docx_path_traversal(client):
    response = client.get("/api/reports/../../secret/docx")
    assert response.status_code in [400, 404]

    response = client.get("/api/reports/invalid$id/docx")
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid identifier parameter."}


def test_get_batch_details_path_traversal(client):
    response = client.get("/api/batch/../secret_batch")
    assert response.status_code in [400, 404]

    response = client.get("/api/batch/batch_123;drop")
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid identifier parameter."}


def test_download_batch_markdown_path_traversal(client):
    response = client.get("/api/batch/../../etc/download/markdown")
    assert response.status_code in [400, 404]

    response = client.get("/api/batch/batch!invalid/download/markdown")
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid identifier parameter."}


def test_valid_identifier_format(client):
    # Valid identifier format should pass validate_identifier and return 404 (Not Found) if run/batch doesn't exist
    response = client.get("/api/runs/valid_run_123-abc")
    assert response.status_code == 404
    assert response.json() == {"detail": "Test run not found"}
