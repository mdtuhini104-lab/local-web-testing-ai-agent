import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_path_traversal_download_json():
    response = client.get("/api/runs/..%2F..%2Fapp%2Fconfig.py/download/json")
    assert response.status_code in [400, 404]

def test_path_traversal_download_markdown():
    response = client.get("/api/runs/..%2F..%2Fapp%2Fconfig.py/download/markdown")
    assert response.status_code in [400, 404]

def test_path_traversal_download_docx():
    response = client.get("/api/runs/..%2F..%2Fapp%2Fconfig.py/download/docx")
    assert response.status_code in [400, 404]

def test_path_traversal_batch_download_markdown():
    response = client.get("/api/batch/..%2F..%2Fapp%2Fconfig.py/download/markdown")
    assert response.status_code in [400, 404]

def test_path_traversal_get_run_details():
    response = client.get("/api/runs/..%2F..%2Fapp%2Fconfig.py")
    assert response.status_code in [400, 404]

def test_path_traversal_get_batch_details():
    response = client.get("/api/batch/..%2F..%2Fapp%2Fconfig.py")
    assert response.status_code in [400, 404]
