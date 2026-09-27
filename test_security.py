import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_download_json_path_traversal():
    response = client.get("/api/runs/..%2F..%2Frequirements.txt/download/json")
    assert response.status_code in [400, 404]


def test_download_markdown_path_traversal():
    response = client.get("/api/runs/..%2F..%2Frequirements.txt/download/markdown")
    assert response.status_code in [400, 404]


def test_download_docx_path_traversal():
    response = client.get("/api/runs/..%2F..%2Frequirements.txt/download/docx")
    assert response.status_code in [400, 404]


def test_download_batch_details_path_traversal():
    response = client.get("/api/batch/..%2F..%2Frequirements.txt")
    assert response.status_code in [400, 404]


def test_download_batch_markdown_path_traversal():
    response = client.get("/api/batch/..%2F..%2Frequirements.txt/download/markdown")
    assert response.status_code in [400, 404]
