import pytest
import asyncio
from fastapi.testclient import TestClient
from app.main import app
from app.db import database


@pytest.fixture(autouse=True, scope="module")
def setup_database():
    """Ensure database is initialized before running tests."""
    asyncio.run(database.init_db())


client = TestClient(app)


def test_path_traversal_run_details():
    """Verify that path traversal or illegal characters in run_id are rejected with HTTP 400."""
    # Identifier containing path traversal or invalid characters matching route
    response = client.get("/api/runs/run_123_..")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid identifier parameter"


def test_path_traversal_download_json():
    """Verify path traversal prevention when downloading JSON report."""
    response = client.get("/api/runs/run_123_../download/json")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid identifier parameter"


def test_path_traversal_download_markdown():
    """Verify path traversal prevention when downloading Markdown report."""
    response = client.get("/api/runs/run_123_../download/markdown")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid identifier parameter"


def test_path_traversal_download_docx():
    """Verify path traversal prevention when downloading DOCX report."""
    response = client.get("/api/runs/run_123_../download/docx")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid identifier parameter"


def test_path_traversal_batch_details():
    """Verify path traversal prevention on batch details endpoint."""
    response = client.get("/api/batch/batch_123_..")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid identifier parameter"


def test_path_traversal_batch_markdown():
    """Verify path traversal prevention on batch markdown download endpoint."""
    response = client.get("/api/batch/batch_123_../download/markdown")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid identifier parameter"


def test_valid_identifier_format():
    """Verify that valid identifiers pass validation and return 404 if run does not exist in DB."""
    response = client.get("/api/runs/run_20231004_120000")
    assert response.status_code == 404
    assert response.json()["detail"] == "Test run not found"
