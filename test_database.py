"""
Unit tests for database schema, indexing, persistence, and pruning functions.
"""

import os
import sqlite3
import asyncio
import pytest
from app.config import settings
from app.db import database


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Fixture to redirect DB_PATH and STORAGE_DIR to temporary directory for isolated testing."""
    test_db_path = str(tmp_path / "test_runs.db")
    test_storage_dir = str(tmp_path / "storage")

    monkeypatch.setattr(settings, "DB_PATH", test_db_path)
    monkeypatch.setattr(settings, "STORAGE_DIR", test_storage_dir)
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(tmp_path / "storage" / "chroma_db"))

    os.makedirs(test_storage_dir, exist_ok=True)
    yield test_db_path, test_storage_dir


def test_init_db_creates_tables_and_indexes(temp_db):
    test_db_path, _ = temp_db

    async def _test():
        await database.init_db()

    asyncio.run(_test())

    assert os.path.exists(test_db_path)

    # Inspect sqlite_master for created tables and indexes
    with sqlite3.connect(test_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "test_runs" in tables
        assert "test_steps" in tables

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
        indexes = [row[0] for row in cursor.fetchall()]
        assert "idx_test_runs_created_at" in indexes
        assert "idx_test_steps_run_id_step" in indexes


def test_create_and_get_test_runs(temp_db):
    async def _test():
        await database.init_db()

        run_id = "test_run_001"
        target_url = "http://localhost:3000"
        username = "admin@example.com"

        await database.create_test_run(run_id, target_url, username)

        runs = await database.get_test_runs(limit=10)
        assert len(runs) == 1
        assert runs[0]["run_id"] == run_id
        assert runs[0]["target_url"] == target_url
        assert runs[0]["username"] == username
        assert runs[0]["status"] == "running"

    asyncio.run(_test())


def test_add_test_step_and_get_details(temp_db):
    async def _test():
        await database.init_db()

        run_id = "test_run_002"
        await database.create_test_run(run_id, "http://example.com")

        step_data = {
            "step_number": 1,
            "action": "click",
            "target_selector": "#submit-btn",
            "reasoning": "Submitting login form",
            "screenshot_path": "step_01.png",
            "observed_issues": ["Visual alignment warning"],
            "ux_feedback": ["Button text clear"],
        }
        await database.add_test_step(run_id, step_data)

        details = await database.get_test_run_details(run_id)
        assert details is not None
        assert details["run_id"] == run_id
        assert details["total_steps"] == 1
        assert len(details["steps"]) == 1

        step = details["steps"][0]
        assert step["step_number"] == 1
        assert step["action"] == "click"
        assert step["issues"] == ["Visual alignment warning"]
        assert step["ux_feedback"] == ["Button text clear"]

    asyncio.run(_test())


def test_prune_old_test_runs(temp_db):
    _, test_storage = temp_db

    async def _test():
        await database.init_db()

        # Create 5 runs
        for i in range(1, 6):
            run_id = f"run_{i:03d}"
            await database.create_test_run(run_id, f"http://example.com/{i}")
            await database.add_test_step(run_id, {"step_number": 1, "action": "click"})

        runs_before = await database.get_test_runs(limit=10)
        assert len(runs_before) == 5

        # Prune keeping latest 3
        result = await database.prune_old_test_runs(keep_latest=3, storage_dir=test_storage)
        assert result["status"] == "success"
        assert result["deleted_runs_count"] == 2

        runs_after = await database.get_test_runs(limit=10)
        assert len(runs_after) == 3

    asyncio.run(_test())


def test_is_safe_to_delete_run_dir():
    allowed_parent = "/tmp/storage/runs"

    # Valid child
    assert database.is_safe_to_delete_run_dir("/tmp/storage/runs/run_001", allowed_parent) is True

    # Disallowed parent or root
    assert database.is_safe_to_delete_run_dir("/tmp/storage/runs", allowed_parent) is False
    assert database.is_safe_to_delete_run_dir("/tmp/storage", allowed_parent) is False

    # Path traversal attack
    assert database.is_safe_to_delete_run_dir("/tmp/storage/runs/../other", allowed_parent) is False
