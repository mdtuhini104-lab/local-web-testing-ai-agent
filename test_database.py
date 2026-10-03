"""
Unit tests for database initialization, indexing optimizations, and CRUD operations.
"""

import os
import tempfile
import pytest
import sqlite3
import aiosqlite
from app.config import settings
from app.db import database


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Overrides DB_PATH and STORAGE_DIR to use temporary directory during testing."""
    test_db_path = str(tmp_path / "test_runs.db")
    test_storage_dir = str(tmp_path / "storage")
    monkeypatch.setattr(settings, "DB_PATH", test_db_path)
    monkeypatch.setattr(settings, "STORAGE_DIR", test_storage_dir)
    return test_db_path


@pytest.mark.asyncio
async def test_init_db_creates_indexes():
    """Verifies that init_db creates tables and performance indexes."""
    await database.init_db()

    async with aiosqlite.connect(settings.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT name FROM sqlite_master WHERE type='index'") as cursor:
            rows = await cursor.fetchall()
            index_names = [row["name"] for row in rows]

        assert "idx_test_steps_run_id_step" in index_names
        assert "idx_test_runs_created_at" in index_names


@pytest.mark.asyncio
async def test_query_explain_plan_uses_indexes():
    """Verifies EXPLAIN QUERY PLAN demonstrates index usage for steps and runs queries."""
    await database.init_db()

    async with aiosqlite.connect(settings.DB_PATH) as db:
        # Check query execution plan for steps retrieval
        async with db.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM test_steps WHERE run_id = 'test_123' ORDER BY step_number ASC"
        ) as cursor:
            plan_rows = await cursor.fetchall()
            plan_str = " ".join([str(row) for row in plan_rows]).lower()
            assert "idx_test_steps_run_id_step" in plan_str

        # Check query execution plan for recent runs retrieval
        async with db.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM test_runs ORDER BY created_at DESC LIMIT 50"
        ) as cursor:
            plan_rows = await cursor.fetchall()
            plan_str = " ".join([str(row) for row in plan_rows]).lower()
            assert "idx_test_runs_created_at" in plan_str


@pytest.mark.asyncio
async def test_create_and_get_test_run_details():
    """Verifies test run creation, step addition, and detail fetching."""
    await database.init_db()
    run_id = "run_test_001"
    target_url = "http://localhost:3000"

    await database.create_test_run(run_id, target_url, "admin")
    await database.add_test_step(run_id, {
        "step_number": 1,
        "action": "click",
        "target_selector": "button#submit",
        "reasoning": "Submitting login form",
        "observed_issues": ["Low contrast"],
        "ux_feedback": ["Button is small"],
    })

    details = await database.get_test_run_details(run_id)
    assert details is not None
    assert details["run_id"] == run_id
    assert details["target_url"] == target_url
    assert len(details["steps"]) == 1
    assert details["steps"][0]["action"] == "click"
    assert details["steps"][0]["issues"] == ["Low contrast"]


@pytest.mark.asyncio
async def test_prune_old_test_runs():
    """Verifies retention pruning of old runs."""
    await database.init_db()

    for i in range(5):
        run_id = f"run_{i:03d}"
        await database.create_test_run(run_id, "http://localhost:3000")

    runs_before = await database.get_test_runs(limit=10)
    assert len(runs_before) == 5

    res = await database.prune_old_test_runs(keep_latest=3)
    assert res["status"] == "success"
    assert res["deleted_runs_count"] == 2

    runs_after = await database.get_test_runs(limit=10)
    assert len(runs_after) == 3
