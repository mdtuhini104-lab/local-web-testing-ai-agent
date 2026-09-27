"""
Unit tests for app.db.database module.
Tests schema initialization, index creation, run insertion, step execution logging,
retention pruning, and query plan index utilization.
"""

import os
import sqlite3
import tempfile
import pytest
import aiosqlite

from app.config import settings
import app.db.database as db_module


@pytest.fixture(autouse=True)
def temp_db_path(monkeypatch, tmp_path):
    test_db = str(tmp_path / "test_runs.db")
    test_storage = str(tmp_path / "storage")
    monkeypatch.setattr(settings, "DB_PATH", test_db)
    monkeypatch.setattr(settings, "STORAGE_DIR", test_storage)
    os.makedirs(test_storage, exist_ok=True)
    yield test_db


@pytest.mark.asyncio
async def test_init_db_creates_indexes():
    """Verify init_db initializes tables and performance indexes."""
    await db_module.init_db()

    async with aiosqlite.connect(settings.DB_PATH) as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ) as cursor:
            rows = await cursor.fetchall()
            index_names = [row[0] for row in rows]

    assert "idx_test_steps_run_id_step" in index_names
    assert "idx_test_runs_created_at" in index_names


@pytest.mark.asyncio
async def test_crud_and_query_plans():
    """Verify test run creation, step logging, status update, details fetching, and index usage."""
    await db_module.init_db()

    run_id = "run_test_001"
    target_url = "http://localhost:3000"

    # Create test run
    await db_module.create_test_run(run_id, target_url, username="testuser")

    # Log step
    await db_module.add_test_step(
        run_id,
        {
            "step_number": 1,
            "action": "click",
            "target_selector": "button#submit",
            "reasoning": "Submitting login form",
            "screenshot_path": "step1.png",
            "observed_issues": ["Low contrast"],
            "ux_feedback": ["Consider brighter primary button"],
        },
    )

    # Update run status
    await db_module.update_test_run_status(
        run_id=run_id,
        status="completed",
        duration_seconds=5.2,
        total_steps=1,
        overall_ux_rating="Good",
        summary_data={"completed_reason": "All checks passed"},
    )

    # Fetch list
    runs = await db_module.get_test_runs(limit=10)
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id

    # Fetch details
    details = await db_module.get_test_run_details(run_id)
    assert details is not None
    assert details["run_id"] == run_id
    assert len(details["steps"]) == 1
    assert details["steps"][0]["action"] == "click"
    assert details["steps"][0]["issues"] == ["Low contrast"]

    # Verify EXPLAIN QUERY PLAN uses indexes
    async with aiosqlite.connect(settings.DB_PATH) as db:
        # Step query index check
        async with db.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM test_steps WHERE run_id = ? ORDER BY step_number ASC",
            (run_id,),
        ) as cursor:
            step_qp_rows = await cursor.fetchall()
            step_plan = " ".join(str(r) for r in step_qp_rows)
            assert "idx_test_steps_run_id_step" in step_plan

        # Runs listing index check
        async with db.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM test_runs ORDER BY created_at DESC LIMIT 50"
        ) as cursor:
            runs_qp_rows = await cursor.fetchall()
            runs_plan = " ".join(str(r) for r in runs_qp_rows)
            assert "idx_test_runs_created_at" in runs_plan


@pytest.mark.asyncio
async def test_retention_pruning():
    """Verify prune_old_test_runs keeps only keep_latest runs and cleans DB/files."""
    await db_module.init_db()

    # Create 5 runs
    for i in range(5):
        rid = f"run_batch_{i}"
        await db_module.create_test_run(rid, f"http://app{i}.com")
        await db_module.add_test_step(rid, {"step_number": 1, "action": "navigate"})

    res = await db_module.prune_old_test_runs(keep_latest=3, storage_dir=settings.STORAGE_DIR)
    assert res["status"] == "success"
    assert res["deleted_runs_count"] == 2

    remaining_runs = await db_module.get_test_runs(limit=10)
    assert len(remaining_runs) == 3
