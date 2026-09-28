import asyncio
import os
import sqlite3
import tempfile
import pytest
import aiosqlite

from app.config import settings
from app.db.database import (
    init_db,
    create_test_run,
    add_test_step,
    get_test_runs,
    get_test_run_details,
    update_test_run_status,
    prune_old_test_runs,
)


@pytest.fixture(autouse=True)
def temp_db_path(monkeypatch, tmp_path):
    """Fixture to use a temporary database file for isolation."""
    db_file = tmp_path / "test_runs.db"
    storage_dir = tmp_path / "storage"
    monkeypatch.setattr(settings, "DB_PATH", str(db_file))
    monkeypatch.setattr(settings, "STORAGE_DIR", str(storage_dir))
    return db_file


@pytest.mark.anyio
async def test_init_db_creates_indexes_and_tables(temp_db_path):
    """Verifies init_db creates tables and database indexes for test_steps and test_runs."""
    await init_db()

    assert os.path.exists(temp_db_path)

    async with aiosqlite.connect(temp_db_path) as db:
        # Check tables exist
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ) as cursor:
            rows = await cursor.fetchall()
            table_names = [r[0] for r in rows]
            assert "test_runs" in table_names
            assert "test_steps" in table_names

        # Check indexes exist
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='index';"
        ) as cursor:
            rows = await cursor.fetchall()
            index_names = [r[0] for r in rows]
            assert "idx_test_steps_run_id_step" in index_names
            assert "idx_test_runs_created_at" in index_names


@pytest.mark.anyio
async def test_test_runs_crud_and_indexing(temp_db_path):
    """Verifies creating, adding steps, updating status, and retrieving test runs."""
    await init_db()

    run_id = "test_run_001"
    target_url = "http://localhost:3000"
    username = "admin"

    # Create run
    await create_test_run(run_id, target_url, username)

    # Add steps
    step_data_1 = {
        "step_number": 1,
        "action": "click",
        "target_selector": "#login-btn",
        "reasoning": "Clicking login",
        "screenshot_path": "step_01.png",
        "observed_issues": ["Low contrast"],
        "ux_feedback": ["Improve button contrast"],
    }
    step_data_2 = {
        "step_number": 2,
        "action": "type",
        "target_selector": "#username-input",
        "reasoning": "Typing username",
        "screenshot_path": "step_02.png",
        "observed_issues": [],
        "ux_feedback": [],
    }
    await add_test_step(run_id, step_data_1)
    await add_test_step(run_id, step_data_2)

    # Update status
    await update_test_run_status(
        run_id=run_id,
        status="completed",
        duration_seconds=12.5,
        total_steps=2,
        overall_ux_rating="Good (4/5)",
        summary_data={"total_steps": 2},
    )

    # Retrieve list
    runs = await get_test_runs(limit=10)
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["status"] == "completed"

    # Retrieve details
    details = await get_test_run_details(run_id)
    assert details is not None
    assert details["run_id"] == run_id
    assert len(details["steps"]) == 2
    assert details["steps"][0]["action"] == "click"
    assert details["steps"][1]["action"] == "type"


@pytest.mark.anyio
async def test_prune_old_test_runs(temp_db_path):
    """Verifies retention policy prunes older runs beyond keep_latest."""
    await init_db()

    # Create 5 runs
    for i in range(1, 6):
        r_id = f"run_{i:03d}"
        await create_test_run(r_id, f"http://localhost:300{i}")
        await add_test_step(r_id, {"step_number": 1, "action": "navigate"})

    runs_before = await get_test_runs(limit=10)
    assert len(runs_before) == 5

    # Prune keeping latest 2
    prune_res = await prune_old_test_runs(keep_latest=2)
    assert prune_res["status"] == "success"
    assert prune_res["deleted_runs_count"] == 3

    runs_after = await get_test_runs(limit=10)
    assert len(runs_after) == 2
