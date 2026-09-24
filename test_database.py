"""
Unit tests for app.db.database module.
Tests database initialization, schema indexes, CRUD operations, and retention pruning.
"""

import os
import tempfile
import pytest
import sqlite3
import aiosqlite

from app.db import database
from app.config import settings


@pytest.fixture
async def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        original_db_path = settings.DB_PATH
        original_storage_dir = settings.STORAGE_DIR
        db_path = os.path.join(tmpdir, "test.db")
        settings.DB_PATH = db_path
        settings.STORAGE_DIR = tmpdir

        await database.init_db()
        yield db_path, tmpdir

        settings.DB_PATH = original_db_path
        settings.STORAGE_DIR = original_storage_dir


@pytest.mark.anyio
async def test_init_db_creates_indexes(temp_db):
    db_path, _ = temp_db

    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ) as cursor:
            rows = await cursor.fetchall()
            indexes = [r[0] for r in rows]

    assert "idx_test_steps_run_id" in indexes
    assert "idx_test_runs_created_at" in indexes


@pytest.mark.anyio
async def test_create_and_get_test_run(temp_db):
    db_path, _ = temp_db
    run_id = "run_test_001"
    target_url = "http://localhost:3000"

    await database.create_test_run(run_id, target_url, username="admin")
    runs = await database.get_test_runs(limit=10)

    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["target_url"] == target_url
    assert runs[0]["username"] == "admin"
    assert runs[0]["status"] == "running"


@pytest.mark.anyio
async def test_add_test_step_and_get_details(temp_db):
    db_path, _ = temp_db
    run_id = "run_test_002"
    target_url = "http://localhost:3000/login"

    await database.create_test_run(run_id, target_url)

    step_data_1 = {
        "step_number": 1,
        "action": "type",
        "target_selector": "input#username",
        "reasoning": "Entering username",
        "screenshot_path": "step_01.png",
        "observed_issues": [],
        "ux_feedback": ["Good field clarity"],
    }
    step_data_2 = {
        "step_number": 2,
        "action": "click",
        "target_selector": "button#submit",
        "reasoning": "Submitting login form",
        "screenshot_path": "step_02.png",
        "observed_issues": ["Slow API response"],
        "ux_feedback": [],
    }

    await database.add_test_step(run_id, step_data_1)
    await database.add_test_step(run_id, step_data_2)

    details = await database.get_test_run_details(run_id)

    assert details is not None
    assert details["run_id"] == run_id
    assert details["total_steps"] == 2
    assert len(details["steps"]) == 2
    assert details["steps"][0]["action"] == "type"
    assert details["steps"][1]["action"] == "click"
    assert details["steps"][1]["issues"] == ["Slow API response"]


@pytest.mark.anyio
async def test_prune_old_test_runs(temp_db):
    db_path, tmpdir = temp_db

    # Create 5 test runs
    for i in range(1, 6):
        run_id = f"run_00{i}"
        await database.create_test_run(run_id, f"http://localhost:3000/page{i}")
        await database.add_test_step(run_id, {"step_number": 1, "action": "navigate"})

    runs_before = await database.get_test_runs(limit=10)
    assert len(runs_before) == 5

    prune_res = await database.prune_old_test_runs(keep_latest=3, storage_dir=tmpdir)

    assert prune_res["status"] == "success"
    assert prune_res["deleted_runs_count"] == 2

    runs_after = await database.get_test_runs(limit=10)
    assert len(runs_after) == 3
