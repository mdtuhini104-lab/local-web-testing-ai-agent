"""
Unit tests for app.db.database module.
Verifies table creation, index initialization, test run operations, step logging, and pruning.
"""

import asyncio
import os
import sqlite3
import pytest
import aiosqlite
from app.config import settings
from app.db import database


@pytest.fixture
def setup_test_db(tmp_path):
    test_db_path = str(tmp_path / "test_app.db")
    original_db_path = settings.DB_PATH
    settings.DB_PATH = test_db_path
    yield test_db_path
    settings.DB_PATH = original_db_path


@pytest.mark.asyncio
async def test_database_init_and_indexes(setup_test_db):
    test_db_path = setup_test_db
    await database.init_db()

    assert os.path.exists(test_db_path)

    async with aiosqlite.connect(test_db_path) as db:
        # Verify tables
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
            tables = [row[0] for row in await cursor.fetchall()]
            assert "test_runs" in tables
            assert "test_steps" in tables

        # Verify indexes
        async with db.execute("SELECT name FROM sqlite_master WHERE type='index'") as cursor:
            indexes = [row[0] for row in await cursor.fetchall()]
            assert "idx_test_steps_run_id_step" in indexes
            assert "idx_test_runs_created_at" in indexes


@pytest.mark.asyncio
async def test_create_and_get_test_run(setup_test_db):
    await database.init_db()
    run_id = "test_run_001"
    target_url = "http://localhost:3000"
    username = "admin"

    await database.create_test_run(run_id, target_url, username)

    runs = await database.get_test_runs(limit=10)
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["target_url"] == target_url
    assert runs[0]["status"] == "running"


@pytest.mark.asyncio
async def test_add_test_steps_and_get_details(setup_test_db):
    await database.init_db()
    run_id = "test_run_002"
    await database.create_test_run(run_id, "http://localhost:3000")

    step1 = {
        "step_number": 1,
        "action": "click",
        "target_selector": "#login-btn",
        "reasoning": "Clicking login button",
        "screenshot_path": "step1.png",
        "observed_issues": [],
        "ux_feedback": ["Good placement"],
    }
    step2 = {
        "step_number": 2,
        "action": "type",
        "target_selector": "#username",
        "reasoning": "Filling username field",
        "screenshot_path": "step2.png",
        "observed_issues": ["Low contrast"],
        "ux_feedback": [],
    }

    await database.add_test_step(run_id, step1)
    await database.add_test_step(run_id, step2)

    details = await database.get_test_run_details(run_id)
    assert details is not None
    assert details["run_id"] == run_id
    assert details["total_steps"] == 2
    assert len(details["steps"]) == 2
    assert details["steps"][0]["step_number"] == 1
    assert details["steps"][1]["step_number"] == 2
    assert details["steps"][0]["action"] == "click"
    assert details["steps"][1]["action"] == "type"


@pytest.mark.asyncio
async def test_prune_old_test_runs(setup_test_db, tmp_path):
    await database.init_db()
    storage_dir = str(tmp_path / "storage")

    for i in range(5):
        run_id = f"run_prune_{i}"
        await database.create_test_run(run_id, f"http://localhost:300{i}")
        await database.add_test_step(run_id, {"step_number": 1, "action": "navigate"})

    result = await database.prune_old_test_runs(keep_latest=3, storage_dir=storage_dir)

    assert result["status"] == "success"
    assert len(result["kept_runs"]) == 3
    assert result["deleted_runs_count"] == 2

    runs_remaining = await database.get_test_runs(limit=10)
    assert len(runs_remaining) == 3
