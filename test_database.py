import asyncio
import os
import sqlite3
import pytest
import aiosqlite

from app.config import settings
from app.db import database


@pytest.mark.asyncio
async def test_init_db_creates_indexes(tmp_path):
    test_db_path = str(tmp_path / "test_runs.db")
    settings.DB_PATH = test_db_path

    await database.init_db()

    async with aiosqlite.connect(test_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT name FROM sqlite_master WHERE type='index'") as cursor:
            rows = await cursor.fetchall()
            index_names = [row["name"] for row in rows]

    assert "idx_test_steps_run_id" in index_names
    assert "idx_test_runs_created_at" in index_names


@pytest.mark.asyncio
async def test_database_operations(tmp_path):
    test_db_path = str(tmp_path / "test_runs.db")
    settings.DB_PATH = test_db_path

    await database.init_db()
    run_id = "test_run_123"

    await database.create_test_run(run_id=run_id, target_url="http://example.com", username="admin")

    step_data = {
        "step_number": 1,
        "action": "click",
        "target_selector": "#submit",
        "reasoning": "Submitting form",
        "observed_issues": [],
        "ux_feedback": []
    }
    await database.add_test_step(run_id=run_id, step_data=step_data)

    details = await database.get_test_run_details(run_id)
    assert details is not None
    assert details["run_id"] == run_id
    assert len(details["steps"]) == 1
    assert details["steps"][0]["action"] == "click"
