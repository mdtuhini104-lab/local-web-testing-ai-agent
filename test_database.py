import os
import tempfile
import pytest
import pytest_asyncio
import sqlite3
import aiosqlite
from unittest.mock import patch

from app.db import database


@pytest_asyncio.fixture
async def temp_db():
    """Sets up a temporary SQLite database for testing database operations."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db_path = f.name

    with patch("app.config.settings.DB_PATH", temp_db_path):
        await database.init_db()
        yield temp_db_path

    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)


@pytest.mark.asyncio
async def test_init_db_creates_tables_and_indexes(temp_db):
    """Verifies that init_db creates test_runs, test_steps, and performance indexes."""
    async with aiosqlite.connect(temp_db) as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ) as cursor:
            tables = [row[0] for row in await cursor.fetchall()]
            assert "test_runs" in tables
            assert "test_steps" in tables

        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ) as cursor:
            indexes = [row[0] for row in await cursor.fetchall()]
            assert "idx_test_runs_created_at" in indexes
            assert "idx_test_steps_run_id_step" in indexes


@pytest.mark.asyncio
async def test_query_plans_use_indexes(temp_db):
    """Verifies that SQLite query planner uses indexes for queries on test_runs and test_steps."""
    async with aiosqlite.connect(temp_db) as db:
        # Check query plan for test_runs created_at ordering
        async with db.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM test_runs ORDER BY created_at DESC LIMIT 10"
        ) as cursor:
            rows = await cursor.fetchall()
            plan_detail = " ".join(str(r) for r in rows)
            assert "idx_test_runs_created_at" in plan_detail

        # Check query plan for test_steps run_id filtering
        async with db.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM test_steps WHERE run_id = 'run_1' ORDER BY step_number ASC"
        ) as cursor:
            rows = await cursor.fetchall()
            plan_detail = " ".join(str(r) for r in rows)
            assert "idx_test_steps_run_id_step" in plan_detail


@pytest.mark.asyncio
async def test_create_and_get_test_run(temp_db):
    """Verifies creation and retrieval of test run records."""
    run_id = "test_run_001"
    target_url = "http://example.com"
    username = "test_user"

    with patch("app.config.settings.DB_PATH", temp_db):
        await database.create_test_run(run_id, target_url, username)
        runs = await database.get_test_runs(limit=10)

        assert len(runs) == 1
        assert runs[0]["run_id"] == run_id
        assert runs[0]["target_url"] == target_url
        assert runs[0]["username"] == username
        assert runs[0]["status"] == "running"


@pytest.mark.asyncio
async def test_add_test_steps_and_fetch_details(temp_db):
    """Verifies adding execution steps and fetching ordered run details."""
    run_id = "test_run_002"
    target_url = "http://example.com/login"

    with patch("app.config.settings.DB_PATH", temp_db):
        await database.create_test_run(run_id, target_url)

        # Add steps out of order to verify index ordering
        await database.add_test_step(
            run_id,
            {
                "step_number": 2,
                "action": "click",
                "target_selector": "#submit",
                "reasoning": "Click submit button",
                "observed_issues": [],
                "ux_feedback": ["Fast response"],
            },
        )
        await database.add_test_step(
            run_id,
            {
                "step_number": 1,
                "action": "type",
                "target_selector": "#username",
                "reasoning": "Enter username",
                "observed_issues": ["Low contrast"],
                "ux_feedback": [],
            },
        )

        details = await database.get_test_run_details(run_id)

        assert details is not None
        assert details["run_id"] == run_id
        assert details["total_steps"] == 2
        assert len(details["steps"]) == 2

        # Verify steps are ordered by step_number ASC
        assert details["steps"][0]["step_number"] == 1
        assert details["steps"][0]["action"] == "type"
        assert details["steps"][0]["issues"] == ["Low contrast"]

        assert details["steps"][1]["step_number"] == 2
        assert details["steps"][1]["action"] == "click"
        assert details["steps"][1]["ux_feedback"] == ["Fast response"]


@pytest.mark.asyncio
async def test_prune_old_test_runs(temp_db):
    """Verifies retention policy prunes old runs while keeping specified latest runs."""
    with patch("app.config.settings.DB_PATH", temp_db):
        for i in range(5):
            run_id = f"run_{i}"
            await database.create_test_run(run_id, f"http://example.com/{i}")
            await database.add_test_step(
                run_id, {"step_number": 1, "action": f"action_{i}"}
            )

        runs_before = await database.get_test_runs(limit=10)
        assert len(runs_before) == 5

        result = await database.prune_old_test_runs(keep_latest=2)
        assert result["status"] == "success"
        assert result["deleted_runs_count"] == 3

        runs_after = await database.get_test_runs(limit=10)
        assert len(runs_after) == 2
