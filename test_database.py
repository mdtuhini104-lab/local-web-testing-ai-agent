"""
Unit tests for SQLite database initialization, index creation, test run CRUD operations, and retention pruning.
"""

import asyncio
import os
import pytest
import aiosqlite

from app.config import settings
from app.db import database


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Fixture to redirect DB_PATH and STORAGE_DIR to a temporary path for isolated testing."""
    test_db_path = str(tmp_path / "test_runs.db")
    test_storage_dir = str(tmp_path / "storage")
    os.makedirs(test_storage_dir, exist_ok=True)

    monkeypatch.setattr(settings, "DB_PATH", test_db_path)
    monkeypatch.setattr(settings, "STORAGE_DIR", test_storage_dir)
    return test_db_path


def test_init_db_creates_tables_and_indexes():
    """Verify that init_db creates required tables and performance indexes."""

    async def _test():
        await database.init_db()

        assert os.path.exists(settings.DB_PATH)

        async with aiosqlite.connect(settings.DB_PATH) as db:
            # Check tables
            async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
                tables = [row[0] for row in await cursor.fetchall()]
                assert "test_runs" in tables
                assert "test_steps" in tables

            # Check indexes
            async with db.execute("SELECT name FROM sqlite_master WHERE type='index'") as cursor:
                indexes = [row[0] for row in await cursor.fetchall()]
                assert "idx_test_runs_created_at" in indexes
                assert "idx_test_steps_run_id_step" in indexes

    asyncio.run(_test())


def test_create_and_get_test_runs():
    """Verify test run creation, step addition, and details retrieval."""

    async def _test():
        await database.init_db()

        run_id = "run_test_001"
        target_url = "http://localhost:3000"

        await database.create_test_run(run_id, target_url, username="admin")

        runs = await database.get_test_runs(limit=10)
        assert len(runs) == 1
        assert runs[0]["run_id"] == run_id
        assert runs[0]["status"] == "running"

        # Add steps
        step1 = {
            "step_number": 1,
            "action": "click",
            "target_selector": "button#submit",
            "reasoning": "Submitting login form",
            "screenshot_path": "step1.png",
            "observed_issues": ["Low contrast"],
            "ux_feedback": ["Add loading state"],
        }
        await database.add_test_step(run_id, step1)

        # Update status
        await database.update_test_run_status(
            run_id=run_id,
            status="completed",
            duration_seconds=12.5,
            total_steps=1,
            overall_ux_rating="A",
            summary_data={"total_steps": 1, "completed_reason": "Finished"},
        )

        details = await database.get_test_run_details(run_id)
        assert details is not None
        assert details["run_id"] == run_id
        assert details["status"] == "completed"
        assert len(details["steps"]) == 1
        assert details["steps"][0]["action"] == "click"
        assert details["steps"][0]["issues"] == ["Low contrast"]

    asyncio.run(_test())


def test_prune_old_test_runs():
    """Verify retention policy prunes older runs beyond keep_latest."""

    async def _test():
        await database.init_db()

        # Create 5 runs
        for i in range(1, 6):
            run_id = f"run_00{i}"
            await database.create_test_run(run_id, f"http://app{i}.com")
            await database.add_test_step(run_id, {"step_number": 1, "action": "navigate"})

        # Prune keeping latest 3
        res = await database.prune_old_test_runs(keep_latest=3)
        assert res["status"] == "success"
        assert res["deleted_runs_count"] == 2

        remaining_runs = await database.get_test_runs(limit=10)
        assert len(remaining_runs) == 3

    asyncio.run(_test())
