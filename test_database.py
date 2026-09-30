"""
Unit tests for app/db/database.py performance optimizations and CRUD operations.
"""

import asyncio
import os
import sqlite3
import tempfile
import uuid
import pytest
import aiosqlite

from app.config import settings
from app.db import database


@pytest.fixture(autouse=True)
def setup_tmp_db(monkeypatch):
    """Isolated temporary database for unit tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_db_path = tmp.name
    monkeypatch.setattr(settings, "DB_PATH", tmp_db_path)
    yield tmp_db_path
    if os.path.exists(tmp_db_path):
        try:
            os.remove(tmp_db_path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_database_indexes_and_wal():
    """Verifies SQLite WAL mode and index creation on init_db()."""
    await database.init_db()

    # Verify via standard sqlite3 connection
    with sqlite3.connect(settings.DB_PATH) as con:
        cur = con.cursor()
        cur.execute("PRAGMA journal_mode;")
        mode = cur.fetchone()[0]
        assert mode.lower() == "wal", f"Expected WAL journal mode, got {mode}"

        cur.execute("SELECT name FROM sqlite_master WHERE type='index';")
        indexes = {row[0] for row in cur.fetchall()}
        assert "idx_test_steps_run_id" in indexes, "idx_test_steps_run_id index missing"
        assert "idx_test_runs_created_at" in indexes, "idx_test_runs_created_at index missing"


@pytest.mark.asyncio
async def test_database_crud_and_pruning():
    """Verifies test run creation, step addition, detail retrieval, and retention pruning."""
    await database.init_db()

    uid = str(uuid.uuid4())[:8]
    run_id_1 = f"test_run_{uid}_1"
    run_id_2 = f"test_run_{uid}_2"
    run_id_3 = f"test_run_{uid}_3"
    run_id_4 = f"test_run_{uid}_4"

    for r_id in [run_id_1, run_id_2, run_id_3, run_id_4]:
        await database.create_test_run(r_id, "http://example.com", "admin")
        await database.add_test_step(r_id, {
            "step_number": 1,
            "action": "click",
            "target_selector": "#btn-submit",
            "reasoning": "Submitting test form",
            "observed_issues": ["Minor warning"],
            "ux_feedback": ["Good responsiveness"]
        })
        await database.update_test_run_status(r_id, "completed", duration_seconds=5.0, total_steps=1)

    runs = await database.get_test_runs(limit=10)
    assert len(runs) >= 4, f"Expected at least 4 test runs, got {len(runs)}"

    details = await database.get_test_run_details(run_id_4)
    assert details is not None
    assert details["run_id"] == run_id_4
    assert len(details["steps"]) == 1
    assert details["steps"][0]["action"] == "click"

    # Test retention pruning (keeping latest 3)
    prune_result = await database.prune_old_test_runs(keep_latest=3)
    assert prune_result["status"] == "success"
    assert prune_result["deleted_runs_count"] >= 1


if __name__ == "__main__":
    asyncio.run(test_database_indexes_and_wal())
    asyncio.run(test_database_crud_and_pruning())
    print("✅ All database unit tests passed successfully!")
