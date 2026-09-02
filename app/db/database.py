"""
SQLite Database module using aiosqlite.
Manages schema creation, test run persistence, step log saving, and report retrieval.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import aiosqlite
from app.config import settings

logger = logging.getLogger("database")


async def init_db() -> None:
    """Initializes the SQLite database tables."""
    os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
    async with aiosqlite.connect(settings.DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS test_runs (
                run_id TEXT PRIMARY KEY,
                target_url TEXT NOT NULL,
                username TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                duration_seconds REAL DEFAULT 0.0,
                total_steps INTEGER DEFAULT 0,
                overall_ux_rating TEXT,
                summary_json TEXT
            );
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS test_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                action TEXT NOT NULL,
                target_selector TEXT,
                reasoning TEXT,
                screenshot_path TEXT,
                issues_json TEXT,
                ux_feedback_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES test_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        await db.commit()
    logger.info(f"💾 SQLite Database initialized at {settings.DB_PATH}")


async def create_test_run(run_id: str, target_url: str, username: Optional[str] = None) -> None:
    """Creates a new test run record with 'running' status."""
    now = os.sys.modules["datetime"].datetime.now().isoformat()
    async with aiosqlite.connect(settings.DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO test_runs (run_id, target_url, username, status, created_at, updated_at)
            VALUES (?, ?, ?, 'running', ?, ?)
            """,
            (run_id, target_url, username, now, now),
        )
        await db.commit()


async def update_test_run_status(
    run_id: str,
    status: str,
    duration_seconds: float = 0.0,
    total_steps: int = 0,
    overall_ux_rating: Optional[str] = None,
    summary_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Updates test run status upon completion or failure."""
    now = os.sys.modules["datetime"].datetime.now().isoformat()
    summary_json = json.dumps(summary_data) if summary_data else None

    async with aiosqlite.connect(settings.DB_PATH) as db:
        await db.execute(
            """
            UPDATE test_runs
            SET status = ?, updated_at = ?, duration_seconds = ?, total_steps = ?, overall_ux_rating = ?, summary_json = ?
            WHERE run_id = ?
            """,
            (status, now, duration_seconds, total_steps, overall_ux_rating, summary_json, run_id),
        )
        await db.commit()


async def add_test_step(run_id: str, step_data: Dict[str, Any]) -> None:
    """Saves individual step execution logs and dynamically updates total_steps counter."""
    now = os.sys.modules["datetime"].datetime.now().isoformat()
    issues_json = json.dumps(step_data.get("observed_issues", []))
    ux_json = json.dumps(step_data.get("ux_feedback", []))
    step_num = step_data.get("step_number", 0)

    async with aiosqlite.connect(settings.DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO test_steps (run_id, step_number, action, target_selector, reasoning, screenshot_path, issues_json, ux_feedback_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                step_num,
                step_data.get("action", ""),
                step_data.get("target_selector", ""),
                step_data.get("reasoning", ""),
                step_data.get("screenshot_path", ""),
                issues_json,
                ux_json,
                now,
            ),
        )
        await db.execute(
            """
            UPDATE test_runs
            SET total_steps = MAX(total_steps, ?), updated_at = ?
            WHERE run_id = ?
            """,
            (step_num, now, run_id),
        )
        await db.commit()


async def get_test_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """Fetches list of test runs ordered by creation date."""
    async with aiosqlite.connect(settings.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM test_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def get_test_run_details(run_id: str) -> Optional[Dict[str, Any]]:
    """Fetches full details of a specific test run including all steps."""
    async with aiosqlite.connect(settings.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM test_runs WHERE run_id = ?", (run_id,)) as cursor:
            run_row = await cursor.fetchone()
            if not run_row:
                return None

            run_dict = dict(run_row)
            if run_dict.get("summary_json"):
                run_dict["summary"] = json.loads(run_dict["summary_json"])

        async with db.execute(
            "SELECT * FROM test_steps WHERE run_id = ? ORDER BY step_number ASC", (run_id,)
        ) as cursor:
            step_rows = await cursor.fetchall()
            steps = []
            for row in step_rows:
                s = dict(row)
                s["issues"] = json.loads(s["issues_json"]) if s.get("issues_json") else []
                s["ux_feedback"] = json.loads(s["ux_feedback_json"]) if s.get("ux_feedback_json") else []
                steps.append(s)

            run_dict["steps"] = steps
            return run_dict
