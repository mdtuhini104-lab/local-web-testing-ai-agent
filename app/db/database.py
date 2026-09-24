"""
SQLite Database module using aiosqlite.
Manages schema creation, test run persistence, step log saving, and report retrieval.
"""

import json
import logging
import os
import shutil
import sqlite3
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
        # Bolt Performance Optimization:
        # Create database indexes on foreign key `run_id` and creation timestamp `created_at`.
        # Prevents full table scans on `test_steps` during `get_test_run_details` and `prune_old_test_runs`.
        # Accelerates `get_test_runs` ordering by `created_at DESC`.
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_test_steps_run_id ON test_steps (run_id, step_number);"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_test_runs_created_at ON test_runs (created_at DESC);"
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


def is_safe_to_delete_run_dir(run_dir: str, allowed_parent: str) -> bool:
    """
    Defensively validates that run_dir is strictly a direct child of allowed_parent,
    and is never a protected directory (like chroma_db, model_cache, or root storage).
    """
    try:
        norm_run_dir = os.path.abspath(run_dir)
        norm_parent = os.path.abspath(allowed_parent)

        # Must be within allowed_parent and cannot be allowed_parent itself
        if not norm_run_dir.startswith(norm_parent) or norm_run_dir == norm_parent:
            return False

        # Must be a direct child (e.g. storage/runs/run_xxx)
        if os.path.dirname(norm_run_dir) != norm_parent:
            return False

        basename = os.path.basename(norm_run_dir).lower()
        # Protect system, root, and AI memory directories
        protected_basenames = {"chroma_db", "model_cache", "reports", "storage", ".", ".."}
        if basename in protected_basenames:
            return False

        # Strictly verify ChromaDB path is NEVER touched
        chroma_abs = os.path.abspath(settings.CHROMA_DB_PATH)
        if norm_run_dir == chroma_abs or norm_run_dir.startswith(chroma_abs):
            return False

        return True
    except Exception:
        return False


def get_dir_size_bytes(path: str) -> int:
    """Calculates total size of a directory in bytes."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size_bytes(entry.path)
    except Exception:
        pass
    return total


async def prune_old_test_runs(keep_latest: int = 3, storage_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Automatic Retention Policy:
    1. Retains only the latest `keep_latest` audit runs in SQLite DB (test_runs.db) and physical filesystem (./storage/runs).
    2. Identifies all completed runs ordered by creation date descending.
    3. Deletes physical artifact directories/files (screenshots, .docx, .json) of older runs.
    4. Removes pruned run metadata from SQLite test_runs.db (test_steps and test_runs).
    5. STRICTLY PRESERVES ChromaDB collections, vector embeddings, and AI knowledge store.
    """
    effective_storage_dir = storage_dir or settings.STORAGE_DIR
    runs_dir = os.path.abspath(os.path.join(effective_storage_dir, "runs"))
    reports_dir = os.path.abspath(os.path.join(effective_storage_dir, "reports"))

    deleted_runs: List[str] = []
    kept_runs: List[str] = []
    freed_bytes: int = 0

    # Step 1: Query DB to find runs to keep vs prune
    async with aiosqlite.connect(settings.DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT run_id, created_at FROM test_runs ORDER BY created_at DESC") as cursor:
            rows = await cursor.fetchall()

        all_db_runs = [row["run_id"] for row in rows]
        kept_runs = all_db_runs[:keep_latest]
        runs_to_prune_db = all_db_runs[keep_latest:]

        if runs_to_prune_db:
            logger.info(f"🧹 Pruning {len(runs_to_prune_db)} old test runs from SQLite DB (keeping latest {len(kept_runs)}).")
            # Delete in chunks to respect SQLite variable limits
            chunk_size = 50
            for i in range(0, len(runs_to_prune_db), chunk_size):
                chunk = runs_to_prune_db[i:i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                await db.execute(f"DELETE FROM test_steps WHERE run_id IN ({placeholders})", chunk)
                await db.execute(f"DELETE FROM test_runs WHERE run_id IN ({placeholders})", chunk)
            await db.commit()
            deleted_runs.extend(runs_to_prune_db)

    # Step 2: Physical Filesystem Cleanup in ./storage/runs/
    if os.path.exists(runs_dir):
        kept_set = set(kept_runs)

        # Gather all run directory entries
        physical_entries = []
        try:
            for entry in os.scandir(runs_dir):
                if entry.is_dir(follow_symlinks=False):
                    physical_entries.append(entry)
        except Exception as scan_err:
            logger.warning(f"⚠️ Error scanning runs_dir {runs_dir}: {scan_err}")

        # If DB had fewer than keep_latest runs, use directory mtime to keep the latest ones
        if len(kept_set) < keep_latest and physical_entries:
            sorted_entries = sorted(physical_entries, key=lambda e: e.stat().st_mtime, reverse=True)
            for entry in sorted_entries[:keep_latest]:
                kept_set.add(entry.name)

        for entry in physical_entries:
            run_name = entry.name
            if run_name in kept_set:
                continue

            target_path = entry.path
            if is_safe_to_delete_run_dir(target_path, runs_dir):
                size = get_dir_size_bytes(target_path)
                try:
                    shutil.rmtree(target_path, ignore_errors=True)
                    freed_bytes += size
                    if run_name not in deleted_runs:
                        deleted_runs.append(run_name)
                    logger.info(f"🗑️ Cleaned up old audit artifact directory: {run_name} ({size / (1024*1024):.2f} MB freed)")
                except Exception as del_err:
                    logger.warning(f"⚠️ Failed to remove run dir {target_path}: {del_err}")

        # Also clean up old loose master batch reports if their runs are not kept
        try:
            for file_entry in os.scandir(runs_dir):
                if file_entry.is_file(follow_symlinks=False) and file_entry.name.startswith("batch_"):
                    if not any(k in file_entry.name for k in kept_set):
                        size = file_entry.stat().st_size
                        os.remove(file_entry.path)
                        freed_bytes += size
        except Exception:
            pass

    # Step 3: Clean up loose docx reports in ./storage/reports/
    if os.path.exists(reports_dir):
        try:
            for entry in os.scandir(reports_dir):
                if entry.is_file(follow_symlinks=False) and entry.name.endswith(".docx"):
                    run_prefix = entry.name.replace("_audit_report.docx", "")
                    if run_prefix not in set(kept_runs):
                        size = entry.stat().st_size
                        os.remove(entry.path)
                        freed_bytes += size
                        logger.info(f"🗑️ Removed old report file: {entry.name}")
        except Exception as rep_err:
            logger.warning(f"⚠️ Error cleaning reports_dir: {rep_err}")

    # Step 4: Verification that AI Vector Memory is intact
    chroma_path = os.path.abspath(settings.CHROMA_DB_PATH)
    if os.path.exists(chroma_path):
        logger.info(f"🧠 [AI Retention Guard] Vector Memory at {chroma_path} is strictly preserved intact.")

    total_mb_freed = round(freed_bytes / (1024 * 1024), 2)
    logger.info(f"✅ Retention policy complete: {len(deleted_runs)} runs pruned, {total_mb_freed} MB disk space reclaimed. Retained runs: {kept_runs}")

    return {
        "status": "success",
        "kept_runs": kept_runs,
        "deleted_runs_count": len(deleted_runs),
        "deleted_runs": deleted_runs,
        "freed_mb": total_mb_freed,
        "vector_memory_preserved": True,
    }


def prune_old_test_runs_sync(keep_latest: int = 3, storage_dir: Optional[str] = None) -> Dict[str, Any]:
    """Synchronous version of prune_old_test_runs using standard sqlite3."""
    effective_storage_dir = storage_dir or settings.STORAGE_DIR
    runs_dir = os.path.abspath(os.path.join(effective_storage_dir, "runs"))
    reports_dir = os.path.abspath(os.path.join(effective_storage_dir, "reports"))

    deleted_runs: List[str] = []
    kept_runs: List[str] = []
    freed_bytes: int = 0

    if os.path.exists(settings.DB_PATH):
        try:
            with sqlite3.connect(settings.DB_PATH) as con:
                con.row_factory = sqlite3.Row
                cur = con.cursor()
                cur.execute("SELECT run_id, created_at FROM test_runs ORDER BY created_at DESC")
                rows = cur.fetchall()
                all_db_runs = [row["run_id"] for row in rows]
                kept_runs = all_db_runs[:keep_latest]
                runs_to_prune_db = all_db_runs[keep_latest:]

                if runs_to_prune_db:
                    chunk_size = 50
                    for i in range(0, len(runs_to_prune_db), chunk_size):
                        chunk = runs_to_prune_db[i:i + chunk_size]
                        placeholders = ",".join("?" for _ in chunk)
                        cur.execute(f"DELETE FROM test_steps WHERE run_id IN ({placeholders})", chunk)
                        cur.execute(f"DELETE FROM test_runs WHERE run_id IN ({placeholders})", chunk)
                    con.commit()
                    deleted_runs.extend(runs_to_prune_db)
        except Exception as db_err:
            logger.warning(f"⚠️ Sync DB prune warning: {db_err}")

    # Filesystem cleanup
    if os.path.exists(runs_dir):
        kept_set = set(kept_runs)
        physical_entries = []
        try:
            for entry in os.scandir(runs_dir):
                if entry.is_dir(follow_symlinks=False):
                    physical_entries.append(entry)
        except Exception:
            pass

        if len(kept_set) < keep_latest and physical_entries:
            sorted_entries = sorted(physical_entries, key=lambda e: e.stat().st_mtime, reverse=True)
            for entry in sorted_entries[:keep_latest]:
                kept_set.add(entry.name)

        for entry in physical_entries:
            run_name = entry.name
            if run_name in kept_set:
                continue

            target_path = entry.path
            if is_safe_to_delete_run_dir(target_path, runs_dir):
                size = get_dir_size_bytes(target_path)
                try:
                    shutil.rmtree(target_path, ignore_errors=True)
                    freed_bytes += size
                    if run_name not in deleted_runs:
                        deleted_runs.append(run_name)
                except Exception:
                    pass

        try:
            for file_entry in os.scandir(runs_dir):
                if file_entry.is_file(follow_symlinks=False) and file_entry.name.startswith("batch_"):
                    if not any(k in file_entry.name for k in kept_set):
                        size = file_entry.stat().st_size
                        os.remove(file_entry.path)
                        freed_bytes += size
        except Exception:
            pass

    if os.path.exists(reports_dir):
        try:
            for entry in os.scandir(reports_dir):
                if entry.is_file(follow_symlinks=False) and entry.name.endswith(".docx"):
                    run_prefix = entry.name.replace("_audit_report.docx", "")
                    if run_prefix not in set(kept_runs):
                        size = entry.stat().st_size
                        os.remove(entry.path)
                        freed_bytes += size
        except Exception:
            pass

    total_mb_freed = round(freed_bytes / (1024 * 1024), 2)
    return {
        "status": "success",
        "kept_runs": kept_runs,
        "deleted_runs_count": len(deleted_runs),
        "deleted_runs": deleted_runs,
        "freed_mb": total_mb_freed,
        "vector_memory_preserved": True,
    }

