## 2025-05-18 - SQLite Foreign Key & Sorting Indexes for Audit Runs
**Learning:** SQLite does not automatically create indexes for foreign key relationships (`test_steps.run_id`) or sorted list queries (`test_runs.created_at`). Without explicit indexes, queries like `get_test_run_details` and retention pruning (`prune_old_test_runs`) force full table scans over `test_steps`.
**Action:** Always create compound indexes (`run_id`, `step_number`) on child step tables and timestamp indexes (`created_at DESC`) on run tables in `init_db()`.
