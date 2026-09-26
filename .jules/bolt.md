## 2026-03-01 - Missing Indexes on SQLite Step and Run Queries
**Learning:** SQLite database queries filtering test_steps by `run_id` and sorting `test_runs` by `created_at DESC` caused full table scans as step count grew. Adding `idx_test_runs_created_at` on `test_runs (created_at DESC)` and `idx_test_steps_run_id_step` on `test_steps (run_id, step_number ASC)` achieved a ~2x query speedup.
**Action:** Always create appropriate indexes for foreign keys and frequent filter/order-by columns on SQLite database tables to avoid performance degradation as table size increases.
