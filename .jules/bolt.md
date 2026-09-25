# Bolt's Journal - Critical Performance Learnings

## 2025-05-18 - Database Indexes on SQLite Test Logs
**Learning:** SQLite foreign key columns (like `test_steps.run_id`) are not automatically indexed. Without indexes, queries like `SELECT * FROM test_steps WHERE run_id = ?` and `DELETE FROM test_steps WHERE run_id IN (...)` force full table scans across all historical test steps. Adding composite index `(run_id, step_number)` and `created_at` index on `test_runs` reduces lookup overhead from O(N) full table scan to O(log N) B-tree lookup.
**Action:** Always create SQLite indexes on foreign key columns and frequently filtered/sorted fields in relational schemas.
