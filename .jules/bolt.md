## 2026-09-23 - Indexing SQLite Test Run and Step Log Tables
**Learning:** In a test automation system with heavy log appending and polling, queries like `ORDER BY created_at DESC` on `test_runs` and `WHERE run_id = ? ORDER BY step_number ASC` on `test_steps` cause full table scans $O(N)$ as run history grows.
**Action:** Always create composite/ordering indexes (`idx_test_runs_created_at` and `idx_test_steps_run_step`) during `init_db()` using `CREATE INDEX IF NOT EXISTS` to maintain constant $O(\log N)$ retrieval speed for UI dashboard polling and retention pruning.
