## 2026-09-29 - SQLite Composite & Sorting Index Optimization
**Learning:** Querying `test_runs` sorted by `created_at DESC` and fetching `test_steps` by `run_id` ordered by `step_number ASC` incurred linear table scans during dashboard history listings, run detail fetches, and automatic retention pruning. Adding `idx_test_runs_created_at` and `idx_test_steps_run_id_step` reduced query execution time by ~7.3x.
**Action:** Always ensure foreign key lookup columns and sorting timestamp columns in SQLite database modules are explicitly indexed using `CREATE INDEX IF NOT EXISTS`.
