## 2025-05-18 - SQLite Database Indexing on Frequent Audit Queries
**Learning:** SQLite query execution on `test_steps` (filtered by `run_id` and ordered by `step_number`) and `test_runs` (ordered by `created_at DESC`) performs full table scans without explicit secondary indexes.
**Action:** Always verify SQLite database schema for missing indexes on foreign keys (`run_id`) and timestamp fields (`created_at`) used in `WHERE` and `ORDER BY` clauses to achieve ~10x-15x performance improvements on lookup queries.
