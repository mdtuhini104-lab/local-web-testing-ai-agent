## 2025-05-20 - SQLite Composite Index for Foreign Key Filtering & Ordering

**Learning:** In SQLite, querying child table entries filtered by parent foreign key (`run_id`) and sorted by step order (`step_number ASC`) results in a full table scan and temporary B-tree sorting unless a composite index `(run_id, step_number ASC)` is defined. With the composite index, SQLite performs a single covering index search (`SEARCH test_steps USING COVERING INDEX`) with zero temp B-tree sorting.

**Action:** Always create composite indexes on foreign key + sort column pairs for child step/log tables to keep details lookup and retention pruning deletions O(K) instead of O(N).
