## 2026-10-01 - Fast Selector Validation via Pre-computed Sets and Suffix Tuples
**Learning:** In DOM snapshot processing, repeatedly calling `split(">")` and iterating over layout class sets inside inner loops introduces significant string allocation and iteration overhead on every element check.
**Action:** Pre-compute exact match sets (`DISALLOWED_LAYOUT_EXACT`, `BARE_NON_ACTIONABLE`) and tuple prefixes/suffixes (`ACTIONABLE_STARTS`, `DISALLOWED_LAYOUT_ENDINGS`) at module scope to replace linear loops and splits with fast O(1) set membership and `rpartition` checks.
