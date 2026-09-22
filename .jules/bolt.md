## 2026-09-22 - Pre-computing CSS Selector Validation Sets
**Learning:** In hot DOM snapshot inspection loops, re-generating f-strings and iterating over disallowed class sets ($O(N)$) for every selector causes noticeable latency (~12.6µs per call). Pre-computing set lookups and tuple matches reduces runtime to $O(1)$ set checks (~1.15µs per call, ~10x speedup).
**Action:** When validating DOM selectors or interactive elements in hot loops, pre-compute lookup sets and tuples outside the function scope.
