# A-004 read-only fixed-budget audit

The common result covers every query through cap 10000. The staged result adds cap 100000 only for queries whose complete rows changed between caps 1000 and 10000.

## Held-out results

| Strategy | Partition | Result | Prohibited queries | Denominator | Rate | Prohibited pairs |
|---|---|---|---:|---:|---:|---:|
| group_aware | test | `common_all_query_10000` | 1 | 1 | 100.000000% | 1 |
| group_aware | test | `staged_union_with_changed_query_100000` | 1 | 1 | 100.000000% | 1 |
| group_aware | validation | `common_all_query_10000` | 1 | 1 | 100.000000% | 1 |
| group_aware | validation | `staged_union_with_changed_query_100000` | 1 | 1 | 100.000000% | 1 |
| random | test | `common_all_query_10000` | 1 | 1 | 100.000000% | 1 |
| random | test | `staged_union_with_changed_query_100000` | 1 | 1 | 100.000000% | 1 |
| random | validation | `common_all_query_10000` | 1 | 1 | 100.000000% | 1 |
| random | validation | `staged_union_with_changed_query_100000` | 1 | 1 | 100.000000% | 1 |

## Per-cap evidence

| Strategy | Partition | Pass | Source | Cap | Query scope | Prohibited queries | Denominator | Rate | Prohibited pairs |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| group_aware | test | enforcement | `executed_a004` | 1000 | all_queries | 0 | 1 | 0.000000% | 0 |
| group_aware | test | enforcement | `executed_a004` | 10000 | all_queries | 1 | 1 | 100.000000% | 1 |
| group_aware | test | enforcement | `executed_a004` | 100000 | changed_queries_1000_to_10000 | 1 | 1 | 100.000000% | 1 |
| group_aware | test | residual | `executed_a004` | 1000 | all_queries | 0 | 1 | 0.000000% | 0 |
| group_aware | test | residual | `executed_a004` | 10000 | all_queries | 1 | 1 | 100.000000% | 1 |
| group_aware | test | residual | `executed_a004` | 100000 | changed_queries_1000_to_10000 | 1 | 1 | 100.000000% | 1 |
| group_aware | validation | enforcement | `executed_a004` | 1000 | all_queries | 0 | 1 | 0.000000% | 0 |
| group_aware | validation | enforcement | `executed_a004` | 10000 | all_queries | 1 | 1 | 100.000000% | 1 |
| group_aware | validation | enforcement | `executed_a004` | 100000 | changed_queries_1000_to_10000 | 1 | 1 | 100.000000% | 1 |
| group_aware | validation | residual | `executed_a004` | 1000 | all_queries | 0 | 1 | 0.000000% | 0 |
| group_aware | validation | residual | `executed_a004` | 10000 | all_queries | 1 | 1 | 100.000000% | 1 |
| group_aware | validation | residual | `executed_a004` | 100000 | changed_queries_1000_to_10000 | 1 | 1 | 100.000000% | 1 |
| random | test | enforcement | `executed_a004` | 1000 | all_queries | 0 | 1 | 0.000000% | 0 |
| random | test | enforcement | `executed_a004` | 10000 | all_queries | 1 | 1 | 100.000000% | 1 |
| random | test | enforcement | `executed_a004` | 100000 | changed_queries_1000_to_10000 | 1 | 1 | 100.000000% | 1 |
| random | test | residual | `executed_a004` | 1000 | all_queries | 0 | 1 | 0.000000% | 0 |
| random | test | residual | `executed_a004` | 10000 | all_queries | 1 | 1 | 100.000000% | 1 |
| random | test | residual | `executed_a004` | 100000 | changed_queries_1000_to_10000 | 1 | 1 | 100.000000% | 1 |
| random | validation | enforcement | `executed_a004` | 1000 | all_queries | 0 | 1 | 0.000000% | 0 |
| random | validation | enforcement | `executed_a004` | 10000 | all_queries | 1 | 1 | 100.000000% | 1 |
| random | validation | enforcement | `executed_a004` | 100000 | changed_queries_1000_to_10000 | 1 | 1 | 100.000000% | 1 |
| random | validation | residual | `imported_a003` | 1000 | all_queries | 0 | 1 | 0.000000% | 0 |
| random | validation | residual | `imported_a003` | 10000 | all_queries | 1 | 1 | 100.000000% | 1 |
| random | validation | residual | `imported_a003` | 100000 | changed_queries_1000_to_10000 | 1 | 1 | 100.000000% | 1 |

## Cap sensitivity

| Strategy | Partition | Pass | Transition | Compared queries | Complete row changes | Newly prohibited | No longer prohibited | Closest-category changes |
|---|---|---|---|---:|---:|---:|---:|---:|
| group_aware | test | enforcement | 1000 to 10000 | 1 | 1 | 1 | 0 | 1 |
| group_aware | test | enforcement | 10000 to 100000 | 1 | 0 | 0 | 0 | 0 |
| group_aware | test | residual | 1000 to 10000 | 1 | 1 | 1 | 0 | 1 |
| group_aware | test | residual | 10000 to 100000 | 1 | 0 | 0 | 0 | 0 |
| group_aware | validation | enforcement | 1000 to 10000 | 1 | 1 | 1 | 0 | 1 |
| group_aware | validation | enforcement | 10000 to 100000 | 1 | 0 | 0 | 0 | 0 |
| group_aware | validation | residual | 1000 to 10000 | 1 | 1 | 1 | 0 | 1 |
| group_aware | validation | residual | 10000 to 100000 | 1 | 0 | 0 | 0 | 0 |
| random | test | enforcement | 1000 to 10000 | 1 | 1 | 1 | 0 | 1 |
| random | test | enforcement | 10000 to 100000 | 1 | 0 | 0 | 0 | 0 |
| random | test | residual | 1000 to 10000 | 1 | 1 | 1 | 0 | 1 |
| random | test | residual | 10000 to 100000 | 1 | 0 | 0 | 0 | 0 |
| random | validation | enforcement | 1000 to 10000 | 1 | 1 | 1 | 0 | 1 |
| random | validation | enforcement | 10000 to 100000 | 1 | 0 | 0 | 0 | 0 |
| random | validation | residual | 1000 to 10000 | 1 | 1 | 1 | 0 | 1 |
| random | validation | residual | 10000 to 100000 | 1 | 0 | 0 | 0 | 0 |

## Staged additions

| Strategy | Partition | Additional pairs | Newly prohibited queries |
|---|---|---:|---:|
| group_aware | test | 0 | 0 |
| group_aware | validation | 0 | 0 |
| random | test | 0 | 0 |
| random | validation | 0 | 0 |

## Limitations

- Every prohibited-match numerator is a lower bound under the fixed search budget.
- The staged result adds 100000-cap evidence only for changed queries.
- Detected overlap is not an exhaustive biological relationship inventory.
- Length-distribution differences remain descriptive limitations.
