# Week 1 Task 7 Diagnostic Similarity Audit

This report compares the frozen Task 5 random diagnostic with the immutable Task 6 failed-balance pre-repair candidate.

It records detected sequence similarity under the pinned MMseqs2 procedure. It does not repair or select a split, authorize Task 8, authorize model use, or prove the absence of homology.

| Strategy | Held-out partition | Records (share) | Residues (share) | Prohibited queries | Queries audited | Rate | Prohibited pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | validation | 2 (5.000000%) | 200 (5.000000%) | 1 | 2 | 50.000000% | 1 |
| random | test | 2 (5.000000%) | 200 (5.000000%) | 1 | 2 | 50.000000% | 1 |
| random | overall held-out | 4 (10.000000%) | 400 (10.000000%) | 2 | 4 | 50.000000% | 2 |
| group_aware | validation | 2 (5.000000%) | 200 (5.000000%) | 1 | 2 | 50.000000% | 1 |
| group_aware | test | 2 (5.000000%) | 200 (5.000000%) | 1 | 2 | 50.000000% | 1 |
| group_aware | overall held-out | 4 (10.000000%) | 400 (10.000000%) | 2 | 4 | 50.000000% | 2 |

## Frozen membership structure

| Strategy | Exact-hash crossings | UniRef50 crossings | Retained records | Retained residues | Excluded records | Largest group or unit (records) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | 1 | 1 | 100 | 10000 | 0 | 5 |
| group_aware | 1 | 1 | 100 | 10000 | 0 | 5 |

## Held-out query status categories

| Strategy | Partition | Category | Queries |
| --- | --- | --- | ---: |
| random | validation | prohibited | 1 |
| random | validation | identity_ge_50_below_bidirectional_coverage | 0 |
| random | validation | identity_40_to_under_50 | 0 |
| random | validation | identity_30_to_under_40 | 0 |
| random | validation | identity_under_30_or_no_residual_hit | 1 |
| random | test | prohibited | 1 |
| random | test | identity_ge_50_below_bidirectional_coverage | 0 |
| random | test | identity_40_to_under_50 | 0 |
| random | test | identity_30_to_under_40 | 0 |
| random | test | identity_under_30_or_no_residual_hit | 1 |
| group_aware | validation | prohibited | 1 |
| group_aware | validation | identity_ge_50_below_bidirectional_coverage | 0 |
| group_aware | validation | identity_40_to_under_50 | 0 |
| group_aware | validation | identity_30_to_under_40 | 0 |
| group_aware | validation | identity_under_30_or_no_residual_hit | 1 |
| group_aware | test | prohibited | 1 |
| group_aware | test | identity_ge_50_below_bidirectional_coverage | 0 |
| group_aware | test | identity_40_to_under_50 | 0 |
| group_aware | test | identity_30_to_under_40 | 0 |
| group_aware | test | identity_under_30_or_no_residual_hit | 1 |

## Closest residual-match categories

These categories describe the selected closest residual row itself. A query can be prohibited by a different returned pair.

| Strategy | Partition | Category | Queries |
| --- | --- | --- | ---: |
| random | validation | closest_match_prohibited | 1 |
| random | validation | identity_ge_50_below_bidirectional_coverage | 0 |
| random | validation | identity_40_to_under_50 | 0 |
| random | validation | identity_30_to_under_40 | 0 |
| random | validation | identity_under_30_or_no_residual_hit | 1 |
| random | test | closest_match_prohibited | 1 |
| random | test | identity_ge_50_below_bidirectional_coverage | 0 |
| random | test | identity_40_to_under_50 | 0 |
| random | test | identity_30_to_under_40 | 0 |
| random | test | identity_under_30_or_no_residual_hit | 1 |
| group_aware | validation | closest_match_prohibited | 1 |
| group_aware | validation | identity_ge_50_below_bidirectional_coverage | 0 |
| group_aware | validation | identity_40_to_under_50 | 0 |
| group_aware | validation | identity_30_to_under_40 | 0 |
| group_aware | validation | identity_under_30_or_no_residual_hit | 1 |
| group_aware | test | closest_match_prohibited | 1 |
| group_aware | test | identity_ge_50_below_bidirectional_coverage | 0 |
| group_aware | test | identity_40_to_under_50 | 0 |
| group_aware | test | identity_30_to_under_40 | 0 |
| group_aware | test | identity_under_30_or_no_residual_hit | 1 |

## Interpretation boundary

The candidate held-out partitions contain shorter proteins on average than its training partition. The comparison therefore cannot isolate grouping as the only cause of a difference.

Even zero detected prohibited matches would not cure the Task 6 balance failure. Jose must review this diagnostic before any later adjustment is considered.
