# Week 1 Task 7 Read-Only Fixed-Budget Audit

**Status:** Completed August 5, 2026

**Adjustment:** A-004

**Scope:** Aggregate diagnostic comparison only

## Result

The UniRef50-group assignment had a lower detected strong-overlap rate than
the independent-random assignment in both held-out partitions under the same
pinned procedure.

| Strategy | Partition | Prohibited queries | Held-out queries | Rate |
|---|---|---:|---:|---:|
| Independent random | Validation | 24,261 | 27,757 | 87.404979% |
| UniRef50 group | Validation | 10,255 | 28,550 | 35.919440% |
| Independent random | Test | 24,619 | 28,117 | 87.559128% |
| UniRef50 group | Test | 10,690 | 28,583 | 37.399853% |

The random rate was higher by 51.485539 percentage points for validation and
50.159275 percentage points for test.

These are all-query results through the common `10,000` candidate cap. A
prohibited query had at least one detected training match at 50 percent
identity and 80 percent coverage of both proteins.

## Staged Result

The staged union added `100,000`-cap evidence only for queries whose complete
rows changed from cap `1,000` to `10,000`. It is not an all-query `100,000`
result.

| Strategy | Partition | Pairs added at staged cap | Newly prohibited queries |
|---|---|---:|---:|
| Independent random | Validation | 17 | 0 |
| UniRef50 group | Validation | 0 | 0 |
| Independent random | Test | 4 | 0 |
| UniRef50 group | Test | 1 | 0 |

The staged evidence did not change any prohibited-query numerator or rate.
It did show that many complete residual row sets remained cap-sensitive:

| Strategy | Partition | Rows changed, 1,000 to 10,000 | Rows changed, 10,000 to staged 100,000 |
|---|---|---:|---:|
| Independent random | Validation | 20,939 | 10,538 |
| UniRef50 group | Validation | 20,049 | 9,416 |
| Independent random | Test | 21,273 | 10,554 |
| UniRef50 group | Test | 19,968 | 8,932 |

No query became newly prohibited or stopped being prohibited in these
residual transitions. Closest-match categories still changed for 173, 704,
186, and 629 queries, respectively, from `1,000` to `10,000`, and for 21, 39,
21, and 41 queries at the staged `100,000` cap.

## Balance Context

Both assignments cover the same eligible population, but their held-out
length distributions differ.

| Strategy | Partition | Record share | Residue share |
|---|---|---:|---:|
| Independent random | Training | 89.981675% | 89.944871% |
| Independent random | Validation | 4.976888% | 5.010008% |
| Independent random | Test | 5.041437% | 5.045121% |
| UniRef50 group | Training | 89.755934% | 92.551547% |
| UniRef50 group | Validation | 5.119075% | 3.747097% |
| UniRef50 group | Test | 5.124991% | 3.701356% |

The UniRef50-group assignment therefore still fails the approved residue
balance tolerance. The comparison describes the two full assignment policies.
It does not isolate grouping as the sole cause of the observed rate difference.

## Interpretation

The A-004 exploratory expectation is descriptively supported under the pinned
fixed-budget procedure. The UniRef50-group assignment showed substantially
less detected strong training-to-held-out overlap than the random assignment.

The original A-001 hypothesis remains unsupported. Grouping did not produce a
balanced, zero-detected-leakage, training-ready split. Detected prohibited
overlap remained about 36 to 37 percent in its held-out partitions, and its
residue balance remained outside tolerance.

Neither diagnostic assignment is approved for training:

```text
model_use=prohibited
task8_membership_use_authorized=false
diagnostic_assignments_unchanged=true
```

## Evidence Provenance

- A-004 fingerprint:
  `55a1042a476027e56178ad921ad026322119991b89e4828e3605901f79efe852`
- Code revision:
  `a939446e847f9c9260989f48be1a5ea24e742211`
- MMseqs2 version: `18-8cc5c`
- Hardware: Apple arm64 Mac, 10 logical CPUs, macOS 26.5.2
- Executed A-004 search-runtime sum: `15,646.068` seconds
- Imported A-003 search-runtime sum: `6,011.032` seconds
- Imported track: independent-random validation residual at all-query caps
  `1,000` and `10,000`, then `100,000` for changed queries
- Newly executed tracks: the other seven strategy, partition, and pass
  combinations

The ignored local receipt preserves exact commands, input identities, output
checksums, runtime, and the before-and-after assignment hashes. Those hashes
prove that Task 5 and Task 6 memberships did not change.

| Local evidence | SHA-256 |
|---|---|
| A-004 JSON report | `d6a19b12101a4246ba07ff44128ae381be61c92070b320b7fef1ff619bcd29ea` |
| A-004 Markdown report | `2456f606e376b206d994bec9259cb775d79c3b7e594e9c7a8f6715c8954777f3` |
| A-004 import receipt | `e8e26b15b32682f715edc2ae5a8355525949e83f5312669f0f3a22dfd02a3967` |
| A-004 completion marker | `83b6ab1cf7ffd0ec91e1dce3a73dc014ab20a4ab206539de03ff23a3831e63ee` |

The detailed local evidence remains Git-ignored because it contains query-level
records, alignments, databases, and machine-specific paths. This tracked record
contains aggregate findings only.

## Limitations

- Every prohibited-query numerator is a lower bound under a finite search
  budget.
- No detected match does not prove that no biological relationship exists.
- The staged result searches only escalated queries at `100,000` candidates.
- Complete residual rows did not converge within the approved caps.
- Length-distribution differences limit causal interpretation.
- One frozen random assignment does not describe every possible random seed.

Task 7 is closed as a read-only diagnostic result. It does not select, repair,
or authorize a model dataset.
