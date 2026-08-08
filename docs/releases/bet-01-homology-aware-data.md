# Bet 01: Homology-Aware Protein Data

## Result

This release compares two immutable assignments of the same 557,718 eligible
Swiss-Prot proteins:

1. independent assignment by a frozen accession-hash rule; and
2. assignment that keeps UniRef50 groups, exact duplicates, and represented
   ProteinGym-reserved groups together.

Under the same fixed-budget MMseqs2 procedure, the UniRef50-group assignment
had lower detected strong training-to-held-out overlap in both held-out
partitions.

| Assignment | Partition | Detected strong-overlap rate |
|---|---|---:|
| Independent random | Validation | 87.404979% |
| Independent random | Test | 87.559128% |
| UniRef50 group | Validation | 35.919440% |
| UniRef50 group | Test | 37.399853% |

A prohibited query had at least one detected training match at 50 percent
identity and 80 percent coverage of both proteins. These rates are lower
bounds under the finite search budget.

## What The Result Means

The exploratory A-004 expectation was descriptively supported. Group-aware
assignment reduced detected strong overlap relative to independent random
assignment under the pinned procedure.

The original Week 1 hypothesis was not supported. The UniRef50-group candidate
failed the frozen residue-balance tolerance and retained detected overlap in
about 36 to 37 percent of held-out queries. It is not a balanced,
zero-detected-leakage, training-ready split.

Neither assignment is approved for model training.

## Included Artifacts

- frozen Week 1 experiment configurations;
- aggregate corpus, eligibility, assignment, and similarity reports;
- label-free public diagnostic manifests and checksum sidecars;
- strict 24-token protein tokenization;
- deterministic synthetic dataset, collation, and loading utilities;
- a fixture-only PyTorch batch-to-gradient smoke notebook;
- unit, invariant, characterization, and synthetic integration tests; and
- source, license, and third-party provenance.

Raw protein sequences, query-level similarity rows, MMseqs2 databases, model
weights, private evidence, and external caches remain outside Git.

## Reproduction Status

A separate clean-environment reproduction was not performed before this
release. The release relies on the original frozen runs, repeated-pass checks
for Tasks 2, 4, 5, and 6, committed checksums, characterization tests, and the
preserved A-004 evidence receipt.

The offline commands and source requirements remain documented in
[`scripts/README.md`](../../scripts/README.md) and
[`DATA_SOURCES.md`](../../DATA_SOURCES.md). Skipping the separate reproduction
is a release limitation, not evidence that a new reproduction passed.

## Limitations

- The finite MMseqs2 search budget makes every prohibited-query count a lower
  bound.
- No detected match does not prove that two proteins are unrelated.
- The staged 100,000-candidate search covered only queries whose complete rows
  changed between the 1,000 and 10,000 caps.
- The two assignments have different held-out length distributions, so the
  comparison does not isolate grouping as the only cause of the observed
  difference.
- One frozen random assignment does not represent every possible random seed.
- No clean-environment reproduction was completed before release.
- No model was trained or evaluated, and no biological claim is made.

## Evidence

- [Task 2 corpus audit](../../reports/week_01/task_02_corpus_audit.md)
- [Task 4 eligible population](../../reports/week_01/task_04_eligible_records.md)
- [Task 5 independent-random diagnostic](../../reports/week_01/task_05_random_diagnostic.md)
- [Task 6 UniRef50-group candidate](../../reports/week_01/task_06_group_aware_pre_repair.md)
- [Task 7 A-004 fixed-budget comparison](../../reports/week_01/task_07_read_only_fixed_budget_audit_a004.md)
- [Public manifest documentation](../../manifests/README.md)
- [Data and tool provenance](../../DATA_SOURCES.md)

## Release Boundary

The public release tag is `bet-01-homology-aware-data`. The release closes the
Week 1 diagnostic artifact. It does not select a training dataset or authorize
Week 2 model training.
