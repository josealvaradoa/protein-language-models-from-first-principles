# Reports

Weekly reports record aggregate metrics, failures, limitations, and run
provenance. They must distinguish smoke, development, validation, and final
test results.

The Week 1 Task 2 corpus audit writes:

```text
reports/week_01/task_02_corpus_audit.json
reports/week_01/task_02_corpus_audit.md
reports/week_01/task_02_corpus_audit.sha256
```

The JSON is the canonical machine-readable aggregate. The Markdown is rendered
from the same object, and the sidecar records the JSON SHA-256. Runtime is
printed to the console rather than embedded because changing runtime would
break repeated-run checksum agreement.

The Week 1 Task 4 eligible-record preparation writes:

```text
reports/week_01/task_04_eligible_records.json
reports/week_01/task_04_eligible_records.md
reports/week_01/task_04_eligible_records.sha256
```

These files record aggregate filter, mapping, duplicate, group, and
ProteinGym-reservation evidence. They do not contain sequences, accessions,
family identifiers, split membership, labels, or model results. The canonical
Task 4 JSON SHA-256 is
`be791d35b39c4bf1337c121ed830ab01de1d9e73adee77e9eb5d24b0bf64bc5d`.

The Week 1 Task 5 random diagnostic writes:

```text
reports/week_01/task_05_random_diagnostic.json
reports/week_01/task_05_random_diagnostic.md
reports/week_01/task_05_random_diagnostic.sha256
reports/week_01/task_05_random_diagnostic.complete.json
```

The report records target and realized record and residue proportions,
two-pass equality, input and output checksums, and the model-use prohibition.
The completion index is written last and covers the byte size and SHA-256 of
every preceding public Task 5 artifact. The canonical Task 5 JSON SHA-256 is
`403d76db01632a875b2a1d549e08ef4ac979557bb77513f685606f9be3500c44`.

The Week 1 Task 6 group-aware pre-repair candidate writes:

```text
reports/week_01/task_06_group_aware_pre_repair.json
reports/week_01/task_06_group_aware_pre_repair.md
reports/week_01/task_06_group_aware_pre_repair.sha256
reports/week_01/task_06_group_aware_pre_repair.complete.json
```

The report records exact grouping, reservation, balance, integrity, state-zero,
and two-pass reproducibility evidence. The frozen candidate passed record
balance but failed residue balance in every partition, so Task 7 and model use
remain unauthorized. The canonical Task 6 JSON SHA-256 is
`9ebb8093962bf31cc010b160f229efb3bbd73ed6d688c54d277db4e6ee683ebf`.

`reports/week_01/task_07_diagnostic_continuation.md` records the later A-003
authorization to run only the frozen MMseqs2 audit against the immutable Task
5 and Task 6 assignments. It does not rewrite the Task 6 result or authorize
repair, selection, Task 8 membership use, or model training.

`reports/week_01/task_07_read_only_fixed_budget_audit_a004.md` records the
completed A-004 aggregate comparison. Under the same pinned procedure, the
UniRef50-group assignment had lower detected strong-overlap rates than the
independent-random assignment in both held-out partitions. The report also
preserves the failed residue-balance result, finite-search limitations, exact
evidence checksums, and the prohibition on using either assignment for model
training.
