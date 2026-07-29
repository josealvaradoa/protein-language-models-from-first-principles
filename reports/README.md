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
