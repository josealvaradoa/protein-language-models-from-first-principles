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
