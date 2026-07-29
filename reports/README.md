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
