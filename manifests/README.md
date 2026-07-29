# Manifests

Public manifests may contain stable identifiers, partition assignments, source
revisions, and checksums. Raw datasets, sealed evaluation labels, and external
model assets do not belong here.

The Week 1 Task 5 random diagnostic manifest is:

```text
manifests/week_01/task_05_random_diagnostic.tsv
```

Its five columns are primary accession, partition, sequence SHA-256,
biological length, and UniRef50 group. It contains 557,718 data rows and has
SHA-256
`bd0f4e376df0afa785bfef0153e072470c733cb8f9afe06c5bf973cb12a39c3e`.
The adjacent sidecar records the same checksum.

This manifest deliberately does not keep exact duplicates, UniRef50 groups, or
ProteinGym families together. It exists only as the comparison baseline and
must not be used for model training.
