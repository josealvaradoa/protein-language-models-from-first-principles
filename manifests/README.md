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

The Week 1 Task 6 group-aware pre-repair manifest is:

```text
manifests/week_01/task_06_group_aware_pre_repair.tsv
```

It uses the same five-column public schema and contains the same 557,718
eligible records. Its SHA-256 is
`f6ee25d078ca6df864e8e30bba848025d5b45810ed60605f09255c767c77f71a`.
The adjacent sidecar records the same checksum.

The manifest keeps complete UniRef50 groups, exact-duplicate-connected units,
and represented ProteinGym-reserved groups intact. It is preserved as
pre-repair evidence, but it failed the frozen residue-balance tolerance and
must not be used for Task 7 or model training.
