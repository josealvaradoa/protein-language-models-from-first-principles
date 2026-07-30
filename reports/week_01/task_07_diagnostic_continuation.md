# Week 1 Task 7 Diagnostic Continuation

**Status:** Approved July 30, 2026

**Adjustment:** A-003

**Scope:** MMseqs2 audit only

## Why This Exists

The frozen Task 6 group-aware candidate preserved every structural invariant
and 100 percent of the eligible population, but failed residue balance:

| Partition | Record share | Residue share |
|---|---:|---:|
| Training | 89.755934% | 92.551547% |
| Validation | 5.119075% | 3.747097% |
| Test | 5.124991% | 3.701356% |

The original report remains unchanged with `candidate_status=failed_balance`,
`task6_gates_passed=false`, `task7_authorized=false`, and
`model_use=prohibited`.

Jose approved continuing with this exact assignment as a diagnostic
experiment. The purpose is to measure how the observed balance tradeoff
coexists with detected training-to-held-out sequence similarity, not to turn
the failed candidate into a passing split.

## Frozen Inputs

| Input | SHA-256 |
|---|---|
| Task 5 random public manifest | `bd0f4e376df0afa785bfef0153e072470c733cb8f9afe06c5bf973cb12a39c3e` |
| Task 5 random local assignment | `3b8687cdd5c7477b114fef7f425a68516ffd093f084aefee19d98272f397bfe6` |
| Task 6 candidate public manifest | `f6ee25d078ca6df864e8e30bba848025d5b45810ed60605f09255c767c77f71a` |
| Task 6 candidate local assignment | `7a89c5d432ff9b2a2a787cae9d5584754ec8a027df4118aaedcc2bd4c41221d1` |
| Task 6 repair state zero | `0ed690a0e048c2c0c735d1caf796487be47b3700eb90ad9fcafeec0a315ce176` |

Both strategies cover the same 557,718 eligible proteins and 197,375,585
residues.

## Authorized Next Step

Run the exact frozen W1-D09 MMseqs2 `18-8cc5c` procedure against:

1. the Task 5 random diagnostic; and
2. the Task 6 failed-balance pre-repair candidate.

Report aggregate baseline and pre-repair similarity results, the existing
balance evidence, checksums, commands, runtime, and hardware provenance. Then
stop and return to Jose.

## Still Prohibited

- allocator reruns, alternate seeds, or changed balance rules;
- assignment moves, merges, exclusions, or repair cycles;
- post-repair or selected manifests;
- Task 8 membership use;
- model training or evaluation;
- test-label or ProteinGym assay access; and
- any claim that zero detected matches would cure the balance failure or prove
  absence of homology.

No MMseqs2 result, model result, assay label, or test score had been observed
when this diagnostic continuation was approved.
