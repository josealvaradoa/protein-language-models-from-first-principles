# Week 4 foundations reproduction contract

The frozen machine-readable contract is
[`experiments/week_04/foundations_reproduction_v1.toml`](../experiments/week_04/foundations_reproduction_v1.toml).
It is the source of truth for the Week 4 foundations reproduction framework.

## Purpose

Week 4 will reproduce the closed Week 1 through Week 3 evidence through one
local framework without changing the historical data, configurations, models,
seeds, splits, or metrics. This contract does not add a model or an execution
path. It freezes the requirements that later implementation must satisfy.

The required evidence is the complete closed Week 1 A-004 aggregate audit, the
complete Week 2 final bigram evaluation report, and the Week 3 C10, C20, and
E64 three-seed comparison. The public repository remains aggregate-only.

## Identity and evidence rules

The contract pins the exact bytes of every tracked historical configuration and
report it references. It also fixes configuration checksums, manifest
memberships, dataset and token counts, parameter counts, seeds, prediction
budgets, model and collection names, Week 1 aggregate counts, sealed-test
denial, and zero network requests.

Reevaluation compares each primary cross-entropy value against historical
evidence with an absolute tolerance of `0.000001`. It also requires exact
correct-prediction counts. Retraining compares each model and seed with an
absolute cross-entropy tolerance of `0.0001`. It must preserve every frozen
ranking and every material cross-entropy gap of at least `0.001`. Accuracy,
runtime, memory, and throughput may be reported, but are not independent pass
gates.

Missing, duplicate, unexpected, NaN, or infinite metrics fail the comparison.
No statistical-significance or biological-function claim is permitted.

## Stages and gates

`Verify` checks the frozen contract, source bytes, local identities, named clean
branch state, sealed-test denial, and offline operation. `Reevaluate` scores
frozen artifacts without modifying them. Week 1 reevaluation is the closed
read-only split audit, not a model evaluation. `Retrain` creates only new,
ignored candidates and checkpoints, then evaluates them. Week 1 has no
retraining stage. `Compare` applies the frozen rules automatically.

A failed verification locks that week's reevaluation and retraining. A failed
reevaluation locks only the dependent retraining path. A failed retraining
fails the milestone. Independent paths can continue, but the final comparison
requires all listed dependencies. Failures and logs are preserved. Historical
evidence is never overwritten, and there is no manual dashboard override.

## Operator and storage boundary

Jose is the sole operator. The later coordinator must run locally and offline.
It must retain the existing Week 1 through Week 3 command names, flags,
preflight behavior, output protections, and sealed-data restrictions. The
browser may submit only a fixed job identifier. It may not choose commands,
paths, seeds, configurations, or arguments.

Each attempt will use one ignored, append-only bundle at
`runs/week_04/<run-id>/`:

```text
contract.toml
run.json
log.txt
metrics.json
comparison.json
provenance.json
```

JSON writes will use atomic temporary-file replacement. Completed bundles are
immutable, retries use a new run ID, and cancellation or runner restart is a
terminal state. No SQLite, cloud service, queue, remote worker, FastAPI
migration, or multi-user execution is part of this contract.

## Current status

This task freezes the contract and characterizes current safe CLI boundaries.
It does not implement adapters, a coordinator, storage, comparison code, or
dashboard launches. The dashboard's five reproduction jobs remain blocked until
those later components are implemented and synthetically verified.
