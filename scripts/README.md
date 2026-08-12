# Scripts

Scripts in this directory provide reproducible setup, data preparation,
training, evaluation, and audit entrypoints. They must fail clearly rather than
silently replacing missing data, models, or compute backends.

`prepare_week2_model_data.py` and `validate_week2_model_data.py` implement the
pre-run Week 2 Candidate v1 contract. With no flag, candidate preparation
parses the frozen configuration and verifies and parses the four pinned local
inputs; readiness validation parses the frozen configuration and prints its
planned hard gates. Neither creates output, loads a model, or runs MMseqs2.
Candidate creation requires `--execute-candidate`. Readiness evidence requires
`--execute-readiness-validation` after a candidate exists. Jose runs those
production commands, not automated implementation checks.

`run_esmc_300m_smoke.py` is the private Task 11B local ESMC-300M inference
smoke. It requires Jose to install the optional `esmc` group and update the
lockfile separately, provide the manually acquired pinned model directory, and
choose a new result JSON path. The command is explicitly offline: it validates
local config and tokenizer files, stream-hashes the local weights before model
load, and uses `local_files_only=True`. It accepts only the two public
synthetic fixtures. It does not download, train, score, evaluate, checkpoint,
or write embeddings or logits.

Run the MPS smoke first. CPU is available only as a separately explicit
diagnostic invocation and is never an automatic fallback:

```bash
env PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --locked --offline --group esmc python scripts/run_esmc_300m_smoke.py \
  --execute-esmc-smoke --model-dir /absolute/local/model-directory \
  --device mps --result-path /absolute/new-result.json
```

The script refuses to overwrite a result path and preserves a failed JSON
record after the explicit path has been accepted. Do not substitute an
existing private evidence path in documentation or source control.

`validate_acquisition.py` checks the frozen Week 1 source config, uses Git's
own ignore matcher to prove the raw destinations are excluded, and can verify
already acquired local files. It never makes network requests.

`run_corpus_audit.py` verifies the local UniProt and ProteinGym sources, runs
the aggregate-only Week 1 Task 2 audit, and writes deterministic JSON,
Markdown, and SHA-256 outputs under `reports/week_01/`.

The acceptance run recalculates the complete audit twice:

```bash
uv run --locked --offline python scripts/run_corpus_audit.py --repeat-check
```

The command requires committed execution code, makes no network requests, and
does not produce a split, leakage result, or model result.

`prepare_eligible_records.py` verifies the same pinned sources, applies the
approved Task 4 filters, maps UniRef50 groups, detects exact duplicates, and
reserves the full resolvable ProteinGym family universe. It writes the detailed
catalog and reserved-family list under ignored `data/processed/week_01/`, then
writes aggregate-only evidence under `reports/week_01/`.

The acceptance command always performs two complete passes:

```bash
uv run --locked --offline python scripts/prepare_eligible_records.py --repeat-check
```

The command stops on source, policy, Task 2 anchor, mapping, or repeated-run
drift. It makes no network requests and does not assign a split or train a
model.

`build_random_diagnostic.py` verifies the pinned Task 4 catalog and report,
then assigns each eligible primary accession independently with the frozen
SHA-256 boundaries. It writes an ignored local assignment history, a
label-free public manifest, aggregate reports, checksums, and a last-written
completion index.

The acceptance command performs two complete builds and requires identical
artifact evidence:

```bash
uv run --locked --offline python scripts/build_random_diagnostic.py --repeat-check
```

The command makes no network requests. Its output is diagnostic only and the
training guard rejects it as a selected model split.

`build_group_aware_candidate.py` verifies the same Task 4 inputs, keeps each
UniRef50 group and exact-duplicate-connected component intact, reserves the
approved ProteinGym families for test, and applies the frozen exact two-axis
allocator. It writes the ignored pre-repair assignment history, a label-free
public manifest, aggregate reports, checksums, and a last-written completion
index.

The acceptance command performs two complete builds:

```bash
uv run --locked --offline python scripts/build_group_aware_candidate.py --repeat-check
```

The command makes no network requests and never searches for another seed. A
balance failure is preserved as Task 6 evidence, remains prohibited for model
use, and does not authorize Task 7.

`run_diagnostic_similarity_audit.py` implements the separate A-003
authorization. It verifies the immutable Task 5 and Task 6 assignments,
materializes six ignored FASTA files, builds one training target database per
strategy, and runs the frozen W1-D09 enforcement and residual searches for
validation and test. Every track compares complete per-query rows at caps
1,000 and 10,000, escalating only changed queries to 100,000.

The command is intentionally explicit because it starts the long local corpus
audit:

```bash
env PYTHONPATH=src uv run --locked --offline \
  python scripts/run_diagnostic_similarity_audit.py \
  --execute-searches
```

The runner requires committed execution code and MMseqs2 `18-8cc5c`. It is
resumable, makes no network requests, keeps sequences and alignment rows under
ignored `data/`, and stops without a public report if any staged-cap result
fails to converge. It has no repair, reassignment, selection, training, or
alternate-threshold option. A completed report still records the Task 6
candidate as `failed_balance` and returns control to Jose for review.

For a human code review, read Task 7 in this order:

1. `diagnostic_similarity_audit.toml` freezes the authority and search policy.
2. `fixed_budget_audit/diagnostic_workflow.py` shows the complete audit from
   preflight to publication.
3. `similarity_manifests.py` and `similarity_fastas.py` prove the memberships
   and build the six query and training files.
4. `fixed_budget_audit/search.py` owns staged MMseqs2 search and resumability,
   while `fixed_budget_audit/execution.py` isolates machine safety.
5. `similarity_alignment.py`, `similarity_results.py`, and
   `similarity_evidence.py` follow the scientific data flow from strict rows to
   cap comparison to aggregate evidence.
6. `fixed_budget_audit/diagnostic_reporting.py` assembles, validates, renders, and
   publishes the diagnostic report's public JSON and Markdown.

The small `similarity_audit.py` and `similarity_inputs.py` files only preserve
older imports. Task 7 implementation modules import their concrete owners
directly.

`run_read_only_fixed_budget_audit.py` is the separate A-004 fixed-budget
entrypoint. With no arguments it validates the pinned configuration and prints
the complete stage plan without creating databases, searches, or evidence.
Explicit consent is required to start its local MMseqs2 work:

```bash
env PYTHONPATH=src uv run --locked --offline \
  python scripts/run_read_only_fixed_budget_audit.py \
  --execute-searches
```

The orchestration is owned by `fixed_budget_audit/workflow.py`. It imports the
historical A-003 evidence read-only and does not import or invoke
`fixed_budget_audit/diagnostic_workflow.py`.
