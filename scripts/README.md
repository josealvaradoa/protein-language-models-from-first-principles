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

`promote_week2_model_data.py` is the final Week 2 model-data handoff. With no
flag it verifies the approved readiness identity, candidate revision and
inventory, the three public source manifests, and the sealed aggregate
commitment without opening sealed membership. `--execute-promotion` additionally
requires a clean committed revision and atomically creates `manifests/week_02`.
It copies only shared validation, random-arm, family-aware-arm, and the small
public registry. Jose runs this production command after reviewing the
preflight output.

`audit_week2_training_streams.py` freezes and audits the exact 100-million-pair
bigram stream for each promoted training arm. With no flag it reads only the
small public stream configuration and prints the plan. It does not load a
training collection or create output. Jose runs the explicit production audit
only from a clean committed revision:

```bash
env PYTHONPATH=src uv run --locked --offline \
  python scripts/audit_week2_training_streams.py --execute-stream-audit
```

The command uses only `random_training` and `family_aware_training` through the
approved loader. It writes new aggregate-only JSON, Markdown, and SHA-256
evidence under `reports/week_02/`, refusing to overwrite existing evidence.
It does not load shared validation or sealed membership, train a model, score,
evaluate, or run MMseqs2.

`train_week2_bigrams.py` first verifies the frozen training configuration, the
stream configuration, and the already-published aggregate stream commitment.
With no flags it prints the planned two-arm, one-pass CPU run without loading a
collection or creating output. A future operator run must name a new local
candidate explicitly:

```bash
env PYTHONPATH=src uv run --locked --offline \
  python scripts/train_week2_bigrams.py \
  --execute-candidate --candidate-id week2-bigram-v1-001
```

That command will create an ignored local candidate under
`data/processed/week_02/bigram_model_candidates/`. It fits only
`random_training` and `family_aware_training`, writes six logical models in
both JSON and Safetensors forms, and preserves an aggregate-only passed or
failed run record. It does not load validation or test collections, evaluate,
promote, or make network requests. No production model candidate evidence is
claimed here.

`validate_week2_bigram_candidate.py` is read-only and requires one existing
candidate identity. It never calls a model-data collection loader:

```bash
env PYTHONPATH=src uv run --locked --offline \
  python scripts/validate_week2_bigram_candidate.py \
  --candidate-id week2-bigram-v1-001
```

`publish_week2_bigram_sampling.py` is the separate synthetic-output diagnostic.
With no flags it validates the byte-pinned passed candidate and the two final
neural bigram artifacts without loading a dataset, validation/test collection,
or producing output. Jose runs the explicit command only after committing the
implementation:

```bash
env PYTHONPATH=src uv run --locked --offline \
  python scripts/publish_week2_bigram_sampling.py --execute-publication
```

It samples ten sequences per arm from BOS at temperature 1.0, without top-k or
top-p filtering, stopping at EOS or 128 residues. Seeds are derived from the
frozen base seed and arm namespace. It atomically creates the deterministic
JSON, Markdown, and SHA-256 files under `reports/week_02/`, refusing existing
outputs. The output is synthetic, non-functional educational material only and
is never used for selection or biological claims.

`validate_week2_bigram_sampling.py` is the separate read-only validator. It
checks the output schema, provenance, checksums, deterministic Markdown, and
independently regenerates all twenty samples from the pinned neural artifacts.

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

`evaluate_week2_bigrams.py` is the Week 2 evaluation-only entrypoint. With no
flags it checks the byte-pinned evaluation contract and the candidate run-record
and registry hashes, plus the pinned promoted model-data registry. It does not
load a validation collection or write output.
The explicit command validates the complete frozen six-model candidate before
loading any validation data, then loads random native validation, family-aware
native validation, and shared validation once each. It creates an ignored,
immutable local evaluation candidate containing 12 metric records and never
loads the shared sealed test:

```bash
env PYTHONPATH=src uv run --locked --offline \
  python scripts/evaluate_week2_bigrams.py \
  --execute-evaluation --evaluation-id NEW_EVALUATION_ID
```

The command requires a clean committed revision and refuses an existing output
directory. It is evaluation only: it does not retrain, select, promote, or
publish results. `validate_week2_bigram_evaluation.py --evaluation-id ID`
checks an existing candidate, its artifact checksums, the pinned input model,
and all metric arithmetic without loading a collection.

`publish_week2_bigram_evaluation.py` is the separate aggregate-only public
report publisher. With no flags it validates the byte-pinned evaluation
candidate, its twelve aggregate records, and source provenance without loading
models or collections and without writing output. The explicit command requires
a clean committed revision, refuses any existing report artifact, and installs
the JSON, Markdown, and checksum sidecar together:

```bash
env PYTHONPATH=src uv run --locked --offline \
  python scripts/publish_week2_bigram_evaluation.py \
  --execute-publication
```

The report contains aggregate metrics, length buckets, fixed comparison
arithmetic, source identities, and collection-load accounting only. It excludes
sequences, accessions, family identifiers, and membership rows. It does not
load a model or collection, access the sealed test, retrain, select, or make a
network request. `validate_week2_bigram_public_report.py` is a separate
read-only check of the exact public inventory, checksums, source provenance,
derived arithmetic, and deterministic Markdown rendering.

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
