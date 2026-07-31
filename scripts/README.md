# Scripts

Scripts in this directory provide reproducible setup, data preparation,
training, evaluation, and audit entrypoints. They must fail clearly rather than
silently replacing missing data, models, or compute backends.

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
  --execute-diagnostic-audit
```

The runner requires committed execution code and MMseqs2 `18-8cc5c`. It is
resumable, makes no network requests, keeps sequences and alignment rows under
ignored `data/`, and stops without a public report if any staged-cap result
fails to converge. It has no repair, reassignment, selection, training, or
alternate-threshold option. A completed report still records the Task 6
candidate as `failed_balance` and returns control to Jose for review.
