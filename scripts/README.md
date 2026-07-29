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
