# Protein Language Models From First Principles

An experiment-driven protein language-model series that begins by measuring
related-sequence contamination and freezing reproducible evaluation data and
PyTorch inputs. It then builds toward a small masked protein Transformer
evaluated against simple baselines and an external ESMC-300M reference model.

Scalar automatic differentiation is the private conceptual foundation for the
training mechanics used later in the series. The first public result addresses
a separate prerequisite: an auditable protein evaluation split.

The goal is understanding and reproducibility, not state-of-the-art protein
modeling. Generated sequences are model outputs only. They are not evidence of
biological function, safety, or therapeutic value.

## Status

Week 1 technical implementation and experimental evidence are complete. The
release identifier is `bet-01-homology-aware-data`. The independent-random
and UniRef50-group diagnostic assignments remain prohibited for model
training. No model result or biological claim is included in this release.

The main Week 1 finding is that the UniRef50-group assignment had much less
detected strong training-to-held-out overlap than the random assignment, but
it still failed the frozen residue-balance tolerance and retained substantial
detected overlap. See the
[Week 1 release notes](docs/releases/bet-01-homology-aware-data.md) and the
[aggregate Task 7 report](reports/week_01/task_07_read_only_fixed_budget_audit_a004.md).

## Experiment Progression

1. Homology-aware protein data and deterministic PyTorch inputs
2. Amino-acid unigram and bigram models
3. Context MLPs and residue embeddings
4. Training diagnostics and manual backpropagation
5. One-dimensional convolutional sequence models
6. Self-attention and causal Transformers
7. Masked protein Transformers
8. Frozen evaluation and ESMC-300M comparison

Each stage adds tests, experiment configuration, measured results, failure
analysis, and a public explanation. Later stages reuse earlier components
instead of replacing their history.

## Repository Layout

```text
src/protein_lm/     reusable project-owned Python package
tests/              correctness and regression tests
experiments/        versioned experiment configurations and entrypoints
notebooks/          original explanatory analyses and visualizations
reports/            aggregate metrics, limitations, and run provenance
manifests/          identifiers and checksums, never raw datasets
scripts/            reproducible setup and data-preparation commands
```

Raw data, third-party weights, checkpoints, caches, and large run outputs are
kept outside Git. See [DATA_SOURCES.md](DATA_SOURCES.md) for the external-data
policy and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

## Local Setup

Requirements:

- Python 3.12
- `uv`
- macOS or another platform supported by the selected PyTorch release

```bash
uv sync
uv run pytest
uv run python scripts/check_setup.py
```

Launch the notebook environment with:

```bash
uv run jupyter lab
```

The exact Python dependency resolution is recorded in `uv.lock`.

## Reproducibility Boundary

Public results must identify the code revision, configuration, seed, machine,
Python and PyTorch versions, compute backend, precision, runtime, and relevant
dataset or model revisions. A result is not promoted from a development run to
a published result without that provenance.

## Week 2 pre-run model-data boundary

Week 2 Candidate v1 is not yet created or approved for model use. The two
entrypoints below are safe preflights with no flags. Candidate preparation
validates the frozen configuration and four pinned local inputs; readiness
validation validates the frozen configuration and prints its planned checks.
Neither creates a candidate, report, dataset, model, or network request.

```bash
env PYTHONPATH=src uv run --locked --offline python scripts/prepare_week2_model_data.py
env PYTHONPATH=src uv run --locked --offline python scripts/validate_week2_model_data.py
```

Candidate creation and readiness evidence require separate explicit operator
flags. The candidate stays in ignored local storage until passing review.

After a passing readiness review, the next safe command is the no-flag public
promotion preflight:

```bash
env PYTHONPATH=src uv run --locked --offline python scripts/promote_week2_model_data.py
```

It verifies the approved candidate and readiness identities without creating
output or opening sealed membership. Jose alone runs the separately explicit
promotion flag after reviewing that output.

## Learning Sources

The project is inspired by Andrej Karpathy's *Neural Networks: Zero to Hero*
course. Applicable public model implementations are independently reconstructed
for protein sequences after private study. See
[LEARNING_SOURCES.md](LEARNING_SOURCES.md) for the source map and attribution
policy.

## AI Assistance

AI coding assistants may be used for brainstorming, implementation support,
test suggestions, debugging, and review. The project author reviews public
changes and remains responsible for technical decisions, experimental design,
results, and claims. AI-generated text or model output is never treated as
experimental evidence by itself.

## License

Project-owned source code and documentation are licensed under the MIT License.
External datasets, software, and model weights retain their own licenses and
terms.
