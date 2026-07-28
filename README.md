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

The repository is in its setup phase. No model result or biological claim has
been published yet.

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
