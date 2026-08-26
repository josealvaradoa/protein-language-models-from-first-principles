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

Week 2 engineering, aggregate evidence, and verbal defense are complete. The
Week 2 release identifier is `bet-02-protein-bigrams`. Its public
documentation is prepared in the [Week 2 release notes](docs/releases/bet-02-protein-bigrams.md),
with the [aggregate evaluation report](reports/week_02/bigram_evaluation_v1.md)
and [synthetic sampling diagnostic](reports/week_02/bigram_sampling_v1.md).
Under the approved Week 2 and Week 3 publication exception, the combined
article remains pending and must be published before Week 4 begins.

Week 3 aggregate validation evidence is final. The frozen C20/E32/H800 MLP
was evaluated against the Week 2 family-aware neural bigram baseline and an
E64 challenger on native validation. See the
[Week 3 release notes](docs/releases/bet-03-mlp-protein-context.md) and
[aggregate evaluation report](reports/week_03/mlp_evaluation_v1.md).

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

## Week 3 position-availability diagnostic

The Week 3 position-availability diagnostic is a local, descriptive,
no-training check of where the already-observed C20 advantage occurs on the
family-aware native validation collection. It compares only matched final C20
and E64 seed checkpoints and assigns each residue target and EOS target once by
the number of preceding real residues. It records metrics by the frozen 0-10,
11-19, and 20+ bins without reopening model selection, producing significance
claims, biological conclusions, or a final report.
The frozen aggregate C20-versus-E64 comparison remains provenance only and
cannot be reopened by this diagnostic.

## Week 2 frozen model-data boundary

The Week 2 model-data boundary is frozen and promoted for the two matched
training arms. It uses separate random and family-aware native validation
collections, a shared UniRef50-family-isolated validation collection, and a
shared sealed test collection. The readiness evidence passed without running
MMseqs2 or making network requests. See the [readiness report](reports/week_02/model_data_readiness_v1.md),
[public manifest documentation](manifests/README.md), and [frozen readiness configuration](experiments/week_02/model_data_readiness.toml).

The sealed test remains inaccessible to Weeks 2 through 11. Raw source data,
membership rows, and generated local artifacts remain outside Git. The
[scripts guide](scripts/README.md) and [external-data policy](DATA_SOURCES.md)
document the reproduction boundary.

## Week 2 sampling diagnostic

The sampling diagnostic is a separate, synthetic-only educational artifact. It
uses only the two pinned final neural bigram models. It never loads a dataset,
validation or test collection, scores samples, or makes biological claims.
The no-flag command validates local byte-pinned inputs without writing output:

```bash
env PYTHONPATH=src uv run --locked --offline python scripts/publish_week2_bigram_sampling.py
```

After a code review and clean commit, Jose can run the explicit publisher:

```bash
env PYTHONPATH=src uv run --locked --offline \
  python scripts/publish_week2_bigram_sampling.py --execute-publication
```

It creates the deterministic JSON, Markdown, and checksum sidecar under
`reports/week_02/`, refusing an existing report. The separate validator checks
all report bytes and independently regenerates the saved samples:

```bash
env PYTHONPATH=src uv run --locked --offline python scripts/validate_week2_bigram_sampling.py
```

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
