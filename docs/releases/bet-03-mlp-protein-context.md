# Bet 03: MLP Protein Context

## Pre-release Result

This pre-release package prepares the validation-only evidence for the future
`bet-03-mlp-protein-context` release. It does not claim that a tag exists.

The final fixed-context MLP uses context length 20, embedding width 32, hidden
width 800, 530,293 parameters, and three fixed seeds. It is compared with the
frozen Week 2 family-aware neural bigram on the same native validation
collection and the same 100,000,000-prediction budget. The aggregate report
will record the result whether the MLP wins or loses.

The public evidence package also retains negative and exploratory outcomes:
the 25M capacity screen, LR-tail experiments, and one-epoch continuations. The
E=64 challenger has a nearly matched parameter count and the same flattened
input width as C=20, so it distinguishes context allocation from parameter
count within this architecture and budget. The no-training position diagnostic
is descriptive only. It does not reopen model selection or make a significance
claim.

## Evidence And Reproduction

After Jose runs the operator-gated publisher at a clean committed revision, the
aggregate report will be at
[`reports/week_03/mlp_evaluation_v1.json`](../../reports/week_03/mlp_evaluation_v1.json)
and its deterministic Markdown and SHA-256 companion files will be adjacent.
The original public notebook at
[`notebooks/week_03/week_03_mlp_protein_context.ipynb`](../../notebooks/week_03/week_03_mlp_protein_context.ipynb)
reads only that JSON report.

The publisher verifies frozen source bytes before prose or tensor reads. It
publishes aggregate metrics, PCA coordinates, and seed-aware residue cosine
summaries only. Raw sequences, accessions, family IDs, checkpoint tensors, and
model weights are excluded.

## Boundaries And Limitations

All reported scores are native-validation scores. The sealed test collection is
not loaded by the publisher, validator, or notebook. There is no test claim,
significance claim, mechanism claim, structure claim, or function claim.

A causal protein language model is a statistical factorization. Ribosomes read
mRNA codons and do not choose the next amino acid from earlier amino acids.
Remaining cross-entropy can reflect genuine conditional variability and missing
family, function, global-fold, distant-residue, or future-context information.
Embedding plots are descriptive diagnostics, not biological explanations.
