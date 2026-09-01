# Bet 03: MLP Protein Context

## Result

The frozen final model is a fixed-context MLP with context length 20,
embedding width 32, and hidden width 800 (C20/E32/H800). It has 530,293
parameters and was evaluated for 100,000,000 predictions with seeds 20260821,
20260822, and 20260823.

| Model | Parameters | Mean native CE | Sample SD |
|---|---:|---:|---:|
| C20/E32/H800 final MLP | 530,293 | 2.863666 | 0.000020 |
| C10/E64/H800 challenger | 530,965 | 2.870825 | 0.000008 |
| Week 2 family-aware neural bigram baseline | n/a | 2.898575 | n/a |

The C20/E32/H800 mean native cross-entropy was 2.863666, compared with
2.898575 for the frozen Week 2 family-aware neural bigram baseline. The
C10/E64/H800 challenger has a similar parameter count and the same flattened
input width as C20/E32/H800. Its mean native cross-entropy was 2.870825, a
0.007159 disadvantage relative to C20/E32/H800. This is a fixed-architecture,
fixed-budget comparison, not a significance or mechanism claim.

## Evidence And Reproduction

- [Aggregate evaluation report](../../reports/week_03/mlp_evaluation_v1.md)
- [Aggregate evaluation JSON](../../reports/week_03/mlp_evaluation_v1.json)
- [Aggregate evaluation SHA-256](../../reports/week_03/mlp_evaluation_v1.sha256)
- [Original public notebook](../../notebooks/week_03/week_03_mlp_protein_context.ipynb)
- [Frozen final-model configuration](../../experiments/week_03/mlp_context20_100m_continuation_v1.toml)
- [Frozen E64 challenger configuration](../../experiments/week_03/mlp_embedding64_100m_challenger_v1.toml)

The aggregate report records the complete evidence trail, including the 25M
capacity screen, the final C20 continuation, the E64 challenger, exploratory
learning-rate tails and one-epoch continuation, and the post-freeze position
diagnostic. The publisher verifies frozen source bytes before reading prose or
tensors. It publishes aggregate metrics, PCA coordinates, and seed-aware
residue cosine summaries only. Raw sequences, accessions, family IDs,
checkpoint tensors, and model weights are excluded.

## Publication And Defense

[*The MLP Beat the Bigram. It Still Couldn't See the Whole Protein.*](https://josealvaradoalvarenga.substack.com/p/the-mlp-beat-the-bigram-it-still)
was published August 28, 2026 and satisfies both the Week 2 and Week 3
publication obligations. Under workspace adjustment A-009, the Week 3 verbal
defense was waived. Jose's authored article is the written defense artifact;
no Week 3 verbal defense occurred.

## Boundaries And Limitations

All scores here are descriptive native-validation results. The sealed test
collection is not loaded by the publisher, validator, or notebook. These
results make no sealed-test, statistical-significance, biological-mechanism,
structure, or function claim.

The post-freeze position diagnostic does not train models and does not reopen
model selection. PCA panels and residue cosine summaries are descriptive only;
they are not biological explanations. Exploratory learning-rate tails and the
one-epoch continuation are retained as exploratory outcomes and do not reopen
model selection.

A causal protein language model is a statistical factorization over residue
sequences. It does not reproduce biological translation. Ribosomes translate
mRNA codons; they do not choose the next amino acid by conditioning on earlier
amino-acid residues. Remaining cross-entropy can reflect genuine conditional
variability and information absent from this fixed-context model, including
family, function, global fold, distant residues, and future context.
