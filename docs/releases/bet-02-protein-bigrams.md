# Bet 02: Protein Bigrams

## Result

This release asks how much adjacent-residue statistics learn beyond amino-acid
frequencies, and how the split policy changes the apparent validation result.
It evaluates six models: unigram, count-based smoothed bigram, and one-layer
neural bigram models for each of the random and family-aware training arms.

The neural models used the same frozen initial parameters, optimizer, batch
policy, final training step, and equal 100,000,000-pair training budget. Native
validation measures each arm on its own validation collection. Shared
validation measures both neural models on the same UniRef50-family-isolated
collection and is the head-to-head comparison.

| Neural model | Native CE | Shared CE | Optimism gap |
|---|---:|---:|---:|
| Random arm | 2.894124 | 2.901534 | 0.007410 |
| Family-aware arm | 2.898575 | 2.901497 | 0.002922 |

The random-minus-family optimism-gap difference was +0.004488, so the
prospectively specified hypothesis was supported. On shared validation, the
two neural models were effectively tied. This release makes no statistical
significance claim and does not identify grouping as the isolated cause of the
difference.

Within both arms, the count bigram slightly beat the neural bigram. Exact
counts can fit the bigram statistics directly; finite SGD optimization adds no
extra expressiveness to this one-step architecture.

The family-aware neural bigram is the prospectively frozen Week 3 lineage.

## Release Boundary

The public release tag identifier is `bet-02-protein-bigrams`.

The sealed test collection was untouched. All reported scores are validation
scores. Sampling is a synthetic, non-functional educational diagnostic and
makes no biological claim.

The combined Week 2 and Week 3 article, [*The MLP Beat the Bigram. It Still Couldn't See the Whole Protein.*](https://josealvaradoalvarenga.substack.com/p/the-mlp-beat-the-bigram-it-still),
was published August 28, 2026. It cross-links both releases and satisfies both
the Week 2 and Week 3 publication obligations.

## Evidence And Reproduction

- [Model-data readiness report](../../reports/week_02/model_data_readiness_v1.md)
- [Training-stream audit](../../reports/week_02/bigram_training_streams_v1.md)
- [Aggregate evaluation report](../../reports/week_02/bigram_evaluation_v1.md)
- [Synthetic sampling diagnostic](../../reports/week_02/bigram_sampling_v1.md)
- [Frozen model-data configuration](../../experiments/week_02/model_data_readiness.toml)
- [Frozen training configuration](../../experiments/week_02/bigram_training_v1.toml)
- [Frozen evaluation configuration](../../experiments/week_02/bigram_evaluation_v1.toml)
- [Frozen sampling configuration](../../experiments/week_02/bigram_sampling_v1.toml)
- [Public manifest documentation](../../manifests/README.md)
- [Experiment documentation](../../experiments/README.md)
- [Reproduction and operator documentation](../../scripts/README.md)
- [Data and tool provenance](../../DATA_SOURCES.md)

Raw sequences, membership rows, model candidates, checkpoints, and private
evidence remain outside Git.

## Limitations

- This is a comparison of complete data-arm policies, not grouping-only
  causality.
- Shared validation is not a sealed test.
- Adjacent-residue prediction does not establish biological understanding or
  function.
- Generated sequences are not functional proteins, safety evidence, or
  therapeutic evidence.
