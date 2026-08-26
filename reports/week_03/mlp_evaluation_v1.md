# Week 3: MLP Protein Context

## Result

The fixed-context C=20 MLP reached lower native-validation cross-entropy than the frozen Week 2 family-aware neural bigram at the same 100,000,000-prediction budget. This is validation-only evidence, with no sealed-test access and no significance claim.

| Model | Parameters | Mean CE | Sample SD | Mean accuracy |
|---|---:|---:|---:|---:|
| C=10, E=32, H=800 original control | 274293 | 2.870821 | 0.000005 | n/a |
| C=20, E=32, H=800 | 530293 | 2.863666 | 0.000020 | 0.111249 |
| C=10, E=64, H=800 challenger | 530965 | 2.870825 | 0.000008 | 0.108922 |
| Week 2 family-aware neural bigram | n/a | 2.898575 | n/a | 0.098853 |

The E=64 challenger had a matched flattened input width and a similar parameter count. Its mean CE was higher by 0.007159, exceeding the frozen 0.001 material threshold. This supports context allocation over parameter count within this architecture and contract only.

## Architecture

Each context token is looked up in the embedding table. The context embeddings are flattened, passed through one tanh hidden layer, and projected to next-token logits. A causal protein language model is a statistical factorization, not a model of ribosomal residue choice. Ribosomes read mRNA codons, not prior amino-acid residues.

## Experimental evidence trail

The report separates the original C10 learning curve, the C20 25M capacity screen and its 100M continuation, the final C20 versus E64 challenger, exploratory LR tails and one-epoch continuation, and the post-freeze position diagnostic. H=1600 appears only in the 25M screen.

## Runtimes and parameters

Observed staged CPU wall time includes harness, validation, and checkpoint overhead. It is not pure training time. The Week 2 public baseline does not provide a comparable training runtime and is not compared on runtime.

## Descriptive Position Diagnostic

The post-freeze position diagnostic did not train models and did not generate significance evidence. It places each native-validation target by the number of real preceding residues available to the context window.

| Available prior residues | Targets | C20 mean CE | E64 mean CE | Share of mean-NLL advantage |
|---|---:|---:|---:|---:|
| available_prior_residues_0_10 | 29095 | 2.568766 | 2.569888 | 0.004559 |
| available_prior_residues_11_19 | 23805 | 2.822816 | 2.831139 | 0.027662 |
| available_prior_residues_20_plus | 947595 | 2.873747 | 2.881062 | 0.967779 |

## Exploratory outcomes

LR tails and the one-epoch continuation are recorded as exploratory negative or descriptive outcomes. They do not reopen model selection.

## Learning curves

| Model | Predictions | Mean CE | Sample SD | Stage |
|---|---:|---:|---:|---|
| C10_E32_H800 | 1000000 | 2.886002 | 0.000448 | original_100m_run |
| C10_E32_H800 | 5000000 | 2.874654 | 0.000054 | original_100m_run |
| C10_E32_H800 | 10000000 | 2.873227 | 0.000041 | original_100m_run |
| C10_E32_H800 | 25000000 | 2.871545 | 0.000034 | original_100m_run |
| C10_E32_H800 | 50000000 | 2.871320 | 0.000026 | original_100m_run |
| C10_E32_H800 | 100000000 | 2.870821 | 0.000005 | original_100m_run |
| C20_E32_H800 | 1000000 | 2.884442 | 0.000490 | capacity_screen_25m |
| C20_E32_H800 | 5000000 | 2.868884 | 0.000195 | capacity_screen_25m |
| C20_E32_H800 | 10000000 | 2.866665 | 0.000215 | capacity_screen_25m |
| C20_E32_H800 | 25000000 | 2.864817 | 0.000192 | capacity_screen_25m |
| C20_E32_H800 | 50000000 | 2.864277 | 0.000024 | final_continuation_100m |
| C20_E32_H800 | 100000000 | 2.863666 | 0.000020 | final_continuation_100m |

## 25M capacity screen

| Arm | Parameters | Mean CE | Sample SD | Stage |
|---|---:|---:|---:|---|
| context_20 | 530293 | 2.864817 | 0.000192 | exploratory_25m_screen |
| embedding_64 | 530965 | 2.871389 | 0.000019 | exploratory_25m_screen |
| hidden_1600 | 547893 | 2.871764 | 0.000049 | exploratory_25m_screen |

H1600 ended at 25M and was not run to 100M.

## Observed staged CPU wall time

| Model | Mean seconds | Sample SD |
|---|---:|---:|
| C20_E32_H800 | 520.256980 | 8.790953 |
| C10_E64_H800 | 462.763755 | 5.472673 |

## Exploratory LR tails and one epoch

| Run | Position | Mean CE | Sample SD |
|---|---:|---:|---:|
| cosine_90m_100m_001 | 100000000 | 2.870883 | 0.000004 |
| staged_97m_003 | 100000000 | 2.870871 | 0.000006 |
| one_epoch | 124999936 | 2.870644 | 0.000008 |
| one_epoch | 149999872 | 2.870590 | 0.000013 |
| one_epoch | 171329454 | 2.870563 | 0.000005 |

Embedding PCA panels and within-seed residue cosine summaries are descriptive only. PCA is centered NumPy SVD run separately for each seed with canonicalized component signs. Its axes are not treated as directly comparable across seeds, and BOS is excluded from residue similarity summaries.

## Limits

Remaining cross-entropy can reflect genuine conditional variability and information absent from this model's fixed context, including family, function, global fold, distant residues, and future context. These results do not claim biological mechanism, structure learning, function, or test performance.
