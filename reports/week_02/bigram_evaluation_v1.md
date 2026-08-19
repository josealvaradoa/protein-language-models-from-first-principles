# Week 2 Bigram Evaluation v1

Aggregate-only evaluation evidence. It excludes sequences, accessions, family identifiers, and membership rows.

## Overall cross-entropy

| Arm | Model | Collection | Cross-entropy | Accuracy | Tokens | Proteins |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| random_training | unigram | random_native_validation | 2.901202 | 0.097073 | 1000151 | 2719 |
| random_training | count_bigram | random_native_validation | 2.886545 | 0.100113 | 1000151 | 2719 |
| random_training | neural_bigram | random_native_validation | 2.894124 | 0.100113 | 1000151 | 2719 |
| family_aware_training | unigram | family_aware_native_validation | 2.906191 | 0.096074 | 1000495 | 2645 |
| family_aware_training | count_bigram | family_aware_native_validation | 2.891525 | 0.098853 | 1000495 | 2645 |
| family_aware_training | neural_bigram | family_aware_native_validation | 2.898575 | 0.098853 | 1000495 | 2645 |
| random_training | unigram | shared_validation | 2.911016 | 0.107143 | 1000014 | 2926 |
| random_training | count_bigram | shared_validation | 2.895921 | 0.109295 | 1000014 | 2926 |
| random_training | neural_bigram | shared_validation | 2.901534 | 0.109295 | 1000014 | 2926 |
| family_aware_training | unigram | shared_validation | 2.911025 | 0.107143 | 1000014 | 2926 |
| family_aware_training | count_bigram | shared_validation | 2.895935 | 0.109295 | 1000014 | 2926 |
| family_aware_training | neural_bigram | shared_validation | 2.901497 | 0.109295 | 1000014 | 2926 |

## Hypothesis and optimism gaps

Random neural optimism gap: 0.007410
Family-aware neural optimism gap: 0.002922
Random minus family-aware gap: 0.004488
Prospective hypothesis supported: true

## Family-aware shared-validation length buckets

| Length bucket | Unigram CE | Count CE | Neural CE | Tokens | Proteins |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32-127 | 2.932618 | 2.892290 | 2.891351 | 40214 | 414 |
| 128-255 | 2.896144 | 2.872761 | 2.879477 | 140339 | 699 |
| 256-511 | 2.918122 | 2.904121 | 2.908399 | 532844 | 1445 |
| 512-1023 | 2.905051 | 2.894643 | 2.902594 | 212796 | 313 |
| 1024-2046 | 2.893539 | 2.886617 | 2.895901 | 73821 | 55 |

## Week 3 baseline

Family-aware neural bigram native CE / accuracy: 2.898575 / 0.098853
Family-aware neural bigram shared CE / accuracy: 2.901497 / 0.109295
Family-aware neural optimism gap: 0.002922

## Source and checksum provenance

Source evaluation: `data/processed/week_02/bigram_evaluation_candidates/week2-bigram-eval-v1-001`
Evaluation SHA-256: `b531e45391e4f7e8ae30a031fb0ef8dc14beaca37279d8fbcda6a226344b2bf8`
Run record SHA-256: `f5a34bf7e2ad8289000a773c6d0809c6d9c2090c016ab03632ee5e7e66440211`
Registry SHA-256: `c98d9e82b95e7380b363e472d67d33fd05485d3f00ea27bc72ef1b576200d09d`
Source evaluation code revision: `36c8cc9d964421e67bd16e6aa5ebcea14f76c80e`
Publication code revision: `9c8bde9f400264b713256fc76e8c86721a1be9d6`
Evaluation configuration SHA-256: `219e7a3bc06a6c227ed27b9b4b7e917083b537bd5ac5d11a7526ee8415c2d97c`
Evaluation runtime seconds: 11.541754

## Limitations

- This compares complete data-arm policies, not grouping-only causality.
- This report makes no statistical-significance claim.
- Adjacent-residue prediction is not biological understanding or function.
- Shared validation is not a sealed test. The sealed test remained inaccessible.
- This report makes no generated-sequence or function claim.
