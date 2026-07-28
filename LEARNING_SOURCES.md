# Learning Sources

This project adapts a neural-network learning progression to protein sequences.
Applicable public model implementations must be independently reconstructed,
tested, and explained in original language. The scalar-autograd code-along is
a private conceptual foundation, not a public project deliverable.

## Core Course

- Andrej Karpathy, *Neural Networks: Zero to Hero*:
  <https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ>
- Official course repository:
  <https://github.com/karpathy/nn-zero-to-hero>

## Course-Derived Neural-Network Concepts

| Course topic | Protein-project adaptation |
|---|---|
| Scalar autograd | Private foundation for understanding PyTorch training and debugging |
| Bigram language model | Adjacent amino-acid statistics and neural bigram model |
| MLP language model | Fixed residue context and learned amino-acid embeddings |
| Training diagnostics | Activation, gradient, initialization, and normalization studies |
| Manual backpropagation | Tensor-level gradients for the protein MLP |
| WaveNet-style model | Boundary-safe one-dimensional causal convolutions |
| GPT | Self-attention and a tiny causal protein Transformer |
| Tokenization | Explicit residue-level tokenizer and masked-token contract |

## Project-Specific Protein Evaluation Methods

The Week 1 data comparison is not derived from the neural-network course. It
uses protein-specific sources and methods:

- Swiss-Prot release and archive policy:
  <https://www.uniprot.org/help/synchronization>
- UniProt `2026_02` release notes:
  <https://www.uniprot.org/release-notes/2026-06-10-release>
- Swiss-Prot current-release distribution:
  <https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/>
- UniProtKB ID-mapping distribution:
  <https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/>
- UniProt license: <https://www.uniprot.org/help/license>
- UniRef clustering definitions: <https://www.uniprot.org/help/uniref>
- Official MMseqs2 repository and documentation:
  <https://github.com/soedinglab/MMseqs2>
- GraphPart homology-aware partitioning:
  <https://pubmed.ncbi.nlm.nih.gov/37850036/>
- Protein-language-model pretraining leakage analysis:
  <https://proceedings.mlr.press/v261/hermann24a.html>

## Additional Text

- Chip Huyen, *AI Engineering*: <https://www.oreilly.com/library/view/ai-engineering/9781098166298/>

Additional primary papers and domain references will be added to the relevant
experiment or report before each weekly release. A reference is not evidence
that a local implementation or result is correct; correctness must come from
tests and measured evaluation.
