# Learning Sources

This project adapts a neural-network learning progression to protein sequences.
The public implementation must be independently reconstructed, tested, and
explained in original language.

## Core Course

- Andrej Karpathy, *Neural Networks: Zero to Hero*:
  <https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ>
- Official course repository:
  <https://github.com/karpathy/nn-zero-to-hero>

## Project Mapping

| Course topic | Protein-project adaptation |
|---|---|
| Scalar autograd | Project-owned scalar computation graph and gradient checks |
| Bigram language model | Adjacent amino-acid statistics and neural bigram model |
| MLP language model | Fixed residue context and learned amino-acid embeddings |
| Training diagnostics | Activation, gradient, initialization, and normalization studies |
| Manual backpropagation | Tensor-level gradients for the protein MLP |
| WaveNet-style model | Boundary-safe one-dimensional causal convolutions |
| GPT | Self-attention and a tiny causal protein Transformer |
| Tokenization | Explicit residue-level tokenizer and masked-token contract |

## Additional Text

- Chip Huyen, *AI Engineering*: <https://www.oreilly.com/library/view/ai-engineering/9781098166298/>

Primary papers and domain references will be added to the relevant experiment
or report before each weekly release. A reference is not evidence that a local
implementation or result is correct; correctness must come from tests and
measured evaluation.
