# Third-Party Notices

This repository does not bundle third-party datasets, model weights, tutorial
source code, or native binaries during setup. Those assets remain external and
retain their original licenses and terms.

## Learning Reference

The learning progression is inspired by Andrej Karpathy's *Neural Networks:
Zero to Hero* course and repository:

- Course: <https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ>
- Repository: <https://github.com/karpathy/nn-zero-to-hero>

Public project code is independently reconstructed for amino-acid sequences.
Attribution does not mean that tutorial source code is redistributed here.

## External Resources

The project uses the following external resources without committing their
data, weights, or binaries to this repository:

| Resource | Role | Repository policy |
|---|---|---|
| UniProtKB/Swiss-Prot 2026_02 | Curated protein sequence corpus | Download externally; record source and checksum |
| Release-matched UniRef50 2026_02 | Homology-aware grouping | Download externally; record source and checksum |
| [MMseqs2 `18-8cc5c`](https://github.com/soedinglab/MMseqs2) | Local sequence-similarity audit | Installed externally with Homebrew; binary not committed |
| ProteinGym v1.3 | Mutation-effect evaluation panel | Download externally; record source and checksum |
| Biohub ESMC-300M | Private synthetic local inference smoke | Cache weights externally; pin revision, checksum, and terms |

Before any resource is used in a public release, its exact revision, source,
license or terms, and checksum must be added here or to `DATA_SOURCES.md`.

## Biohub ESMC-300M Code And Weights

Task 11B uses the external `biohub/ESMC-300M` model revision
`a59b831785f907e96e6a246b1d142bfb76df31ee` only from a manually acquired
local cache. The local `model.safetensors` SHA-256 is
`0772d8fe64bb25e14fe6f23b80e3c9a7d215d0da3c6cba5bd356d7c0e0bb22cc`.
Neither the model files nor their cache location are committed.

- Biohub ESM code is pinned to revision
  `26b0bc2b771e3e419ea74f445a5f35cc094a1509`, project version `3.3.0`, and is
  licensed under the [MIT License](https://github.com/Biohub/esm/blob/26b0bc2b771e3e419ea74f445a5f35cc094a1509/LICENSE).
  Its frozen repository is <https://github.com/Biohub/esm.git>.
- ESMC model weights and their use are governed separately by the
  [Cambrian Open License](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement)
  and [acceptable-use policy](https://www.evolutionaryscale.ai/policies/acceptable-use-policy).
  Those terms are not replaced by the code's MIT license.

## Python Dependencies

Python packages are resolved in `uv.lock` and installed into a local virtual
environment. They are not vendored. Each package retains its upstream license.
