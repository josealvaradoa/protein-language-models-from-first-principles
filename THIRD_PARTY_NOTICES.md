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

## Planned External Resources

The project expects to use the following external resources without committing
their data, weights, or binaries to this repository:

| Resource | Planned role | Repository policy |
|---|---|---|
| UniProtKB/Swiss-Prot 2026_02 | Curated protein sequence corpus | Download externally; record source and checksum |
| Release-matched UniRef50 2026_02 | Homology-aware grouping | Download externally; record source and checksum |
| [MMseqs2 `18-8cc5c`](https://github.com/soedinglab/MMseqs2) | Local sequence-similarity audit | Installed externally with Homebrew; binary not committed |
| ProteinGym v1.3 | Mutation-effect evaluation panel | Download externally; record source and checksum |
| Biohub ESMC-300M | Final external model baseline | Cache weights externally; pin revision and terms |

Before any resource is used in a public release, its exact revision, source,
license or terms, and checksum must be added here or to `DATA_SOURCES.md`.

## Python Dependencies

Python packages are resolved in `uv.lock` and installed into a local virtual
environment. They are not vendored. Each package retains its upstream license.
