# Data Sources And Model Assets

No biological dataset or external model weight is committed to this
repository. Generation and download steps must be reproducible, but downloaded
assets stay in ignored local directories.

## Frozen Source Contract

| Source | Frozen version | Planned use | Setup status |
|---|---|---|---|
| UniProtKB/Swiss-Prot | `2026_01` | Canonical curated protein sequences | Not downloaded |
| UniRef50 | Release matched to Swiss-Prot `2026_01` | Group-level train, validation, and test splitting | Not downloaded |
| ProteinGym | `v1.3` | Small mutation-effect evaluation panel | Not downloaded |
| Biohub ESMC | `biohub/ESMC-300M`, revision to be pinned | External masked-model and representation baseline | Not downloaded |

MMseqs2 will be used as an external local binary to audit held-out proteins
against the training partition. Its version will be recorded before the first
audit.

## Primary Corpus Eligibility

The primary corpus uses canonical protein sequences containing only the 20
standard amino-acid symbols. Sequences containing `B`, `J`, `X`, `Z`, `U`, or
`O` are excluded and counted rather than silently rewritten.

Raw records, annotations, sequence files, split working data, and external
labels remain outside Git. Public manifests may contain stable identifiers,
partition assignments, source revisions, and checksums, but not raw labels
sealed by the evaluation firewall.

## Required Provenance Before Use

Every downloaded source must record:

- the official download URL;
- source release or immutable revision;
- retrieval date;
- file name and byte size;
- SHA-256 checksum;
- applicable license or terms; and
- the script and command that produced any derived artifact.

Download commands and checksums will be added when the corresponding data task
is specified and executed. No placeholder checksum is treated as evidence.
