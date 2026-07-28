# Data Sources And Model Assets

No biological dataset or external model weight is committed to this
repository. Generation and download steps must be reproducible, but downloaded
assets stay in ignored local directories.

## Frozen Source Contract

| Source | Frozen version | Planned use | Setup status |
|---|---|---|---|
| UniProtKB/Swiss-Prot | `2026_02` | Canonical curated protein sequences | Not downloaded |
| UniRef50 | Release matched to Swiss-Prot `2026_02` | Group-level train, validation, and test splitting | Not downloaded |
| ProteinGym | `v1.3` | Small mutation-effect evaluation panel | Not downloaded |
| Biohub ESMC | `biohub/ESMC-300M`, revision to be pinned | External masked-model and representation baseline | Not downloaded |

The `2026_02` source pair is frozen for this project. Swiss-Prot records and
UniRef50 membership must come from the same release.

The machine-readable source pin is
`experiments/week_01/acquisition.toml`. Validate its schema and prove its raw
destinations are ignored with:

```bash
uv run python scripts/validate_acquisition.py
```

This validation command is offline. It does not create directories, contact
UniProt, or download data.

## Approved Source Artifacts

| Use | File | Published bytes | Published MD5 | Local SHA-256 |
|---|---|---:|---|---|
| Swiss-Prot records | `uniprot_sprot.dat.gz` | 699,031,150 | `868b301a6ec93955f4e4355d579d8683` | Pending acquisition |
| UniRef50 membership, column 10 | `idmapping_selected.tab.gz` | 7,066,467,385 | `f426a0ee61882f4c86f1b0d616ae53ec` | Pending acquisition |

The upstream files currently appear below `current_release`. Before any
transfer, acquisition must verify that `reldate.txt` still identifies release
`2026_02` of June 10, 2026 and that published sizes and MD5 values match this
table. A mismatch stops acquisition rather than advancing the project to a
newer release.

The published MD5 values verify that local files match UniProt's listings.
After acquisition, the project also calculates SHA-256 values for its own
provenance record. Those local values are evidence from actual files, so they
are not prefilled in the acquisition config.

## External Tool Provenance

| Tool | Version | Installation | Executable | Verified |
|---|---|---|---|---|
| MMseqs2 | `18-8cc5c` | Homebrew | `/opt/homebrew/bin/mmseqs` | July 22, 2026 |

MMseqs2 will be used to audit held-out proteins against the training partition.
The executable remains an external system dependency and is not committed to
this repository.

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
