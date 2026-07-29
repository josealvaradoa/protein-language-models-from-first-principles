# Data Sources And Model Assets

No biological dataset or external model weight is committed to this
repository. Generation and download steps must be reproducible, but downloaded
assets stay in ignored local directories.

## Frozen Source Contract

| Source | Frozen version | Planned use | Setup status |
|---|---|---|---|
| UniProtKB/Swiss-Prot | `2026_02` | Canonical curated protein sequences | Present locally; size, MD5, and SHA-256 verified |
| UniRef50 | Release matched to Swiss-Prot `2026_02` | Group-level train, validation, and test splitting | Present locally; size, MD5, and SHA-256 verified |
| ProteinGym | `v1.3` | Small mutation-effect evaluation panel | Metadata present and verified; assay files not acquired |
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
| Swiss-Prot records | `uniprot_sprot.dat.gz` | 699,031,150 | `868b301a6ec93955f4e4355d579d8683` | `741bcb144f98b8d10f0369b145d562b6751bfd17c285e936553aeb9cb54ab592` |
| UniRef50 membership, column 10 | `idmapping_selected.tab.gz` | 7,066,467,385 | `f426a0ee61882f4c86f1b0d616ae53ec` | `96707b0430f76e78f708eaa70d6cae7ccb4bb8e4b6b981c7d169215d650cc605` |

Before use, local verification must confirm that `reldate.txt` identifies
release `2026_02` of June 10, 2026 and that file sizes and calculated MD5
values match this table. A mismatch stops the task rather than advancing the
project to a newer release.

The published MD5 values verify that local files match UniProt's listings.
The same local verification pass also calculates SHA-256 values for the
project's provenance record. Those values are evidence from actual files, so
they are not prefilled in the acquisition config.

Both UniProt files were retrieved manually on July 28, 2026. No project
download script was executed.

The Task 2 audit verified the release metadata, published sizes, published MD5
values, and local SHA-256 values on July 29, 2026:

```bash
uv run --locked --offline python scripts/run_corpus_audit.py --repeat-check
```

The repeated audit produced the canonical aggregate JSON checksum
`ab83d9a3341694dab9b4097334f43b2036e5b4fb0417c8b3a028e54f679cdd0f`.
The corresponding machine-readable, human-readable, and checksum outputs are
stored under `reports/week_01/`.

## ProteinGym v1.3 Metadata Pin

The local ProteinGym metadata file is pinned to the official `PG_v1.3` release:

| Field | Frozen value |
|---|---|
| Official repository | [`OATML-Markslab/ProteinGym`](https://github.com/OATML-Markslab/ProteinGym) |
| Release tag | [`PG_v1.3`](https://github.com/OATML-Markslab/ProteinGym/releases/tag/PG_v1.3) |
| Release date | April 28, 2025 |
| Immutable commit | `1f8de974dead8ff7501eff087b725d14a965e9f9` |
| Official artifact | [`reference_files/DMS_substitutions.csv`](https://github.com/OATML-Markslab/ProteinGym/blob/1f8de974dead8ff7501eff087b725d14a965e9f9/reference_files/DMS_substitutions.csv) |
| Published bytes | 208,734 |
| Official Git blob SHA-1 | `8d1ea9a19c0404b511cd24378b25c2a5f86f10e9` |
| Local SHA-256 | `a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308` |
| Retrieval date | July 28, 2026 |
| Retrieval method | Manual user download; no project download script executed |
| Local path | `data/raw/proteingym/v1.3/DMS_substitutions.csv` |
| License | [MIT](https://github.com/OATML-Markslab/ProteinGym/blob/1f8de974dead8ff7501eff087b725d14a965e9f9/LICENSE) |

The local file has the same byte count and Git blob identity as the artifact at
the immutable upstream commit. The project-specific SHA-256 above identifies
the local bytes independently of Git.

The file's `UniProt_ID` field contains UniProtKB entry names, not stable primary
accessions. Corpus support is therefore resolved through the parsed Swiss-Prot
entry name and primary accession. Column 2 of the release-matched UniProt
identifier mapping is used only as a fallback for ProteinGym targets outside
the Swiss-Prot population so that their UniRef50 families can still be
reserved.

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
`O` are excluded and counted rather than silently rewritten. Records marked as
fragments, records outside the inclusive 32 through 2046 amino-acid range, and
records with blank UniRef50 mappings are also ineligible.

The Task 4 eligibility policy is pinned in
`experiments/week_01/eligibility.toml`. On July 29, 2026, committed revision
`a7c8b59f38b3599fb9541f2336e4d276c0dccc23` ran the required two-pass offline
preparation:

```bash
uv run --locked --offline python scripts/prepare_eligible_records.py --repeat-check
```

Both passes agreed. The source population contained 575,503 records and
208,906,902 residues. After applying the frozen filters, 557,718 records and
197,375,585 residues were eligible.

The detailed sequence-bearing artifacts remain ignored:

| Artifact | Data rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `data/processed/week_01/task_04_record_catalog.tsv` | 575,503 | 286,813,587 | `7d619d7853eb6165786c0e0aca4f50ed66f5b69dfbed134a81d789d9c6dbcb70` |
| `data/processed/week_01/task_04_candidate_test_reserved_families.txt` | 175 | 2,812 | `7b0f3389c74ad31f849a120c6d944ffdfc64a838a636cb0ee546bec9ce2d2a07` |

The aggregate public JSON has SHA-256
`be791d35b39c4bf1337c121ed830ab01de1d9e73adee77e9eb5d24b0bf64bc5d`.
It contains counts and provenance, but no sequences, record identifiers,
family identifiers, split membership, labels, or model results.

## Task 5 Random Diagnostic

Committed revision `976af20de2e834e7576b89e69cb8a18b09d818fb` assigned
the 557,718 eligible records twice with the frozen accession-hash algorithm:

```bash
uv run --locked --offline python scripts/build_random_diagnostic.py --repeat-check
```

Both passes agreed. The public manifest has 557,718 data rows, 56,286,073
bytes, and SHA-256
`bd0f4e376df0afa785bfef0153e072470c733cb8f9afe06c5bf973cb12a39c3e`.
The ignored local assignment history has SHA-256
`3b8687cdd5c7477b114fef7f425a68516ffd093f084aefee19d98272f397bfe6`.

This split is an intentionally unprotected diagnostic baseline. It is
prohibited for model training and is not the selected Week 1 split.

## Task 6 Group-Aware Pre-Repair Candidate

Committed revision `bfa191d788954c0e4b8a26f03ee2b2eeca1e6339` assigned
the same 557,718 eligible records twice with the frozen group-aware allocator:

```bash
uv run --locked --offline python scripts/build_group_aware_candidate.py --repeat-check
```

Both passes agreed. The 185,344 UniRef50 groups remained 185,344 assignment
units because no exact sequence hash crossed groups. All 157 represented
ProteinGym-reserved groups remained in test, no accession, exact hash, or
UniRef50 group crossed partitions, and retention was 100 percent.

The public pre-repair manifest has 557,718 data rows, 56,285,795 bytes, and
SHA-256
`f6ee25d078ca6df864e8e30bba848025d5b45810ed60605f09255c767c77f71a`.
The ignored local assignment history has SHA-256
`7a89c5d432ff9b2a2a787cae9d5584754ec8a027df4118aaedcc2bd4c41221d1`.
The exact repair-state-zero digest is
`0ed690a0e048c2c0c735d1caf796487be47b3700eb90ad9fcafeec0a315ce176`.

Record shares passed the frozen tolerance, but residue shares did not:

| Partition | Records | Record share | Residues | Residue share |
|---|---:|---:|---:|---:|
| Training | 500,585 | 89.755934% | 182,674,157 | 92.551547% |
| Validation | 28,550 | 5.119075% | 7,395,854 | 3.747097% |
| Test | 28,583 | 5.124991% | 7,305,574 | 3.701356% |

The candidate status is `failed_balance`. The frozen seed, allocator, and
tolerance were not changed after observing the result. Task 7 and model use
remain unauthorized.

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

Verification commands and checksums are added only after the corresponding data
task is executed. No placeholder checksum is treated as evidence.
