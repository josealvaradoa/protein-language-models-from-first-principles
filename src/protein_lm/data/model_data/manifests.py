"""Exact Week 2 candidate manifest writing and checksum verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from protein_lm.data.model_data.allocation import CandidateAllocation
from protein_lm.data.model_data.contracts import Alias, CandidateRecord, ModelDataError

MEMBERSHIP_HEADER = (
    "primary_accession",
    "partition",
    "sequence_sha256",
    "biological_length",
    "length_bucket",
    "uniref50_group",
)
ALIAS_HEADER = ("sequence_sha256", "canonical_accession", "alias_accession")
TSV_FILENAMES = (
    "shared_validation.tsv",
    "shared_sealed_test.tsv",
    "random_arm.tsv",
    "family_aware_arm.tsv",
    "deduplication_aliases.tsv",
)


@dataclass(frozen=True)
class ArtifactEvidence:
    row_count: int
    byte_size: int
    sha256: str


def write_candidate_tsvs(
    directory: Path, allocation: CandidateAllocation, aliases: tuple[Alias, ...]
) -> dict[str, ArtifactEvidence]:
    """Write all membership and alias TSVs with deterministic ordering."""

    evidence = {
        "shared_validation.tsv": _write_membership(
            directory / "shared_validation.tsv",
            allocation.shared_validation,
            "shared_validation",
        ),
        "shared_sealed_test.tsv": _write_membership(
            directory / "shared_sealed_test.tsv",
            allocation.shared_sealed_test,
            "shared_sealed_test",
        ),
        "random_arm.tsv": _write_membership(
            directory / "random_arm.tsv",
            allocation.random_training,
            "training",
            allocation.random_native_validation,
        ),
        "family_aware_arm.tsv": _write_membership(
            directory / "family_aware_arm.tsv",
            allocation.family_training,
            "training",
            allocation.family_native_validation,
        ),
        "deduplication_aliases.tsv": _write_aliases(
            directory / "deduplication_aliases.tsv", aliases
        ),
    }
    return evidence


def write_checksums(path: Path, evidence: dict[str, ArtifactEvidence]) -> None:
    """Write the exact evidence object for the five local TSV artifacts."""

    payload = {
        filename: {
            "row_count": item.row_count,
            "byte_size": item.byte_size,
            "sha256": item.sha256,
        }
        for filename, item in sorted(evidence.items())
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_and_verify_checksums(directory: Path) -> dict[str, ArtifactEvidence]:
    """Require each expected TSV identity and reject extra checksum entries."""

    expected_files = {
        *TSV_FILENAMES,
        "artifact_checksums.json",
        "preparation_record.json",
    }
    try:
        actual_files = {path.name for path in directory.iterdir()}
    except OSError as error:
        raise ModelDataError("candidate directory cannot be inspected") from error
    if actual_files != expected_files:
        raise ModelDataError("candidate artifact inventory drifted")
    try:
        raw = json.loads(
            (directory / "artifact_checksums.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError("artifact checksums are malformed") from error
    if not isinstance(raw, dict) or set(raw) != set(TSV_FILENAMES):
        raise ModelDataError("artifact checksums do not cover exactly five TSV files")
    evidence = {}
    for filename in TSV_FILENAMES:
        item = raw[filename]
        if not isinstance(item, dict) or set(item) != {
            "row_count",
            "byte_size",
            "sha256",
        }:
            raise ModelDataError("artifact checksum item is malformed")
        row_count = item["row_count"]
        byte_size = item["byte_size"]
        digest = item["sha256"]
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 0
            or isinstance(byte_size, bool)
            or not isinstance(byte_size, int)
            or byte_size < 0
            or not isinstance(digest, str)
        ):
            raise ModelDataError("artifact checksum values are malformed")
        actual = _evidence(directory / filename)
        expected = ArtifactEvidence(row_count, byte_size, digest)
        if actual != expected:
            raise ModelDataError(f"artifact identity drifted: {filename}")
        evidence[filename] = actual
    return evidence


def _write_membership(
    path: Path,
    first: tuple[CandidateRecord, ...],
    first_partition: str,
    second: tuple[CandidateRecord, ...] = (),
) -> ArtifactEvidence:
    rows = [(record, first_partition) for record in first] + [
        (record, "native_validation") for record in second
    ]
    rows.sort(key=lambda item: item[0].primary_accession)
    content = ["\t".join(MEMBERSHIP_HEADER)]
    content.extend(
        "\t".join(
            (
                record.primary_accession,
                partition,
                record.sequence_sha256,
                str(record.biological_length),
                record.length_bucket,
                record.uniref50_group,
            )
        )
        for record, partition in rows
    )
    return _write(path, "\n".join(content) + "\n", len(rows))


def _write_aliases(path: Path, aliases: tuple[Alias, ...]) -> ArtifactEvidence:
    content = ["\t".join(ALIAS_HEADER)]
    content.extend(
        "\t".join(
            (alias.sequence_sha256, alias.canonical_accession, alias.alias_accession)
        )
        for alias in aliases
    )
    return _write(path, "\n".join(content) + "\n", len(aliases))


def _write(path: Path, text: str, row_count: int) -> ArtifactEvidence:
    raw = text.encode("utf-8")
    path.write_bytes(raw)
    return ArtifactEvidence(row_count, len(raw), hashlib.sha256(raw).hexdigest())


def _evidence(path: Path) -> ArtifactEvidence:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ModelDataError(f"candidate artifact is missing: {path.name}") from error
    lines = raw.splitlines()
    return ArtifactEvidence(len(lines) - 1, len(raw), hashlib.sha256(raw).hexdigest())
