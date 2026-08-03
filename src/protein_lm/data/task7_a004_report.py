"""Validated JSON and concise Markdown reporting for the A-004 audit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.task7_a004_aggregation import PairUnionBundle, TrackEvidence
from protein_lm.data.task7_a004_policy import A004Policy
from protein_lm.data.task7_a004_report_payload import (
    build_report_payload,
    validate_report_payload,
)
from protein_lm.data.task7_a004_report_render import render_markdown_report
from protein_lm.data.task7_checkpoints import (
    file_identity,
    read_json,
    require_marker_identity,
    verify_file,
    write_json_atomic,
)

@dataclass(frozen=True)
class ReportPublication:
    """Expected identities and payload for one published A-004 report."""

    directory: Path
    json_path: Path
    markdown_path: Path
    marker_path: Path
    payload: Mapping[str, object]
    json_identity: Mapping[str, object]
    markdown_identity: Mapping[str, object]
    marker_identity: Mapping[str, object]


def publish_a004_report(
    *,
    workspace: Path,
    fingerprint: str,
    policy: A004Policy,
    hardware: Mapping[str, object],
    assignment_balances: Mapping[str, object],
    assignments_unchanged: bool,
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
    unions: Mapping[tuple[str, str], PairUnionBundle],
) -> ReportPublication:
    """Write or verify the complete A-004 JSON and Markdown report pair."""

    payload = build_report_payload(
        fingerprint=fingerprint,
        policy=policy,
        hardware=hardware,
        assignment_balances=assignment_balances,
        assignments_unchanged=assignments_unchanged,
        tracks=tracks,
        unions=unions,
    )
    validate_report_payload(payload, fingerprint=fingerprint)
    markdown = render_markdown_report(payload)
    directory = workspace / "evidence" / "report"
    marker_path = directory / "complete.json"
    json_path = directory / "a004_report.json"
    markdown_path = directory / "a004_report.md"
    if marker_path.exists():
        publication = _publication_from_disk(directory, payload, markdown, fingerprint)
        verify_report_publication(publication, fingerprint=fingerprint)
        return publication
    if directory.exists() and any(directory.iterdir()):
        raise SimilarityAuditError("A-004 report output lacks its completion marker")
    staging = directory.with_name(".report.incomplete")
    if staging.exists():
        raise SimilarityAuditError("A-004 report has an unmarked staging directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    write_json_atomic(staging / json_path.name, payload)
    (staging / markdown_path.name).write_text(markdown, encoding="utf-8")
    marker = {
        "schema_version": 1,
        "stage": "a004_report_artifacts",
        "fingerprint": fingerprint,
        "json": file_identity(staging / json_path.name),
        "markdown": file_identity(staging / markdown_path.name),
    }
    write_json_atomic(staging / marker_path.name, marker)
    if directory.exists():
        directory.rmdir()
    staging.replace(directory)
    publication = _publication_from_disk(directory, payload, markdown, fingerprint)
    verify_report_publication(publication, fingerprint=fingerprint)
    return publication


def verify_report_publication(
    publication: ReportPublication, *, fingerprint: str
) -> None:
    """Re-read and verify report bytes, marker, schema, and expected payload."""

    marker = read_json(publication.marker_path)
    require_marker_identity(marker, fingerprint, "a004_report_artifacts")
    expected_marker = {
        "schema_version": 1,
        "stage": "a004_report_artifacts",
        "fingerprint": fingerprint,
        "json": dict(publication.json_identity),
        "markdown": dict(publication.markdown_identity),
    }
    if marker != expected_marker:
        raise SimilarityAuditError("A-004 report marker identity drifted")
    _verify_identity(publication.marker_path, publication.marker_identity)
    _verify_identity(publication.json_path, publication.json_identity)
    _verify_identity(publication.markdown_path, publication.markdown_identity)
    current = read_json(publication.json_path)
    validate_report_payload(current, fingerprint=fingerprint)
    if current != publication.payload:
        raise SimilarityAuditError("A-004 JSON report payload drifted")


def _publication_from_disk(
    directory: Path,
    payload: Mapping[str, object],
    markdown: str,
    fingerprint: str,
) -> ReportPublication:
    json_path = directory / "a004_report.json"
    markdown_path = directory / "a004_report.md"
    marker_path = directory / "complete.json"
    expected_markdown = markdown.encode("utf-8")
    if not markdown_path.is_file() or markdown_path.read_bytes() != expected_markdown:
        raise SimilarityAuditError("A-004 Markdown report payload drifted")
    marker = read_json(marker_path)
    require_marker_identity(marker, fingerprint, "a004_report_artifacts")
    json_identity = marker.get("json")
    markdown_identity = marker.get("markdown")
    if not isinstance(json_identity, dict) or not isinstance(markdown_identity, dict):
        raise SimilarityAuditError("A-004 report marker evidence is malformed")
    return ReportPublication(
        directory=directory,
        json_path=json_path,
        markdown_path=markdown_path,
        marker_path=marker_path,
        payload=payload,
        json_identity=json_identity,
        markdown_identity=markdown_identity,
        marker_identity=file_identity(marker_path),
    )


def _verify_identity(path: Path, identity: Mapping[str, object]) -> None:
    size = identity.get("byte_size")
    digest = identity.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or not isinstance(digest, str):
        raise SimilarityAuditError("A-004 report file identity is malformed")
    verify_file(path, size, digest)
