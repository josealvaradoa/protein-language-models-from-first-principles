"""Immutable local A-004 import receipt and completion-marker publication."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)
from protein_lm.data.similarity_fastas import MaterializedInputs
from protein_lm.data.task7_a003_import import A003Import
from protein_lm.data.task7_a003_stages import ImportedStage
from protein_lm.data.task7_a004_aggregation import PairUnionBundle, TrackEvidence
from protein_lm.data.task7_a004_database import A004Database
from protein_lm.data.task7_a004_policy import (
    APPROVED_A004_CONFIG_SHA256,
    A004Policy,
)
from protein_lm.data.task7_a004_report import ReportPublication
from protein_lm.data.task7_a004_runtime import validate_hardware_provenance
from protein_lm.data.task7_checkpoints import (
    file_identity,
    read_json,
    require_marker_identity,
    verify_file,
    write_json_atomic,
)
from protein_lm.data.task7_fixed_budget_contract import FixedBudgetStage


def a004_fingerprint(
    *,
    policy: A004Policy,
    source_policy: SimilarityAuditPolicy,
    code_revision: str,
    mmseqs_version: str,
) -> str:
    """Bind A-004 output to its code, two policies, tool, and frozen inputs."""

    payload = {
        "a004_policy_sha256": APPROVED_A004_CONFIG_SHA256,
        "a003_policy_sha256": policy.source_policy_sha256,
        "a003_run_fingerprint": policy.source_run_fingerprint,
        "a003_code_revision": policy.source_code_revision,
        "a004_code_revision": code_revision,
        "mmseqs_version": mmseqs_version,
        "task4_catalog_sha256": source_policy.task4_catalog_sha256,
        "task5_local_assignment_sha256": source_policy.task5_local_assignment_sha256,
        "task6_local_assignment_sha256": source_policy.task6_local_assignment_sha256,
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def frozen_assignment_identities(paths: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    """Capture only immutable Task 5/6 evidence identities, never memberships."""

    names = (
        "task5_public",
        "task5_local",
        "task5_report",
        "task6_public",
        "task6_local",
        "task6_report",
    )
    return {name: file_identity(paths[name]) for name in names}


def require_same_six_fastas(inputs: MaterializedInputs, imported: A003Import) -> None:
    """Require newly materialized A-004 inputs to equal all preserved A-003 FASTAs."""

    for strategy, partitions in inputs.fastas.items():
        for partition, evidence in partitions.items():
            if evidence != imported.fasta(strategy, partition):
                raise SimilarityAuditError("A-004 FASTA differs from preserved A-003 input")


@dataclass(frozen=True)
class ReceiptPublication:
    """Expected bytes and payload for the deterministic A-004 receipt."""

    path: Path
    payload: Mapping[str, object]
    identity: Mapping[str, object]


@dataclass(frozen=True)
class CompletionAuthorization:
    """Disk-backed final-validation result required for completion."""

    fingerprint: str
    receipt_identity: Mapping[str, object]
    report_identities: Mapping[str, Mapping[str, object]]


def publish_receipt(
    *,
    workspace: Path,
    fingerprint: str,
    policy: A004Policy,
    source_policy_path: Path,
    code_revision: str,
    mmseqs_version: str,
    hardware: Mapping[str, object],
    assignments_before: Mapping[str, Mapping[str, object]],
    assignments_after: Mapping[str, Mapping[str, object]],
    imported: A003Import,
    databases: Mapping[str, A004Database],
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
    unions: Mapping[tuple[str, str], PairUnionBundle],
    report: ReportPublication,
) -> ReceiptPublication:
    """Write or verify the deterministic receipt without completing A-004."""

    receipt_path = workspace / "a004_import_receipt.json"
    validate_hardware_provenance(hardware)
    if assignments_before != assignments_after:
        raise SimilarityAuditError("A-004 receipt cannot claim changed assignments")
    receipt = _normalize_json(
        _receipt(
            fingerprint=fingerprint,
            policy=policy,
            source_policy_path=source_policy_path,
            code_revision=code_revision,
            mmseqs_version=mmseqs_version,
            hardware=hardware,
            assignments_before=assignments_before,
            assignments_after=assignments_after,
            imported=imported,
            databases=databases,
            tracks=tracks,
            unions=unions,
            report=report,
        )
    )
    _write_or_verify(receipt_path, receipt, "a004_import_receipt", fingerprint)
    publication = ReceiptPublication(
        path=receipt_path,
        payload=receipt,
        identity=file_identity(receipt_path),
    )
    verify_receipt_publication(publication, fingerprint=fingerprint)
    return publication


def publish_completion_marker(
    *,
    workspace: Path,
    fingerprint: str,
    receipt: ReceiptPublication,
    report: ReportPublication,
    authorization: CompletionAuthorization,
) -> Path:
    """Publish only after the caller supplies the final disk-validation result."""

    expected_report_identities = {
        "json": dict(report.json_identity),
        "markdown": dict(report.markdown_identity),
        "marker": dict(report.marker_identity),
    }
    if (
        authorization.fingerprint != fingerprint
        or dict(authorization.receipt_identity) != dict(receipt.identity)
        or authorization.report_identities != expected_report_identities
    ):
        raise SimilarityAuditError("A-004 completion authorization is invalid")
    completion_path = workspace / "a004_complete.json"
    completion = {
        "schema_version": 1,
        "stage": "a004_workflow_complete",
        "fingerprint": fingerprint,
        "receipt": dict(receipt.identity),
        "report": {
            "json": dict(report.json_identity),
            "markdown": dict(report.markdown_identity),
            "marker": dict(report.marker_identity),
        },
        "model_use": "prohibited",
        "task8_membership_use_authorized": False,
        "diagnostic_assignments_unchanged": True,
    }
    _write_or_verify(completion_path, completion, "a004_workflow_complete", fingerprint)
    return completion_path


def verify_receipt_publication(
    publication: ReceiptPublication, *, fingerprint: str
) -> None:
    """Re-read and compare receipt bytes against the expected payload."""

    size = publication.identity.get("byte_size")
    digest = publication.identity.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or not isinstance(digest, str):
        raise SimilarityAuditError("A-004 receipt identity is malformed")
    verify_file(publication.path, size, digest)
    current = read_json(publication.path)
    require_marker_identity(current, fingerprint, "a004_import_receipt")
    if current != publication.payload:
        raise SimilarityAuditError("A-004 receipt payload drifted")


def _receipt(
    *,
    fingerprint: str,
    policy: A004Policy,
    source_policy_path: Path,
    code_revision: str,
    mmseqs_version: str,
    hardware: Mapping[str, object],
    assignments_before: Mapping[str, Mapping[str, object]],
    assignments_after: Mapping[str, Mapping[str, object]],
    imported: A003Import,
    databases: Mapping[str, A004Database],
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
    unions: Mapping[tuple[str, str], PairUnionBundle],
    report: ReportPublication,
) -> dict[str, object]:
    track_records = [_track_record(track) for _, track in sorted(tracks.items())]
    return {
        "schema_version": 1,
        "stage": "a004_import_receipt",
        "fingerprint": fingerprint,
        "authority": {
            "adjustment_id": policy.adjustment_id,
            "source_adjustment_id": policy.source_adjustment_id,
            "read_only": policy.read_only,
            "repair_authorized": policy.repair_authorized,
            "selected_split_authorized": policy.selected_split_authorized,
            "model_use": policy.model_use,
            "task8_membership_use_authorized": policy.task8_membership_use_authorized,
        },
        "code_revision": code_revision,
        "mmseqs_version": mmseqs_version,
        "hardware": dict(hardware),
        "source_policy": file_identity(source_policy_path),
        "diagnostic_assignments": {
            "unchanged": True,
            "before": assignments_before,
            "after": assignments_after,
        },
        "imported_a003": {
            "fingerprint": imported.fingerprint,
            "fastas": {
                strategy: {
                    partition: asdict(evidence)
                    for partition, evidence in partitions.items()
                }
                for strategy, partitions in imported.fastas.items()
            },
            "database": asdict(imported.database),
            "escalated_query_ids": list(imported.escalated_query_ids),
            "stages": [_stage_record(stage) for stage in imported.stages],
        },
        "fresh_a004_databases": {
            strategy: {
                "command": list(database.command),
                "runtime_seconds": database.runtime_seconds,
                "identity": database.identity,
            }
            for strategy, database in databases.items()
        },
        "imported_tracks": [track for track in track_records if track["origin"] == "imported_a003"],
        "executed_tracks": [track for track in track_records if track["origin"] == "executed_a004"],
        "pair_unions": {
            f"{strategy}_{partition}": {
                "common_all_query_10000": {
                    "marker": bundle.common_all_query_10000.marker_identity,
                    "evidence": asdict(bundle.common_all_query_10000.evidence),
                },
                "staged_union_with_changed_query_100000": {
                    "marker": (
                        bundle.staged_union_with_changed_query_100000.marker_identity
                    ),
                    "evidence": asdict(
                        bundle.staged_union_with_changed_query_100000.evidence
                    ),
                },
                "comparison": asdict(bundle.comparison),
            }
            for (strategy, partition), bundle in sorted(unions.items())
        },
        "report": {
            "directory": str(report.directory),
            "json": dict(report.json_identity),
            "markdown": dict(report.markdown_identity),
            "marker": dict(report.marker_identity),
        },
    }


def _track_record(track: TrackEvidence) -> dict[str, object]:
    return {
        "strategy": track.plan.strategy,
        "partition": track.plan.partition,
        "pass_name": track.plan.pass_name,
        "origin": track.plan.origin,
        "all_query_count": len(track.all_query_ids),
        "changed_query_ids_between_1000_and_10000": list(track.changed_query_ids),
        "stages": [_stage_record(stage) for stage in track.stages],
        "cap_sensitivity": [asdict(comparison) for comparison in track.comparisons],
        "cap_summaries": {
            str(cap): {
                "source_label": summary.source_label,
                "marker": dict(summary.marker_identity),
                "evidence": asdict(summary.evidence),
            }
            for cap, summary in track.summaries.items()
        },
        "pass_marker": track.pass_marker_identity,
    }


def _stage_record(stage: FixedBudgetStage | ImportedStage) -> dict[str, object]:
    marker = (
        asdict(stage.marker)
        if isinstance(stage, ImportedStage)
        else file_identity(stage.marker_path)
    )
    return {
        "cap": stage.cap,
        "origin": "imported_a003" if isinstance(stage, ImportedStage) else "executed_a004",
        "marker": marker,
        "query_fasta": asdict(stage.query_fasta),
        "canonical": asdict(stage.canonical),
        "command": list(stage.command),
        "runtime_seconds": stage.runtime_seconds,
    }


def _write_or_verify(path: Path, payload: Mapping[str, object], stage: str, fingerprint: str) -> None:
    if path.exists():
        existing = read_json(path)
        require_marker_identity(existing, fingerprint, stage)
        if existing != payload:
            raise SimilarityAuditError(f"A-004 {stage} identity drifted")
        return
    write_json_atomic(path, payload)


def _normalize_json(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    return value
