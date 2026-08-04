"""Final frozen-source and disk-backed gate before A-004 completion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from protein_lm.data.artifacts import (
    file_identity,
    read_json,
    require_marker_identity,
    verify_file,
)
from protein_lm.data.fixed_budget_audit.config import (
    A004Policy,
    AuditPass,
    CandidateCap,
    PairUnionKind,
    TrackOrigin,
    load_a004_policy,
)
from protein_lm.data.fixed_budget_audit.evidence import (
    compare_caps,
    compare_pair_unions,
    verify_cap_summary,
    verify_pair_union,
)
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    AuditValidationError,
    SourceEvidenceError,
)
from protein_lm.data.fixed_budget_audit.provenance import (
    frozen_assignment_identities,
)
from protein_lm.data.fixed_budget_audit.reporting import (
    CompletionAuthorization,
    ReceiptPublication,
    ReportPublication,
    verify_receipt_publication,
    verify_report_publication,
)
from protein_lm.data.fixed_budget_audit.search import (
    A004Database,
    FixedBudgetStage,
    verify_a004_database,
)
from protein_lm.data.fixed_budget_audit.source import (
    A003Import,
    ImportedStage,
    reverify_frozen_run_state,
    verify_a003_residual_import,
)
from protein_lm.data.fixed_budget_audit.tracks import (
    PairUnionBundle,
    TrackEvidence,
)
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)

__all__ = [
    "FinalValidationContext",
    "build_final_validation_context",
    "revalidate_before_completion",
]


@dataclass(frozen=True)
class FinalValidationContext:
    """Immutable baseline required to authorize the completion marker."""

    source_paths: Mapping[str, Path]
    source_policy: SimilarityAuditPolicy
    a004_policy: A004Policy
    a004_config_path: Path
    source_policy_path: Path
    project_root: Path
    code_revision: str
    mmseqs_version: str
    baseline_assignment_identities: Mapping[str, Mapping[str, object]]
    baseline_imported_a003: A003Import


def _immutable_a003_snapshot(imported: A003Import) -> A003Import:
    """Return a detached A-003 value graph with immutable mapping containers."""

    fastas = MappingProxyType(
        {
            strategy: MappingProxyType(
                {
                    partition: replace(evidence)
                    for partition, evidence in partitions.items()
                }
            )
            for strategy, partitions in imported.fastas.items()
        }
    )
    database = replace(imported.database, marker=replace(imported.database.marker))
    stages = tuple(
        replace(
            stage,
            marker=replace(stage.marker),
            query_fasta=replace(stage.query_fasta),
            canonical=replace(stage.canonical),
            command=tuple(item for item in stage.command),
        )
        for stage in imported.stages
    )
    return replace(
        imported,
        fastas=fastas,
        database=database,
        stages=stages,
        escalated_query_ids=tuple(item for item in imported.escalated_query_ids),
    )


def build_final_validation_context(
    *,
    source_paths: Mapping[str, Path],
    source_policy: SimilarityAuditPolicy,
    a004_policy: A004Policy,
    a004_config_path: Path,
    source_policy_path: Path,
    project_root: Path,
    code_revision: str,
    mmseqs_version: str,
    baseline_assignment_identities: Mapping[str, Mapping[str, object]],
    baseline_imported_a003: A003Import,
) -> FinalValidationContext:
    """Freeze the already-proven source state before public artifacts are written."""

    frozen_paths = MappingProxyType(dict(source_paths))
    frozen_assignments = MappingProxyType(
        {
            name: MappingProxyType(dict(identity))
            for name, identity in baseline_assignment_identities.items()
        }
    )
    return FinalValidationContext(
        source_paths=frozen_paths,
        source_policy=source_policy,
        a004_policy=a004_policy,
        a004_config_path=a004_config_path,
        source_policy_path=source_policy_path,
        project_root=project_root,
        code_revision=code_revision,
        mmseqs_version=mmseqs_version,
        baseline_assignment_identities=frozen_assignments,
        baseline_imported_a003=_immutable_a003_snapshot(baseline_imported_a003),
    )


def revalidate_before_completion(
    *,
    context: FinalValidationContext,
    fingerprint: str,
    databases: Mapping[str, A004Database],
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
    unions: Mapping[tuple[str, str], PairUnionBundle],
    report: ReportPublication,
    receipt: ReceiptPublication,
) -> CompletionAuthorization:
    """Re-read frozen sources and every mutable A-004 artifact without writing."""

    _revalidate_frozen_sources(context)
    try:
        return _revalidate_a004_artifacts(
            fingerprint=fingerprint,
            databases=databases,
            tracks=tracks,
            unions=unions,
            report=report,
            receipt=receipt,
        )
    except AuditValidationError:
        raise
    except SimilarityAuditError as error:
        raise AuditValidationError(str(error)) from error


def _revalidate_a004_artifacts(
    *,
    fingerprint: str,
    databases: Mapping[str, A004Database],
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
    unions: Mapping[tuple[str, str], PairUnionBundle],
    report: ReportPublication,
    receipt: ReceiptPublication,
) -> CompletionAuthorization:
    for database in databases.values():
        verify_a004_database(database, fingerprint=fingerprint)
    for track in tracks.values():
        _verify_track(track, fingerprint=fingerprint)
    for key, bundle in unions.items():
        _verify_union_bundle(
            key=key,
            bundle=bundle,
            tracks=tracks,
            fingerprint=fingerprint,
        )
    verify_report_publication(report, fingerprint=fingerprint)
    verify_receipt_publication(receipt, fingerprint=fingerprint)
    report_record = receipt.payload.get("report")
    expected_report_record = {
        "directory": str(report.directory),
        "json": dict(report.json_identity),
        "markdown": dict(report.markdown_identity),
        "marker": dict(report.marker_identity),
    }
    if report_record != expected_report_record:
        raise AuditValidationError("A-004 receipt does not identify its report")
    return CompletionAuthorization(
        fingerprint=fingerprint,
        receipt_identity=file_identity(receipt.path),
        report_identities={
            "json": dict(report.json_identity),
            "markdown": dict(report.markdown_identity),
            "marker": dict(report.marker_identity),
        },
    )


def _revalidate_frozen_sources(context: FinalValidationContext) -> None:
    reverify_frozen_run_state(
        paths=context.source_paths,
        policy=context.source_policy,
        code_revision=context.code_revision,
        mmseqs_version=context.mmseqs_version,
        config_path=context.source_policy_path,
        project_root=context.project_root,
    )
    if load_a004_policy(context.a004_config_path) != context.a004_policy:
        raise AuditConfigurationError("A-004 policy changed during the audit")
    try:
        imported = verify_a003_residual_import(
            project_root=context.project_root,
            policy=context.a004_policy,
        )
    except SourceEvidenceError:
        raise
    except SimilarityAuditError as error:
        raise SourceEvidenceError(str(error)) from error
    if imported != context.baseline_imported_a003:
        raise SourceEvidenceError("A-003 imported evidence changed during A-004")
    if (
        frozen_assignment_identities(context.source_paths)
        != context.baseline_assignment_identities
    ):
        raise SourceEvidenceError("immutable Task 5 or Task 6 assignment changed")


def _verify_track(track: TrackEvidence, *, fingerprint: str) -> None:
    initial_cap = CandidateCap.INITIAL.value
    comparison_cap = CandidateCap.COMPARISON.value
    escalation_cap = CandidateCap.ESCALATION.value
    by_cap = {stage.cap: stage for stage in track.stages}
    if set(by_cap) != set(track.summaries):
        raise AuditValidationError("A-004 final track inventory drifted")
    for cap, stage in by_cap.items():
        _verify_stage(track, stage, fingerprint=fingerprint)
        query_ids = (
            track.changed_query_ids if cap == escalation_cap else track.all_query_ids
        )
        query_fasta = (
            stage.canonical_path.parent.parent / "escalated_queries.fasta"
            if cap == escalation_cap
            else track.query_fasta
        )
        current = verify_cap_summary(
            source_label=track.plan.origin.value,
            cap=cap,
            canonical_path=stage.canonical_path,
            canonical_evidence=stage.canonical,
            query_fasta=query_fasta,
            query_fasta_evidence=stage.query_fasta,
            expected_query_ids=query_ids,
            output_directory=track.summaries[cap].directory,
            fingerprint=fingerprint,
        )
        if current != track.summaries[cap]:
            raise AuditValidationError(
                "A-004 cap-summary evidence changed before completion"
            )
    comparisons = [
        _compare(
            track,
            by_cap,
            initial_cap,
            comparison_cap,
            track.all_query_ids,
        )
    ]
    if track.changed_query_ids:
        comparisons.append(
            _compare(
                track,
                by_cap,
                comparison_cap,
                escalation_cap,
                track.changed_query_ids,
                baseline_contains_other_queries=True,
            )
        )
    if tuple(comparisons) != track.comparisons:
        raise AuditValidationError("A-004 cap sensitivity changed before completion")
    _verify_pass_marker(track, fingerprint=fingerprint)


def _verify_stage(
    track: TrackEvidence,
    stage: FixedBudgetStage | ImportedStage,
    *,
    fingerprint: str,
) -> None:
    verify_file(stage.canonical_path, stage.canonical.byte_size, stage.canonical.sha256)
    marker_path = (
        stage.canonical_path.parent / "complete.json"
        if isinstance(stage, ImportedStage)
        else stage.marker_path
    )
    expected_identity = track.stage_marker_identities.get(stage.cap)
    if expected_identity is None or file_identity(marker_path) != expected_identity:
        raise AuditValidationError(
            "A-004 search-stage marker changed before completion"
        )
    marker = read_json(marker_path)
    expected_stage = (
        "search_stage"
        if isinstance(stage, ImportedStage)
        else ("a004_fixed_budget_search_stage")
    )
    expected_fingerprint = (
        marker.get("fingerprint") if isinstance(stage, ImportedStage) else fingerprint
    )
    if not isinstance(expected_fingerprint, str):
        raise AuditValidationError("A-004 search-stage fingerprint is malformed")
    require_marker_identity(marker, expected_fingerprint, expected_stage)
    if marker.get("cap") != stage.cap:
        raise AuditValidationError("A-004 search-stage cap changed before completion")


def _compare(
    track: TrackEvidence,
    by_cap: Mapping[int, FixedBudgetStage | ImportedStage],
    baseline: int,
    comparison: int,
    query_ids: tuple[str, ...],
    *,
    baseline_contains_other_queries: bool = False,
):
    return compare_caps(
        baseline_cap=baseline,
        comparison_cap=comparison,
        baseline_canonical_path=by_cap[baseline].canonical_path,
        comparison_canonical_path=by_cap[comparison].canonical_path,
        baseline_summary_path=track.summaries[baseline].directory
        / "query_summaries.tsv",
        comparison_summary_path=(
            track.summaries[comparison].directory / "query_summaries.tsv"
        ),
        expected_query_ids=query_ids,
        baseline_contains_other_queries=baseline_contains_other_queries,
    )


def _verify_pass_marker(track: TrackEvidence, *, fingerprint: str) -> None:
    if track.plan.origin is TrackOrigin.IMPORTED_A003:
        if track.pass_marker_path is not None or track.pass_marker_identity is not None:
            raise AuditValidationError("A-004 imported track has a fresh pass marker")
        return
    if track.pass_marker_path is None or track.pass_marker_identity is None:
        raise AuditValidationError("A-004 executed track lacks its pass marker")
    if file_identity(track.pass_marker_path) != track.pass_marker_identity:
        raise AuditValidationError("A-004 pass marker changed before completion")
    marker = read_json(track.pass_marker_path)
    require_marker_identity(marker, fingerprint, "a004_fixed_budget_pass")
    expected_cap_outputs = {f"cap_{cap}" for cap in track.stage_marker_identities}
    actual_cap_outputs = {
        path.name
        for path in track.pass_marker_path.parent.iterdir()
        if path.name.startswith("cap_")
    }
    if actual_cap_outputs != expected_cap_outputs:
        raise AuditValidationError("A-004 final pass cap inventory drifted")
    stage_records = marker.get("stages")
    if not isinstance(stage_records, dict) or set(stage_records) != {
        str(cap) for cap in track.stage_marker_identities
    }:
        raise AuditValidationError("A-004 pass stage inventory drifted")
    for cap, identity in track.stage_marker_identities.items():
        record = stage_records[str(cap)]
        if not isinstance(record, dict) or record.get("marker") != identity:
            raise AuditValidationError(
                "A-004 pass marker no longer identifies its stage"
            )
    escalation = marker.get("escalation")
    if track.changed_query_ids:
        if not isinstance(escalation, dict) or not isinstance(
            escalation.get("marker"), dict
        ):
            raise AuditValidationError("A-004 pass escalation marker is malformed")
        marker_path = track.pass_marker_path.parent / "escalated_queries.complete.json"
        if file_identity(marker_path) != escalation["marker"]:
            raise AuditValidationError(
                "A-004 escalation marker changed before completion"
            )
    elif escalation is not None:
        raise AuditValidationError("A-004 pass has unexpected escalation evidence")


def _verify_union_bundle(
    *,
    key: tuple[str, str],
    bundle: PairUnionBundle,
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
    fingerprint: str,
) -> None:
    common_caps = (
        CandidateCap.INITIAL.value,
        CandidateCap.COMPARISON.value,
    )
    escalation_cap = CandidateCap.ESCALATION.value
    strategy, partition = key
    passes = {
        name: tracks[(strategy, partition, name)]
        for name in (AuditPass.ENFORCEMENT.value, AuditPass.RESIDUAL.value)
    }
    common_sources = {
        f"{name}_{item.plan.origin.value}_cap_{cap}": (
            item.summaries[cap].directory / "prohibited_pairs.tsv"
        )
        for name, item in passes.items()
        for cap in common_caps
    }
    common_kind = PairUnionKind.COMMON_ALL_QUERY_10000.value
    staged_kind = PairUnionKind.STAGED_UNION_WITH_CHANGED_QUERY_100000.value
    common = verify_pair_union(
        label=f"{common_kind}_{strategy}_{partition}",
        source_paths=common_sources,
        output_directory=bundle.common_all_query_10000.directory,
        fingerprint=fingerprint,
    )
    staged_sources = {common_kind: common.directory / "prohibited_pairs.tsv"}
    for name, item in passes.items():
        if escalation_cap in item.summaries:
            staged_sources[f"{name}_{item.plan.origin.value}_cap_{escalation_cap}"] = (
                item.summaries[escalation_cap].directory / "prohibited_pairs.tsv"
            )
    staged = verify_pair_union(
        label=f"{staged_kind}_{strategy}_{partition}",
        source_paths=staged_sources,
        output_directory=bundle.staged_union_with_changed_query_100000.directory,
        fingerprint=fingerprint,
    )
    comparison = compare_pair_unions(
        common_path=common.directory / "prohibited_pairs.tsv",
        staged_path=staged.directory / "prohibited_pairs.tsv",
    )
    if (
        common != bundle.common_all_query_10000
        or staged != bundle.staged_union_with_changed_query_100000
        or comparison != bundle.comparison
    ):
        raise AuditValidationError(
            "A-004 pair-union evidence changed before completion"
        )
