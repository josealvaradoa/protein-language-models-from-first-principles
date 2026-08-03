"""Final disk-backed evidence gate before A-004 completion publication."""

from __future__ import annotations

from collections.abc import Mapping

from protein_lm.data.similarity_audit_policy import SimilarityAuditError
from protein_lm.data.task7_a003_stages import ImportedStage
from protein_lm.data.task7_a004_aggregation import PairUnionBundle, TrackEvidence
from protein_lm.data.task7_a004_database import A004Database, verify_a004_database
from protein_lm.data.task7_a004_evidence import verify_cap_summary, verify_pair_union
from protein_lm.data.task7_a004_receipt import (
    CompletionAuthorization,
    ReceiptPublication,
    verify_receipt_publication,
)
from protein_lm.data.task7_a004_report import (
    ReportPublication,
    verify_report_publication,
)
from protein_lm.data.task7_checkpoints import (
    file_identity,
    read_json,
    require_marker_identity,
    verify_file,
)
from protein_lm.data.task7_fixed_budget import compare_caps
from protein_lm.data.task7_fixed_budget_contract import FixedBudgetStage
from protein_lm.data.task7_pair_union import compare_pair_unions


def revalidate_before_completion(
    *,
    fingerprint: str,
    databases: Mapping[str, A004Database],
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
    unions: Mapping[tuple[str, str], PairUnionBundle],
    report: ReportPublication,
    receipt: ReceiptPublication,
) -> CompletionAuthorization:
    """Re-read every mutable A-004 artifact without writing any evidence."""

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
        raise SimilarityAuditError("A-004 receipt does not identify its report")
    return CompletionAuthorization(
        fingerprint=fingerprint,
        receipt_identity=file_identity(receipt.path),
        report_identities={
            "json": dict(report.json_identity),
            "markdown": dict(report.markdown_identity),
            "marker": dict(report.marker_identity),
        },
    )


def _verify_track(track: TrackEvidence, *, fingerprint: str) -> None:
    by_cap = {stage.cap: stage for stage in track.stages}
    if set(by_cap) != set(track.summaries):
        raise SimilarityAuditError("A-004 final track inventory drifted")
    for cap, stage in by_cap.items():
        _verify_stage(track, stage, fingerprint=fingerprint)
        query_ids = track.changed_query_ids if cap == 100_000 else track.all_query_ids
        query_fasta = (
            stage.canonical_path.parent.parent / "escalated_queries.fasta"
            if cap == 100_000
            else track.query_fasta
        )
        current = verify_cap_summary(
            source_label=track.plan.origin,
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
            raise SimilarityAuditError("A-004 cap-summary evidence changed before completion")
    comparisons = [_compare(track, by_cap, 1_000, 10_000, track.all_query_ids)]
    if track.changed_query_ids:
        comparisons.append(
            _compare(
                track,
                by_cap,
                10_000,
                100_000,
                track.changed_query_ids,
                baseline_contains_other_queries=True,
            )
        )
    if tuple(comparisons) != track.comparisons:
        raise SimilarityAuditError("A-004 cap sensitivity changed before completion")
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
        raise SimilarityAuditError("A-004 search-stage marker changed before completion")
    marker = read_json(marker_path)
    expected_stage = "search_stage" if isinstance(stage, ImportedStage) else (
        "a004_fixed_budget_search_stage"
    )
    expected_fingerprint = (
        marker.get("fingerprint") if isinstance(stage, ImportedStage) else fingerprint
    )
    if not isinstance(expected_fingerprint, str):
        raise SimilarityAuditError("A-004 search-stage fingerprint is malformed")
    require_marker_identity(marker, expected_fingerprint, expected_stage)
    if marker.get("cap") != stage.cap:
        raise SimilarityAuditError("A-004 search-stage cap changed before completion")


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
        baseline_summary_path=track.summaries[baseline].directory / "query_summaries.tsv",
        comparison_summary_path=(
            track.summaries[comparison].directory / "query_summaries.tsv"
        ),
        expected_query_ids=query_ids,
        baseline_contains_other_queries=baseline_contains_other_queries,
    )


def _verify_pass_marker(track: TrackEvidence, *, fingerprint: str) -> None:
    if track.plan.origin == "imported_a003":
        if track.pass_marker_path is not None or track.pass_marker_identity is not None:
            raise SimilarityAuditError("A-004 imported track has a fresh pass marker")
        return
    if track.pass_marker_path is None or track.pass_marker_identity is None:
        raise SimilarityAuditError("A-004 executed track lacks its pass marker")
    if file_identity(track.pass_marker_path) != track.pass_marker_identity:
        raise SimilarityAuditError("A-004 pass marker changed before completion")
    marker = read_json(track.pass_marker_path)
    require_marker_identity(marker, fingerprint, "a004_fixed_budget_pass")
    expected_cap_outputs = {f"cap_{cap}" for cap in track.stage_marker_identities}
    actual_cap_outputs = {
        path.name
        for path in track.pass_marker_path.parent.iterdir()
        if path.name.startswith("cap_")
    }
    if actual_cap_outputs != expected_cap_outputs:
        raise SimilarityAuditError("A-004 final pass cap inventory drifted")
    stage_records = marker.get("stages")
    if not isinstance(stage_records, dict) or set(stage_records) != {
        str(cap) for cap in track.stage_marker_identities
    }:
        raise SimilarityAuditError("A-004 pass stage inventory drifted")
    for cap, identity in track.stage_marker_identities.items():
        record = stage_records[str(cap)]
        if not isinstance(record, dict) or record.get("marker") != identity:
            raise SimilarityAuditError("A-004 pass marker no longer identifies its stage")
    escalation = marker.get("escalation")
    if track.changed_query_ids:
        if not isinstance(escalation, dict) or not isinstance(escalation.get("marker"), dict):
            raise SimilarityAuditError("A-004 pass escalation marker is malformed")
        marker_path = track.pass_marker_path.parent / "escalated_queries.complete.json"
        if file_identity(marker_path) != escalation["marker"]:
            raise SimilarityAuditError("A-004 escalation marker changed before completion")
    elif escalation is not None:
        raise SimilarityAuditError("A-004 pass has unexpected escalation evidence")


def _verify_union_bundle(
    *,
    key: tuple[str, str],
    bundle: PairUnionBundle,
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
    fingerprint: str,
) -> None:
    strategy, partition = key
    passes = {
        name: tracks[(strategy, partition, name)] for name in ("enforcement", "residual")
    }
    common_sources = {
        f"{name}_{item.plan.origin}_cap_{cap}": (
            item.summaries[cap].directory / "prohibited_pairs.tsv"
        )
        for name, item in passes.items()
        for cap in (1_000, 10_000)
    }
    common = verify_pair_union(
        label=f"common_all_query_10000_{strategy}_{partition}",
        source_paths=common_sources,
        output_directory=bundle.common_all_query_10000.directory,
        fingerprint=fingerprint,
    )
    staged_sources = {
        "common_all_query_10000": common.directory / "prohibited_pairs.tsv"
    }
    for name, item in passes.items():
        if 100_000 in item.summaries:
            staged_sources[f"{name}_{item.plan.origin}_cap_100000"] = (
                item.summaries[100_000].directory / "prohibited_pairs.tsv"
            )
    staged = verify_pair_union(
        label=f"staged_union_with_changed_query_100000_{strategy}_{partition}",
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
        raise SimilarityAuditError("A-004 pair-union evidence changed before completion")
