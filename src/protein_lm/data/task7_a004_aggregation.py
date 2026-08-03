"""A-004 per-cap summaries, comparisons, and explicitly staged pair unions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError,
    SimilarityAuditPolicy,
)
from protein_lm.data.similarity_fastas import MaterializedInputs
from protein_lm.data.similarity_manifests import StrategyManifest, metadata_by_partition
from protein_lm.data.task7_a003_import import A003Import
from protein_lm.data.task7_a003_stages import ImportedStage
from protein_lm.data.task7_a004_database import A004Database
from protein_lm.data.task7_a004_evidence import (
    CapSummary,
    StoredPairUnion,
    ensure_cap_summary,
    ensure_pair_union,
)
from protein_lm.data.task7_a004_plan import PlannedTrack, fixed_budget_stage_plan
from protein_lm.data.task7_fixed_budget import CapComparison, compare_caps
from protein_lm.data.task7_fixed_budget_contract import FixedBudgetStage, SearchRunner
from protein_lm.data.task7_fixed_budget_execution import ensure_fixed_budget_pass
from protein_lm.data.task7_inputs import fasta_path
from protein_lm.data.task7_pair_union import PairUnionComparison, compare_pair_unions
from protein_lm.data.task7_checkpoints import file_identity


@dataclass(frozen=True)
class TrackEvidence:
    """Stages, cap summaries, and cap sensitivity for one audit pass."""

    plan: PlannedTrack
    all_query_ids: tuple[str, ...]
    changed_query_ids: tuple[str, ...]
    stages: tuple[FixedBudgetStage | ImportedStage, ...]
    query_fasta: Path
    summaries: Mapping[int, CapSummary]
    comparisons: tuple[CapComparison, ...]
    stage_marker_identities: Mapping[int, Mapping[str, object]]
    pass_marker_path: Path | None
    pass_marker_identity: Mapping[str, object] | None

    @property
    def summary_paths(self) -> dict[int, Path]:
        """Return cap-summary directories without discarding their evidence."""

        return {cap: summary.directory for cap, summary in self.summaries.items()}


@dataclass(frozen=True)
class PairUnionBundle:
    """Separate all-query common-cap and staged-union pair evidence."""

    common_all_query_10000: StoredPairUnion
    staged_union_with_changed_query_100000: StoredPairUnion
    comparison: PairUnionComparison


def run_tracks(
    *,
    project_root: Path,
    workspace: Path,
    source_workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
    manifests: Mapping[str, StrategyManifest],
    inputs: MaterializedInputs,
    imported: A003Import,
    databases: Mapping[str, A004Database],
    search_runner: SearchRunner | None,
) -> dict[tuple[str, str, str], TrackEvidence]:
    """Import one historical residual pass and execute exactly seven fresh passes."""

    tracks: dict[tuple[str, str, str], TrackEvidence] = {}
    for plan in fixed_budget_stage_plan():
        query_metadata = metadata_by_partition(manifests[plan.strategy], plan.partition)
        if plan.origin == "imported_a003":
            stages = imported.stages
            changed = imported.escalated_query_ids
            query_fasta = source_workspace / "fastas" / "random_validation.fasta"
        else:
            completed = ensure_fixed_budget_pass(
                strategy=plan.strategy,
                partition=plan.partition,
                pass_name=plan.pass_name,
                query_fasta=fasta_path(workspace, plan.strategy, plan.partition),
                query_fasta_evidence=inputs.fastas[plan.strategy][plan.partition],
                query_metadata=query_metadata,
                target_database=databases[plan.strategy].prefix,
                target_database_identity=databases[plan.strategy].identity,
                target_metadata=metadata_by_partition(manifests[plan.strategy], "training"),
                project_root=project_root,
                workspace=workspace,
                policy=policy,
                fingerprint=fingerprint,
                command_runner=search_runner,
            )
            stages = completed.stages
            changed = completed.changed_query_ids
            query_fasta = fasta_path(workspace, plan.strategy, plan.partition)
        tracks[(plan.strategy, plan.partition, plan.pass_name)] = summarize_track(
            plan=plan,
            stages=stages,
            all_query_ids=tuple(sorted(query_metadata)),
            changed_query_ids=changed,
            query_fasta=query_fasta,
            workspace=workspace,
            fingerprint=fingerprint,
        )
    return tracks


def summarize_track(
    *,
    plan: PlannedTrack,
    stages: tuple[FixedBudgetStage | ImportedStage, ...],
    all_query_ids: tuple[str, ...],
    changed_query_ids: tuple[str, ...],
    query_fasta: Path,
    workspace: Path,
    fingerprint: str,
) -> TrackEvidence:
    """Summarize every executed cap without treating 100k as all-query evidence."""

    by_cap = {stage.cap: stage for stage in stages}
    expected_caps = {1_000, 10_000, 100_000} if changed_query_ids else {1_000, 10_000}
    if set(by_cap) != expected_caps:
        raise SimilarityAuditError("A-004 fixed-budget stages do not reconcile")
    summaries: dict[int, CapSummary] = {}
    for cap, stage in by_cap.items():
        expected_ids = changed_query_ids if cap == 100_000 else all_query_ids
        cap_query_fasta = (
            query_fasta
            if cap != 100_000
            else stage.canonical_path.parent.parent / "escalated_queries.fasta"
        )
        summaries[cap] = ensure_cap_summary(
            source_label=plan.origin,
            cap=cap,
            canonical_path=stage.canonical_path,
            canonical_evidence=stage.canonical,
            query_fasta=cap_query_fasta,
            query_fasta_evidence=stage.query_fasta,
            expected_query_ids=expected_ids,
            output_directory=(
                workspace
                / "evidence"
                / plan.origin
                / plan.strategy
                / plan.partition
                / plan.pass_name
                / f"cap_{cap}"
            ),
            fingerprint=fingerprint,
        )
    common_change = compare_caps(
        baseline_cap=1_000,
        comparison_cap=10_000,
        baseline_canonical_path=by_cap[1_000].canonical_path,
        comparison_canonical_path=by_cap[10_000].canonical_path,
        baseline_summary_path=summaries[1_000].directory / "query_summaries.tsv",
        comparison_summary_path=summaries[10_000].directory / "query_summaries.tsv",
        expected_query_ids=all_query_ids,
    )
    if common_change.complete_row_change_query_ids != changed_query_ids:
        raise SimilarityAuditError("A-004 100k query set differs from full-row cap changes")
    comparisons = [common_change]
    if changed_query_ids:
        comparisons.append(
            compare_caps(
                baseline_cap=10_000,
                comparison_cap=100_000,
                baseline_canonical_path=by_cap[10_000].canonical_path,
                comparison_canonical_path=by_cap[100_000].canonical_path,
                baseline_summary_path=summaries[10_000].directory / "query_summaries.tsv",
                comparison_summary_path=summaries[100_000].directory / "query_summaries.tsv",
                expected_query_ids=changed_query_ids,
                baseline_contains_other_queries=True,
            )
        )
    return TrackEvidence(
        plan=plan,
        all_query_ids=all_query_ids,
        changed_query_ids=changed_query_ids,
        stages=tuple(by_cap[cap] for cap in sorted(by_cap)),
        query_fasta=query_fasta,
        summaries=summaries,
        comparisons=tuple(comparisons),
        stage_marker_identities={
            stage.cap: (
                asdict(stage.marker)
                if isinstance(stage, ImportedStage)
                else file_identity(stage.marker_path)
            )
            for stage in stages
        },
        pass_marker_path=(
            None
            if isinstance(stages[0], ImportedStage)
            else stages[0].marker_path.parent.parent / "complete.json"
        ),
        pass_marker_identity=(
            None
            if isinstance(stages[0], ImportedStage)
            else file_identity(stages[0].marker_path.parent.parent / "complete.json")
        ),
    )


def build_pair_unions(
    *,
    workspace: Path,
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
    fingerprint: str,
) -> dict[tuple[str, str], PairUnionBundle]:
    """Keep common all-query 10k unions distinct from staged 100k additions."""

    bundles: dict[tuple[str, str], PairUnionBundle] = {}
    for strategy in ("random", "group_aware"):
        for partition in ("validation", "test"):
            key = (strategy, partition)
            passes = {name: tracks[(strategy, partition, name)] for name in ("enforcement", "residual")}
            common_sources = {
                f"{name}_{item.plan.origin}_cap_{cap}": item.summary_paths[cap]
                / "prohibited_pairs.tsv"
                for name, item in passes.items()
                for cap in (1_000, 10_000)
            }
            root = workspace / "pair_unions" / strategy / partition
            common = ensure_pair_union(
                label=f"common_all_query_10000_{strategy}_{partition}",
                source_paths=common_sources,
                output_directory=root / "common_all_query_10000",
                fingerprint=fingerprint,
            )
            staged_sources = {
                "common_all_query_10000": common.directory / "prohibited_pairs.tsv"
            }
            for name, item in passes.items():
                if 100_000 in item.summaries:
                    staged_sources[f"{name}_{item.plan.origin}_cap_100000"] = (
                        item.summary_paths[100_000] / "prohibited_pairs.tsv"
                    )
            staged = ensure_pair_union(
                label=(
                    "staged_union_with_changed_query_100000_"
                    f"{strategy}_{partition}"
                ),
                source_paths=staged_sources,
                output_directory=root / "staged_union_with_changed_query_100000",
                fingerprint=fingerprint,
            )
            bundles[key] = PairUnionBundle(
                common_all_query_10000=common,
                staged_union_with_changed_query_100000=staged,
                comparison=compare_pair_unions(
                    common_path=common.directory / "prohibited_pairs.tsv",
                    staged_path=staged.directory / "prohibited_pairs.tsv",
                ),
            )
    return bundles
