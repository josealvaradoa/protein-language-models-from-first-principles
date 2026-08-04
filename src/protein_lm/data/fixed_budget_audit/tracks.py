"""Frozen track plan, execution, summaries, and pair-union orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from protein_lm.data.artifacts import (
    file_identity,
    read_json,
    require_marker_identity,
    write_json_atomic,
)
from protein_lm.data.fixed_budget_audit.config import (
    AuditPass,
    CandidateCap,
    DatasetPartition,
    PairUnionKind as _PairUnionKind,
    SplitStrategy,
    TrackOrigin as _TrackOrigin,
)
from protein_lm.data.fixed_budget_audit.evidence import (
    CapComparison,
    CapSummary,
    PairUnionComparison,
    StoredPairUnion,
    compare_caps,
    compare_pair_unions,
    ensure_cap_summary,
    ensure_pair_union,
)
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    AuditExecutionError,
    AuditValidationError,
)
from protein_lm.data.fixed_budget_audit.search import (
    A004Database,
    FixedBudgetStage,
    SearchRunner,
    ensure_escalation_fasta,
    ensure_search_stage,
    pass_marker,
    require_fixed_policy_caps,
    verify_query_fasta,
)
from protein_lm.data.fixed_budget_audit.source import (
    A003Import,
    ImportedStage,
    fasta_path,
)
from protein_lm.data.similarity_audit_models import SequenceMetadata
from protein_lm.data.similarity_audit_policy import SimilarityAuditPolicy
from protein_lm.data.similarity_fastas import FastaEvidence, MaterializedInputs
from protein_lm.data.similarity_manifests import StrategyManifest, metadata_by_partition
from protein_lm.data.similarity_results import compare_canonical_results


# Closed track topology


_TRACK_TOPOLOGY = (
    (
        SplitStrategy.RANDOM,
        DatasetPartition.VALIDATION,
        AuditPass.RESIDUAL,
        _TrackOrigin.IMPORTED_A003,
    ),
    (
        SplitStrategy.RANDOM,
        DatasetPartition.VALIDATION,
        AuditPass.ENFORCEMENT,
        _TrackOrigin.EXECUTED_A004,
    ),
    (
        SplitStrategy.RANDOM,
        DatasetPartition.TEST,
        AuditPass.ENFORCEMENT,
        _TrackOrigin.EXECUTED_A004,
    ),
    (
        SplitStrategy.RANDOM,
        DatasetPartition.TEST,
        AuditPass.RESIDUAL,
        _TrackOrigin.EXECUTED_A004,
    ),
    (
        SplitStrategy.GROUP_AWARE,
        DatasetPartition.VALIDATION,
        AuditPass.ENFORCEMENT,
        _TrackOrigin.EXECUTED_A004,
    ),
    (
        SplitStrategy.GROUP_AWARE,
        DatasetPartition.VALIDATION,
        AuditPass.RESIDUAL,
        _TrackOrigin.EXECUTED_A004,
    ),
    (
        SplitStrategy.GROUP_AWARE,
        DatasetPartition.TEST,
        AuditPass.ENFORCEMENT,
        _TrackOrigin.EXECUTED_A004,
    ),
    (
        SplitStrategy.GROUP_AWARE,
        DatasetPartition.TEST,
        AuditPass.RESIDUAL,
        _TrackOrigin.EXECUTED_A004,
    ),
)
_TRACK_KEYS = tuple(
    (strategy.value, partition.value, pass_name.value)
    for strategy, partition, pass_name, _ in _TRACK_TOPOLOGY
)
_TRACK_ORIGINS = {
    (strategy.value, partition.value, pass_name.value): origin
    for strategy, partition, pass_name, origin in _TRACK_TOPOLOGY
}
_EXECUTED_TRACK_KEYS = frozenset(
    (strategy.value, partition.value, pass_name.value)
    for strategy, partition, pass_name, origin in _TRACK_TOPOLOGY
    if origin is _TrackOrigin.EXECUTED_A004
)
_COMMON_CAPS = (
    CandidateCap.INITIAL.value,
    CandidateCap.COMPARISON.value,
)


# Track records and fixed plan


@dataclass(frozen=True)
class PlannedTrack:
    """One imported or freshly executed fixed-budget pass."""

    strategy: SplitStrategy
    partition: DatasetPartition
    pass_name: AuditPass
    origin: _TrackOrigin


@dataclass(frozen=True)
class FixedBudgetPass:
    """All completed stages for one immutable strategy, partition, and pass."""

    strategy: SplitStrategy
    partition: DatasetPartition
    pass_name: AuditPass
    all_query_ids: tuple[str, ...]
    changed_query_ids: tuple[str, ...]
    escalation_fasta: FastaEvidence | None
    stages: tuple[FixedBudgetStage, ...]
    marker_path: Path

    def stage(self, cap: int) -> FixedBudgetStage:
        """Return a completed stage by its frozen cap."""

        try:
            return next(stage for stage in self.stages if stage.cap == cap)
        except StopIteration as error:
            raise AuditConfigurationError(
                f"fixed-budget cap is unavailable: {cap}"
            ) from error


def fixed_budget_stage_plan() -> tuple[PlannedTrack, ...]:
    """Return the one A-003 import plus seven fresh A-004 passes."""

    return tuple(
        PlannedTrack(
            strategy=strategy,
            partition=partition,
            pass_name=pass_name,
            origin=origin,
        )
        for strategy, partition, pass_name, origin in _TRACK_TOPOLOGY
    )


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


# Per-track fixed-budget execution


def ensure_fixed_budget_pass(
    *,
    strategy: SplitStrategy,
    partition: DatasetPartition,
    pass_name: AuditPass,
    query_fasta: Path,
    query_fasta_evidence: FastaEvidence,
    query_metadata: Mapping[str, SequenceMetadata],
    target_database: Path,
    target_database_identity: Mapping[str, object],
    target_metadata: Mapping[str, SequenceMetadata],
    project_root: Path,
    workspace: Path,
    policy: SimilarityAuditPolicy,
    fingerprint: str,
    command_runner: SearchRunner | None = None,
) -> FixedBudgetPass:
    """Run 1k and 10k for all queries, then 100k only for changed rows.

    A-004 records any 10k-to-100k difference as cap sensitivity. It never
    requires convergence and retains every completed canonical TSV.
    """

    _require_executed_track(strategy, partition, pass_name)
    require_fixed_policy_caps(policy)
    all_query_ids = verify_query_fasta(
        query_fasta, query_fasta_evidence, query_metadata
    )
    strategy_value = strategy.value
    partition_value = partition.value
    pass_value = pass_name.value
    pass_directory = (
        workspace / "tracks" / strategy_value / partition_value / pass_value
    )
    marker_path = pass_directory / "complete.json"
    resumed_caps: frozenset[int] | None = None
    if marker_path.exists():
        marker = read_json(marker_path)
        require_marker_identity(marker, fingerprint, "a004_fixed_budget_pass")
        resumed_caps = _resumed_cap_inventory(marker, pass_directory)
    pass_directory.mkdir(parents=True, exist_ok=True)
    common_kwargs = {
        "strategy": strategy_value,
        "partition": partition_value,
        "pass_name": pass_value,
        "target_database": target_database,
        "target_database_identity": target_database_identity,
        "target_metadata": target_metadata,
        "pass_directory": pass_directory,
        "project_root": project_root,
        "workspace": workspace,
        "policy": policy,
        "fingerprint": fingerprint,
        "command_runner": command_runner,
    }
    first = ensure_search_stage(
        cap=CandidateCap.INITIAL.value,
        query_fasta=query_fasta,
        query_fasta_evidence=query_fasta_evidence,
        query_ids=all_query_ids,
        query_metadata=query_metadata,
        **common_kwargs,
    )
    comparison = ensure_search_stage(
        cap=CandidateCap.COMPARISON.value,
        query_fasta=query_fasta,
        query_fasta_evidence=query_fasta_evidence,
        query_ids=all_query_ids,
        query_metadata=query_metadata,
        **common_kwargs,
    )
    changed = compare_canonical_results(
        first.canonical_path,
        comparison.canonical_path,
        expected_query_ids=all_query_ids,
    )
    escalation_fasta: FastaEvidence | None = None
    escalation_marker: Path | None = None
    stages = [first, comparison]
    if changed:
        if (
            resumed_caps is not None
            and CandidateCap.ESCALATION.value not in resumed_caps
        ):
            raise AuditValidationError("A-004 fixed-budget pass identity drifted")
        escalation_path, escalation_fasta, escalation_marker = ensure_escalation_fasta(
            pass_directory=pass_directory,
            source_fasta=query_fasta,
            source_evidence=query_fasta_evidence,
            source_query_ids=all_query_ids,
            changed_query_ids=changed,
            fingerprint=fingerprint,
        )
        stages.append(
            ensure_search_stage(
                cap=CandidateCap.ESCALATION.value,
                query_fasta=escalation_path,
                query_fasta_evidence=escalation_fasta,
                query_ids=changed,
                query_metadata={key: query_metadata[key] for key in changed},
                **common_kwargs,
            )
        )
    elif (pass_directory / "escalated_queries.fasta").exists() or (
        pass_directory / "escalated_queries.complete.json"
    ).exists():
        error_type = (
            AuditValidationError if resumed_caps is not None else AuditExecutionError
        )
        raise error_type("A-004 has an unexpected escalation FASTA")
    expected_cap_directories = {f"cap_{stage.cap}" for stage in stages}
    actual_cap_outputs = {
        path.name for path in pass_directory.iterdir() if path.name.startswith("cap_")
    }
    if actual_cap_outputs != expected_cap_directories:
        error_type = (
            AuditValidationError if resumed_caps is not None else AuditExecutionError
        )
        raise error_type("A-004 fixed-budget cap inventory drifted")
    marker = pass_marker(
        fingerprint=fingerprint,
        strategy=strategy_value,
        partition=partition_value,
        pass_name=pass_value,
        query_fasta=query_fasta_evidence,
        query_ids=all_query_ids,
        target_database=target_database,
        target_database_identity=target_database_identity,
        changed_query_ids=changed,
        stages=tuple(stages),
        escalation_fasta=escalation_fasta,
        escalation_marker=escalation_marker,
    )
    if marker_path.exists():
        existing = read_json(marker_path)
        require_marker_identity(existing, fingerprint, "a004_fixed_budget_pass")
        if existing != marker:
            raise AuditValidationError("A-004 fixed-budget pass identity drifted")
    else:
        write_json_atomic(marker_path, marker)
    return FixedBudgetPass(
        strategy=strategy,
        partition=partition,
        pass_name=pass_name,
        all_query_ids=all_query_ids,
        changed_query_ids=changed,
        escalation_fasta=escalation_fasta,
        stages=tuple(stages),
        marker_path=marker_path,
    )


# Track summaries and pair-union orchestration


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

    _require_run_inputs(manifests, inputs, databases)
    tracks: dict[tuple[str, str, str], TrackEvidence] = {}
    for plan in fixed_budget_stage_plan():
        strategy = plan.strategy.value
        partition = plan.partition.value
        pass_name = plan.pass_name.value
        query_metadata = metadata_by_partition(manifests[strategy], partition)
        if plan.origin is _TrackOrigin.IMPORTED_A003:
            stages = imported.stages
            changed = imported.escalated_query_ids
            query_fasta = (
                source_workspace
                / "fastas"
                / (
                    f"{SplitStrategy.RANDOM.value}_"
                    f"{DatasetPartition.VALIDATION.value}.fasta"
                )
            )
        else:
            completed = ensure_fixed_budget_pass(
                strategy=plan.strategy,
                partition=plan.partition,
                pass_name=plan.pass_name,
                query_fasta=fasta_path(workspace, strategy, partition),
                query_fasta_evidence=inputs.fastas[strategy][partition],
                query_metadata=query_metadata,
                target_database=databases[strategy].prefix,
                target_database_identity=databases[strategy].identity,
                target_metadata=metadata_by_partition(manifests[strategy], "training"),
                project_root=project_root,
                workspace=workspace,
                policy=policy,
                fingerprint=fingerprint,
                command_runner=search_runner,
            )
            stages = completed.stages
            changed = completed.changed_query_ids
            query_fasta = fasta_path(workspace, strategy, partition)
        tracks[(strategy, partition, pass_name)] = summarize_track(
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

    _require_track_inputs(plan, stages, all_query_ids, changed_query_ids)
    initial_cap = CandidateCap.INITIAL.value
    comparison_cap = CandidateCap.COMPARISON.value
    escalation_cap = CandidateCap.ESCALATION.value
    by_cap = {stage.cap: stage for stage in stages}
    expected_caps = set(_COMMON_CAPS) | (
        {escalation_cap} if changed_query_ids else set()
    )
    if len(by_cap) != len(stages) or set(by_cap) != expected_caps:
        raise AuditValidationError("A-004 fixed-budget stages do not reconcile")
    summaries: dict[int, CapSummary] = {}
    for cap, stage in by_cap.items():
        expected_ids = changed_query_ids if cap == escalation_cap else all_query_ids
        cap_query_fasta = (
            query_fasta
            if cap != escalation_cap
            else stage.canonical_path.parent.parent / "escalated_queries.fasta"
        )
        summaries[cap] = ensure_cap_summary(
            source_label=plan.origin.value,
            cap=cap,
            canonical_path=stage.canonical_path,
            canonical_evidence=stage.canonical,
            query_fasta=cap_query_fasta,
            query_fasta_evidence=stage.query_fasta,
            expected_query_ids=expected_ids,
            output_directory=(
                workspace
                / "evidence"
                / plan.origin.value
                / plan.strategy.value
                / plan.partition.value
                / plan.pass_name.value
                / f"cap_{cap}"
            ),
            fingerprint=fingerprint,
        )
    common_change = compare_caps(
        baseline_cap=initial_cap,
        comparison_cap=comparison_cap,
        baseline_canonical_path=by_cap[initial_cap].canonical_path,
        comparison_canonical_path=by_cap[comparison_cap].canonical_path,
        baseline_summary_path=summaries[initial_cap].directory / "query_summaries.tsv",
        comparison_summary_path=summaries[comparison_cap].directory
        / "query_summaries.tsv",
        expected_query_ids=all_query_ids,
    )
    if common_change.complete_row_change_query_ids != changed_query_ids:
        raise AuditValidationError(
            "A-004 100k query set differs from full-row cap changes"
        )
    comparisons = [common_change]
    if changed_query_ids:
        comparisons.append(
            compare_caps(
                baseline_cap=comparison_cap,
                comparison_cap=escalation_cap,
                baseline_canonical_path=by_cap[comparison_cap].canonical_path,
                comparison_canonical_path=by_cap[escalation_cap].canonical_path,
                baseline_summary_path=summaries[comparison_cap].directory
                / "query_summaries.tsv",
                comparison_summary_path=summaries[escalation_cap].directory
                / "query_summaries.tsv",
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
            if plan.origin is _TrackOrigin.IMPORTED_A003
            else stages[0].marker_path.parent.parent / "complete.json"
        ),
        pass_marker_identity=(
            None
            if plan.origin is _TrackOrigin.IMPORTED_A003
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

    _require_track_inventory(tracks)
    escalation_cap = CandidateCap.ESCALATION.value
    common_kind = _PairUnionKind.COMMON_ALL_QUERY_10000.value
    staged_kind = _PairUnionKind.STAGED_UNION_WITH_CHANGED_QUERY_100000.value
    bundles: dict[tuple[str, str], PairUnionBundle] = {}
    for strategy_member in (SplitStrategy.RANDOM, SplitStrategy.GROUP_AWARE):
        strategy = strategy_member.value
        for partition_member in (
            DatasetPartition.VALIDATION,
            DatasetPartition.TEST,
        ):
            partition = partition_member.value
            key = (strategy, partition)
            passes = {
                pass_name.value: tracks[(strategy, partition, pass_name.value)]
                for pass_name in (AuditPass.ENFORCEMENT, AuditPass.RESIDUAL)
            }
            common_sources = {
                f"{name}_{item.plan.origin.value}_cap_{cap}": item.summary_paths[cap]
                / "prohibited_pairs.tsv"
                for name, item in passes.items()
                for cap in _COMMON_CAPS
            }
            root = workspace / "pair_unions" / strategy / partition
            common = ensure_pair_union(
                label=f"{common_kind}_{strategy}_{partition}",
                source_paths=common_sources,
                output_directory=root / common_kind,
                fingerprint=fingerprint,
            )
            staged_sources = {common_kind: common.directory / "prohibited_pairs.tsv"}
            for name, item in passes.items():
                if escalation_cap in item.summaries:
                    staged_sources[
                        f"{name}_{item.plan.origin.value}_cap_{escalation_cap}"
                    ] = item.summary_paths[escalation_cap] / "prohibited_pairs.tsv"
            staged = ensure_pair_union(
                label=f"{staged_kind}_{strategy}_{partition}",
                source_paths=staged_sources,
                output_directory=root / staged_kind,
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


# Topology and resume contracts


def _require_executed_track(
    strategy: SplitStrategy,
    partition: DatasetPartition,
    pass_name: AuditPass,
) -> None:
    if (
        not isinstance(strategy, SplitStrategy)
        or not isinstance(partition, DatasetPartition)
        or not isinstance(pass_name, AuditPass)
        or (strategy.value, partition.value, pass_name.value)
        not in _EXECUTED_TRACK_KEYS
    ):
        raise AuditConfigurationError(
            "A-004 track is outside the fixed execution topology"
        )


def _resumed_cap_inventory(
    marker: Mapping[str, object],
    pass_directory: Path,
) -> frozenset[int]:
    stages = marker.get("stages")
    common = {str(cap) for cap in _COMMON_CAPS}
    allowed = common | {str(CandidateCap.ESCALATION.value)}
    if (
        not isinstance(stages, dict)
        or not common <= set(stages)
        or not set(stages) <= allowed
    ):
        raise AuditValidationError("A-004 fixed-budget pass identity drifted")
    caps = frozenset(int(cap) for cap in stages)
    expected_directories = {f"cap_{cap}" for cap in caps}
    actual_directories = {
        path.name for path in pass_directory.iterdir() if path.name.startswith("cap_")
    }
    if actual_directories != expected_directories:
        raise AuditValidationError("A-004 fixed-budget cap inventory drifted")
    if any(
        not (pass_directory / f"cap_{cap}" / "complete.json").is_file() for cap in caps
    ):
        raise AuditValidationError("A-004 fixed-budget pass identity drifted")
    escalation_files_exist = (
        pass_directory / "escalated_queries.fasta"
    ).is_file() and (pass_directory / "escalated_queries.complete.json").is_file()
    has_escalation = CandidateCap.ESCALATION.value in caps
    if has_escalation:
        if not isinstance(marker.get("escalation"), dict) or not escalation_files_exist:
            raise AuditValidationError("A-004 fixed-budget pass identity drifted")
    elif marker.get("escalation") is not None:
        raise AuditValidationError("A-004 fixed-budget pass identity drifted")
    return caps


def _require_run_inputs(
    manifests: Mapping[str, StrategyManifest],
    inputs: MaterializedInputs,
    databases: Mapping[str, A004Database],
) -> None:
    strategies = {strategy.value for strategy in SplitStrategy}
    partitions = {partition.value for partition in DatasetPartition}
    if (
        set(manifests) != strategies
        or set(inputs.fastas) != strategies
        or set(databases) != strategies
        or any(
            not partitions <= set(inputs.fastas[strategy]) for strategy in strategies
        )
        or any(
            manifests[strategy].strategy != strategy
            or databases[strategy].strategy != strategy
            for strategy in strategies
        )
    ):
        raise AuditConfigurationError(
            "A-004 track inputs do not cover the fixed topology"
        )


def _require_track_inputs(
    plan: PlannedTrack,
    stages: tuple[FixedBudgetStage | ImportedStage, ...],
    all_query_ids: tuple[str, ...],
    changed_query_ids: tuple[str, ...],
) -> None:
    if (
        not isinstance(plan, PlannedTrack)
        or not isinstance(plan.strategy, SplitStrategy)
        or not isinstance(plan.partition, DatasetPartition)
        or not isinstance(plan.pass_name, AuditPass)
        or not isinstance(plan.origin, _TrackOrigin)
    ):
        raise AuditConfigurationError("A-004 track topology is invalid")
    key = (plan.strategy.value, plan.partition.value, plan.pass_name.value)
    if _TRACK_ORIGINS.get(key) is not plan.origin:
        raise AuditConfigurationError("A-004 track topology is invalid")
    if (
        not all_query_ids
        or all_query_ids != tuple(sorted(set(all_query_ids)))
        or changed_query_ids != tuple(sorted(set(changed_query_ids)))
        or not set(changed_query_ids) <= set(all_query_ids)
    ):
        raise AuditConfigurationError("A-004 track query topology is invalid")
    expected_stage_type = (
        ImportedStage if plan.origin is _TrackOrigin.IMPORTED_A003 else FixedBudgetStage
    )
    if stages and any(not isinstance(stage, expected_stage_type) for stage in stages):
        raise AuditConfigurationError("A-004 track stage topology is invalid")


def _require_track_inventory(
    tracks: Mapping[tuple[str, str, str], TrackEvidence],
) -> None:
    if set(tracks) != set(_TRACK_KEYS):
        raise AuditConfigurationError(
            "A-004 track inventory differs from the fixed topology"
        )
    for key in _TRACK_KEYS:
        track = tracks[key]
        if (
            not isinstance(track, TrackEvidence)
            or not isinstance(track.plan, PlannedTrack)
            or not isinstance(track.plan.strategy, SplitStrategy)
            or not isinstance(track.plan.partition, DatasetPartition)
            or not isinstance(track.plan.pass_name, AuditPass)
            or not isinstance(track.plan.origin, _TrackOrigin)
        ):
            raise AuditConfigurationError(
                "A-004 track inventory differs from the fixed topology"
            )
        plan_key = (
            track.plan.strategy.value,
            track.plan.partition.value,
            track.plan.pass_name.value,
        )
        expected_caps = set(_COMMON_CAPS) | (
            {CandidateCap.ESCALATION.value} if track.changed_query_ids else set()
        )
        if plan_key != key or _TRACK_ORIGINS.get(key) is not track.plan.origin:
            raise AuditConfigurationError(
                "A-004 track summary inventory differs from the fixed topology"
            )
        if set(track.summaries) != expected_caps:
            raise AuditValidationError(
                "A-004 track summary inventory differs from the fixed topology"
            )
