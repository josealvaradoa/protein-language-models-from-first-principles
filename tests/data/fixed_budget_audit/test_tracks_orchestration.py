"""Synthetic orchestration tests for imported tracks and pair-union labels."""

from pathlib import Path
from types import SimpleNamespace

import protein_lm.data.fixed_budget_audit.tracks as tracks_module
import pytest
from protein_lm.data.fixed_budget_audit.config import (
    AuditPass,
    CandidateCap,
    DatasetPartition,
    SplitStrategy,
    TrackOrigin,
)
from protein_lm.data.fixed_budget_audit.evidence import CapComparison
from protein_lm.data.fixed_budget_audit.errors import (
    AuditConfigurationError,
    AuditValidationError,
)
from protein_lm.data.fixed_budget_audit.search import FixedBudgetStage
from protein_lm.data.fixed_budget_audit.tracks import (
    TrackEvidence,
    build_pair_unions,
    fixed_budget_stage_plan,
    run_tracks,
    summarize_track,
)
from protein_lm.data.similarity_audit_models import FileEvidence
from protein_lm.data.similarity_fastas import FastaEvidence


def test_run_tracks_imports_one_track_without_calling_the_executor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    execution_calls = []
    summary_calls = []

    def runner(*args):
        return "unused"

    def fake_executor(**kwargs):
        execution_calls.append(kwargs)
        return SimpleNamespace(
            stages=(f"executed-{kwargs['pass_name'].value}",),
            changed_query_ids=(),
        )

    def fake_summary(**kwargs):
        summary_calls.append(kwargs)
        return kwargs["plan"]

    monkeypatch.setattr(tracks_module, "ensure_fixed_budget_pass", fake_executor)
    monkeypatch.setattr(tracks_module, "summarize_track", fake_summary)
    monkeypatch.setattr(
        tracks_module,
        "metadata_by_partition",
        lambda manifest, partition: {f"{manifest.strategy}-{partition}": object()},
    )
    strategies = tuple(strategy.value for strategy in SplitStrategy)
    partitions = tuple(partition.value for partition in DatasetPartition)
    manifests = {
        strategy: SimpleNamespace(strategy=strategy) for strategy in strategies
    }
    inputs = SimpleNamespace(
        fastas={
            strategy: {partition: object() for partition in partitions}
            for strategy in strategies
        }
    )
    databases = {
        strategy: SimpleNamespace(
            strategy=strategy,
            prefix=tmp_path / "databases" / strategy / "target",
            identity={"strategy": strategy},
        )
        for strategy in strategies
    }
    imported = SimpleNamespace(
        stages=("imported-stage",),
        escalated_query_ids=(),
    )

    observed = run_tracks(
        project_root=tmp_path,
        workspace=tmp_path / "a004",
        source_workspace=tmp_path / "a003",
        policy=object(),
        fingerprint="synthetic-fingerprint",
        manifests=manifests,
        inputs=inputs,
        imported=imported,
        databases=databases,
        search_runner=runner,
    )

    assert tuple(observed) == tuple(
        (item.strategy.value, item.partition.value, item.pass_name.value)
        for item in fixed_budget_stage_plan()
    )
    assert len(execution_calls) == 7
    assert tuple(
        (
            call["strategy"],
            call["partition"],
            call["pass_name"],
        )
        for call in execution_calls
    ) == tuple(
        (item.strategy, item.partition, item.pass_name)
        for item in fixed_budget_stage_plan()
        if item.origin is TrackOrigin.EXECUTED_A004
    )
    assert all(call["command_runner"] is runner for call in execution_calls)
    assert summary_calls[0]["plan"].origin is TrackOrigin.IMPORTED_A003
    assert summary_calls[0]["stages"] == imported.stages
    assert all(
        call["plan"].origin is TrackOrigin.EXECUTED_A004 for call in summary_calls[1:]
    )


def test_build_pair_unions_uses_exact_common_and_staged_wire_labels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    union_calls = []
    comparison = SimpleNamespace(kind="comparison")

    def fake_union(**kwargs):
        union_calls.append(kwargs)
        return SimpleNamespace(directory=kwargs["output_directory"])

    monkeypatch.setattr(tracks_module, "ensure_pair_union", fake_union)
    monkeypatch.setattr(
        tracks_module,
        "compare_pair_unions",
        lambda **kwargs: comparison,
    )
    tracks = {}
    for plan in fixed_budget_stage_plan():
        changed = (
            ("Q1",)
            if plan.origin is TrackOrigin.EXECUTED_A004
            and plan.pass_name is AuditPass.ENFORCEMENT
            else ()
        )
        caps = {
            CandidateCap.INITIAL.value,
            CandidateCap.COMPARISON.value,
        } | ({CandidateCap.ESCALATION.value} if changed else set())
        summaries = {
            cap: SimpleNamespace(
                directory=(
                    tmp_path
                    / "summaries"
                    / plan.origin.value
                    / plan.strategy.value
                    / plan.partition.value
                    / plan.pass_name.value
                    / f"cap_{cap}"
                )
            )
            for cap in caps
        }
        key = (plan.strategy.value, plan.partition.value, plan.pass_name.value)
        tracks[key] = TrackEvidence(
            plan=plan,
            all_query_ids=("Q1",),
            changed_query_ids=changed,
            stages=(),
            query_fasta=tmp_path / "queries.fasta",
            summaries=summaries,
            comparisons=(),
            stage_marker_identities={},
            pass_marker_path=None,
            pass_marker_identity=None,
        )

    bundles = build_pair_unions(
        workspace=tmp_path / "a004",
        tracks=tracks,
        fingerprint="synthetic-fingerprint",
    )

    assert tuple(bundles) == (
        (SplitStrategy.RANDOM.value, DatasetPartition.VALIDATION.value),
        (SplitStrategy.RANDOM.value, DatasetPartition.TEST.value),
        (SplitStrategy.GROUP_AWARE.value, DatasetPartition.VALIDATION.value),
        (SplitStrategy.GROUP_AWARE.value, DatasetPartition.TEST.value),
    )
    assert [call["label"] for call in union_calls] == [
        "common_all_query_10000_random_validation",
        "staged_union_with_changed_query_100000_random_validation",
        "common_all_query_10000_random_test",
        "staged_union_with_changed_query_100000_random_test",
        "common_all_query_10000_group_aware_validation",
        "staged_union_with_changed_query_100000_group_aware_validation",
        "common_all_query_10000_group_aware_test",
        "staged_union_with_changed_query_100000_group_aware_test",
    ]
    assert tuple(union_calls[0]["source_paths"]) == (
        "enforcement_executed_a004_cap_1000",
        "enforcement_executed_a004_cap_10000",
        "residual_imported_a003_cap_1000",
        "residual_imported_a003_cap_10000",
    )
    assert tuple(union_calls[1]["source_paths"]) == (
        "common_all_query_10000",
        "enforcement_executed_a004_cap_100000",
    )


def test_summarize_track_types_topology_and_reconciliation_failures(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = fixed_budget_stage_plan()[1]
    initial = _stage(tmp_path, CandidateCap.INITIAL.value)
    comparison = _stage(tmp_path, CandidateCap.COMPARISON.value)
    escalation = _stage(tmp_path, CandidateCap.ESCALATION.value)

    with pytest.raises(
        AuditConfigurationError,
        match="query topology is invalid",
    ):
        summarize_track(
            plan=plan,
            stages=(initial, comparison),
            all_query_ids=("Q2", "Q1"),
            changed_query_ids=(),
            query_fasta=tmp_path / "queries.fasta",
            workspace=tmp_path / "workspace",
            fingerprint="synthetic-fingerprint",
        )
    with pytest.raises(
        AuditValidationError,
        match="stages do not reconcile",
    ):
        summarize_track(
            plan=plan,
            stages=(initial, initial, comparison),
            all_query_ids=("Q1",),
            changed_query_ids=(),
            query_fasta=tmp_path / "queries.fasta",
            workspace=tmp_path / "workspace",
            fingerprint="synthetic-fingerprint",
        )

    monkeypatch.setattr(
        tracks_module,
        "ensure_cap_summary",
        lambda **kwargs: SimpleNamespace(directory=kwargs["output_directory"]),
    )
    monkeypatch.setattr(
        tracks_module,
        "compare_caps",
        lambda **kwargs: CapComparison(
            baseline_cap=kwargs["baseline_cap"],
            comparison_cap=kwargs["comparison_cap"],
            compared_queries=1,
            complete_row_change_query_ids=(),
            complete_row_changes=0,
            newly_prohibited_queries=0,
            no_longer_prohibited_queries=0,
            closest_category_changes=0,
        ),
    )
    with pytest.raises(
        AuditValidationError,
        match="100k query set differs",
    ):
        summarize_track(
            plan=plan,
            stages=(initial, comparison, escalation),
            all_query_ids=("Q1",),
            changed_query_ids=("Q1",),
            query_fasta=tmp_path / "queries.fasta",
            workspace=tmp_path / "workspace",
            fingerprint="synthetic-fingerprint",
        )


def _stage(tmp_path: Path, cap: int) -> FixedBudgetStage:
    return FixedBudgetStage(
        cap=cap,
        query_fasta=FastaEvidence(1, 4, 0, "0" * 64),
        canonical=FileEvidence(0, 0, "0" * 64),
        canonical_path=tmp_path / "tracks" / f"cap_{cap}" / "canonical.tsv",
        command=(),
        runtime_seconds="0.01",
        marker_path=tmp_path / "tracks" / f"cap_{cap}" / "complete.json",
    )
