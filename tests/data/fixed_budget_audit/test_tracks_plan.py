"""Closed vocabulary and exact ordering for the fixed eight-track plan."""

import protein_lm.data.fixed_budget_audit.tracks as tracks_module

from protein_lm.data.fixed_budget_audit.config import (
    AuditPass,
    DatasetPartition,
    PairUnionKind,
    QueryScope,
    SplitStrategy,
    TrackOrigin,
)
from protein_lm.data.fixed_budget_audit.tracks import fixed_budget_stage_plan


def test_shared_enums_have_config_as_their_only_public_owner() -> None:
    for enum_type in (TrackOrigin, PairUnionKind, QueryScope):
        assert enum_type.__module__ == "protein_lm.data.fixed_budget_audit.config"
    for enum_name in ("TrackOrigin", "PairUnionKind", "QueryScope"):
        assert not hasattr(tracks_module, enum_name)


def test_fixed_budget_plan_has_exact_enum_identity_and_order() -> None:
    plan = fixed_budget_stage_plan()

    assert tuple(
        (item.strategy, item.partition, item.pass_name, item.origin) for item in plan
    ) == (
        (
            SplitStrategy.RANDOM,
            DatasetPartition.VALIDATION,
            AuditPass.RESIDUAL,
            TrackOrigin.IMPORTED_A003,
        ),
        (
            SplitStrategy.RANDOM,
            DatasetPartition.VALIDATION,
            AuditPass.ENFORCEMENT,
            TrackOrigin.EXECUTED_A004,
        ),
        (
            SplitStrategy.RANDOM,
            DatasetPartition.TEST,
            AuditPass.ENFORCEMENT,
            TrackOrigin.EXECUTED_A004,
        ),
        (
            SplitStrategy.RANDOM,
            DatasetPartition.TEST,
            AuditPass.RESIDUAL,
            TrackOrigin.EXECUTED_A004,
        ),
        (
            SplitStrategy.GROUP_AWARE,
            DatasetPartition.VALIDATION,
            AuditPass.ENFORCEMENT,
            TrackOrigin.EXECUTED_A004,
        ),
        (
            SplitStrategy.GROUP_AWARE,
            DatasetPartition.VALIDATION,
            AuditPass.RESIDUAL,
            TrackOrigin.EXECUTED_A004,
        ),
        (
            SplitStrategy.GROUP_AWARE,
            DatasetPartition.TEST,
            AuditPass.ENFORCEMENT,
            TrackOrigin.EXECUTED_A004,
        ),
        (
            SplitStrategy.GROUP_AWARE,
            DatasetPartition.TEST,
            AuditPass.RESIDUAL,
            TrackOrigin.EXECUTED_A004,
        ),
    )

    assert tuple(
        (
            item.strategy.value,
            item.partition.value,
            item.pass_name.value,
            item.origin.value,
        )
        for item in plan
    ) == (
        ("random", "validation", "residual", "imported_a003"),
        ("random", "validation", "enforcement", "executed_a004"),
        ("random", "test", "enforcement", "executed_a004"),
        ("random", "test", "residual", "executed_a004"),
        ("group_aware", "validation", "enforcement", "executed_a004"),
        ("group_aware", "validation", "residual", "executed_a004"),
        ("group_aware", "test", "enforcement", "executed_a004"),
        ("group_aware", "test", "residual", "executed_a004"),
    )


def test_track_wire_vocabulary_keeps_exact_external_strings() -> None:
    assert TrackOrigin.IMPORTED_A003.value == "imported_a003"
    assert TrackOrigin.EXECUTED_A004.value == "executed_a004"
    assert PairUnionKind.COMMON_ALL_QUERY_10000.value == "common_all_query_10000"
    assert PairUnionKind.STAGED_UNION_WITH_CHANGED_QUERY_100000.value == (
        "staged_union_with_changed_query_100000"
    )
    assert QueryScope.ALL_QUERIES.value == "all_queries"
    assert QueryScope.CHANGED_QUERIES_1000_TO_10000.value == (
        "changed_queries_1000_to_10000"
    )
