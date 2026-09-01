"""Focused synthetic checks for shared reproduction metric comparison."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import pytest

from protein_lm.reproduction.comparison import (
    ComparisonStage,
    CrossEntropyTolerances,
    MetricKey,
    MetricRecord,
    compare_metric_records,
)


TOLERANCES = CrossEntropyTolerances(reevaluation=0.000001, retraining=0.0001)


def metric(
    model_id: str = "c20",
    collection: str = "native_validation",
    seed: int | None = None,
    cross_entropy: float = 2.5,
    correct_predictions: int = 8,
    token_count: int = 10,
) -> MetricRecord:
    return MetricRecord(
        MetricKey(model_id, collection, seed),
        cross_entropy,
        correct_predictions,
        token_count,
    )


def test_exact_matches_pass_and_value_objects_are_immutable() -> None:
    expected = (metric(), metric("week2_bigram", "shared_validation"))
    report = compare_metric_records(
        expected,
        tuple(reversed(expected)),
        ComparisonStage.REEVALUATION,
        TOLERANCES,
    )

    assert report.passed is True
    assert report.issues == ()
    with pytest.raises(FrozenInstanceError):
        expected[0].key.model_id = "other"  # type: ignore[misc]


def test_empty_expected_inventory_returns_a_structured_failure() -> None:
    report = compare_metric_records(
        (), (), ComparisonStage.REEVALUATION, TOLERANCES
    )

    assert report.passed is False
    assert [issue.code for issue in report.issues] == ["EMPTY_EXPECTED_INVENTORY"]
    assert json.loads(report.to_json()) == report.to_dict()


def test_cross_entropy_tolerance_is_inclusive_and_stage_specific() -> None:
    expected = (metric(cross_entropy=1.0),)
    reevaluation_boundary = (metric(cross_entropy=1.000001),)
    retraining_boundary = (metric(cross_entropy=1.0001),)

    assert compare_metric_records(
        expected,
        reevaluation_boundary,
        ComparisonStage.REEVALUATION,
        TOLERANCES,
    ).passed
    assert compare_metric_records(
        expected,
        retraining_boundary,
        ComparisonStage.RETRAINING,
        TOLERANCES,
    ).passed

    report = compare_metric_records(
        expected,
        retraining_boundary,
        ComparisonStage.REEVALUATION,
        TOLERANCES,
    )
    assert report.passed is False
    assert [issue.code for issue in report.issues] == ["CROSS_ENTROPY_MISMATCH"]

    just_over = compare_metric_records(
        expected,
        (metric(cross_entropy=1.0000011),),
        ComparisonStage.REEVALUATION,
        TOLERANCES,
    )
    assert [issue.code for issue in just_over.issues] == ["CROSS_ENTROPY_MISMATCH"]


def test_counts_are_exact_even_when_cross_entropy_matches() -> None:
    report = compare_metric_records(
        (metric(),),
        (metric(correct_predictions=7, token_count=9),),
        ComparisonStage.REEVALUATION,
        TOLERANCES,
    )

    assert [issue.code for issue in report.issues] == [
        "CORRECT_PREDICTIONS_MISMATCH",
        "TOKEN_COUNT_MISMATCH",
    ]


def test_missing_extra_and_duplicate_keys_fail_closed() -> None:
    shared = metric()
    duplicate = metric(cross_entropy=2.6)
    report = compare_metric_records(
        (shared, duplicate, metric("expected_only")),
        (metric("observed_only"),),
        ComparisonStage.REEVALUATION,
        TOLERANCES,
    )

    assert report.passed is False
    assert {issue.code for issue in report.issues} == {
        "DUPLICATE_EXPECTED_KEY",
        "MISSING_OBSERVED_KEY",
        "UNEXPECTED_OBSERVED_KEY",
    }


@pytest.mark.parametrize(
    ("bad", "code"),
    [
        (metric(cross_entropy=math.nan), "INVALID_CROSS_ENTROPY"),
        (metric(cross_entropy=math.inf), "INVALID_CROSS_ENTROPY"),
        (metric(cross_entropy=-0.1), "INVALID_CROSS_ENTROPY"),
        (metric(cross_entropy=2), "INVALID_CROSS_ENTROPY"),
        (metric(cross_entropy=True), "INVALID_CROSS_ENTROPY"),
        (metric(correct_predictions=True), "INVALID_CORRECT_PREDICTIONS"),
        (metric(token_count=False), "INVALID_TOKEN_COUNT"),
        (metric(correct_predictions=-1), "INVALID_CORRECT_PREDICTIONS"),
        (metric(token_count=-1), "INVALID_TOKEN_COUNT"),
        (metric(correct_predictions=11), "CORRECT_PREDICTIONS_EXCEED_TOKEN_COUNT"),
        (metric(model_id=""), "INVALID_MODEL_ID"),
        (metric(collection=""), "INVALID_COLLECTION"),
        (metric(seed=True), "INVALID_SEED"),
    ],
)
def test_malformed_records_return_structured_failures(
    bad: MetricRecord, code: str
) -> None:
    report = compare_metric_records(
        (bad,),
        (metric(),),
        ComparisonStage.REEVALUATION,
        TOLERANCES,
    )

    assert report.passed is False
    assert code in {issue.code for issue in report.issues}


def test_seeded_and_seedless_keys_have_distinct_inventory_identity() -> None:
    seedless = metric(seed=None)
    seeded = metric(seed=20260821)
    report = compare_metric_records(
        (seedless,),
        (seeded,),
        ComparisonStage.REEVALUATION,
        TOLERANCES,
    )

    assert [issue.code for issue in report.issues] == [
        "MISSING_OBSERVED_KEY",
        "UNEXPECTED_OBSERVED_KEY",
    ]


def test_report_order_and_serialization_do_not_depend_on_input_order() -> None:
    expected = (metric("z"), metric("a", cross_entropy=1.0))
    observed = (metric("a", cross_entropy=2.0), metric("z", token_count=9))
    first = compare_metric_records(
        expected, observed, ComparisonStage.REEVALUATION, TOLERANCES
    )
    second = compare_metric_records(
        tuple(reversed(expected)),
        tuple(reversed(observed)),
        ComparisonStage.REEVALUATION,
        TOLERANCES,
    )

    assert first == second
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()
    assert [issue.key.model_id for issue in first.issues if issue.key] == ["a", "z"]


@pytest.mark.parametrize(
    "tolerances",
    [
        CrossEntropyTolerances(reevaluation=-1.0, retraining=0.0),
        CrossEntropyTolerances(reevaluation=math.nan, retraining=0.0),
        CrossEntropyTolerances(reevaluation=1, retraining=0.0),
        CrossEntropyTolerances(reevaluation=True, retraining=0.0),
    ],
)
def test_invalid_tolerance_policy_raises_value_error(
    tolerances: CrossEntropyTolerances,
) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        compare_metric_records(
            (metric(),), (metric(),), ComparisonStage.REEVALUATION, tolerances
        )


@pytest.mark.parametrize("stage", list(ComparisonStage))
def test_invalid_unused_stage_tolerance_also_raises_value_error(
    stage: ComparisonStage,
) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        compare_metric_records(
            (metric(),),
            (metric(),),
            stage,
            CrossEntropyTolerances(reevaluation=0.0, retraining=math.nan),
        )


def test_unsupported_stage_raises_value_error() -> None:
    with pytest.raises(ValueError, match="stage"):
        compare_metric_records((metric(),), (metric(),), "reevaluation", TOLERANCES)  # type: ignore[arg-type]
