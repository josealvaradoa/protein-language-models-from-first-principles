"""Exact-inventory comparison of scalar reproduction metrics.

This module deliberately knows nothing about model training, configuration
files, or run storage. Callers supply already-collected metric records and the
tolerance policy approved for their comparison stage.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class ComparisonStage(str, Enum):
    """Stages whose cross-entropy tolerances differ."""

    REEVALUATION = "reevaluation"
    RETRAINING = "retraining"


@dataclass(frozen=True)
class MetricKey:
    """The identity of one reported metric population."""

    model_id: str
    collection: str
    seed: int | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "model_id": self.model_id,
            "collection": self.collection,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class MetricRecord:
    """The three scalar metrics compared for one metric population."""

    key: MetricKey
    cross_entropy: float
    correct_predictions: int
    token_count: int


@dataclass(frozen=True)
class CrossEntropyTolerances:
    """Caller-supplied absolute tolerances for the two comparison stages."""

    reevaluation: float
    retraining: float


@dataclass(frozen=True)
class ComparisonIssue:
    """One deterministic reason a metric comparison cannot pass."""

    code: str
    side: str | None = None
    key: MetricKey | None = None
    field: str | None = None
    expected_value: int | float | None = None
    observed_value: int | float | None = None
    tolerance: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "side": self.side,
            "key": None if self.key is None else self.key.to_dict(),
            "field": self.field,
            "expected_value": self.expected_value,
            "observed_value": self.observed_value,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class ComparisonReport:
    """An immutable result that can be written directly through ``to_dict``."""

    stage: ComparisonStage
    cross_entropy_tolerance: float
    passed: bool
    issues: tuple[ComparisonIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "cross_entropy_tolerance": self.cross_entropy_tolerance,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self) -> str:
        """Return a stable JSON representation suitable for an evidence file."""

        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


@dataclass(frozen=True)
class _RecordEntry:
    key: MetricKey | None
    record: MetricRecord | None
    valid: bool


def compare_metric_records(
    expected: Iterable[object],
    observed: Iterable[object],
    stage: ComparisonStage,
    tolerances: CrossEntropyTolerances,
) -> ComparisonReport:
    """Compare two complete metric inventories without raising for bad records.

    Record problems are reported as issues so a caller can preserve a complete
    failed comparison result. Policy problems are different: they prevent a
    meaningful comparison and therefore raise ``ValueError``.
    """

    tolerance = _tolerance_for(stage, tolerances)
    issues: list[ComparisonIssue] = []
    expected_entries = _validate_records(expected, "expected", issues)
    observed_entries = _validate_records(observed, "observed", issues)
    if not expected_entries:
        issues.append(ComparisonIssue("EMPTY_EXPECTED_INVENTORY", side="expected"))

    expected_by_key = _index_entries(expected_entries, "expected", issues)
    observed_by_key = _index_entries(observed_entries, "observed", issues)

    expected_keys = set(expected_by_key)
    observed_keys = set(observed_by_key)
    for key in expected_keys - observed_keys:
        issues.append(ComparisonIssue("MISSING_OBSERVED_KEY", key=key))
    for key in observed_keys - expected_keys:
        issues.append(ComparisonIssue("UNEXPECTED_OBSERVED_KEY", key=key))

    for key in expected_keys & observed_keys:
        expected_entry = expected_by_key[key]
        observed_entry = observed_by_key[key]
        if not expected_entry.valid or not observed_entry.valid:
            continue
        assert expected_entry.record is not None
        assert observed_entry.record is not None
        _compare_metrics(
            expected_entry.record,
            observed_entry.record,
            tolerance,
            issues,
        )

    ordered_issues = tuple(sorted(issues, key=_issue_sort_key))
    return ComparisonReport(
        stage=stage,
        cross_entropy_tolerance=tolerance,
        passed=not ordered_issues,
        issues=ordered_issues,
    )


def _tolerance_for(stage: ComparisonStage, tolerances: CrossEntropyTolerances) -> float:
    if not isinstance(stage, ComparisonStage):
        raise ValueError("comparison stage must be a ComparisonStage")
    if not isinstance(tolerances, CrossEntropyTolerances):
        raise ValueError("cross-entropy tolerances are invalid")
    for value in (tolerances.reevaluation, tolerances.retraining):
        if type(value) is not float or not math.isfinite(value) or value < 0:
            raise ValueError(
                "cross-entropy tolerance must be a finite nonnegative number"
            )
    value = (
        tolerances.reevaluation
        if stage is ComparisonStage.REEVALUATION
        else tolerances.retraining
    )
    return float(value)


def _validate_records(
    values: Iterable[object], side: str, issues: list[ComparisonIssue]
) -> tuple[_RecordEntry, ...]:
    entries: list[_RecordEntry] = []
    try:
        records = tuple(values)
    except TypeError:
        records = (values,)
    for value in records:
        if not isinstance(value, MetricRecord):
            issues.append(ComparisonIssue("INVALID_RECORD", side=side))
            entries.append(_RecordEntry(None, None, False))
            continue
        key_valid = _validate_key(value.key, side, issues)
        metrics_valid = _validate_metrics(value, side, issues) if key_valid else False
        entries.append(
            _RecordEntry(value.key if key_valid else None, value, key_valid and metrics_valid)
        )
    return tuple(entries)


def _validate_key(key: object, side: str, issues: list[ComparisonIssue]) -> bool:
    if not isinstance(key, MetricKey):
        issues.append(ComparisonIssue("INVALID_KEY", side=side, field="key"))
        return False
    valid = True
    if not isinstance(key.model_id, str) or not key.model_id.strip():
        issues.append(ComparisonIssue("INVALID_MODEL_ID", side=side, field="model_id"))
        valid = False
    if not isinstance(key.collection, str) or not key.collection.strip():
        issues.append(ComparisonIssue("INVALID_COLLECTION", side=side, field="collection"))
        valid = False
    if key.seed is not None and type(key.seed) is not int:
        issues.append(ComparisonIssue("INVALID_SEED", side=side, field="seed"))
        valid = False
    return valid


def _validate_metrics(
    record: MetricRecord, side: str, issues: list[ComparisonIssue]
) -> bool:
    valid = True
    key = record.key
    if (
        type(record.cross_entropy) is not float
        or not math.isfinite(record.cross_entropy)
        or record.cross_entropy < 0
    ):
        issues.append(
            ComparisonIssue("INVALID_CROSS_ENTROPY", side=side, key=key, field="cross_entropy")
        )
        valid = False
    if type(record.correct_predictions) is not int or record.correct_predictions < 0:
        issues.append(
            ComparisonIssue(
                "INVALID_CORRECT_PREDICTIONS",
                side=side,
                key=key,
                field="correct_predictions",
            )
        )
        valid = False
    if type(record.token_count) is not int or record.token_count <= 0:
        issues.append(
            ComparisonIssue("INVALID_TOKEN_COUNT", side=side, key=key, field="token_count")
        )
        valid = False
    if (
        type(record.correct_predictions) is int
        and type(record.token_count) is int
        and record.correct_predictions > record.token_count
    ):
        issues.append(
            ComparisonIssue(
                "CORRECT_PREDICTIONS_EXCEED_TOKEN_COUNT",
                side=side,
                key=key,
            )
        )
        valid = False
    return valid


def _index_entries(
    entries: tuple[_RecordEntry, ...], side: str, issues: list[ComparisonIssue]
) -> dict[MetricKey, _RecordEntry]:
    grouped: dict[MetricKey, list[_RecordEntry]] = {}
    for entry in entries:
        if entry.key is not None:
            grouped.setdefault(entry.key, []).append(entry)
    result: dict[MetricKey, _RecordEntry] = {}
    for key, key_entries in grouped.items():
        if len(key_entries) != 1:
            issues.append(ComparisonIssue(f"DUPLICATE_{side.upper()}_KEY", side=side, key=key))
            result[key] = _RecordEntry(key, None, False)
        else:
            result[key] = key_entries[0]
    return result


def _compare_metrics(
    expected: MetricRecord,
    observed: MetricRecord,
    tolerance: float,
    issues: list[ComparisonIssue],
) -> None:
    cross_entropy_delta = abs(expected.cross_entropy - observed.cross_entropy)
    if cross_entropy_delta > tolerance:
        issues.append(
            ComparisonIssue(
                "CROSS_ENTROPY_MISMATCH",
                key=expected.key,
                field="cross_entropy",
                expected_value=expected.cross_entropy,
                observed_value=observed.cross_entropy,
                tolerance=tolerance,
            )
        )
    if expected.correct_predictions != observed.correct_predictions:
        issues.append(
            ComparisonIssue(
                "CORRECT_PREDICTIONS_MISMATCH",
                key=expected.key,
                field="correct_predictions",
                expected_value=expected.correct_predictions,
                observed_value=observed.correct_predictions,
            )
        )
    if expected.token_count != observed.token_count:
        issues.append(
            ComparisonIssue(
                "TOKEN_COUNT_MISMATCH",
                key=expected.key,
                field="token_count",
                expected_value=expected.token_count,
                observed_value=observed.token_count,
            )
        )


def _issue_sort_key(issue: ComparisonIssue) -> tuple[tuple[str, str, int, int], str, str, str]:
    return (
        _key_sort_key(issue.key),
        issue.side or "",
        issue.code,
        issue.field or "",
    )


def _key_sort_key(key: MetricKey | None) -> tuple[str, str, int, int]:
    if key is None:
        return ("", "", 0, 0)
    seed = 0 if key.seed is None else key.seed
    return (key.model_id, key.collection, 0 if key.seed is None else 1, seed)
