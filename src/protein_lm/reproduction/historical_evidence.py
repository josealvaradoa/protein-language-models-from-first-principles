"""Read-only normalization of the source-pinned Week 2 and Week 3 evidence.

This adapter is deliberately narrow.  It validates the schemas of the three
historical inputs used by the frozen Week 4 contract, converts their overall
metrics into the contract inventory, and evaluates only the contract claims
whose inputs are present in that inventory.  It never trains, scores, or
writes evidence.
"""

from __future__ import annotations

import json
import math
import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path

from protein_lm.reproduction.comparison import (
    ComparisonReport,
    ComparisonStage,
    MetricKey,
    MetricRecord,
    compare_metric_records,
)
from protein_lm.reproduction.contract import (
    ComparisonClaim,
    FoundationsContract,
    ReproductionContractError,
    SourcePinVerificationReport,
    load_foundations_contract,
    verify_source_pins,
)


_WEEK2_REPORT = "reports/week_02/bigram_evaluation_v1.json"
_WEEK3_REPORT = "reports/week_03/mlp_evaluation_v1.json"
_WEEK3_C10_CONFIG = "experiments/week_03/mlp_context20_100m_continuation_v1.toml"
_CONTRACT_PATH = "experiments/week_04/foundations_reproduction_v1.toml"
_WEEK2_REPORT_SCOPE = "week_02_bigram_evaluation_public_report"
_WEEK3_REPORT_SCOPE = "week_03_mlp_public_report"
_C10_SCOPE = "week_03_mlp_context20_100m_continuation_exploratory"
_C10_CONTRACT_IDENTIFIER = "2026-08-25-week-03-context20-100m-continuation-v1"
_WEEK2_MODEL_IDS = {
    ("random_training", "unigram"): "week2_random_unigram",
    ("random_training", "count_bigram"): "week2_random_count_bigram",
    ("random_training", "neural_bigram"): "week2_random_neural_bigram",
    ("family_aware_training", "unigram"): "week2_family_aware_unigram",
    ("family_aware_training", "count_bigram"): "week2_family_aware_count_bigram",
    ("family_aware_training", "neural_bigram"): "week2_family_aware_neural_bigram",
}
_ARITHMETIC_ABSOLUTE_TOLERANCE = 1e-9


class HistoricalEvidenceError(ValueError):
    """Raised when pinned historical evidence cannot be safely normalized."""


@dataclass(frozen=True)
class ClaimOutcome:
    """One ordered frozen claim result, including intentionally unevaluated ones."""

    claim_id: str
    rule: str
    evaluated: bool
    passed: bool | None
    observed_cross_entropy_gap: float | None
    minimum_cross_entropy_gap: float
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "rule": self.rule,
            "evaluated": self.evaluated,
            "passed": self.passed,
            "observed_cross_entropy_gap": self.observed_cross_entropy_gap,
            "minimum_cross_entropy_gap": self.minimum_cross_entropy_gap,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HistoricalEvidenceReport:
    """Immutable, JSON-safe result of the historical evidence adapter."""

    source_verification: SourcePinVerificationReport
    observed_records: tuple[MetricRecord, ...]
    metric_comparison: ComparisonReport
    claim_outcomes: tuple[ClaimOutcome, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_verification": self.source_verification.to_dict(),
            "observed_records": [_record_to_dict(record) for record in self.observed_records],
            "metric_comparison": self.metric_comparison.to_dict(),
            "claim_outcomes": [outcome.to_dict() for outcome in self.claim_outcomes],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


def evaluate_historical_evidence(
    project_root: Path | str, contract: FoundationsContract
) -> HistoricalEvidenceReport:
    """Verify all contract pins, normalize evidence, and compare it to the contract.

    A failed pin or any malformed historical source raises a domain error rather
    than returning a partial inventory.  This makes the adapter fail closed.
    """

    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as error:
        raise HistoricalEvidenceError("historical evidence project root is unavailable") from error
    try:
        canonical_contract = load_foundations_contract(root / _CONTRACT_PATH)
        if canonical_contract != contract:
            raise HistoricalEvidenceError(
                "supplied contract does not match the canonical frozen contract"
            )
        verification = verify_source_pins(root, canonical_contract)
    except ReproductionContractError as error:
        raise HistoricalEvidenceError(
            "could not load or verify the canonical frozen contract"
        ) from error
    if not verification.passed:
        raise HistoricalEvidenceError("historical evidence source pins did not verify")
    _require_source_dependencies(canonical_contract)
    week2 = _load_json(root / _WEEK2_REPORT, "Week 2 report")
    week3 = _load_json(root / _WEEK3_REPORT, "Week 3 report")
    c10 = _load_toml(root / _WEEK3_C10_CONFIG, "Week 3 C10 configuration")
    _validate_source_identities(week2, week3, c10, canonical_contract.metric_targets)
    observed = _normalize_week2(week2) + _normalize_week3(c10, week3)
    _require_exact_inventory(observed, canonical_contract.metric_targets)
    observed = _order_by_contract_targets(observed, canonical_contract.metric_targets)
    comparison = compare_metric_records(
        canonical_contract.metric_targets,
        observed,
        ComparisonStage.REEVALUATION,
        canonical_contract.tolerances,
    )
    if not comparison.passed:
        raise HistoricalEvidenceError("historical metric records do not match the frozen contract")
    return HistoricalEvidenceReport(
        source_verification=verification,
        observed_records=observed,
        metric_comparison=comparison,
        claim_outcomes=_evaluate_claims(canonical_contract.comparison_claims, observed),
    )


def _validate_source_identities(
    week2_raw: object,
    week3_raw: object,
    c10_raw: object,
    targets: tuple[MetricRecord, ...],
) -> None:
    week2 = _mapping(week2_raw, "Week 2 report")
    _require_identity(
        week2,
        {
            "schema_version": 1,
            "scope": _WEEK2_REPORT_SCOPE,
            "status": "passed",
            "network_requests_made": 0,
        },
        "Week 2 report",
    )
    _require_true_gates(
        _mapping(week2.get("hard_gates"), "Week 2 report hard gates"),
        (
            "aggregate_only",
            "exact_twelve_records",
            "no_network_requests",
            "sealed_test_inaccessible",
            "validated_source_evaluation",
        ),
        "Week 2 report",
    )
    week3 = _mapping(week3_raw, "Week 3 report")
    _require_identity(
        week3,
        {
            "schema_version": 1,
            "scope": _WEEK3_REPORT_SCOPE,
            "status": "passed",
            "network_requests_made": 0,
        },
        "Week 3 report",
    )
    _require_true_gates(
        _mapping(week3.get("hard_gates"), "Week 3 report hard gates"),
        (
            "aggregate_only",
            "no_network_requests",
            "no_training_or_evaluation",
            "sealed_test_inaccessible",
            "validated_pinned_sources",
        ),
        "Week 3 report",
    )
    c10 = _mapping(c10_raw, "Week 3 C10 configuration")
    _require_identity(
        c10,
        {
            "schema_version": 1,
            "scope": _C10_SCOPE,
            "contract_identifier": _C10_CONTRACT_IDENTIFIER,
            "context_length": 20,
        },
        "Week 3 C10 configuration",
    )
    c10_targets = [record for record in targets if record.key.model_id == "c10"]
    if len(c10_targets) != 3:
        raise HistoricalEvidenceError("canonical contract has invalid C10 metric identity")
    target_collections = {record.key.collection for record in c10_targets}
    target_token_counts = {record.token_count for record in c10_targets}
    if len(target_collections) != 1 or len(target_token_counts) != 1:
        raise HistoricalEvidenceError("canonical contract has inconsistent C10 identity")
    if (
        c10.get("native_validation_collection") != next(iter(target_collections))
        or c10.get("native_validation_prediction_tokens") != next(iter(target_token_counts))
    ):
        raise HistoricalEvidenceError("Week 3 C10 configuration identity does not match contract")


def _require_identity(
    value: dict[str, object], expected: dict[str, object], label: str
) -> None:
    for field, expected_value in expected.items():
        if value.get(field) != expected_value or type(value.get(field)) is not type(expected_value):
            raise HistoricalEvidenceError(f"{label} {field} identity is invalid")


def _require_true_gates(
    gates: dict[str, object], required: tuple[str, ...], label: str
) -> None:
    if any(gates.get(name) is not True for name in required):
        raise HistoricalEvidenceError(f"{label} hard gates are invalid")


def _normalize_week2(raw: object) -> tuple[MetricRecord, ...]:
    report = _mapping(raw, "Week 2 report")
    records = _list(report.get("records"), "Week 2 report records")
    normalized: list[MetricRecord] = []
    identities: set[MetricKey] = set()
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"Week 2 record {index}")
        arm = _string(record.get("model_arm"), f"Week 2 record {index} model_arm")
        model_type = _string(record.get("model_type"), f"Week 2 record {index} model_type")
        collection = _string(record.get("collection"), f"Week 2 record {index} collection")
        model_id = _WEEK2_MODEL_IDS.get((arm, model_type))
        if model_id is None:
            raise HistoricalEvidenceError("Week 2 report has an unexpected model identity")
        expected_collection = (
            "random_native_validation"
            if arm == "random_training"
            else "family_aware_native_validation"
        )
        if collection not in {expected_collection, "shared_validation"}:
            raise HistoricalEvidenceError("Week 2 report has an unexpected collection")
        metrics = _mapping(record.get("metrics"), f"Week 2 record {index} metrics")
        overall = _mapping(metrics.get("overall"), f"Week 2 record {index} overall")
        parsed = _metric_from_overall(overall, f"Week 2 record {index} overall")
        key = MetricKey(model_id, collection)
        if key in identities:
            raise HistoricalEvidenceError("Week 2 report has duplicate primary metric identity")
        identities.add(key)
        normalized.append(MetricRecord(key, *parsed))
    if len(normalized) != 12:
        raise HistoricalEvidenceError("Week 2 report must contain exactly twelve primary records")
    return tuple(normalized)


def _normalize_week3(c10_raw: object, report_raw: object) -> tuple[MetricRecord, ...]:
    config = _mapping(c10_raw, "Week 3 C10 configuration")
    tokens = _positive_int(
        config.get("native_validation_prediction_tokens"), "Week 3 C10 native token count"
    )
    control_runs = _list(config.get("control_runs"), "Week 3 C10 control_runs")
    result = list(_normalize_c10_runs(control_runs, tokens))
    report = _mapping(report_raw, "Week 3 report")
    comparison = _mapping(
        report.get("final_three_seed_comparison"), "Week 3 final three-seed comparison"
    )
    result.extend(
        _normalize_week3_report_arm(
            _mapping(comparison.get("context20"), "Week 3 C20 comparison"), "c20"
        )
    )
    result.extend(
        _normalize_week3_report_arm(
            _mapping(comparison.get("embedding64_challenger"), "Week 3 E64 comparison"), "e64"
        )
    )
    return tuple(result)


def _normalize_c10_runs(runs: list[object], token_count: int) -> tuple[MetricRecord, ...]:
    records: list[MetricRecord] = []
    seeds: set[int] = set()
    for index, raw_run in enumerate(runs):
        run = _mapping(raw_run, f"Week 3 C10 control run {index}")
        seed = _seed(run.get("seed"), f"Week 3 C10 control run {index} seed")
        ce = _finite_nonnegative_float(
            run.get("native_cross_entropy"), f"Week 3 C10 control run {index} CE"
        )
        accuracy = _probability(run.get("native_accuracy"), f"Week 3 C10 control run {index} accuracy")
        nll = _finite_nonnegative_float(
            run.get("native_nll_numerator"), f"Week 3 C10 control run {index} NLL"
        )
        correct = _nonnegative_int(
            run.get("native_correct_predictions"), f"Week 3 C10 control run {index} correct predictions"
        )
        _validate_metric_arithmetic(ce, accuracy, nll, correct, token_count, "Week 3 C10 control run")
        if seed in seeds:
            raise HistoricalEvidenceError("Week 3 C10 control runs have duplicate seeds")
        seeds.add(seed)
        records.append(
            MetricRecord(MetricKey("c10", "family_aware_native_validation", seed), ce, correct, token_count)
        )
    if len(records) != 3:
        raise HistoricalEvidenceError("Week 3 C10 control runs must contain exactly three records")
    return tuple(records)


def _normalize_week3_report_arm(section: dict[str, object], model_id: str) -> tuple[MetricRecord, ...]:
    records = _list(section.get("records"), f"Week 3 {model_id} records")
    aggregate = _mapping(section.get("aggregate"), f"Week 3 {model_id} aggregate")
    normalized: list[MetricRecord] = []
    seeds: set[int] = set()
    cross_entropies: list[float] = []
    accuracies: list[float] = []
    for index, raw_record in enumerate(records):
        item = _mapping(raw_record, f"Week 3 {model_id} record {index}")
        seed = _seed(item.get("seed"), f"Week 3 {model_id} record {index} seed")
        ce = _finite_nonnegative_float(item.get("cross_entropy"), f"Week 3 {model_id} record {index} CE")
        accuracy = _probability(item.get("accuracy"), f"Week 3 {model_id} record {index} accuracy")
        nll = _finite_nonnegative_float(item.get("nll_numerator"), f"Week 3 {model_id} record {index} NLL")
        correct = _nonnegative_int(item.get("correct_predictions"), f"Week 3 {model_id} record {index} correct predictions")
        tokens = _positive_int(item.get("token_count"), f"Week 3 {model_id} record {index} token count")
        _validate_metric_arithmetic(ce, accuracy, nll, correct, tokens, f"Week 3 {model_id} record")
        if seed in seeds:
            raise HistoricalEvidenceError(f"Week 3 {model_id} records have duplicate seeds")
        seeds.add(seed)
        cross_entropies.append(ce)
        accuracies.append(accuracy)
        normalized.append(MetricRecord(MetricKey(model_id, "family_aware_native_validation", seed), ce, correct, tokens))
    if len(normalized) != 3:
        raise HistoricalEvidenceError(f"Week 3 {model_id} records must contain exactly three records")
    _validate_aggregate(aggregate, cross_entropies, accuracies, f"Week 3 {model_id} aggregate")
    return tuple(normalized)


def _metric_from_overall(overall: dict[str, object], label: str) -> tuple[float, int, int]:
    ce = _finite_nonnegative_float(overall.get("cross_entropy"), f"{label} CE")
    accuracy = _probability(overall.get("accuracy"), f"{label} accuracy")
    nll = _finite_nonnegative_float(overall.get("total_nll"), f"{label} total NLL")
    correct = _nonnegative_int(overall.get("correct_tokens"), f"{label} correct tokens")
    tokens = _positive_int(overall.get("token_count"), f"{label} token count")
    _validate_metric_arithmetic(ce, accuracy, nll, correct, tokens, label)
    return ce, correct, tokens


def _validate_metric_arithmetic(
    ce: float, accuracy: float, nll: float, correct: int, tokens: int, label: str
) -> None:
    if correct > tokens:
        raise HistoricalEvidenceError(f"{label} correct predictions exceed token count")
    if not _close(ce, nll / tokens) or not _close(accuracy, correct / tokens):
        raise HistoricalEvidenceError(f"{label} metric arithmetic is inconsistent")


def _validate_aggregate(
    aggregate: dict[str, object], ces: list[float], accuracies: list[float], label: str
) -> None:
    expected = {
        "mean_cross_entropy": statistics.fmean(ces),
        "mean_accuracy": statistics.fmean(accuracies),
        "sample_standard_deviation_cross_entropy": statistics.stdev(ces),
        "sample_standard_deviation_accuracy": statistics.stdev(accuracies),
    }
    for field, value in expected.items():
        observed = _finite_nonnegative_float(aggregate.get(field), f"{label} {field}")
        if not _close(observed, value):
            raise HistoricalEvidenceError(f"{label} arithmetic is inconsistent")


def _evaluate_claims(
    claims: tuple[ComparisonClaim, ...], records: tuple[MetricRecord, ...]
) -> tuple[ClaimOutcome, ...]:
    by_key = {record.key: record for record in records}
    means = {
        model_id: statistics.fmean(
            record.cross_entropy
            for key, record in by_key.items()
            if key.model_id == model_id
            and key.collection == "family_aware_native_validation"
            and key.seed is not None
        )
        for model_id in ("c10", "c20", "e64")
    }
    outcomes: list[ClaimOutcome] = []
    for claim in claims:
        if claim.claim_id == "week1_group_aware_lower_detected_strong_overlap":
            outcomes.append(ClaimOutcome(claim.claim_id, claim.rule, False, None, None, claim.minimum_cross_entropy_gap, "not_evaluated_by_historical_evidence_pipeline"))
            continue
        if claim.claim_id == "week2_prospective_comparison_supported":
            random_gap = _ce(by_key, "week2_random_neural_bigram", "shared_validation") - _ce(by_key, "week2_random_neural_bigram", "random_native_validation")
            family_gap = _ce(by_key, "week2_family_aware_neural_bigram", "shared_validation") - _ce(by_key, "week2_family_aware_neural_bigram", "family_aware_native_validation")
            gap = random_gap - family_gap
            outcomes.append(ClaimOutcome(claim.claim_id, claim.rule, True, gap > claim.minimum_cross_entropy_gap, gap, claim.minimum_cross_entropy_gap))
            continue
        if claim.claim_id == "c20_beats_week2_family_aware_neural_bigram":
            gap = _ce(by_key, "week2_family_aware_neural_bigram", "family_aware_native_validation") - means["c20"]
        elif claim.claim_id == "c20_beats_c10":
            gap = means["c10"] - means["c20"]
        elif claim.claim_id == "c20_beats_e64":
            gap = means["e64"] - means["c20"]
        else:
            raise HistoricalEvidenceError("contract contains an unsupported historical claim")
        outcomes.append(ClaimOutcome(claim.claim_id, claim.rule, True, gap >= claim.minimum_cross_entropy_gap, gap, claim.minimum_cross_entropy_gap))
    return tuple(outcomes)


def _ce(records: dict[MetricKey, MetricRecord], model_id: str, collection: str) -> float:
    return records[MetricKey(model_id, collection)].cross_entropy


def _require_exact_inventory(observed: tuple[MetricRecord, ...], expected: tuple[MetricRecord, ...]) -> None:
    observed_keys = tuple(record.key for record in observed)
    expected_keys = tuple(record.key for record in expected)
    if len(observed_keys) != 21 or len(set(observed_keys)) != len(observed_keys):
        raise HistoricalEvidenceError("historical evidence primary metric inventory is invalid")
    if set(observed_keys) != set(expected_keys):
        raise HistoricalEvidenceError("historical evidence primary metric identities do not match contract")


def _order_by_contract_targets(
    observed: tuple[MetricRecord, ...], expected: tuple[MetricRecord, ...]
) -> tuple[MetricRecord, ...]:
    by_key = {record.key: record for record in observed}
    return tuple(by_key[target.key] for target in expected)


def _require_source_dependencies(contract: FoundationsContract) -> None:
    paths = {pin.relative_path for pin in contract.source_pins}
    required = {_WEEK2_REPORT, _WEEK3_REPORT, _WEEK3_C10_CONFIG}
    if not required <= paths:
        raise HistoricalEvidenceError("contract does not pin all historical evidence inputs")


def _load_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HistoricalEvidenceError(f"could not parse {label}") from error


def _load_toml(path: Path, label: str) -> object:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise HistoricalEvidenceError(f"could not parse {label}") from error


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise HistoricalEvidenceError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise HistoricalEvidenceError(f"{label} must be a list")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HistoricalEvidenceError(f"{label} must be a nonempty string")
    return value


def _finite_nonnegative_float(value: object, label: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise HistoricalEvidenceError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise HistoricalEvidenceError(f"{label} must be a finite nonnegative number")
    return result


def _probability(value: object, label: str) -> float:
    result = _finite_nonnegative_float(value, label)
    if result > 1:
        raise HistoricalEvidenceError(f"{label} must be between zero and one")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise HistoricalEvidenceError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise HistoricalEvidenceError(f"{label} must be a positive integer")
    return value


def _seed(value: object, label: str) -> int:
    return _positive_int(value, label)


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=_ARITHMETIC_ABSOLUTE_TOLERANCE)


def _record_to_dict(record: MetricRecord) -> dict[str, object]:
    return {
        **record.key.to_dict(),
        "cross_entropy": record.cross_entropy,
        "correct_predictions": record.correct_predictions,
        "token_count": record.token_count,
    }
