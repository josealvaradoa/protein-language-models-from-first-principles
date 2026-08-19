"""Result schema, aggregate arithmetic, and hypothesis calculations without loaders."""

from __future__ import annotations

import math

from protein_lm.bigram.evaluation_contract import EvaluationConfig
from protein_lm.data.model_data.contracts import ModelDataError


def result_payload(
    *,
    config: EvaluationConfig,
    config_sha256: str,
    evaluation_id: str,
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Create the complete local result evidence after exact record validation."""

    validate_records(records, config)
    return {
        "schema_version": 1,
        "scope": "week_02_bigram_evaluation_results",
        "contract_identifier": config.contract_identifier,
        "evaluation_id": evaluation_id,
        "metric_dtype": "float64",
        "provenance": provenance(config, config_sha256),
        "records": records,
        "hypothesis": hypothesis(records),
    }


def provenance(config: EvaluationConfig, config_sha256: str) -> dict[str, object]:
    """Return the exact model and data identities carried by every output."""

    return {
        "configuration_sha256": config_sha256,
        "model_candidate": {
            "candidate_id": config.model_candidate_id,
            "relative_path": config.model_candidate_relative_path,
            "candidate_registry_sha256": config.model_candidate_registry_sha256,
            "run_record_sha256": config.model_candidate_run_record_sha256,
        },
        "model_data_registry": {
            "relative_path": config.model_data_registry_relative_path,
            "sha256": config.model_data_registry_sha256,
        },
    }


def hypothesis(records: list[dict[str, object]]) -> dict[str, object]:
    """Calculate the frozen neural optimism comparison and report either outcome."""

    def ce(arm: str, collection: str) -> float:
        record = next(
            item
            for item in records
            if item["model_arm"] == arm
            and item["model_type"] == "neural_bigram"
            and item["collection"] == collection
        )
        metrics = record["metrics"]
        assert isinstance(metrics, dict)
        overall = metrics["overall"]
        assert isinstance(overall, dict)
        value = overall["cross_entropy"]
        assert type(value) is float
        return value

    random_gap = ce("random_training", "shared_validation") - ce(
        "random_training", "random_native_validation"
    )
    family_gap = ce("family_aware_training", "shared_validation") - ce(
        "family_aware_training", "family_aware_native_validation"
    )
    comparison = random_gap - family_gap
    return {
        "random_neural_optimism_gap": random_gap,
        "family_neural_optimism_gap": family_gap,
        "comparison": comparison,
        "supports_hypothesis": comparison > 0.0,
    }


def validate_records(
    records: list[dict[str, object]], config: EvaluationConfig
) -> None:
    """Require the exact 12 principal records and complete aggregate accounting."""

    expected = (
        {
            ("random_training", model, "random_native_validation")
            for model in config.model_types
        }
        | {
            ("family_aware_training", model, "family_aware_native_validation")
            for model in config.model_types
        }
        | {
            (arm, model, "shared_validation")
            for arm in config.model_arms
            for model in config.model_types
        }
    )
    fields = {"model_arm", "model_type", "collection", "metrics"}
    if any(
        set(record) != fields
        or not isinstance(record["model_arm"], str)
        or not isinstance(record["model_type"], str)
        or not isinstance(record["collection"], str)
        for record in records
    ):
        raise ModelDataError("evaluation record schema is invalid")
    found = {
        (item["model_arm"], item["model_type"], item["collection"]) for item in records
    }
    if len(records) != 12 or found != expected:
        raise ModelDataError(
            "evaluation record inventory is not exactly the approved twelve records"
        )
    for record in records:
        metrics = record["metrics"]
        if not isinstance(metrics, dict):
            raise ModelDataError("evaluation record metrics are invalid")
        validate_metric_payload(metrics, config.length_buckets)
    _validate_peer_populations(records)


def _validate_peer_populations(records: list[dict[str, object]]) -> None:
    """Ensure models compared on one collection saw the same proteins and tokens."""

    by_collection: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_collection.setdefault(record["collection"], []).append(record)
    for collection, peers in by_collection.items():
        signatures = {_population_signature(peer) for peer in peers}
        if len(signatures) != 1:
            raise ModelDataError(
                f"evaluation records for {collection} disagree on collection population"
            )


def _population_signature(record: dict[str, object]) -> tuple[object, ...]:
    metrics = record["metrics"]
    assert isinstance(metrics, dict)
    overall = metrics["overall"]
    buckets = metrics["length_buckets"]
    assert isinstance(overall, dict) and isinstance(buckets, dict)
    return (
        overall["token_count"],
        overall["protein_count"],
        tuple(
            (name, value["token_count"], value["protein_count"])
            for name, value in sorted(buckets.items())
            if isinstance(value, dict)
        ),
    )


def validate_metric_payload(
    metrics: dict[str, object], buckets: tuple[str, ...]
) -> None:
    """Check overall and bucket arithmetic, including reproducible median evidence."""

    overall = metrics.get("overall")
    by_bucket = metrics.get("length_buckets")
    if (
        not isinstance(overall, dict)
        or not isinstance(by_bucket, dict)
        or set(by_bucket) != set(buckets)
    ):
        raise ModelDataError("evaluation metric buckets are invalid")
    validate_aggregate(overall)
    for item in by_bucket.values():
        if not isinstance(item, dict):
            raise ModelDataError("evaluation bucket metric is invalid")
        validate_aggregate(item)
    for key in ("token_count", "protein_count", "correct_tokens"):
        if overall[key] != sum(item[key] for item in by_bucket.values()):
            raise ModelDataError(
                "evaluation bucket accounting does not equal the overall metric"
            )
    bucket_nll = sum(item["total_nll"] for item in by_bucket.values())
    if not math.isclose(overall["total_nll"], bucket_nll, rel_tol=1e-12, abs_tol=1e-10):
        raise ModelDataError("evaluation bucket NLL does not equal the overall metric")


def validate_aggregate(value: dict[str, object]) -> None:
    """Check finite floats, non-empty populations, and all derived metric equations."""

    expected = {
        "token_count",
        "protein_count",
        "total_nll",
        "correct_tokens",
        "cross_entropy",
        "accuracy",
        "median_per_protein_nll",
        "median_lower_per_protein_nll",
        "median_upper_per_protein_nll",
    }
    if set(value) != expected:
        raise ModelDataError("evaluation aggregate metric schema is invalid")
    tokens, proteins, nll, correct, ce, accuracy, median, lower, upper = (
        value[key]
        for key in (
            "token_count",
            "protein_count",
            "total_nll",
            "correct_tokens",
            "cross_entropy",
            "accuracy",
            "median_per_protein_nll",
            "median_lower_per_protein_nll",
            "median_upper_per_protein_nll",
        )
    )
    if (
        type(tokens) is not int
        or type(proteins) is not int
        or type(correct) is not int
        or tokens <= 0
        or proteins <= 0
        or correct < 0
        or correct > tokens
    ):
        raise ModelDataError("evaluation aggregate integer accounting is invalid")
    if any(
        type(item) is not float or not math.isfinite(item)
        for item in (nll, ce, accuracy, median, lower, upper)
    ):
        raise ModelDataError("evaluation aggregate float accounting is invalid")
    if (
        nll < 0
        or median < 0
        or lower < 0
        or upper < 0
        or lower > upper
        or (proteins % 2 == 1 and lower != upper)
        or not math.isclose(ce, nll / tokens, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(accuracy, correct / tokens, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(median, (lower + upper) / 2.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ModelDataError("evaluation aggregate arithmetic is inconsistent")
