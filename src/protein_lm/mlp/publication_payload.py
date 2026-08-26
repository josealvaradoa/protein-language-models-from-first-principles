"""Arithmetic and aggregate-only payload construction for Week 3 evidence."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from statistics import mean, stdev

import numpy as np

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.publication_config import PublicationConfig


RESIDUES = ("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")
_SEEDS = {20260821, 20260822, 20260823}
_NATIVE_TOKENS = 1_000_495
_NLL_SUM_ABS_TOLERANCE = 1e-6


def final_metrics(statuses: Iterable[dict[str, object]], arm: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for status in statuses:
        if (
            status.get("status") != "passed"
            or status.get("network_requests_made") != 0
            or status.get("exploratory_only") is not True
            or status.get("automatic_selection_generated") is not False
            or status.get("automatic_report_generated") is not False
            or status.get("sealed_test_accessed") is True
            or status.get("sealed_test_collection") is True
        ):
            raise ModelDataError("final source status did not pass local-only gates")
        seed = _int(status.get("seed"), "seed")
        model = _mapping(status.get("model"), "model")
        expected_shape = (20, 32, 800, 530293) if arm == "context20" else (10, 64, 800, 530965)
        if tuple(model.get(key) for key in ("context_length", "embedding_width", "hidden_width", "parameter_count")) != expected_shape:
            raise ModelDataError("final model shape or parameter count drifted")
        milestones = status.get("native_validation_milestones") if arm == "context20" else status.get("metrics")
        if not isinstance(milestones, list) or [item.get("prediction_position") for item in milestones if isinstance(item, dict)] != [50000000, 100000000]:
            raise ModelDataError("final metric endpoints drifted")
        milestone = _endpoint(milestones, 100000000)
        records.append({"seed": seed, **_metric(milestone)})
    if len(records) != 3 or {record["seed"] for record in records} != _SEEDS:
        raise ModelDataError("final source seeds are missing or extra")
    return sorted(records, key=lambda record: int(record["seed"]))


def aggregate(records: list[dict[str, object]]) -> dict[str, float]:
    ce = [_number(record["cross_entropy"], "cross entropy") for record in records]
    accuracy = [_number(record["accuracy"], "accuracy") for record in records]
    if len(records) != 3:
        raise ModelDataError("aggregate requires exactly three seeds")
    return {
        "mean_cross_entropy": mean(ce), "sample_standard_deviation_cross_entropy": stdev(ce),
        "mean_accuracy": mean(accuracy), "sample_standard_deviation_accuracy": stdev(accuracy),
    }


def capacity_screen(statuses: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = {}
    parameters: dict[str, int] = {}
    seeds: dict[str, set[int]] = {}
    milestones = (1_000_000, 5_000_000, 10_000_000, 25_000_000)
    for status in statuses:
        arm = status.get("arm")
        if not isinstance(arm, str) or status.get("status") != "passed":
            raise ModelDataError("capacity screen status is invalid")
        if status.get("network_requests_made") != 0:
            raise ModelDataError("capacity screen network gate drifted")
        seed = _int(status.get("seed"), "capacity seed")
        seeds.setdefault(arm, set())
        if seed in seeds[arm]:
            raise ModelDataError("capacity screen seed is duplicated")
        seeds[arm].add(seed)
        model = _mapping(status.get("model"), "capacity model")
        parameter_count = _int(model.get("parameter_count"), "capacity parameter count")
        if parameter_count <= 0 or parameters.setdefault(arm, parameter_count) != parameter_count:
            raise ModelDataError("capacity parameter count drifted")
        metric_rows = status.get("native_validation_milestones")
        if not isinstance(metric_rows, list) or tuple(
            item.get("prediction_position") for item in metric_rows if isinstance(item, dict)
        ) != milestones:
            raise ModelDataError("capacity milestones drifted")
        for position in milestones:
            _metric(_endpoint(metric_rows, position))
        grouped.setdefault(arm, []).append(_metric(_endpoint(metric_rows, 25_000_000))["cross_entropy"])
    expected = {"context_20", "embedding_64", "hidden_1600"}
    if (
        set(grouped) != expected
        or any(len(values) != 3 or seeds[arm] != _SEEDS for arm, values in grouped.items())
    ):
        raise ModelDataError("capacity screen arms are incomplete")
    return [{"arm": arm, "parameter_count": parameters[arm], "prediction_budget": 25000000, "mean_cross_entropy": mean(grouped[arm]), "sample_standard_deviation_cross_entropy": stdev(grouped[arm]), "stage": "exploratory_25m_screen"} for arm in sorted(grouped)]


def position_summary(statuses: Iterable[dict[str, object]]) -> dict[str, object]:
    bins = ("available_prior_residues_0_10", "available_prior_residues_11_19", "available_prior_residues_20_plus")
    statuses = list(statuses)
    if len(statuses) != 3 or {item.get("seed") for item in statuses} != _SEEDS:
        raise ModelDataError("position diagnostic seeds are incomplete or duplicated")
    rows: list[dict[str, object]] = []
    total_advantage = 0.0
    by_bin: dict[str, list[dict[str, object]]] = {name: [] for name in bins}
    expected_counts: dict[str, int] = {}
    for status in statuses:
        if status.get("status") != "passed" or any(status.get(key) is not value for key, value in (("no_training", True), ("no_gradients", True), ("sealed_test_accessed", False), ("significance_generated", False), ("automatic_report_generated", False), ("automatic_selection_generated", False), ("training_predictions", 0), ("optimizer_steps", 0), ("backward_passes", 0))) or status.get("network_requests_made") != 0:
            raise ModelDataError("position diagnostic gates drifted")
        native = _mapping(status.get("native_validation"), "position native identity")
        if (
            native.get("prediction_tokens") != _NATIVE_TOKENS
            or native.get("records") != 2645
            or native.get("ordered_target_and_prior_residue_sha256")
            != "dfac8b1d5eeceadc0428a39568b43b3563acd4a3fabfaac7cf450f724c613097"
        ):
            raise ModelDataError("position diagnostic native identity drifted")
        values = _mapping(_mapping(status.get("results"), "diagnostic results").get("bins"), "diagnostic bins")
        if set(values) != set(bins):
            raise ModelDataError("position diagnostic bins drifted")
        for name in bins:
            item = _mapping(values[name], "diagnostic bin")
            c20_item, e64_item = _mapping(item.get("context20"), "context20 bin"), _mapping(item.get("embedding64"), "embedding64 bin")
            count = _int(c20_item.get("token_count"), "position bin token count")
            _validated_metric(c20_item, count, "position bin")
            _validated_metric(e64_item, count, "position bin")
            if c20_item["token_count"] != e64_item["token_count"]:
                raise ModelDataError("position bin arm counts differ")
            if name in expected_counts and expected_counts[name] != count:
                raise ModelDataError("position bin counts drifted across seeds")
            expected_counts[name] = count
            if not math.isclose(item["embedding64_minus_context20_cross_entropy"], e64_item["cross_entropy"] - c20_item["cross_entropy"], abs_tol=1e-12) or not math.isclose(item["context20_minus_embedding64_accuracy"], c20_item["accuracy"] - e64_item["accuracy"], abs_tol=1e-12):
                raise ModelDataError("position signed delta is inconsistent")
            by_bin[name].append(item)
        overall = _mapping(_mapping(status.get("results"), "diagnostic results").get("overall"), "position overall")
        for arm in ("context20", "embedding64"):
            metric = _mapping(overall.get(arm), "position overall arm")
            _validated_metric(metric, _NATIVE_TOKENS, "position overall")
            bin_nll = _sum_numbers(
                [_mapping(values[name], "bin")[arm]["nll_numerator"] for name in bins]
            )
            bin_correct = sum(
                _int(_mapping(values[name], "bin")[arm]["correct_predictions"], "bin correct")
                for name in bins
            )
            if (
                not math.isclose(
                    _number(metric["nll_numerator"], "overall NLL"), bin_nll,
                    rel_tol=0.0, abs_tol=_NLL_SUM_ABS_TOLERANCE,
                )
                or metric["correct_predictions"] != bin_correct
            ):
                raise ModelDataError("position bins do not reconstruct overall")
        if (
            not math.isclose(
                _number(overall.get("embedding64_minus_context20_cross_entropy"), "overall CE delta"),
                _number(_mapping(overall["embedding64"], "overall E64")["cross_entropy"], "overall E64 CE")
                - _number(_mapping(overall["context20"], "overall C20")["cross_entropy"], "overall C20 CE"),
                abs_tol=1e-12,
            )
            or not math.isclose(
                _number(overall.get("context20_minus_embedding64_accuracy"), "overall accuracy delta"),
                _number(_mapping(overall["context20"], "overall C20")["accuracy"], "overall C20 accuracy")
                - _number(_mapping(overall["embedding64"], "overall E64")["accuracy"], "overall E64 accuracy"),
                abs_tol=1e-12,
            )
        ):
            raise ModelDataError("position overall signed delta is inconsistent")
    if sum(expected_counts.values()) != _NATIVE_TOKENS:
        raise ModelDataError("position bins do not cover native targets")
    for name in bins:
        items = by_bin[name]
        token_count = _int(_mapping(items[0].get("context20"), "context20 bin").get("token_count"), "bin token count")
        c20 = mean(_number(_mapping(item["context20"], "context20 bin")["cross_entropy"], "bin CE") for item in items)
        e64 = mean(_number(_mapping(item["embedding64"], "embedding64 bin")["cross_entropy"], "bin CE") for item in items)
        advantage = (e64 - c20) * token_count
        total_advantage += advantage
        rows.append({"bin": name, "token_count": token_count, "context20_mean_cross_entropy": c20, "embedding64_mean_cross_entropy": e64, "embedding64_minus_context20_cross_entropy": e64 - c20, "mean_nll_advantage": advantage})
    for row in rows:
        row["share_of_total_mean_nll_advantage"] = row["mean_nll_advantage"] / total_advantage
    return {"descriptive_only": True, "no_training": True, "bins": rows}


def deterministic_pca(embedding: np.ndarray) -> list[list[float]]:
    if embedding.shape != (21, 32) or not np.isfinite(embedding).all():
        raise ModelDataError("embedding tensor shape or values are invalid")
    centered = embedding.astype(np.float64, copy=False) - embedding.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].copy()
    for index, component in enumerate(components):
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            components[index] *= -1
    return (centered @ components.T).tolist()


def cosine_summary(embeddings: dict[int, np.ndarray]) -> list[dict[str, object]]:
    if set(embeddings) != {20260821, 20260822, 20260823}:
        raise ModelDataError("embedding summaries require all final seeds")
    output = []
    for left, right in itertools.combinations(range(1, 21), 2):
        values = []
        for embedding in embeddings.values():
            a, b = embedding[left], embedding[right]
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if not math.isfinite(denominator) or denominator == 0.0:
                raise ModelDataError("residue cosine input is nonfinite or zero-norm")
            values.append(float(np.dot(a, b) / denominator))
        output.append({"residue_pair": f"{RESIDUES[left - 1]}-{RESIDUES[right - 1]}", "mean_cosine_similarity": mean(values), "sample_standard_deviation": stdev(values)})
    return output


def reject_forbidden_keys(value: object, forbidden: Iterable[str]) -> None:
    forbidden_normalized = {item.lower() for item in forbidden}
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in forbidden_normalized:
                raise ModelDataError("public payload contains a forbidden key")
            reject_forbidden_keys(item, forbidden)
    elif isinstance(value, list):
        for item in value:
            reject_forbidden_keys(item, forbidden)


def build_payload(*, config: PublicationConfig, config_hash: str, revision: str, context20: list[dict[str, object]], embedding64: list[dict[str, object]], baseline: dict[str, float], capacity: list[dict[str, object]], diagnostic: dict[str, object], pca: list[dict[str, object]], cosine: list[dict[str, object]], source_pins: list[dict[str, str]]) -> dict[str, object]:
    c20_aggregate, e64_aggregate = aggregate(context20), aggregate(embedding64)
    expected = config.expected
    _expect(c20_aggregate["mean_cross_entropy"], expected["context20_mean_cross_entropy"])
    _expect(c20_aggregate["sample_standard_deviation_cross_entropy"], expected["context20_sample_standard_deviation"])
    _expect(e64_aggregate["mean_cross_entropy"], expected["embedding64_mean_cross_entropy"])
    _expect(e64_aggregate["sample_standard_deviation_cross_entropy"], expected["embedding64_sample_standard_deviation"])
    _expect(c20_aggregate["mean_accuracy"], expected["context20_mean_accuracy"])
    _expect(e64_aggregate["mean_accuracy"], expected["embedding64_mean_accuracy"])
    _expect(baseline["cross_entropy"], expected["baseline_cross_entropy"])
    _expect(baseline["accuracy"], expected["baseline_accuracy"])
    _expect(e64_aggregate["mean_cross_entropy"] - c20_aggregate["mean_cross_entropy"], expected["embedding64_minus_context20_mean_cross_entropy"])
    _expect(baseline["cross_entropy"] - c20_aggregate["mean_cross_entropy"], expected["context20_cross_entropy_gain_over_baseline"])
    _expect(c20_aggregate["mean_accuracy"] - baseline["accuracy"], expected["context20_accuracy_gain_over_baseline"])
    by_arm = {str(row["arm"]): row for row in capacity}
    _expect(by_arm["context_20"]["mean_cross_entropy"], expected["capacity_context20_cross_entropy"])
    _expect(by_arm["embedding_64"]["mean_cross_entropy"], expected["capacity_embedding64_cross_entropy"])
    _expect(by_arm["hidden_1600"]["mean_cross_entropy"], expected["capacity_hidden1600_cross_entropy"])
    _expect(float(diagnostic["bins"][2]["share_of_total_mean_nll_advantage"]), expected["position_20_plus_advantage_share"])
    payload = {
        "schema_version": 1, "scope": "week_03_mlp_public_report", "contract_identifier": config.contract_identifier,
        "status": "passed", "publication_configuration_sha256": config_hash, "publication_code_revision": revision,
        "hard_gates": {"validated_pinned_sources": True, "aggregate_only": True, "sealed_test_inaccessible": True, "no_network_requests": True, "no_training_or_evaluation": True},
        "source_pins": source_pins,
        "model": {"architecture": "lookup_flatten_tanh_hidden_logits", "context20": {"context_length": 20, "embedding_width": 32, "hidden_width": 800, "parameter_count": 530293}, "prediction_budget": 100000000, "learning_rate_schedule": "SGD 0.1 before 90M predictions, 0.01 from 90M"},
        "learning_curves": {"stage": "frozen_capacity_and_final_continuation", "points": []},
        "final_three_seed_comparison": {"context20": {"records": context20, "aggregate": c20_aggregate}, "embedding64_challenger": {"records": embedding64, "aggregate": e64_aggregate, "material_gap": expected["material_gap"], "embedding64_minus_context20_mean_cross_entropy": e64_aggregate["mean_cross_entropy"] - c20_aggregate["mean_cross_entropy"]}},
        "fixed_budget_baseline_comparison": {"baseline": {"model": "Week 2 family-aware neural bigram", **baseline, "prediction_budget": 100000000}, "context20_cross_entropy_gain": baseline["cross_entropy"] - c20_aggregate["mean_cross_entropy"], "context20_accuracy_gain": c20_aggregate["mean_accuracy"] - baseline["accuracy"], "significance_claim": False},
        "capacity_screen_25m": capacity,
        "negative_and_exploratory_outcomes": {"label": "exploratory outcomes do not reopen model selection", "lr_tails": [], "one_epoch_continuation": []},
        "position_availability_diagnostic": diagnostic,
        "embedding_diagnostics": {"method": "centered_numpy_svd_pca_per_seed_with_canonical_component_signs", "axes_comparable_across_seeds": False, "tokens": ["BOS", *RESIDUES], "pca_coordinates": pca, "residue_cosine_similarity": {"scope": "within_seed_residue_pairs_only_BOS_excluded", "pairs": cosine}},
        "claim_boundaries": ["Validation only. No sealed-test result.", "PCA and cosine summaries are descriptive, not biological mechanism evidence.", "A causal protein language model is a statistical factorization. Ribosomes read mRNA codons and do not choose residues from prior amino acids.", "Remaining cross-entropy can reflect conditional variability and information absent from this fixed context, including family, function, global fold, distant residues, and future context."],
        "network_requests_made": 0,
    }
    reject_forbidden_keys(payload, config.forbidden_public_keys)
    return payload


def _endpoint(value: object, position: int) -> dict[str, object]:
    if not isinstance(value, list):
        raise ModelDataError("metric milestones are malformed")
    matches = [item for item in value if isinstance(item, dict) and item.get("prediction_position") == position]
    if len(matches) != 1:
        raise ModelDataError("required metric endpoint is missing or duplicated")
    return matches[0]


def _metric(value: dict[str, object]) -> dict[str, object]:
    result = {key: value.get(key) for key in ("cross_entropy", "accuracy", "token_count", "nll_numerator", "correct_predictions")}
    _validated_metric(result, _NATIVE_TOKENS, "native validation")
    return result


def _validated_metric(value: dict[str, object], token_count: int, label: str) -> None:
    if _int(value.get("token_count"), f"{label} token count") != token_count:
        raise ModelDataError(f"{label} token count drifted")
    ce, accuracy, nll = (_number(value.get(key), f"{label} {key}") for key in ("cross_entropy", "accuracy", "nll_numerator"))
    correct = _int(value.get("correct_predictions"), f"{label} correct predictions")
    if (
        token_count <= 0
        or ce < 0
        or nll < 0
        or not 0 <= accuracy <= 1
        or not 0 <= correct <= token_count
        or not math.isclose(ce, nll / token_count, rel_tol=0.0, abs_tol=1e-9)
        or not math.isclose(accuracy, correct / token_count, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ModelDataError(f"{label} metric arithmetic is inconsistent")


def _sum_numbers(values: Iterable[object]) -> float:
    return sum(float(value) for value in values)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ModelDataError(f"{label} is malformed")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ModelDataError(f"{label} is nonfinite or malformed")
    return float(value)


def _int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ModelDataError(f"{label} is malformed")
    return value


def _expect(found: float, expected: object) -> None:
    if not math.isclose(found, _number(expected, "expected aggregate"), rel_tol=0.0, abs_tol=1e-12):
        raise ModelDataError("published aggregate does not match the frozen contract")
