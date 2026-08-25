"""Byte-pinned contract for the exploratory 25M Week 3 capacity screen."""

from __future__ import annotations

import hashlib
import math
import statistics
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError


APPROVED_CAPACITY_SCREEN_CONFIG_SHA256 = (
    "78e52264ed569a4f5b4592cab7dada6dc7e71ded1d40ab507909bc2598ee459b"
)
APPROVED_ARMS = ("context_20", "embedding_64", "hidden_1600")
_SEEDS = (20260821, 20260822, 20260823)
_EVENTS = (1_000_000, 5_000_000, 10_000_000, 25_000_000)
_ARM_VALUES = (
    ("context_20", 20, 32, 800, 530_293),
    ("embedding_64", 10, 64, 800, 530_965),
    ("hidden_1600", 10, 32, 1600, 547_893),
)
_CONTROL_VALUES = (
    (
        20260821,
        "week3-mlp-seed-20260821-cpu",
        "ea2928958daadaa41035688df2a35c5059732010bc72e2f414f2210f13771cdf",
        2.8715134820380035,
        0.10837635370491606,
        2872934.8812116124,
        108430,
    ),
    (
        20260822,
        "week3-mlp-seed-20260822-cpu",
        "4a1fd7745c78eb1a4562d338aab63777b5cc9a349e0b31e82777ded97906db91",
        2.8715403197151392,
        0.10832637844267087,
        2872961.732173398,
        108380,
    ),
    (
        20260823,
        "week3-mlp-seed-20260823-cpu",
        "e1c582c9a2ce4a3cf1f7d29e2878a47fa3f5a34f2cd54fd3e236734c03f703e4",
        2.8715817734363047,
        0.10842632896716126,
        2873003.2064141557,
        108480,
    ),
)


@dataclass(frozen=True)
class CapacityArm:
    name: str
    context_length: int
    embedding_width: int
    hidden_width: int
    parameter_count: int


@dataclass(frozen=True)
class ControlRun:
    seed: int
    run_id: str
    run_status_sha256: str
    native_cross_entropy: float
    native_accuracy: float
    native_nll_numerator: float
    native_correct_predictions: int
    optimizer_steps: int
    cursor_prediction_index: int
    cursor_protein_index: int
    cursor_within_protein_target_offset: int
    active_learning_rate: float


@dataclass(frozen=True)
class MLPCapacityScreenConfig:
    schema_version: int
    scope: str
    contract_identifier: str
    base_contract_identifier: str
    base_config_relative_path: str
    base_config_sha256: str
    control_code_revision: str
    training_collection: str
    native_validation_collection: str
    readiness_report_relative_path: str
    readiness_report_sha256: str
    training_prediction_tokens: int
    training_records: int
    native_validation_prediction_tokens: int
    native_validation_records: int
    prediction_budget: int
    batch_size: int
    event_predictions: tuple[int, ...]
    run_seeds: tuple[int, ...]
    fixed_learning_rate: float
    momentum: float
    weight_decay: float
    output_relative_root: str
    control_three_seed_mean_native_cross_entropy: float
    control_three_seed_sample_standard_deviation: float
    minimum_mean_native_cross_entropy_improvement: float
    qualifying_mean_native_cross_entropy_at_most: float
    arms: tuple[CapacityArm, ...]
    control_runs: tuple[ControlRun, ...]

    def arm(self, name: str) -> CapacityArm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise ModelDataError("capacity-screen arm is not approved")

    def expected_optimizer_steps(self, predictions: int) -> int:
        if (
            type(predictions) is not int
            or predictions < 0
            or predictions > self.prediction_budget
        ):
            raise ModelDataError("capacity-screen prediction position is invalid")
        total = 0
        previous = 0
        for event in self.event_predictions:
            end = min(event, predictions)
            if end > previous:
                total += math.ceil((end - previous) / self.batch_size)
            previous = event
            if event >= predictions:
                return total
        return total


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_capacity_screen_config(path: Path) -> MLPCapacityScreenConfig:
    """Load exact reviewed bytes, then reject every unapproved type or value."""

    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError(
            f"could not load capacity-screen configuration: {error}"
        ) from error
    if hashlib.sha256(content).hexdigest() != APPROVED_CAPACITY_SCREEN_CONFIG_SHA256:
        raise ModelDataError(
            "capacity-screen configuration bytes do not match approval"
        )
    if set(raw) != {field.name for field in fields(MLPCapacityScreenConfig)}:
        raise ModelDataError("capacity-screen configuration keys differ from schema")
    values = dict(raw)
    values["arms"] = _arms(values["arms"])
    values["control_runs"] = _control_runs(values["control_runs"])
    for name in ("event_predictions", "run_seeds"):
        value = values[name]
        if not isinstance(value, list) or any(type(item) is not int for item in value):
            raise ModelDataError(f"{name} must be a list of integers")
        values[name] = tuple(value)
    integer_names = {
        "schema_version",
        "training_prediction_tokens",
        "training_records",
        "native_validation_prediction_tokens",
        "native_validation_records",
        "prediction_budget",
        "batch_size",
    }
    for name in integer_names:
        if type(values[name]) is not int:
            raise ModelDataError(f"{name} must be an integer")
    float_names = {
        "fixed_learning_rate",
        "momentum",
        "weight_decay",
        "control_three_seed_mean_native_cross_entropy",
        "control_three_seed_sample_standard_deviation",
        "minimum_mean_native_cross_entropy_improvement",
        "qualifying_mean_native_cross_entropy_at_most",
    }
    for name in float_names:
        if type(values[name]) not in (int, float) or not math.isfinite(values[name]):
            raise ModelDataError(f"{name} must be a finite number")
        values[name] = float(values[name])
    for name in (
        set(values)
        - integer_names
        - float_names
        - {"event_predictions", "run_seeds", "arms", "control_runs"}
    ):
        if not isinstance(values[name], str) or not values[name]:
            raise ModelDataError(f"{name} must be a nonempty string")
    config = MLPCapacityScreenConfig(**values)
    _validate(config)
    return config


def _arms(value: object) -> tuple[CapacityArm, ...]:
    if not isinstance(value, list) or len(value) != len(APPROVED_ARMS):
        raise ModelDataError("capacity-screen arms are invalid")
    arms = []
    for item in value:
        if not isinstance(item, dict) or set(item) != set(
            CapacityArm.__dataclass_fields__
        ):
            raise ModelDataError("capacity-screen arm schema is invalid")
        if not isinstance(item["name"], str) or any(
            type(item[name]) is not int for name in set(item) - {"name"}
        ):
            raise ModelDataError("capacity-screen arm types are invalid")
        arms.append(CapacityArm(**item))
    return tuple(arms)


def _control_runs(value: object) -> tuple[ControlRun, ...]:
    if not isinstance(value, list) or len(value) != len(_SEEDS):
        raise ModelDataError("capacity-screen control provenance is invalid")
    runs = []
    integers = {
        "seed",
        "native_correct_predictions",
        "optimizer_steps",
        "cursor_prediction_index",
        "cursor_protein_index",
        "cursor_within_protein_target_offset",
    }
    floats = {
        "native_cross_entropy",
        "native_accuracy",
        "native_nll_numerator",
        "active_learning_rate",
    }
    for item in value:
        if not isinstance(item, dict) or set(item) != set(
            ControlRun.__dataclass_fields__
        ):
            raise ModelDataError("capacity-screen control provenance schema is invalid")
        if (
            any(type(item[name]) is not int for name in integers)
            or any(
                type(item[name]) not in (int, float) or not math.isfinite(item[name])
                for name in floats
            )
            or any(
                not isinstance(item[name], str) or not item[name]
                for name in {"run_id", "run_status_sha256"}
            )
        ):
            raise ModelDataError("capacity-screen control provenance types are invalid")
        item = dict(item)
        for name in floats:
            item[name] = float(item[name])
        runs.append(ControlRun(**item))
    return tuple(runs)


def _parameter_count(
    context_length: int, embedding_width: int, hidden_width: int
) -> int:
    return (
        21 * embedding_width
        + context_length * embedding_width * hidden_width
        + hidden_width
        + hidden_width * 21
        + 21
    )


def _validate(config: MLPCapacityScreenConfig) -> None:
    strings = (
        config.scope == "week_03_mlp_capacity_allocation_screen_exploratory"
        and config.contract_identifier == "2026-08-24-week-03-mlp-capacity-screen-v1"
        and config.base_contract_identifier == "2026-08-23-week-03-mlp-training-v1"
        and config.base_config_relative_path
        == "experiments/week_03/mlp_training_v1.toml"
        and config.base_config_sha256
        == "3df5ed5c13e3f562fdf8cdb2166470f2642f9079305347d9d4c513f8b72d18a7"
        and config.control_code_revision == "b8db080d37c6d2eff97e546d2a0026eac5e624dd"
        and config.training_collection == "family_aware_training"
        and config.native_validation_collection == "family_aware_native_validation"
        and config.readiness_report_relative_path
        == "reports/week_02/model_data_readiness_v1.json"
        and config.readiness_report_sha256
        == "19d4ee82eae49b600e9e83e4bb19d468b7a9fc2cfd6b78ffebc995f77db9b881"
        and config.output_relative_root
        == "data/processed/week_03/mlp_capacity_screen_runs"
    )
    numeric = (
        config.schema_version == 1
        and config.training_prediction_tokens == 171_329_454
        and config.training_records == 461_132
        and config.native_validation_prediction_tokens == 1_000_495
        and config.native_validation_records == 2_645
        and config.prediction_budget == 25_000_000
        and config.batch_size == 1_024
        and config.event_predictions == _EVENTS
        and config.run_seeds == _SEEDS
        and config.fixed_learning_rate == 0.1
        and config.momentum == 0.0
        and config.weight_decay == 0.0
        and config.control_three_seed_mean_native_cross_entropy == 2.871545191729816
        and config.control_three_seed_sample_standard_deviation
        == 0.00003440539442791946
        and config.minimum_mean_native_cross_entropy_improvement == 0.001
        and config.qualifying_mean_native_cross_entropy_at_most == 2.870545191729816
    )
    arms = tuple(
        (
            arm.name,
            arm.context_length,
            arm.embedding_width,
            arm.hidden_width,
            arm.parameter_count,
        )
        for arm in config.arms
    )
    controls = tuple(
        (
            run.seed,
            run.run_id,
            run.run_status_sha256,
            run.native_cross_entropy,
            run.native_accuracy,
            run.native_nll_numerator,
            run.native_correct_predictions,
        )
        for run in config.control_runs
    )
    if not strings or not numeric or arms != _ARM_VALUES or controls != _CONTROL_VALUES:
        raise ModelDataError("capacity-screen configuration values are not approved")
    if any(
        _parameter_count(arm.context_length, arm.embedding_width, arm.hidden_width)
        != arm.parameter_count
        for arm in config.arms
    ):
        raise ModelDataError("capacity-screen arm parameter counts are invalid")
    if any(
        sum(
            a != b
            for a, b in zip(
                (arm.context_length, arm.embedding_width, arm.hidden_width),
                (10, 32, 800),
                strict=True,
            )
        )
        != 1
        for arm in config.arms
    ):
        raise ModelDataError(
            "capacity-screen arms must change exactly one allocation axis"
        )
    if config.expected_optimizer_steps(config.prediction_budget) != 24_416:
        raise ModelDataError("capacity-screen event arithmetic is invalid")
    if (
        config.qualifying_mean_native_cross_entropy_at_most
        != config.control_three_seed_mean_native_cross_entropy
        - config.minimum_mean_native_cross_entropy_improvement
    ):
        raise ModelDataError("capacity-screen qualification threshold is invalid")
    control_cross_entropies = [run.native_cross_entropy for run in config.control_runs]
    if statistics.fmean(
        control_cross_entropies
    ) != config.control_three_seed_mean_native_cross_entropy or not math.isclose(
        statistics.stdev(control_cross_entropies),
        config.control_three_seed_sample_standard_deviation,
        rel_tol=0.0,
        abs_tol=1e-20,
    ):
        raise ModelDataError("capacity-screen control summary is invalid")
    if any(
        (
            run.optimizer_steps,
            run.cursor_prediction_index,
            run.cursor_protein_index,
            run.cursor_within_protein_target_offset,
            run.active_learning_rate,
        )
        != (24_416, 25_000_000, 67_233, 99, 0.1)
        for run in config.control_runs
    ):
        raise ModelDataError("capacity-screen control state provenance is invalid")
