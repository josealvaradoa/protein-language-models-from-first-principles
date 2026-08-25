"""Byte-pinned contract for the Week 3 C=20 continuation to 100M."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError


APPROVED_CONTEXT20_CONTINUATION_CONFIG_SHA256 = (
    "a0b1aa37c647aa5dc62c9d7a2b7c051bf66a81380448a5719d9ebe34eb6a3fb6"
)
_SEEDS = (20260821, 20260822, 20260823)
_MILESTONES = (1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000)
_CHECKPOINTS = (
    1_000_000,
    5_000_000,
    10_000_000,
    25_000_000,
    50_000_000,
    90_000_000,
    100_000_000,
)
_PARENTS = (
    (
        20260821,
        "week3-capacity-context-20-seed-20260821-cpu",
        "178a44b3cc02ace3cb2065e2a34f48e6c823f4a7f2166f87779b0552b97ee251",
        "3028e8dd90cf92cfd2827f09ca251e5b8f7a8789412f895bef1a6be6c409c3e1",
        "adaafe22d37b3578b9d941bad84243deece1069740921643b10fb08fa2ee6956",
        71547700.54751587,
    ),
    (
        20260822,
        "week3-capacity-context-20-seed-20260822-cpu",
        "d85c2d8d0b991f8a026a5b9cd6da65911acb82e0e69a315eb411189769c06b86",
        "da084beace76f64d3fa3087a2aee8bc3df23ec1474970b4982dcbdeaab46eb22",
        "bd08fcb2cc9ce350c8d8cf7755d1fe30253cc55a2f4b5a60f9439a70ca5dbb7d",
        71550640.13130188,
    ),
    (
        20260823,
        "week3-capacity-context-20-seed-20260823-cpu",
        "84c73968a1261ecda04d095dd7e67ac1e02a4f1e0a9365cd405a07db8095d323",
        "df303c5617c4994b985f829590d194c5dae1f67e4ef6f11892892d7ff8b8cda2",
        "1c969e4e8513a103330a08ff71387011dedc716c8f1febd1c1ab5da028859786",
        71551996.43241882,
    ),
)
_CONTROLS = (
    (
        20260821,
        "week3-mlp-seed-20260821-cpu",
        "ea2928958daadaa41035688df2a35c5059732010bc72e2f414f2210f13771cdf",
        2.8708225749287593,
        0.10900004497773602,
        2872243.632103349,
        109054,
    ),
    (
        20260822,
        "week3-mlp-seed-20260822-cpu",
        "4a1fd7745c78eb1a4562d338aab63777b5cc9a349e0b31e82777ded97906db91",
        2.8708150645367714,
        0.10889209841128641,
        2872236.117993717,
        108946,
    ),
    (
        20260823,
        "week3-mlp-seed-20260823-cpu",
        "e1c582c9a2ce4a3cf1f7d29e2878a47fa3f5a34f2cd54fd3e236734c03f703e4",
        2.8708239598580225,
        0.10889209841128641,
        2872245.017718152,
        108946,
    ),
)


@dataclass(frozen=True)
class ParentRun:
    seed: int
    run_id: str
    run_status_sha256: str
    metadata_sha256: str
    tensor_sha256: str
    training_loss_numerator: float


@dataclass(frozen=True)
class ControlRun:
    seed: int
    run_id: str
    run_status_sha256: str
    native_cross_entropy: float
    native_accuracy: float
    native_nll_numerator: float
    native_correct_predictions: int


@dataclass(frozen=True)
class Context20ContinuationConfig:
    schema_version: int
    scope: str
    contract_identifier: str
    base_contract_identifier: str
    base_config_relative_path: str
    base_config_sha256: str
    capacity_config_relative_path: str
    capacity_config_sha256: str
    parent_code_revision: str
    training_collection: str
    native_validation_collection: str
    readiness_report_relative_path: str
    readiness_report_sha256: str
    training_prediction_tokens: int
    training_records: int
    native_validation_prediction_tokens: int
    native_validation_records: int
    context_length: int
    embedding_width: int
    hidden_width: int
    parameter_count: int
    parent_prediction_position: int
    parent_optimizer_steps: int
    parent_cursor_prediction_index: int
    parent_cursor_protein_index: int
    parent_cursor_within_protein_target_offset: int
    final_prediction_position: int
    final_optimizer_steps: int
    continuation_optimizer_updates: int
    final_cursor_prediction_index: int
    final_cursor_protein_index: int
    final_cursor_within_protein_target_offset: int
    batch_size: int
    base_learning_rate: float
    post_boundary_learning_rate: float
    learning_rate_boundary_predictions: int
    historical_milestone_predictions: tuple[int, ...]
    historical_checkpoint_predictions: tuple[int, ...]
    continuation_evaluation_predictions: tuple[int, ...]
    continuation_checkpoint_predictions: tuple[int, ...]
    output_relative_root: str
    control_three_seed_mean_native_cross_entropy: float
    control_three_seed_sample_standard_deviation: float
    minimum_mean_native_cross_entropy_improvement: float
    qualifying_mean_native_cross_entropy_at_most: float
    parent_runs: tuple[ParentRun, ...]
    control_runs: tuple[ControlRun, ...]

    def parent_run(self, seed: int) -> ParentRun:
        for parent in self.parent_runs:
            if parent.seed == seed:
                return parent
        raise ModelDataError("continuation seed is not approved")

    @property
    def event_predictions(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                set(
                    self.historical_milestone_predictions
                    + self.historical_checkpoint_predictions
                )
            )
        )

    @property
    def continuation_predictions(self) -> int:
        return self.final_prediction_position - self.parent_prediction_position


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_context20_continuation_config(path: Path) -> Context20ContinuationConfig:
    """Load only exact reviewed bytes, types, and the frozen continuation values."""

    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError(
            f"could not load context20 continuation configuration: {error}"
        ) from error
    if (
        hashlib.sha256(content).hexdigest()
        != APPROVED_CONTEXT20_CONTINUATION_CONFIG_SHA256
    ):
        raise ModelDataError(
            "context20 continuation configuration bytes do not match approval"
        )
    if set(raw) != {field.name for field in fields(Context20ContinuationConfig)}:
        raise ModelDataError(
            "context20 continuation configuration keys differ from schema"
        )
    values = dict(raw)
    values["parent_runs"] = _runs(values["parent_runs"], ParentRun)
    values["control_runs"] = _runs(values["control_runs"], ControlRun)
    list_names = {
        "historical_milestone_predictions",
        "historical_checkpoint_predictions",
        "continuation_evaluation_predictions",
        "continuation_checkpoint_predictions",
    }
    for name in list_names:
        if not isinstance(values[name], list) or any(
            type(item) is not int for item in values[name]
        ):
            raise ModelDataError(f"{name} must be a list of integers")
        values[name] = tuple(values[name])
    integers = {
        "schema_version",
        "training_prediction_tokens",
        "training_records",
        "native_validation_prediction_tokens",
        "native_validation_records",
        "context_length",
        "embedding_width",
        "hidden_width",
        "parameter_count",
        "parent_prediction_position",
        "parent_optimizer_steps",
        "parent_cursor_prediction_index",
        "parent_cursor_protein_index",
        "parent_cursor_within_protein_target_offset",
        "final_prediction_position",
        "final_optimizer_steps",
        "continuation_optimizer_updates",
        "final_cursor_prediction_index",
        "final_cursor_protein_index",
        "final_cursor_within_protein_target_offset",
        "batch_size",
        "learning_rate_boundary_predictions",
    }
    floats = {
        "base_learning_rate",
        "post_boundary_learning_rate",
        "control_three_seed_mean_native_cross_entropy",
        "control_three_seed_sample_standard_deviation",
        "minimum_mean_native_cross_entropy_improvement",
        "qualifying_mean_native_cross_entropy_at_most",
    }
    for name in integers:
        if type(values[name]) is not int:
            raise ModelDataError(f"{name} must be an integer")
    for name in floats:
        if type(values[name]) not in (int, float) or not math.isfinite(values[name]):
            raise ModelDataError(f"{name} must be a finite number")
        values[name] = float(values[name])
    for name in (
        set(values) - integers - floats - list_names - {"parent_runs", "control_runs"}
    ):
        if not isinstance(values[name], str) or not values[name]:
            raise ModelDataError(f"{name} must be a nonempty string")
    config = Context20ContinuationConfig(**values)
    _validate(config)
    return config


def _runs(
    value: object, kind: type[ParentRun] | type[ControlRun]
) -> tuple[ParentRun, ...] | tuple[ControlRun, ...]:
    if not isinstance(value, list) or len(value) != len(_SEEDS):
        raise ModelDataError("continuation run pins are invalid")
    result = []
    if kind is ParentRun:
        integer_names = {"seed"}
        float_names = {"training_loss_numerator"}
    else:
        integer_names = {"seed", "native_correct_predictions"}
        float_names = {
            "native_cross_entropy",
            "native_accuracy",
            "native_nll_numerator",
        }
    for item in value:
        if not isinstance(item, dict) or set(item) != set(kind.__dataclass_fields__):
            raise ModelDataError("continuation run pin schema is invalid")
        if (
            any(type(item[name]) is not int for name in integer_names)
            or any(
                type(item[name]) not in (int, float) or not math.isfinite(item[name])
                for name in float_names
            )
            or any(
                not isinstance(item[name], str) or not item[name]
                for name in set(item) - integer_names - float_names
            )
        ):
            raise ModelDataError("continuation run pin types are invalid")
        normalized = dict(item)
        for name in float_names:
            normalized[name] = float(normalized[name])
        result.append(kind(**normalized))
    return tuple(result)


def _validate(config: Context20ContinuationConfig) -> None:
    if (
        config.schema_version != 1
        or config.scope != "week_03_mlp_context20_100m_continuation_exploratory"
        or config.contract_identifier
        != "2026-08-25-week-03-context20-100m-continuation-v1"
        or config.base_contract_identifier != "2026-08-23-week-03-mlp-training-v1"
        or config.base_config_relative_path
        != "experiments/week_03/mlp_training_v1.toml"
        or config.base_config_sha256
        != "3df5ed5c13e3f562fdf8cdb2166470f2642f9079305347d9d4c513f8b72d18a7"
        or config.capacity_config_relative_path
        != "experiments/week_03/mlp_capacity_screen_v1.toml"
        or config.capacity_config_sha256
        != "78e52264ed569a4f5b4592cab7dada6dc7e71ded1d40ab507909bc2598ee459b"
        or config.parent_code_revision != "17aa23d0a6a7ec070f0d649bb912f2c422245ad5"
        or config.training_collection != "family_aware_training"
        or config.native_validation_collection != "family_aware_native_validation"
        or config.output_relative_root
        != "data/processed/week_03/mlp_context20_100m_continuation_runs"
        or (
            config.context_length,
            config.embedding_width,
            config.hidden_width,
            config.parameter_count,
        )
        != (20, 32, 800, 530_293)
        or config.historical_milestone_predictions != _MILESTONES
        or config.historical_checkpoint_predictions != _CHECKPOINTS
        or config.continuation_evaluation_predictions != (50_000_000, 100_000_000)
        or config.continuation_checkpoint_predictions
        != (50_000_000, 90_000_000, 100_000_000)
        or config.control_three_seed_mean_native_cross_entropy != 2.870820533107851
        or config.control_three_seed_sample_standard_deviation
        != 0.000004786278252060456
        or config.minimum_mean_native_cross_entropy_improvement != 0.001
        or config.qualifying_mean_native_cross_entropy_at_most != 2.869820533107851
        or tuple(item.seed for item in config.parent_runs) != _SEEDS
        or tuple(item.seed for item in config.control_runs) != _SEEDS
        or tuple(
            (
                item.seed,
                item.run_id,
                item.run_status_sha256,
                item.metadata_sha256,
                item.tensor_sha256,
                item.training_loss_numerator,
            )
            for item in config.parent_runs
        )
        != _PARENTS
        or tuple(
            (
                item.seed,
                item.run_id,
                item.run_status_sha256,
                item.native_cross_entropy,
                item.native_accuracy,
                item.native_nll_numerator,
                item.native_correct_predictions,
            )
            for item in config.control_runs
        )
        != _CONTROLS
    ):
        raise ModelDataError(
            "context20 continuation configuration values are not approved"
        )
    if (
        config.parent_prediction_position != 25_000_000
        or config.parent_optimizer_steps != 24_416
        or config.final_prediction_position != 100_000_000
        or config.final_optimizer_steps != 97_660
        or config.continuation_optimizer_updates != 73_244
        or config.batch_size != 1_024
        or config.base_learning_rate != 0.1
        or config.post_boundary_learning_rate != 0.01
        or config.learning_rate_boundary_predictions != 90_000_000
        or config.event_predictions != _CHECKPOINTS
        or config.qualifying_mean_native_cross_entropy_at_most
        != config.control_three_seed_mean_native_cross_entropy
        - config.minimum_mean_native_cross_entropy_improvement
    ):
        raise ModelDataError("context20 continuation schedule is not approved")
    steps = _expected_steps(
        config.event_predictions, config.batch_size, config.parent_prediction_position
    )
    if (
        steps != config.parent_optimizer_steps
        or _expected_steps(config.event_predictions, config.batch_size, 50_000_000)
        != 48_831
        or _expected_steps(config.event_predictions, config.batch_size, 90_000_000)
        != 87_894
        or _expected_steps(
            config.event_predictions,
            config.batch_size,
            config.final_prediction_position,
        )
        != config.final_optimizer_steps
        or config.final_optimizer_steps - config.parent_optimizer_steps
        != config.continuation_optimizer_updates
    ):
        raise ModelDataError("context20 continuation event arithmetic is not approved")


def _expected_steps(events: tuple[int, ...], batch_size: int, position: int) -> int:
    total, previous = 0, 0
    for event in events:
        end = min(event, position)
        if end > previous:
            total += math.ceil((end - previous) / batch_size)
        previous = event
        if event >= position:
            return total
    return total
