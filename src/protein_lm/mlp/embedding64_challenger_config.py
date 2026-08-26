"""Byte-pinned contract for the Week 3 E=64 25M-to-100M challenger."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError


APPROVED_EMBEDDING64_CHALLENGER_CONFIG_SHA256 = (
    "5d441c6e27746f59428ee1a5ee9d3a0aa32cac28b3dea9c188095c873ed7d92a"
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
_PARENT_LOSSES = (71696935.91693115, 71696552.6525116, 71697877.06741333)


@dataclass(frozen=True)
class ParentRun:
    seed: int
    run_id: str
    run_status_sha256: str
    metadata_sha256: str
    tensor_sha256: str
    training_loss_numerator: float


@dataclass(frozen=True)
class ReferenceContext20Run:
    seed: int
    run_id: str
    run_status_sha256: str
    native_cross_entropy: float
    native_accuracy: float
    native_nll_numerator: float
    native_correct_predictions: int


@dataclass(frozen=True)
class Embedding64ChallengerConfig:
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
    reference_context20_contract_identifier: str
    reference_context20_code_revision: str
    reference_context20_parameter_count: int
    reference_context20_three_seed_mean_native_cross_entropy: float
    reference_context20_three_seed_sample_standard_deviation: float
    minimum_material_mean_native_cross_entropy_gap: float
    context20_materially_better_if_embedding64_mean_at_or_above: float
    embedding64_materially_better_if_embedding64_mean_at_or_below: float
    challenger_selection_scope: str
    challenger_selection_basis: str
    embedding64_25m_three_seed_mean_native_cross_entropy: float
    hidden1600_25m_three_seed_mean_native_cross_entropy: float
    parent_runs: tuple[ParentRun, ...]
    reference_context20_runs: tuple[ReferenceContext20Run, ...]

    def parent_run(self, seed: int) -> ParentRun:
        for run in self.parent_runs:
            if run.seed == seed:
                return run
        raise ModelDataError("challenger seed is not approved")

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


def load_embedding64_challenger_config(path: Path) -> Embedding64ChallengerConfig:
    """Load exact approved bytes, then reject every unapproved type and value."""

    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError(
            f"could not load embedding64 challenger configuration: {error}"
        ) from error
    if (
        hashlib.sha256(content).hexdigest()
        != APPROVED_EMBEDDING64_CHALLENGER_CONFIG_SHA256
    ):
        raise ModelDataError(
            "embedding64 challenger configuration bytes do not match approval"
        )
    if set(raw) != {field.name for field in fields(Embedding64ChallengerConfig)}:
        raise ModelDataError(
            "embedding64 challenger configuration keys differ from schema"
        )
    values = dict(raw)
    values["parent_runs"] = _runs(values["parent_runs"], ParentRun)
    values["reference_context20_runs"] = _runs(
        values["reference_context20_runs"], ReferenceContext20Run
    )
    lists = {
        "historical_milestone_predictions",
        "historical_checkpoint_predictions",
        "continuation_evaluation_predictions",
        "continuation_checkpoint_predictions",
    }
    for name in lists:
        if not isinstance(values[name], list) or any(
            type(item) is not int for item in values[name]
        ):
            raise ModelDataError(f"{name} must be a list of integers")
        values[name] = tuple(values[name])
    integers = {
        name
        for name, field in Embedding64ChallengerConfig.__dataclass_fields__.items()
        if field.type == "int"
    }
    floats = {
        name
        for name, field in Embedding64ChallengerConfig.__dataclass_fields__.items()
        if field.type == "float"
    }
    for name in integers:
        if type(values[name]) is not int:
            raise ModelDataError(f"{name} must be an integer")
    for name in floats:
        if type(values[name]) not in (int, float) or not math.isfinite(values[name]):
            raise ModelDataError(f"{name} must be a finite number")
        values[name] = float(values[name])
    for name in (
        set(values)
        - integers
        - floats
        - lists
        - {"parent_runs", "reference_context20_runs"}
    ):
        if not isinstance(values[name], str) or not values[name]:
            raise ModelDataError(f"{name} must be a nonempty string")
    config = Embedding64ChallengerConfig(**values)
    _validate(config)
    return config


def _runs(
    value: object, kind: type[ParentRun] | type[ReferenceContext20Run]
) -> tuple[ParentRun, ...] | tuple[ReferenceContext20Run, ...]:
    if not isinstance(value, list) or len(value) != len(_SEEDS):
        raise ModelDataError("challenger run pins are invalid")
    result = []
    ints = {"seed"} | (
        {"native_correct_predictions"} if kind is ReferenceContext20Run else set()
    )
    floats = (
        {"training_loss_numerator"}
        if kind is ParentRun
        else {"native_cross_entropy", "native_accuracy", "native_nll_numerator"}
    )
    for item in value:
        if not isinstance(item, dict) or set(item) != set(kind.__dataclass_fields__):
            raise ModelDataError("challenger run pin schema is invalid")
        if (
            any(type(item[name]) is not int for name in ints)
            or any(
                type(item[name]) not in (int, float) or not math.isfinite(item[name])
                for name in floats
            )
            or any(
                not isinstance(item[name], str) or not item[name]
                for name in set(item) - ints - floats
            )
        ):
            raise ModelDataError("challenger run pin types are invalid")
        normalized = dict(item)
        for name in floats:
            normalized[name] = float(normalized[name])
        result.append(kind(**normalized))
    return tuple(result)


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


def _validate(config: Embedding64ChallengerConfig) -> None:
    if (
        config.schema_version != 1
        or config.scope != "week_03_mlp_embedding64_100m_challenger_exploratory"
        or config.contract_identifier
        != "2026-08-25-week-03-embedding64-100m-challenger-v1"
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
        or config.readiness_report_sha256
        != "19d4ee82eae49b600e9e83e4bb19d468b7a9fc2cfd6b78ffebc995f77db9b881"
        or (
            config.context_length,
            config.embedding_width,
            config.hidden_width,
            config.parameter_count,
        )
        != (10, 64, 800, 530965)
        or config.historical_milestone_predictions != _MILESTONES
        or config.historical_checkpoint_predictions != _CHECKPOINTS
        or config.continuation_evaluation_predictions != (50_000_000, 100_000_000)
        or config.continuation_checkpoint_predictions
        != (50_000_000, 90_000_000, 100_000_000)
        or tuple(item.seed for item in config.parent_runs) != _SEEDS
        or tuple(item.seed for item in config.reference_context20_runs) != _SEEDS
        or tuple(item.training_loss_numerator for item in config.parent_runs)
        != _PARENT_LOSSES
    ):
        raise ModelDataError(
            "embedding64 challenger configuration values are not approved"
        )
    if (
        config.training_prediction_tokens != 171329454
        or config.training_records != 461132
        or config.native_validation_prediction_tokens != 1000495
        or config.native_validation_records != 2645
        or (config.parent_prediction_position, config.parent_optimizer_steps)
        != (25000000, 24416)
        or (
            config.final_prediction_position,
            config.final_optimizer_steps,
            config.continuation_optimizer_updates,
        )
        != (100000000, 97660, 73244)
        or (
            config.final_cursor_prediction_index,
            config.final_cursor_protein_index,
            config.final_cursor_within_protein_target_offset,
        )
        != (100000000, 269057, 270)
        or config.batch_size != 1024
        or config.base_learning_rate != 0.1
        or config.post_boundary_learning_rate != 0.01
        or config.learning_rate_boundary_predictions != 90000000
        or config.event_predictions != _CHECKPOINTS
        or config.reference_context20_parameter_count != 530293
        or config.minimum_material_mean_native_cross_entropy_gap != 0.001
        or config.context20_materially_better_if_embedding64_mean_at_or_above
        != config.reference_context20_three_seed_mean_native_cross_entropy + 0.001
        or config.embedding64_materially_better_if_embedding64_mean_at_or_below
        != config.reference_context20_three_seed_mean_native_cross_entropy - 0.001
        or config.challenger_selection_scope
        != "post_screen_adversarial_challenger_not_open_model_selection"
        or config.challenger_selection_basis
        != "lowest_25m_mean_native_cross_entropy_non_context_arm"
        or config.embedding64_25m_three_seed_mean_native_cross_entropy
        != 2.8713892507036705
        or config.hidden1600_25m_three_seed_mean_native_cross_entropy
        != 2.871764147540805
        or config.embedding64_25m_three_seed_mean_native_cross_entropy
        >= config.hidden1600_25m_three_seed_mean_native_cross_entropy
        or _expected_steps(config.event_predictions, config.batch_size, 25000000)
        != 24416
        or _expected_steps(config.event_predictions, config.batch_size, 50000000)
        != 48831
        or _expected_steps(config.event_predictions, config.batch_size, 90000000)
        != 87894
        or _expected_steps(config.event_predictions, config.batch_size, 100000000)
        != 97660
    ):
        raise ModelDataError(
            "embedding64 challenger schedule or reference is not approved"
        )
