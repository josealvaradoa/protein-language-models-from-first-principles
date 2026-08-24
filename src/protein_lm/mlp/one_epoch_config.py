"""Frozen contract for the exploratory Week 3 one-epoch continuation."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError


APPROVED_ONE_EPOCH_CONFIG_SHA256 = (
    "3294151b1ccdd496bf42f1b223e5733c900c7401b36bb9da9c2ca2b0b0e811b1"
)
_SEEDS = (20260821, 20260822, 20260823)
_MILESTONES = (124_999_936, 149_999_872, 171_329_454)
_PARENT_PINS = (
    (
        20260821,
        "week3-mlp-seed-20260821-cpu",
        "8da95b90a7b981485e1d50b842280ca04471b7476c7d2fefd92c85b1737ca4a9",
        "5e467dcd5b63f7e93f470e22cdbdaa73044ad73f6b43ad5a6a869eb14cdb6142",
    ),
    (
        20260822,
        "week3-mlp-seed-20260822-cpu",
        "be3346133097de29d8a859a82b745ff63e36151b1cc0874f0ab930586a789ba2",
        "86c039123f18421497f659e05dd2d98cd3f7c95bc01dd426ef4f55d297db36ae",
    ),
    (
        20260823,
        "week3-mlp-seed-20260823-cpu",
        "debbef6655ff5af44d340f06256a7087372bbca6bf0fd690687468b0551239e4",
        "101643b4622050148084ddd1929bc708986dcdd8fc85bdce7065dd2c3207a524",
    ),
)


@dataclass(frozen=True)
class ParentPin:
    seed: int
    run_id: str
    metadata_sha256: str
    tensor_sha256: str


@dataclass(frozen=True)
class OneEpochContinuationConfig:
    schema_version: int
    scope: str
    contract_identifier: str
    base_contract_identifier: str
    base_config_relative_path: str
    base_config_sha256: str
    parent_code_revision: str
    training_collection: str
    native_validation_collection: str
    readiness_report_relative_path: str
    readiness_report_sha256: str
    training_prediction_tokens: int
    training_records: int
    native_validation_prediction_tokens: int
    native_validation_records: int
    parent_prediction_position: int
    parent_optimizer_steps: int
    parent_cursor_prediction_index: int
    parent_cursor_protein_index: int
    parent_cursor_within_protein_target_offset: int
    final_prediction_position: int
    final_optimizer_steps: int
    continuation_optimizer_updates: int
    batch_size: int
    fixed_learning_rate: float
    milestone_predictions: tuple[int, ...]
    output_relative_root: str
    control_three_seed_mean_native_cross_entropy: float
    minimum_mean_native_cross_entropy_improvement: float
    useful_three_seed_mean_native_cross_entropy_at_most: float
    parent_pins: tuple[ParentPin, ...]

    def parent_pin(self, seed: int) -> ParentPin:
        for pin in self.parent_pins:
            if pin.seed == seed:
                return pin
        raise ModelDataError("continuation seed does not have an approved parent pin")

    @property
    def continuation_predictions(self) -> int:
        return self.final_prediction_position - self.parent_prediction_position

    @property
    def final_partial_batch_predictions(self) -> int:
        return self.continuation_predictions % self.batch_size


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_one_epoch_config(path: Path) -> OneEpochContinuationConfig:
    """Accept only the reviewed config bytes, types, and approved values."""

    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError(
            f"could not load one-epoch continuation configuration: {error}"
        ) from error
    if hashlib.sha256(content).hexdigest() != APPROVED_ONE_EPOCH_CONFIG_SHA256:
        raise ModelDataError(
            "one-epoch continuation configuration bytes do not match approval"
        )
    if set(raw) != {field.name for field in fields(OneEpochContinuationConfig)}:
        raise ModelDataError(
            "one-epoch continuation configuration keys differ from schema"
        )
    values = dict(raw)
    pins = values.pop("parent_pins")
    values["parent_pins"] = _parent_pins(pins)
    milestones = values["milestone_predictions"]
    if not isinstance(milestones, list) or any(
        type(item) is not int for item in milestones
    ):
        raise ModelDataError("milestone_predictions must be a list of integers")
    values["milestone_predictions"] = tuple(milestones)
    integer_names = {
        "schema_version",
        "training_prediction_tokens",
        "training_records",
        "native_validation_prediction_tokens",
        "native_validation_records",
        "parent_prediction_position",
        "parent_optimizer_steps",
        "parent_cursor_prediction_index",
        "parent_cursor_protein_index",
        "parent_cursor_within_protein_target_offset",
        "final_prediction_position",
        "final_optimizer_steps",
        "continuation_optimizer_updates",
        "batch_size",
    }
    for name in integer_names:
        if type(values[name]) is not int:
            raise ModelDataError(f"{name} must be an integer")
    float_names = {
        "fixed_learning_rate",
        "control_three_seed_mean_native_cross_entropy",
        "minimum_mean_native_cross_entropy_improvement",
        "useful_three_seed_mean_native_cross_entropy_at_most",
    }
    for name in float_names:
        if type(values[name]) not in (int, float) or not math.isfinite(values[name]):
            raise ModelDataError(f"{name} must be a finite number")
        values[name] = float(values[name])
    for name in (
        set(values)
        - integer_names
        - float_names
        - {
            "milestone_predictions",
            "parent_pins",
        }
    ):
        if not isinstance(values[name], str) or not values[name]:
            raise ModelDataError(f"{name} must be a nonempty string")
    config = OneEpochContinuationConfig(**values)
    _validate(config)
    return config


def _parent_pins(value: object) -> tuple[ParentPin, ...]:
    if not isinstance(value, list) or len(value) != len(_SEEDS):
        raise ModelDataError("continuation parent pins are invalid")
    pins: list[ParentPin] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "seed",
            "run_id",
            "metadata_sha256",
            "tensor_sha256",
        }:
            raise ModelDataError("continuation parent pin schema is invalid")
        if type(item["seed"]) is not int or any(
            not isinstance(item[name], str) or not item[name]
            for name in ("run_id", "metadata_sha256", "tensor_sha256")
        ):
            raise ModelDataError("continuation parent pin types are invalid")
        pins.append(ParentPin(**item))
    return tuple(pins)


def _validate(config: OneEpochContinuationConfig) -> None:
    if (
        config.schema_version != 1
        or config.scope != "week_03_mlp_one_epoch_continuation_exploratory"
        or config.contract_identifier
        != "2026-08-24-week-03-mlp-one-epoch-continuation-v1"
        or config.base_contract_identifier != "2026-08-23-week-03-mlp-training-v1"
        or config.base_config_relative_path
        != "experiments/week_03/mlp_training_v1.toml"
        or config.base_config_sha256
        != "3df5ed5c13e3f562fdf8cdb2166470f2642f9079305347d9d4c513f8b72d18a7"
        or config.parent_code_revision != "b8db080d37c6d2eff97e546d2a0026eac5e624dd"
        or config.training_collection != "family_aware_training"
        or config.native_validation_collection != "family_aware_native_validation"
        or config.readiness_report_relative_path
        != "reports/week_02/model_data_readiness_v1.json"
        or config.readiness_report_sha256
        != "19d4ee82eae49b600e9e83e4bb19d468b7a9fc2cfd6b78ffebc995f77db9b881"
        or config.training_prediction_tokens != 171_329_454
        or config.training_records != 461_132
        or config.native_validation_prediction_tokens != 1_000_495
        or config.native_validation_records != 2_645
        or config.parent_prediction_position != 100_000_000
        or config.parent_optimizer_steps != 97_660
        or config.parent_cursor_prediction_index != 100_000_000
        or config.parent_cursor_protein_index != 269_057
        or config.parent_cursor_within_protein_target_offset != 270
        or config.final_prediction_position != 171_329_454
        or config.final_optimizer_steps != 167_318
        or config.continuation_optimizer_updates != 69_658
        or config.batch_size != 1_024
        or config.fixed_learning_rate != 0.01
        or config.milestone_predictions != _MILESTONES
        or config.output_relative_root
        != "data/processed/week_03/mlp_one_epoch_continuation_runs"
        or config.control_three_seed_mean_native_cross_entropy != 2.870820533107851
        or config.minimum_mean_native_cross_entropy_improvement != 0.001
        or config.useful_three_seed_mean_native_cross_entropy_at_most
        != 2.869820533107851
        or tuple(
            (pin.seed, pin.run_id, pin.metadata_sha256, pin.tensor_sha256)
            for pin in config.parent_pins
        )
        != _PARENT_PINS
    ):
        raise ModelDataError(
            "one-epoch continuation configuration values are not approved"
        )
    if (
        config.continuation_predictions != 71_329_454
        or config.final_partial_batch_predictions != 686
        or (config.continuation_predictions + config.batch_size - 1)
        // config.batch_size
        != config.continuation_optimizer_updates
        or config.parent_optimizer_steps + config.continuation_optimizer_updates
        != config.final_optimizer_steps
        or config.milestone_predictions[:2]
        != (
            config.parent_prediction_position + 24_414 * config.batch_size,
            config.parent_prediction_position + 48_828 * config.batch_size,
        )
        or config.milestone_predictions[-1] != config.final_prediction_position
        or config.useful_three_seed_mean_native_cross_entropy_at_most
        != config.control_three_seed_mean_native_cross_entropy
        - config.minimum_mean_native_cross_entropy_improvement
    ):
        raise ModelDataError("one-epoch continuation arithmetic is not approved")
