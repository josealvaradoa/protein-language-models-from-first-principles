"""Strict configuration for the non-primary Week 3 learning-rate tails."""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError


APPROVED_MLP_LR_TAIL_CONFIG_SHA256 = (
    "76afe3e1a3cef4afa4e4468cb722684746889e0a9b94636fa97f3b35c85648a4"
)
APPROVED_ARMS = ("staged_97m_003", "cosine_90m_100m_001")
_APPROVED_PARENT_PINS = (
    (
        20260821,
        "week3-mlp-seed-20260821-cpu",
        "5a71548408c8619e13c225a69644261d7e670b319d5e6ba98e091327e97eb200",
        "96fd28a532041cf0da77bbeb069d779f7b4a62e13f51844b19990e5b2d156d1d",
    ),
    (
        20260822,
        "week3-mlp-seed-20260822-cpu",
        "4cdf7685ea5b6be8ca602a3f82977e2bf998489ce1ccf8b15eeae3e365c3a0bb",
        "4d85f3a5fb1653154875f7b26ad7d04a9122c21fb620513f15756eec2b4eef40",
    ),
    (
        20260823,
        "week3-mlp-seed-20260823-cpu",
        "f122feb552a9752eb5a579f7cd0cc58ac75b47b2846010aecb55e93659ce94cf",
        "870991be7a797cb93465bc455c5f477d38cd80426d70aa3694d8e0186fc23d14",
    ),
)


@dataclass(frozen=True)
class ParentPin:
    seed: int
    run_id: str
    metadata_sha256: str
    tensor_sha256: str


@dataclass(frozen=True)
class MLPTailConfig:
    schema_version: int
    scope: str
    contract_identifier: str
    base_contract_identifier: str
    base_config_relative_path: str
    base_config_sha256: str
    parent_code_revision: str
    training_collection: str
    native_validation_collection: str
    parent_prediction_position: int
    parent_optimizer_steps: int
    parent_cursor_prediction_index: int
    parent_cursor_protein_index: int
    parent_cursor_within_protein_target_offset: int
    final_prediction_position: int
    final_optimizer_steps: int
    tail_optimizer_updates: int
    batch_size: int
    base_tail_learning_rate: float
    staged_boundary_prediction: int
    staged_lower_learning_rate: float
    cosine_endpoint_learning_rate: float
    output_relative_root: str
    approved_arms: tuple[str, ...]
    parent_pins: tuple[ParentPin, ...]

    def parent_pin(self, seed: int) -> ParentPin:
        for pin in self.parent_pins:
            if pin.seed == seed:
                return pin
        raise ModelDataError("tail seed does not have an approved parent pin")


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tail_config(path: Path) -> MLPTailConfig:
    """Load the approved bytes before accepting any values from this config."""

    try:
        content = path.read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ModelDataError(
            f"could not load MLP tail configuration: {error}"
        ) from error
    if hashlib.sha256(content).hexdigest() != APPROVED_MLP_LR_TAIL_CONFIG_SHA256:
        raise ModelDataError("MLP tail configuration bytes do not match approval")
    expected = set(MLPTailConfig.__dataclass_fields__) - {"parent_pins"}
    if set(raw) != expected | {"parent_pins"}:
        raise ModelDataError("MLP tail configuration keys differ from the schema")
    values = dict(raw)
    pins = values.pop("parent_pins")
    if not isinstance(pins, list) or len(pins) != 3:
        raise ModelDataError("MLP tail parent pins are invalid")
    parent_pins = tuple(_parent_pin(item) for item in pins)
    if not isinstance(values["approved_arms"], list) or any(
        not isinstance(item, str) for item in values["approved_arms"]
    ):
        raise ModelDataError("approved_arms must be a list of strings")
    values["approved_arms"] = tuple(values["approved_arms"])
    integer_names = {
        "schema_version",
        "parent_prediction_position",
        "parent_optimizer_steps",
        "parent_cursor_prediction_index",
        "parent_cursor_protein_index",
        "parent_cursor_within_protein_target_offset",
        "final_prediction_position",
        "final_optimizer_steps",
        "tail_optimizer_updates",
        "batch_size",
        "staged_boundary_prediction",
    }
    for name in integer_names:
        if type(values[name]) is not int:
            raise ModelDataError(f"{name} must be an integer")
    for name in (
        "base_tail_learning_rate",
        "staged_lower_learning_rate",
        "cosine_endpoint_learning_rate",
    ):
        if type(values[name]) not in (int, float) or not math.isfinite(values[name]):
            raise ModelDataError(f"{name} must be a finite number")
        values[name] = float(values[name])
    for name in (
        set(values)
        - integer_names
        - {
            "approved_arms",
            "base_tail_learning_rate",
            "staged_lower_learning_rate",
            "cosine_endpoint_learning_rate",
        }
    ):
        if not isinstance(values[name], str) or not values[name]:
            raise ModelDataError(f"{name} must be a nonempty string")
    config = MLPTailConfig(parent_pins=parent_pins, **values)
    _validate(config)
    return config


def _parent_pin(value: object) -> ParentPin:
    if not isinstance(value, dict) or set(value) != {
        "seed",
        "run_id",
        "metadata_sha256",
        "tensor_sha256",
    }:
        raise ModelDataError("MLP tail parent pin schema is invalid")
    if type(value["seed"]) is not int or any(
        not isinstance(value[name], str) or not value[name]
        for name in ("run_id", "metadata_sha256", "tensor_sha256")
    ):
        raise ModelDataError("MLP tail parent pin types are invalid")
    return ParentPin(**value)


def _validate(config: MLPTailConfig) -> None:
    if (
        config.schema_version != 1
        or config.scope != "week_03_mlp_learning_rate_tail_exploratory"
        or config.contract_identifier != "2026-08-23-week-03-mlp-lr-tail-v1"
        or config.base_contract_identifier != "2026-08-23-week-03-mlp-training-v1"
        or config.base_config_relative_path
        != "experiments/week_03/mlp_training_v1.toml"
        or config.base_config_sha256
        != "3df5ed5c13e3f562fdf8cdb2166470f2642f9079305347d9d4c513f8b72d18a7"
        or config.parent_code_revision != "b8db080d37c6d2eff97e546d2a0026eac5e624dd"
        or config.training_collection != "family_aware_training"
        or config.native_validation_collection != "family_aware_native_validation"
        or config.parent_prediction_position != 90_000_000
        or config.parent_optimizer_steps != 87_894
        or config.parent_cursor_prediction_index != 90_000_000
        or config.parent_cursor_protein_index != 241_967
        or config.parent_cursor_within_protein_target_offset != 221
        or config.final_prediction_position != 100_000_000
        or config.final_optimizer_steps != 97_660
        or config.tail_optimizer_updates != 9_766
        or config.batch_size != 1_024
        or config.base_tail_learning_rate != 0.01
        or config.staged_boundary_prediction != 97_000_000
        or config.staged_lower_learning_rate != 0.003
        or config.cosine_endpoint_learning_rate != 0.001
        or config.output_relative_root != "data/processed/week_03/mlp_lr_tail_runs"
        or config.approved_arms != APPROVED_ARMS
        or tuple(
            (pin.seed, pin.run_id, pin.metadata_sha256, pin.tensor_sha256)
            for pin in config.parent_pins
        )
        != _APPROVED_PARENT_PINS
    ):
        raise ModelDataError("MLP tail configuration values are not approved")
