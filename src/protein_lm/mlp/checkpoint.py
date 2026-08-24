"""Non-pickle, checksummed checkpoints for exact Week 3 continuation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.config import MLPTrainingConfig, config_sha256
from protein_lm.mlp.model import ContextMLP, parameter_count
from protein_lm.mlp.stream import StreamCursor
from protein_lm.mlp.training import (
    TrainingState,
    assert_optimizer_contract,
    learning_rate_for,
)


_METADATA = "checkpoint.json"
_TENSORS = "model.safetensors"
_REVISION = re.compile(r"^[0-9a-f]{40}$")


def checkpoint_seed(source: Path) -> int:
    """Read the isolated run seed without deserializing model data."""

    seed = _metadata(source).get("seed")
    if type(seed) is not int:
        raise ModelDataError("checkpoint seed is invalid")
    return seed


def save_checkpoint(
    destination: Path,
    *,
    model: ContextMLP,
    optimizer: torch.optim.SGD,
    state: TrainingState,
    config: MLPTrainingConfig,
    config_path: Path,
    seed: int,
    run_id: str,
    device_name: str,
    code_revision: str,
) -> Path:
    """Create one new atomic checkpoint directory. Existing paths are refused."""

    if destination.exists():
        raise ModelDataError("checkpoint destination already exists")
    _validate_state(state, config)
    if _REVISION.fullmatch(code_revision) is None:
        raise ModelDataError("checkpoint code revision is invalid")
    assert_optimizer_contract(optimizer, config)
    if state.predictions_seen not in config.checkpoint_predictions:
        raise ModelDataError("checkpoint is not at an approved event")
    if state.optimizer_steps != config.expected_optimizer_steps(state.predictions_seen):
        raise ModelDataError("checkpoint optimizer-step accounting is invalid")
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ModelDataError("checkpoint model parameters are nonfinite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise ModelDataError("checkpoint temporary path already exists")
    temporary.mkdir()
    try:
        tensors = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        }
        tensor_path = temporary / _TENSORS
        save_file(tensors, str(tensor_path))
        payload = {
            "schema_version": 1,
            "contract_identifier": config.contract_identifier,
            "config_sha256": config_sha256(config_path),
            "source_identity": _source_identity(config),
            "seed": seed,
            "run_id": run_id,
            "device": device_name,
            "code_revision": code_revision,
            "parameter_count": parameter_count(model),
            "predictions_seen": state.predictions_seen,
            "optimizer_steps": state.optimizer_steps,
            "training_loss_numerator": state.training_loss_numerator,
            "next_stream_cursor": _cursor_payload(state.cursor),
            "active_learning_rate": learning_rate_for(state.predictions_seen, config),
            "optimizer": {
                "name": "SGD",
                "momentum": 0.0,
                "weight_decay": 0.0,
                "state": "empty",
            },
            "tensor_file": _TENSORS,
            "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        }
        (temporary / _METADATA).write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
        raise
    return destination


def load_checkpoint(
    source: Path,
    *,
    model: ContextMLP,
    optimizer: torch.optim.SGD,
    config: MLPTrainingConfig,
    config_path: Path,
    seed: int,
    run_id: str,
    device_name: str,
    code_revision: str,
) -> TrainingState:
    """Validate strict JSON and exact finite tensor inventory before loading."""

    metadata = _metadata(source)
    if not source.name.startswith("checkpoint-"):
        raise ModelDataError("checkpoint directory name is invalid")
    try:
        directory_prediction = int(source.name.removeprefix("checkpoint-"))
    except ValueError as error:
        raise ModelDataError("checkpoint directory name is invalid") from error
    tensor_path = source / _TENSORS
    try:
        digest = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ModelDataError("checkpoint is unreadable") from error
    expected = {
        "schema_version",
        "contract_identifier",
        "config_sha256",
        "source_identity",
        "seed",
        "run_id",
        "device",
        "code_revision",
        "parameter_count",
        "predictions_seen",
        "optimizer_steps",
        "training_loss_numerator",
        "next_stream_cursor",
        "active_learning_rate",
        "optimizer",
        "tensor_file",
        "tensor_sha256",
    }
    if set(metadata) != expected:
        raise ModelDataError("checkpoint metadata schema is invalid")
    if (
        metadata["schema_version"] != 1
        or metadata["contract_identifier"] != config.contract_identifier
        or metadata["config_sha256"] != config_sha256(config_path)
        or metadata["source_identity"] != _source_identity(config)
        or metadata["seed"] != seed
        or metadata["run_id"] != run_id
        or metadata["device"] != device_name
        or metadata["code_revision"] != code_revision
        or metadata["parameter_count"] != parameter_count(model)
        or metadata["predictions_seen"] != directory_prediction
        or metadata["tensor_file"] != _TENSORS
        or metadata["tensor_sha256"] != digest
        or not isinstance(metadata["code_revision"], str)
        or _REVISION.fullmatch(metadata["code_revision"]) is None
    ):
        raise ModelDataError("checkpoint identity or integrity check failed")
    cursor = _cursor_from_payload(metadata["next_stream_cursor"])
    state = TrainingState(
        metadata["predictions_seen"],
        metadata["optimizer_steps"],
        float(metadata["training_loss_numerator"]),
        cursor,
    )
    _validate_state(state, config)
    if metadata["active_learning_rate"] != learning_rate_for(
        state.predictions_seen, config
    ):
        raise ModelDataError("checkpoint schedule accounting is invalid")
    if metadata["optimizer"] != {
        "name": "SGD",
        "momentum": 0.0,
        "weight_decay": 0.0,
        "state": "empty",
    }:
        raise ModelDataError("checkpoint optimizer contract is invalid")
    assert_optimizer_contract(optimizer, config)
    try:
        tensors = load_file(str(tensor_path), device="cpu")
    except Exception as error:
        raise ModelDataError("checkpoint tensors are invalid") from error
    _validate_tensor_inventory(tensors, model)
    try:
        model.load_state_dict(tensors, strict=True)
    except Exception as error:
        raise ModelDataError("checkpoint tensors are invalid") from error
    for group in optimizer.param_groups:
        group["lr"] = float(metadata["active_learning_rate"])
    return state


def _metadata(source: Path) -> dict[str, object]:
    try:
        value = json.loads((source / _METADATA).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError("checkpoint is unreadable") from error
    if not isinstance(value, dict):
        raise ModelDataError("checkpoint metadata schema is invalid")
    return value


def _validate_state(state: TrainingState, config: MLPTrainingConfig) -> None:
    if (
        type(state.predictions_seen) is not int
        or type(state.optimizer_steps) is not int
        or state.predictions_seen < 0
        or state.predictions_seen > config.prediction_budget
        or state.optimizer_steps <= 0
        or state.optimizer_steps > state.predictions_seen
        or state.optimizer_steps
        != config.expected_optimizer_steps(state.predictions_seen)
        or not math.isfinite(state.training_loss_numerator)
        or state.training_loss_numerator < 0
        or state.cursor.prediction_index != state.predictions_seen
    ):
        raise ModelDataError("checkpoint accounting is invalid")


def _cursor_payload(cursor: StreamCursor) -> dict[str, int]:
    return {
        "prediction_index": cursor.prediction_index,
        "protein_index": cursor.protein_index,
        "within_protein_target_offset": cursor.within_protein_target_offset,
    }


def _cursor_from_payload(value: object) -> StreamCursor:
    if (
        not isinstance(value, dict)
        or set(value)
        != {"prediction_index", "protein_index", "within_protein_target_offset"}
        or any(type(item) is not int for item in value.values())
    ):
        raise ModelDataError("checkpoint cursor is invalid")
    return StreamCursor(
        value["prediction_index"],
        value["protein_index"],
        value["within_protein_target_offset"],
    )


def _source_identity(config: MLPTrainingConfig) -> dict[str, str]:
    return {
        "model_data_config_sha256": config.model_data_config_sha256,
        "model_data_registry_relative_path": config.model_data_registry_relative_path,
        "model_data_registry_sha256": config.model_data_registry_sha256,
        "training_stream_report_relative_path": config.training_stream_report_relative_path,
        "training_stream_report_sha256": config.training_stream_report_sha256,
    }


def _validate_tensor_inventory(
    tensors: dict[str, torch.Tensor], model: ContextMLP
) -> None:
    expected = model.state_dict()
    if set(tensors) != set(expected):
        raise ModelDataError("checkpoint tensor keys are invalid")
    for name, expected_tensor in expected.items():
        tensor = tensors[name]
        if (
            tensor.dtype != expected_tensor.dtype
            or tensor.shape != expected_tensor.shape
            or not torch.isfinite(tensor).all()
        ):
            raise ModelDataError(
                "checkpoint tensor shape, dtype, or values are invalid"
            )
