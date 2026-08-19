"""Strict, dual JSON and Safetensors model artifact serialization."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file

from protein_lm.bigram.training import ROLE_SPACE_SIZE, all_zero_weights_sha256
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.tokenization import CANONICAL_AMINO_ACIDS


ModelType = Literal["unigram", "count_bigram", "neural_bigram"]
_MODEL_TYPES = frozenset(("unigram", "count_bigram", "neural_bigram"))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CONTEXT_ROLES = ("BOS", *CANONICAL_AMINO_ACIDS)
_TARGET_ROLES = (*CANONICAL_AMINO_ACIDS, "EOS")
_REQUIRED_METADATA = frozenset(
    (
        "arm",
        "model_type",
        "context_roles",
        "target_roles",
        "stream_sha256",
        "config_sha256",
        "source_identity",
        "code_revision",
        "seed",
        "prediction_pair_budget",
        "batch_size",
        "batches_consumed",
        "optimizer_steps",
        "smoothing_alpha",
        "initial_weights_sha256",
        "optimizer",
    )
)


def write_model_artifacts(
    *,
    json_path: Path,
    safetensors_path: Path,
    model_type: ModelType,
    tensor: torch.Tensor,
    metadata: Mapping[str, object],
) -> None:
    """Write equivalent human-readable JSON and Safetensors files without overwrite."""

    normalized = _validate_artifact(model_type, tensor, metadata)
    if json_path == safetensors_path or json_path.exists() or safetensors_path.exists():
        raise ModelDataError("model artifact destination already exists")
    if json_path.parent != safetensors_path.parent:
        raise ModelDataError("dual model artifacts must share one directory")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "format": "protein_lm.bigram.json.v1",
        "model": {
            "model_type": model_type,
            "tensor_name": "parameters",
            "dtype": _dtype_name(tensor),
            "shape": list(tensor.shape),
            "values": tensor.detach().cpu().tolist(),
        },
        "metadata": normalized,
    }
    safe_metadata = {
        "schema_version": "1",
        "model_type": model_type,
        "tensor_name": "parameters",
        "dtype": _dtype_name(tensor),
        "shape": ",".join(str(size) for size in tensor.shape),
    }
    created: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix=".bigram-model-", dir=json_path.parent
        ) as directory:
            stage = Path(directory)
            staged_json = stage / json_path.name
            staged_safe = stage / safetensors_path.name
            staged_json.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            save_file(
                {"parameters": tensor.detach().cpu().contiguous()},
                staged_safe,
                metadata=safe_metadata,
            )
            staged_type, staged_tensor, staged_metadata = load_model_artifacts(
                json_path=staged_json,
                safetensors_path=staged_safe,
            )
            if (
                staged_type != model_type
                or not torch.equal(staged_tensor, tensor.detach().cpu())
                or staged_metadata != normalized
            ):
                raise ModelDataError("staged model artifacts are not equivalent")
            for source, destination in (
                (staged_json, json_path),
                (staged_safe, safetensors_path),
            ):
                os.link(source, destination)
                created.append(destination)
    except (
        ModelDataError,
        OSError,
        SafetensorError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        for path in created:
            path.unlink(missing_ok=True)
        if isinstance(error, ModelDataError):
            raise
        raise ModelDataError(f"could not install model artifacts: {error}") from error


def load_model_artifacts(
    *, json_path: Path, safetensors_path: Path
) -> tuple[ModelType, torch.Tensor, dict[str, object]]:
    """Reload both serializations and require their type, shape, dtype, and values to agree."""

    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            safe_metadata = handle.metadata()
        safe_tensors = load_file(safetensors_path, device="cpu")
    except (
        OSError,
        SafetensorError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RuntimeError,
    ) as error:
        raise ModelDataError(f"could not load model artifacts: {error}") from error
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "format",
        "model",
        "metadata",
    }:
        raise ModelDataError("JSON model artifact schema is invalid")
    if raw["schema_version"] != 1 or raw["format"] != "protein_lm.bigram.json.v1":
        raise ModelDataError("JSON model artifact version is invalid")
    model = raw["model"]
    metadata = raw["metadata"]
    if not isinstance(model, dict) or not isinstance(metadata, dict):
        raise ModelDataError("JSON model artifact body is invalid")
    if set(model) != {"model_type", "tensor_name", "dtype", "shape", "values"}:
        raise ModelDataError("JSON model artifact fields are invalid")
    try:
        model_type = model["model_type"]
        _validate_json_values(model["values"], model["dtype"])
        json_tensor = torch.tensor(model["values"], dtype=_torch_dtype(model["dtype"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ModelDataError("JSON model tensor is invalid") from error
    if not isinstance(model_type, str) or model_type not in _MODEL_TYPES:
        raise ModelDataError("JSON model type is invalid")
    if model.get("tensor_name") != "parameters" or model.get("shape") != list(
        json_tensor.shape
    ):
        raise ModelDataError("JSON model tensor shape is invalid")
    _validate_artifact(model_type, json_tensor, metadata)
    if keys != ["parameters"] or set(safe_tensors) != {"parameters"}:
        raise ModelDataError("Safetensors artifact tensor names are invalid")
    safe_tensor = safe_tensors["parameters"]
    expected_safe_metadata = {
        "schema_version": "1",
        "model_type": model_type,
        "tensor_name": "parameters",
        "dtype": _dtype_name(json_tensor),
        "shape": ",".join(str(size) for size in json_tensor.shape),
    }
    if safe_metadata != expected_safe_metadata:
        raise ModelDataError("Safetensors artifact metadata disagrees with JSON")
    if (
        safe_tensor.dtype != json_tensor.dtype
        or tuple(safe_tensor.shape) != tuple(json_tensor.shape)
        or not torch.equal(safe_tensor, json_tensor)
    ):
        raise ModelDataError("JSON and Safetensors model values disagree")
    return model_type, json_tensor, dict(metadata)  # type: ignore[return-value]


def _validate_artifact(
    model_type: str, tensor: torch.Tensor, metadata: Mapping[str, object]
) -> dict[str, object]:
    if model_type not in _MODEL_TYPES:
        raise ModelDataError("model type is invalid")
    expected_dtype = torch.float32 if model_type == "neural_bigram" else torch.int64
    expected_shape = (
        (ROLE_SPACE_SIZE,)
        if model_type == "unigram"
        else (ROLE_SPACE_SIZE, ROLE_SPACE_SIZE)
    )
    if (
        tensor.device.type != "cpu"
        or tensor.dtype != expected_dtype
        or tuple(tensor.shape) != expected_shape
    ):
        raise ModelDataError("model tensor has an invalid device, dtype, or shape")
    if model_type == "neural_bigram" and not torch.isfinite(tensor).all().item():
        raise ModelDataError("neural model tensor contains a nonfinite value")
    if set(metadata) != _REQUIRED_METADATA or metadata.get("model_type") != model_type:
        raise ModelDataError("model metadata fields are invalid")
    if not isinstance(metadata["arm"], str) or not metadata["arm"]:
        raise ModelDataError("model metadata arm is invalid")
    if (
        not isinstance(metadata["context_roles"], list)
        or not isinstance(metadata["target_roles"], list)
        or tuple(metadata["context_roles"]) != _CONTEXT_ROLES
        or tuple(metadata["target_roles"]) != _TARGET_ROLES
    ):
        raise ModelDataError("model metadata roles are invalid")
    for key in ("stream_sha256", "config_sha256"):
        if (
            not isinstance(metadata[key], str)
            or _SHA256.fullmatch(metadata[key]) is None
        ):
            raise ModelDataError("model metadata identity is invalid")
    if (
        not isinstance(metadata["code_revision"], str)
        or _GIT_REVISION.fullmatch(metadata["code_revision"]) is None
    ):
        raise ModelDataError("model metadata code revision is invalid")
    if not _source_identity(metadata["source_identity"]):
        raise ModelDataError("model metadata source identity is invalid")
    for key in ("seed", "optimizer_steps"):
        if type(metadata[key]) is not int or metadata[key] < 0:
            raise ModelDataError("model metadata accounting is invalid")
    for key in ("prediction_pair_budget", "batch_size", "batches_consumed"):
        if type(metadata[key]) is not int or metadata[key] <= 0:
            raise ModelDataError("model metadata accounting is invalid")
    if metadata["batches_consumed"] != math.ceil(
        metadata["prediction_pair_budget"] / metadata["batch_size"]
    ):
        raise ModelDataError("model metadata batch accounting is invalid")
    optimizer = metadata["optimizer"]
    if model_type == "neural_bigram":
        if (
            metadata["smoothing_alpha"] is not None
            or metadata["initial_weights_sha256"] != all_zero_weights_sha256()
            or metadata["optimizer_steps"] != metadata["batches_consumed"]
            or optimizer
            != {
                "name": "SGD",
                "learning_rate": 1.0,
                "momentum": 0.0,
                "weight_decay": 0.0,
            }
        ):
            raise ModelDataError("neural model optimizer metadata is invalid")
    elif (
        metadata["smoothing_alpha"] != 1
        or metadata["initial_weights_sha256"] is not None
        or metadata["optimizer_steps"] != 0
        or optimizer is not None
    ):
        raise ModelDataError("count model metadata is invalid")
    if model_type != "neural_bigram" and (
        torch.any(tensor < 0).item()
        or tensor.sum().item() != metadata["prediction_pair_budget"]
    ):
        raise ModelDataError("count model tensor accounting is invalid")
    return dict(metadata)


def _source_identity(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(key, str)
            and key
            and isinstance(item, str)
            and _SHA256.fullmatch(item) is not None
            for key, item in value.items()
        )
    )


def _dtype_name(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.int64:
        return "int64"
    if tensor.dtype == torch.float32:
        return "float32"
    raise ModelDataError("model tensor dtype is invalid")


def _torch_dtype(value: object) -> torch.dtype:
    if value == "int64":
        return torch.int64
    if value == "float32":
        return torch.float32
    raise ModelDataError("JSON model tensor dtype is invalid")


def _validate_json_values(values: object, dtype: object) -> None:
    """Reject JSON values that would be silently coerced while rebuilding tensors."""

    if isinstance(values, list):
        for value in values:
            _validate_json_values(value, dtype)
        return
    if dtype == "int64" and (type(values) is not int):
        raise ModelDataError("JSON integer model values are invalid")
    if dtype == "float32" and (
        type(values) not in (int, float) or not math.isfinite(float(values))
    ):
        raise ModelDataError("JSON float model values are invalid")
