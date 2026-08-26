"""Pinned local evidence readers for Week 3 publication."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.publication_config import FinalCheckpoint, PublicationConfig


def validate_sources(root: Path, config: PublicationConfig) -> None:
    """Fail closed before parsing statuses, tensors, or any collection."""

    for pin in (*config.sources, *config.diagnostic_statuses):
        verify_bytes(root / pin.relative_path, pin.sha256, "publication source")
    for pin in config.final_checkpoints:
        verify_bytes(root / pin.status_relative_path, pin.status_sha256, "run status")
        checkpoint = root / pin.checkpoint_relative_path
        verify_bytes(checkpoint / "checkpoint.json", pin.metadata_sha256, "checkpoint metadata")
        verify_bytes(checkpoint / "model.safetensors", pin.tensor_sha256, "checkpoint tensor")
        _validate_checkpoint_metadata(load_json(checkpoint / "checkpoint.json", "checkpoint metadata"), pin)


def verify_bytes(path: Path, expected: str, label: str) -> None:
    try:
        found = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ModelDataError(f"{label} is unavailable") from error
    if found != expected:
        raise ModelDataError(f"{label} hash drifted")


def load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite")),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ModelDataError(f"{label} is malformed") from error
    if not isinstance(value, dict) or not _finite(value):
        raise ModelDataError(f"{label} contains invalid values")
    return value


def final_statuses(root: Path, config: PublicationConfig, arm: str) -> list[tuple[FinalCheckpoint, dict[str, object]]]:
    pins = sorted((pin for pin in config.final_checkpoints if pin.arm == arm), key=lambda pin: pin.seed)
    return [(pin, load_json(root / pin.status_relative_path, f"{arm} run status")) for pin in pins]


def diagnostic_statuses(root: Path, config: PublicationConfig) -> list[dict[str, object]]:
    return [
        load_json(root / pin.relative_path, "position diagnostic status")
        for pin in sorted(config.diagnostic_statuses, key=lambda pin: pin.seed)
    ]


def _validate_checkpoint_metadata(value: dict[str, object], pin: FinalCheckpoint) -> None:
    if (
        value.get("seed") != pin.seed
        or value.get("predictions_seen") != 100000000
        or value.get("tensor_file") != "model.safetensors"
        or value.get("tensor_sha256") != pin.tensor_sha256
        or value.get("parameter_count")
        != (530293 if pin.arm == "context20" else 530965)
    ):
        raise ModelDataError("final checkpoint metadata drifted")


def _finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return value is None or isinstance(value, (str, int, bool))
