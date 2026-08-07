"""Read and validate Capacity H evidence that authorizes sustained testing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from protein_lm.benchmarks.capacity import (
    CAPACITY_CANDIDATES,
    CAPACITY_CONTRACT_IDENTIFIER,
    SOFT_ESCALATION_DRIVER_MEMORY_BYTES,
)


CAPACITY_H_PARAMETER_COUNT = 394_923_544
CAPACITY_H_DRIVER_MEMORY_BOUNDARY_BYTES = SOFT_ESCALATION_DRIVER_MEMORY_BYTES


class SourceCapacityResultError(ValueError):
    """Reject capacity evidence that cannot authorize the sustained H run."""

    def __init__(self, message: str, source_hash: str | None = None) -> None:
        super().__init__(message)
        self.source_hash = source_hash


def validate_capacity_h_source(path: Path) -> str:
    """Validate immutable Capacity H evidence and return its SHA-256."""
    source_hash, payload = _read_capacity_source(path)
    try:
        _validate_capacity_h_payload(payload)
    except SourceCapacityResultError as exception:
        if exception.source_hash is None:
            raise SourceCapacityResultError(str(exception), source_hash) from exception
        raise
    return source_hash


def _read_capacity_source(path: Path) -> tuple[str, Any]:
    """Read one source byte stream so the recorded hash matches validated JSON."""
    try:
        raw_bytes = path.read_bytes()
    except OSError as exception:
        raise SourceCapacityResultError("source result cannot be read") from exception
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    try:
        payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exception:
        raise SourceCapacityResultError(
            "source result is not valid JSON", source_hash
        ) from exception
    return source_hash, payload


def _validate_capacity_h_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise SourceCapacityResultError("source result must be a JSON object")
    expected = CAPACITY_CANDIDATES["H"].as_dict()
    if payload.get("contract_identifier") != CAPACITY_CONTRACT_IDENTIFIER:
        raise SourceCapacityResultError(
            "source contract identifier is not Task 11A-2 v1"
        )
    if payload.get("candidate") != expected:
        raise SourceCapacityResultError(
            "source candidate is not the exact capacity H config"
        )
    if payload.get("status") != "completed":
        raise SourceCapacityResultError("source status is not completed")
    if payload.get("next_candidate_allowed") is not True:
        raise SourceCapacityResultError("source does not permit the next operator step")

    base = payload.get("base_benchmark_result")
    if not isinstance(base, dict):
        raise SourceCapacityResultError("source base benchmark result is absent")
    if base.get("candidate") != expected:
        raise SourceCapacityResultError(
            "source base candidate is not the exact capacity H config"
        )
    if base.get("status") != "completed":
        raise SourceCapacityResultError("source base benchmark status is not completed")
    if not _valid_parameter_count(base.get("parameter_count")):
        raise SourceCapacityResultError(
            "source base benchmark parameter count is not exact Capacity H"
        )
    if not _valid_driver_memory(base.get("maximum_sampled_mps_driver_memory_bytes")):
        raise SourceCapacityResultError(
            "source maximum sampled MPS driver memory is not below the 20 GB boundary"
        )
    if base.get("finite_loss") is not True or base.get("finite_gradients") is not True:
        raise SourceCapacityResultError("source base benchmark is not finite")
    if base.get("error") is not None or base.get("stop_reason") is not None:
        raise SourceCapacityResultError(
            "source base benchmark recorded an error or stop"
        )
    if _serialized_swap_grew(base):
        raise SourceCapacityResultError("source base benchmark recorded swap growth")


def _valid_driver_memory(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and 0 <= value < CAPACITY_H_DRIVER_MEMORY_BOUNDARY_BYTES
    )


def _valid_parameter_count(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value == CAPACITY_H_PARAMETER_COUNT
    )


def _serialized_swap_grew(base_result: dict[str, Any]) -> bool:
    before = base_result.get("swap_before")
    after = base_result.get("swap_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return True
    before_used = before.get("used_bytes")
    after_used = after.get("used_bytes")
    if isinstance(before_used, bool) or isinstance(after_used, bool):
        return True
    if not isinstance(before_used, int) or not isinstance(after_used, int):
        return True
    return after_used > before_used
