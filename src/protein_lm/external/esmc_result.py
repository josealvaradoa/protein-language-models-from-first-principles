"""Result construction, acceptance checks, and serialization for ESMC smoke runs."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import torch

from protein_lm.benchmarks.metrics import collect_environment, process_peak_resident_memory_bytes
from protein_lm.external.esmc_contract import ESMCContract
from protein_lm.external.esmc_provenance import lockfile_sha256


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def create_result(
    contract: ESMCContract,
    device: torch.device,
    project_root: Path,
    swap_before: Any,
) -> dict[str, object]:
    """Create the complete record before any model-dependent action occurs."""
    return {
        "status": "completed",
        "decision": "pending",
        "pins": contract.pins(),
        "device": device.type,
        "precision": "float32",
        "fallback": {
            "requested_device": device.type,
            "automatic_cpu_fallback": (
                "prohibited" if device.type == "mps" else "not_applicable"
            ),
        },
        "local_weight_sha256": None,
        "validated_local_config": None,
        "validated_runtime_config": None,
        "installed_packages": None,
        "lockfile_sha256": lockfile_sha256(project_root),
        "parameter_count": None,
        "unmasked_hidden_state_shape": None,
        "pooled_embedding_shape": None,
        "masked_mlm_logit_shape": None,
        "residue_counts": None,
        "masked_residue_counts": None,
        "finite_outputs": None,
        "padding_poison_invariant": None,
        "runtime_seconds": None,
        "maximum_sampled_mps_allocated_memory_bytes": None,
        "maximum_sampled_mps_driver_memory_bytes": None,
        "process_peak_resident_memory_bytes": None,
        "swap_before": _swap_dict(swap_before),
        "swap_after": None,
        "swap_grew": None,
        "error": None,
        "stop_reason": None,
        "environment": collect_environment(device, project_root),
        "acceptance": {},
    }


def finish_result(
    result: dict[str, object],
    contract: ESMCContract,
    device: torch.device,
    *,
    started_at: float,
    maximum_allocated: int | None,
    maximum_driver: int | None,
    swap_before: Any,
    swap_after: Any,
) -> None:
    """Add final measurements and transform all frozen checks into a decision."""
    result["runtime_seconds"] = time.perf_counter() - started_at
    result["maximum_sampled_mps_allocated_memory_bytes"] = maximum_allocated
    result["maximum_sampled_mps_driver_memory_bytes"] = maximum_driver
    result["process_peak_resident_memory_bytes"] = process_peak_resident_memory_bytes()
    result["swap_after"] = _swap_dict(swap_after)
    result["swap_grew"] = _swap_grew(swap_before, swap_after)
    _apply_acceptance(result, contract, device, swap_before, swap_after)


def write_esmc_result(path: Path, result: dict[str, object]) -> None:
    """Create one evidence record without replacing a prior result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def result_reason(result: dict[str, object]) -> str:
    """Return the short operator-facing reason for a completed or failed record."""
    if result["decision"] == "pass":
        return "all frozen acceptance checks passed"
    error = result.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    checks = result.get("acceptance")
    if isinstance(checks, dict):
        failed = [name for name, passed in checks.items() if passed is not True]
        if failed:
            return f"failed acceptance checks: {', '.join(failed)}"
    return "smoke did not complete"


def _apply_acceptance(
    result: dict[str, object],
    contract: ESMCContract,
    device: torch.device,
    swap_before: Any,
    swap_after: Any,
) -> None:
    runtime_seconds = result["runtime_seconds"]
    assert isinstance(runtime_seconds, float)
    maximum_driver = result["maximum_sampled_mps_driver_memory_bytes"]
    checks = {
        "completed": result["status"] == "completed",
        "local_weight_hash_matches": result["local_weight_sha256"]
        == contract.weight_sha256,
        "lockfile_sha256_present": _is_sha256(result["lockfile_sha256"]),
        "expected_config": result["validated_local_config"] is not None
        and result["validated_runtime_config"] is not None,
        "installed_package_provenance": _installed_package_pins_match(
            result["installed_packages"], contract
        ),
        "expected_shapes": result["unmasked_hidden_state_shape"]
        == contract.expected_shapes["hidden_states"]
        and result["pooled_embedding_shape"]
        == contract.expected_shapes["pooled_embeddings"]
        and result["masked_mlm_logit_shape"] == contract.expected_shapes["mlm_logits"],
        "residue_counts_match": result["residue_counts"] == [32, 64]
        and result["masked_residue_counts"] == [1, 1],
        "finite_outputs": result["finite_outputs"] is True,
        "padding_poison_invariant": result["padding_poison_invariant"] is True,
        "runtime_within_120_seconds": runtime_seconds <= contract.runtime_limit_seconds,
        "mps_driver_memory_within_limit": device.type != "mps"
        or (
            isinstance(maximum_driver, int)
            and not isinstance(maximum_driver, bool)
            and maximum_driver <= contract.mps_driver_memory_limit_bytes
        ),
        "swap_usage_measured_and_not_greater": _swap_is_measured_and_not_greater(
            swap_before, swap_after
        ),
    }
    result["acceptance"] = checks
    if all(checks.values()):
        result["decision"] = "pass"
        return
    result["decision"] = "fail"
    if result["status"] == "completed":
        result["status"] = "failed"
        result["stop_reason"] = {
            "type": "AcceptanceCheckFailed",
            "message": "one or more frozen smoke acceptance checks failed",
        }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _swap_is_measured_and_not_greater(before: Any, after: Any) -> bool:
    return (
        _is_int_not_bool(before.used_bytes)
        and _is_int_not_bool(after.used_bytes)
        and after.used_bytes <= before.used_bytes
    )


def _swap_grew(before: Any, after: Any) -> bool | None:
    if not _is_int_not_bool(before.used_bytes) or not _is_int_not_bool(after.used_bytes):
        return None
    return after.used_bytes > before.used_bytes


def _is_int_not_bool(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _installed_package_pins_match(
    installed: object,
    contract: ESMCContract,
) -> bool:
    if not isinstance(installed, dict):
        return False
    esm = installed.get("esm")
    transformers = installed.get("transformers")
    return (
        isinstance(esm, dict)
        and esm.get("version") == contract.code_version
        and esm.get("commit_id") == contract.code_revision
        and isinstance(transformers, dict)
        and isinstance(transformers.get("version"), str)
        and transformers.get("commit_id") == contract.transformers_revision
    )


def _swap_dict(swap: Any) -> dict[str, object]:
    return {
        "raw": swap.raw,
        "used_bytes": swap.used_bytes,
        "total_bytes": swap.total_bytes,
    }
