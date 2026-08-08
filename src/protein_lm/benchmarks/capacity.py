"""Mac mini capacity-staircase orchestration for Task 11A-2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from protein_lm.benchmarks.config import BenchmarkConfig
from protein_lm.benchmarks.metrics import (
    NORMAL_MEMORY_LIMIT_BYTES,
    memory_limit_exceeded,
    swap_grew,
)
from protein_lm.benchmarks.runner import BenchmarkResult, run_synthetic_benchmark


CAPACITY_CONTRACT_IDENTIFIER = "2026-08-07-task-11a-2-mac-mini-capacity-v1"
SOFT_ESCALATION_DRIVER_MEMORY_BYTES = 20_000_000_000


@dataclass(frozen=True)
class MPSAllocatorConfiguration:
    """Recorded result of the capacity runner's per-process MPS safety cap."""

    recommended_max_memory_bytes: int | None
    configured_ceiling_bytes: int | None
    applied_fraction: float | None


class MPSAllocatorConfigurationError(RuntimeError):
    """Retain allocator facts when MPS cannot apply the requested safety cap."""

    def __init__(self, message: str, configuration: MPSAllocatorConfiguration) -> None:
        super().__init__(message)
        self.configuration = configuration


@dataclass(frozen=True)
class CapacityBenchmarkResult:
    """One capacity candidate plus its unchanged base-run measurement record."""

    contract_identifier: str
    candidate: dict[str, int | str | float]
    status: str
    recommended_max_memory_bytes: int | None
    configured_allocator_ceiling_bytes: int | None
    applied_allocator_fraction: float | None
    base_benchmark_result: BenchmarkResult
    next_candidate_allowed: bool
    continuation_reason: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready record without omitting failed base results."""
        return asdict(self)


CAPACITY_CANDIDATES: dict[str, BenchmarkConfig] = {
    "E": BenchmarkConfig(
        identifier="E",
        batch_size=2,
        sequence_length=1024,
        width=512,
        layers=8,
        heads=8,
        warmup_steps=1,
        measured_steps=3,
    ),
    "F": BenchmarkConfig(
        identifier="F",
        batch_size=2,
        sequence_length=1024,
        width=768,
        layers=12,
        heads=12,
        warmup_steps=1,
        measured_steps=3,
    ),
    "G": BenchmarkConfig(
        identifier="G",
        batch_size=2,
        sequence_length=1024,
        width=1024,
        layers=16,
        heads=16,
        warmup_steps=1,
        measured_steps=3,
    ),
    "H": BenchmarkConfig(
        identifier="H",
        batch_size=2,
        sequence_length=1024,
        width=1280,
        layers=20,
        heads=20,
        warmup_steps=1,
        measured_steps=3,
    ),
    "I": BenchmarkConfig(
        identifier="I",
        batch_size=2,
        sequence_length=1024,
        width=1408,
        layers=22,
        heads=22,
        warmup_steps=1,
        measured_steps=3,
    ),
    "J": BenchmarkConfig(
        identifier="J",
        batch_size=2,
        sequence_length=1024,
        width=1536,
        layers=24,
        heads=24,
        warmup_steps=1,
        measured_steps=3,
    ),
}


def run_synthetic_capacity_benchmark(
    config: BenchmarkConfig,
    *,
    project_root: Path,
) -> CapacityBenchmarkResult:
    """Run one MPS capacity candidate with its allocator cap set first."""
    allocation = MPSAllocatorConfiguration(None, None, None)

    def configure_allocator() -> None:
        nonlocal allocation
        try:
            allocation = configure_mps_allocator()
        except MPSAllocatorConfigurationError as exception:
            allocation = exception.configuration
            raise

    base_result = run_synthetic_benchmark(
        config,
        device="mps",
        project_root=project_root,
        before_allocation=configure_allocator,
    )
    next_candidate_allowed, continuation_reason = _continuation_decision(base_result)
    return CapacityBenchmarkResult(
        contract_identifier=CAPACITY_CONTRACT_IDENTIFIER,
        candidate=config.as_dict(),
        status=base_result.status,
        recommended_max_memory_bytes=allocation.recommended_max_memory_bytes,
        configured_allocator_ceiling_bytes=allocation.configured_ceiling_bytes,
        applied_allocator_fraction=allocation.applied_fraction,
        base_benchmark_result=base_result,
        next_candidate_allowed=next_candidate_allowed,
        continuation_reason=continuation_reason,
    )


def write_capacity_benchmark_result(
    path: Path, result: CapacityBenchmarkResult
) -> None:
    """Write a capacity result once, preserving prior evidence paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(result.as_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def configure_mps_allocator() -> MPSAllocatorConfiguration:
    """Apply the Task 11A-2 allocator ceiling before any MPS allocation."""
    unavailable = MPSAllocatorConfiguration(None, None, None)
    try:
        recommended_max_memory = torch.mps.recommended_max_memory()
    except Exception as exception:
        raise MPSAllocatorConfigurationError(
            "MPS recommended maximum memory is unavailable", unavailable
        ) from exception
    if (
        isinstance(recommended_max_memory, bool)
        or not isinstance(recommended_max_memory, int)
        or recommended_max_memory <= 0
    ):
        raise MPSAllocatorConfigurationError(
            "MPS recommended maximum memory is unavailable or nonpositive",
            unavailable,
        )

    configured_ceiling = min(NORMAL_MEMORY_LIMIT_BYTES, recommended_max_memory)
    fraction = configured_ceiling / recommended_max_memory
    pending = MPSAllocatorConfiguration(
        recommended_max_memory,
        configured_ceiling,
        None,
    )
    try:
        torch.mps.set_per_process_memory_fraction(fraction)
    except Exception as exception:
        raise MPSAllocatorConfigurationError(
            "failed to configure the MPS per-process allocator ceiling", pending
        ) from exception
    return MPSAllocatorConfiguration(
        recommended_max_memory,
        configured_ceiling,
        fraction,
    )


def _continuation_decision(base_result: BenchmarkResult) -> tuple[bool, str]:
    """Apply the frozen staircase boundary without scheduling another run."""
    if base_result.status != "completed":
        return False, f"base benchmark did not complete (status={base_result.status})"
    if base_result.error is not None:
        return False, "base benchmark recorded an error"
    if base_result.stop_reason is not None:
        return False, "base benchmark recorded a stop reason"
    if base_result.finite_loss is not True or base_result.finite_gradients is not True:
        return False, "base benchmark did not preserve finite loss and gradients"
    if swap_grew(base_result.swap_before, base_result.swap_after):
        return False, "system swap usage increased during the base benchmark"
    if memory_limit_exceeded(
        base_result.maximum_sampled_mps_driver_memory_bytes
    ) or memory_limit_exceeded(base_result.process_peak_resident_memory_bytes):
        return False, "base benchmark exceeded the 24 GB normal memory limit"
    driver_memory = base_result.maximum_sampled_mps_driver_memory_bytes
    if driver_memory is None:
        return False, "sampled MPS driver memory is unavailable"
    if driver_memory >= SOFT_ESCALATION_DRIVER_MEMORY_BYTES:
        return (
            False,
            "sampled MPS driver memory reached the 20 GB soft escalation boundary",
        )
    return (
        True,
        "completed below the 20 GB soft escalation boundary without a base-run failure",
    )
