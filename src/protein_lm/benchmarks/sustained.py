"""Sustained confirmation of the completed Task 11A-2 H capacity result."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from protein_lm.benchmarks.capacity import (
    CAPACITY_CANDIDATES,
    MPSAllocatorConfiguration,
    MPSAllocatorConfigurationError,
    configure_mps_allocator,
)
from protein_lm.benchmarks.metrics import SwapState, read_swap_state, swap_grew
from protein_lm.benchmarks.runner import BenchmarkResult, run_synthetic_benchmark
from protein_lm.benchmarks.sustained_source import (
    CAPACITY_H_DRIVER_MEMORY_BOUNDARY_BYTES,
    CAPACITY_H_PARAMETER_COUNT,
    SourceCapacityResultError,
    validate_capacity_h_source,
)


SUSTAINED_CONFIRMATION_CONTRACT_IDENTIFIER = "2026-08-07-task-11a-2-sustained-h-v1"
SUSTAINED_MEASURED_STEPS = 60
SUSTAINED_DRIVER_MEMORY_BOUNDARY_BYTES = CAPACITY_H_DRIVER_MEMORY_BOUNDARY_BYTES
SUSTAINED_SLOWDOWN_LIMIT = 0.20
SUSTAINED_H_CONFIG = replace(
    CAPACITY_CANDIDATES["H"], measured_steps=SUSTAINED_MEASURED_STEPS
)


@dataclass(frozen=True)
class SustainedConfirmationResult:
    """Preserved sustained-H outcome, including a truthful preflight failure."""

    contract_identifier: str
    source_capacity_result_path: str
    source_capacity_result_sha256: str | None
    candidate: dict[str, int | str | float]
    status: str
    recommended_max_memory_bytes: int | None
    configured_allocator_ceiling_bytes: int | None
    applied_allocator_fraction: float | None
    preflight_swap: SwapState
    base_benchmark_result: BenchmarkResult | None
    measured_mps_allocated_memory_bytes: list[int | None]
    measured_mps_driver_memory_bytes: list[int | None]
    first_30_median_seconds: float | None
    last_30_median_seconds: float | None
    slowdown_fraction: float | None
    first_sampled_mps_driver_memory_bytes: int | None
    last_sampled_mps_driver_memory_bytes: int | None
    sampled_mps_driver_memory_growth_bytes: int | None
    sustained_confirmation_passed: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        """Return the complete JSON-ready evidence record."""
        return asdict(self)


def run_sustained_h_confirmation(
    source_capacity_result_path: Path,
    *,
    project_root: Path,
) -> SustainedConfirmationResult:
    """Run the approved 60-step confirmation after source and swap preflight."""
    source_path = str(source_capacity_result_path.resolve())
    source_hash: str | None = None
    preflight_swap = read_swap_state()
    allocation = MPSAllocatorConfiguration(None, None, None)

    try:
        source_hash = validate_capacity_h_source(source_capacity_result_path)
    except SourceCapacityResultError as exception:
        source_hash = exception.source_hash or source_hash
        return _preflight_failure(
            source_path,
            source_hash,
            preflight_swap,
            allocation,
            f"source capacity result rejected: {exception}",
        )

    if not _is_exact_zero_swap(preflight_swap.used_bytes):
        return _preflight_failure(
            source_path,
            source_hash,
            preflight_swap,
            allocation,
            "preflight requires measurable system swap usage of exactly zero bytes",
        )

    allocated_samples: list[int | None] = []
    driver_samples: list[int | None] = []

    def configure_allocator() -> None:
        nonlocal allocation
        try:
            allocation = configure_mps_allocator()
        except MPSAllocatorConfigurationError as exception:
            allocation = exception.configuration
            raise

    def record_measured_step(
        _: float, allocated_bytes: int | None, driver_bytes: int | None
    ) -> None:
        allocated_samples.append(allocated_bytes)
        driver_samples.append(driver_bytes)

    base_result = run_synthetic_benchmark(
        SUSTAINED_H_CONFIG,
        device="mps",
        project_root=project_root,
        before_allocation=configure_allocator,
        measured_step_observer=record_measured_step,
    )
    passed, reason, first_median, last_median, slowdown = _confirmation_decision(
        base_result,
        preflight_swap,
        allocated_samples,
        driver_samples,
    )
    first_driver = driver_samples[0] if driver_samples else None
    last_driver = driver_samples[-1] if driver_samples else None
    driver_growth = (
        last_driver - first_driver
        if isinstance(first_driver, int) and isinstance(last_driver, int)
        else None
    )
    return SustainedConfirmationResult(
        contract_identifier=SUSTAINED_CONFIRMATION_CONTRACT_IDENTIFIER,
        source_capacity_result_path=source_path,
        source_capacity_result_sha256=source_hash,
        candidate=SUSTAINED_H_CONFIG.as_dict(),
        status=base_result.status,
        recommended_max_memory_bytes=allocation.recommended_max_memory_bytes,
        configured_allocator_ceiling_bytes=allocation.configured_ceiling_bytes,
        applied_allocator_fraction=allocation.applied_fraction,
        preflight_swap=preflight_swap,
        base_benchmark_result=base_result,
        measured_mps_allocated_memory_bytes=allocated_samples,
        measured_mps_driver_memory_bytes=driver_samples,
        first_30_median_seconds=first_median,
        last_30_median_seconds=last_median,
        slowdown_fraction=slowdown,
        first_sampled_mps_driver_memory_bytes=first_driver,
        last_sampled_mps_driver_memory_bytes=last_driver,
        sampled_mps_driver_memory_growth_bytes=driver_growth,
        sustained_confirmation_passed=passed,
        reason=reason,
    )


def write_sustained_confirmation_result(
    path: Path, result: SustainedConfirmationResult
) -> None:
    """Write once so a sustained result cannot replace prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(result.as_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _preflight_failure(
    source_path: str,
    source_hash: str | None,
    preflight_swap: SwapState,
    allocation: MPSAllocatorConfiguration,
    reason: str,
) -> SustainedConfirmationResult:
    return SustainedConfirmationResult(
        contract_identifier=SUSTAINED_CONFIRMATION_CONTRACT_IDENTIFIER,
        source_capacity_result_path=source_path,
        source_capacity_result_sha256=source_hash,
        candidate=SUSTAINED_H_CONFIG.as_dict(),
        status="failed",
        recommended_max_memory_bytes=allocation.recommended_max_memory_bytes,
        configured_allocator_ceiling_bytes=allocation.configured_ceiling_bytes,
        applied_allocator_fraction=allocation.applied_fraction,
        preflight_swap=preflight_swap,
        base_benchmark_result=None,
        measured_mps_allocated_memory_bytes=[],
        measured_mps_driver_memory_bytes=[],
        first_30_median_seconds=None,
        last_30_median_seconds=None,
        slowdown_fraction=None,
        first_sampled_mps_driver_memory_bytes=None,
        last_sampled_mps_driver_memory_bytes=None,
        sampled_mps_driver_memory_growth_bytes=None,
        sustained_confirmation_passed=False,
        reason=reason,
    )


def _confirmation_decision(
    base_result: BenchmarkResult,
    preflight_swap: SwapState,
    allocated_samples: list[int | None],
    driver_samples: list[int | None],
) -> tuple[bool, str, float | None, float | None, float | None]:
    first_median, last_median, slowdown = _sustained_timing_metrics(
        base_result.measured_step_seconds
    )
    if base_result.status != "completed":
        return (
            False,
            f"base benchmark did not complete (status={base_result.status})",
            first_median,
            last_median,
            slowdown,
        )
    if len(base_result.measured_step_seconds) != SUSTAINED_MEASURED_STEPS:
        return (
            False,
            "base benchmark did not record exactly 60 measured steps",
            first_median,
            last_median,
            slowdown,
        )
    if base_result.parameter_count != CAPACITY_H_PARAMETER_COUNT:
        return (
            False,
            "base benchmark parameter count is not exact Capacity H",
            first_median,
            last_median,
            slowdown,
        )
    if (
        len(allocated_samples) != SUSTAINED_MEASURED_STEPS
        or len(driver_samples) != SUSTAINED_MEASURED_STEPS
    ):
        return (
            False,
            "per-step MPS memory samples are missing or short",
            first_median,
            last_median,
            slowdown,
        )
    if any(value is None for value in allocated_samples + driver_samples):
        return (
            False,
            "per-step MPS memory sampling is unavailable",
            first_median,
            last_median,
            slowdown,
        )
    if base_result.finite_loss is not True or base_result.finite_gradients is not True:
        return (
            False,
            "base benchmark did not preserve finite loss and gradients",
            first_median,
            last_median,
            slowdown,
        )
    if base_result.error is not None or base_result.stop_reason is not None:
        return (
            False,
            "base benchmark recorded an error or stop reason",
            first_median,
            last_median,
            slowdown,
        )
    if not _is_exact_zero_swap(preflight_swap.used_bytes) or not _is_exact_zero_swap(
        base_result.swap_before.used_bytes
    ):
        return (
            False,
            "starting system swap usage was not exactly zero bytes",
            first_median,
            last_median,
            slowdown,
        )
    if not _is_exact_zero_swap(base_result.swap_after.used_bytes):
        return (
            False,
            "ending system swap usage was not exactly zero bytes",
            first_median,
            last_median,
            slowdown,
        )
    if swap_grew(base_result.swap_before, base_result.swap_after):
        return (
            False,
            "system swap usage increased during the confirmation",
            first_median,
            last_median,
            slowdown,
        )
    maximum_driver = base_result.maximum_sampled_mps_driver_memory_bytes
    if maximum_driver is None:
        return (
            False,
            "maximum sampled MPS driver memory is unavailable",
            first_median,
            last_median,
            slowdown,
        )
    if maximum_driver >= SUSTAINED_DRIVER_MEMORY_BOUNDARY_BYTES:
        return (
            False,
            "maximum sampled MPS driver memory reached the 20 GB boundary",
            first_median,
            last_median,
            slowdown,
        )
    if slowdown is None:
        return (
            False,
            "sustained timing measurements are incomplete",
            first_median,
            last_median,
            slowdown,
        )
    if slowdown > SUSTAINED_SLOWDOWN_LIMIT:
        return (
            False,
            "last-30 median slowdown exceeded the 20 percent limit",
            first_median,
            last_median,
            slowdown,
        )
    return (
        True,
        "all sustained H confirmation checks passed",
        first_median,
        last_median,
        slowdown,
    )


def _is_exact_zero_swap(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value == 0


def _sustained_timing_metrics(
    measured_seconds: list[float],
) -> tuple[float | None, float | None, float | None]:
    if len(measured_seconds) != SUSTAINED_MEASURED_STEPS:
        return None, None, None
    first_median = statistics.median(measured_seconds[:30])
    last_median = statistics.median(measured_seconds[30:])
    if first_median <= 0:
        return first_median, last_median, None
    return first_median, last_median, last_median / first_median - 1
