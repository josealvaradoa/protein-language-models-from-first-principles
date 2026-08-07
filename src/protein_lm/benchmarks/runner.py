"""Orchestrate one synthetic device-envelope candidate and preserve its record."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from protein_lm.benchmarks.config import LEARNING_RATE, WEIGHT_DECAY, BenchmarkConfig
from protein_lm.benchmarks.metrics import (
    DEVELOPMENT_TIME_LIMIT_SECONDS,
    ReadinessChecks,
    SwapState,
    calculate_readiness_checks,
    collect_environment,
    error_details,
    mean_or_none,
    memory_limit_exceeded,
    process_peak_resident_memory_bytes,
    projected_seconds,
    read_swap_state,
    sample_mps_memory,
    swap_grew,
    synchronize,
)
from protein_lm.benchmarks.workload import (
    NonFiniteGradientError,
    NonFiniteLossError,
    SyntheticDeviceWorkload,
    create_synthetic_token_tensors,
    run_one_training_step,
)


@dataclass(frozen=True)
class BenchmarkResult:
    """The complete, serializable outcome of one candidate invocation."""

    candidate: dict[str, int | str | float]
    status: str
    parameter_count: int | None
    batch_shape: list[int] | None
    output_shape: list[int] | None
    target_shape: list[int] | None
    measured_step_seconds: list[float]
    mean_step_seconds: float | None
    median_step_seconds: float | None
    tokens_per_second: float | None
    projected_seconds: dict[str, float] | None
    readiness_checks: ReadinessChecks
    maximum_sampled_mps_allocated_memory_bytes: int | None
    maximum_sampled_mps_driver_memory_bytes: int | None
    process_peak_resident_memory_bytes: int | None
    swap_before: SwapState
    swap_after: SwapState
    finite_loss: bool | None
    finite_gradients: bool | None
    error: dict[str, str] | None
    stop_reason: dict[str, str] | None
    environment: dict[str, str | int | None]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready record without hidden runtime state."""
        return asdict(self)


def run_synthetic_benchmark(
    config: BenchmarkConfig,
    *,
    device: str | torch.device,
    project_root: Path,
) -> BenchmarkResult:
    """Run one candidate and return a completed, stopped, or failed record.

    The caller chooses the device explicitly. Requesting MPS when it is absent
    returns a failed record and never changes the request to CPU.
    """
    requested_device = torch.device(device)
    swap_before = read_swap_state()
    environment = collect_environment(requested_device, project_root)
    maximum_allocated: int | None = None
    maximum_driver: int | None = None
    parameter_count: int | None = None
    batch_shape: list[int] | None = None
    output_shape: list[int] | None = None
    target_shape: list[int] | None = None
    measured_seconds: list[float] = []
    finite_loss: bool | None = None
    finite_gradients: bool | None = None
    error: dict[str, str] | None = None
    stop_reason: dict[str, str] | None = None
    status = "completed"

    try:
        _require_requested_device(requested_device)
        torch.manual_seed(config.seed)
        token_ids, target_ids = create_synthetic_token_tensors(config, requested_device)
        batch_shape = list(token_ids.shape)
        target_shape = list(target_ids.shape)
        model = SyntheticDeviceWorkload(config).to(
            requested_device,
            dtype=torch.float32,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        benchmark_started_at = time.perf_counter()
        for _ in range(config.warmup_steps):
            output_shape = list(
                run_one_training_step(model, optimizer, token_ids, target_ids)
            )
            finite_loss = True
            finite_gradients = True
            synchronize(requested_device)
            maximum_allocated, maximum_driver = sample_mps_memory(
                requested_device,
                maximum_allocated,
                maximum_driver,
            )
            _raise_for_runtime_limits(swap_before, read_swap_state(), maximum_driver)
            if _development_time_limit_exceeded(benchmark_started_at):
                status = "outside_normal_development_envelope"
                stop_reason = _development_stop_reason()
                break

        if status == "completed":
            for _ in range(config.measured_steps):
                synchronize(requested_device)
                step_started_at = time.perf_counter()
                output_shape = list(
                    run_one_training_step(model, optimizer, token_ids, target_ids)
                )
                finite_loss = True
                finite_gradients = True
                synchronize(requested_device)
                measured_seconds.append(time.perf_counter() - step_started_at)
                maximum_allocated, maximum_driver = sample_mps_memory(
                    requested_device,
                    maximum_allocated,
                    maximum_driver,
                )
                _raise_for_runtime_limits(
                    swap_before,
                    read_swap_state(),
                    maximum_driver,
                )
                if _development_time_limit_exceeded(benchmark_started_at):
                    status = "outside_normal_development_envelope"
                    stop_reason = _development_stop_reason()
                    break
    except NonFiniteLossError as exception:
        finite_loss = False
        status = "failed"
        error = error_details(exception)
    except NonFiniteGradientError as exception:
        finite_loss = True
        finite_gradients = False
        status = "failed"
        error = error_details(exception)
    except Exception as exception:  # Preserve device and allocation failures.
        status = "failed"
        error = error_details(exception)

    swap_after = read_swap_state()
    peak_resident = process_peak_resident_memory_bytes()
    if memory_limit_exceeded(peak_resident):
        status = "failed"
        error = error or {
            "type": "MemoryLimitExceeded",
            "message": "process peak resident memory exceeded the 24 GB limit",
        }
    if swap_grew(swap_before, swap_after):
        status = "failed"
        error = error or {
            "type": "SwapGrowthDetected",
            "message": "system swap usage increased during the benchmark",
        }

    mean_seconds = mean_or_none(measured_seconds)
    tokens_per_second = (
        config.tokens_per_step / mean_seconds if mean_seconds is not None else None
    )
    return BenchmarkResult(
        candidate=config.as_dict(),
        status=status,
        parameter_count=parameter_count,
        batch_shape=batch_shape,
        output_shape=output_shape,
        target_shape=target_shape,
        measured_step_seconds=measured_seconds,
        mean_step_seconds=mean_seconds,
        median_step_seconds=(
            statistics.median(measured_seconds) if measured_seconds else None
        ),
        tokens_per_second=tokens_per_second,
        projected_seconds=projected_seconds(tokens_per_second),
        readiness_checks=calculate_readiness_checks(
            status=status,
            tokens_per_second=tokens_per_second,
        ),
        maximum_sampled_mps_allocated_memory_bytes=maximum_allocated,
        maximum_sampled_mps_driver_memory_bytes=maximum_driver,
        process_peak_resident_memory_bytes=peak_resident,
        swap_before=swap_before,
        swap_after=swap_after,
        finite_loss=finite_loss,
        finite_gradients=finite_gradients,
        error=error,
        stop_reason=stop_reason,
        environment=environment,
    )


def write_benchmark_result(path: Path, result: BenchmarkResult) -> None:
    """Write one result to a new path so prior evidence cannot be overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(result.as_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _require_requested_device(device: torch.device) -> None:
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is not available; CPU fallback is prohibited"
        )


def _raise_for_runtime_limits(
    swap_before: SwapState,
    current_swap: SwapState,
    maximum_driver_memory: int | None,
) -> None:
    if memory_limit_exceeded(maximum_driver_memory):
        raise RuntimeError("sampled MPS driver memory exceeded the 24 GB limit")
    if swap_grew(swap_before, current_swap):
        raise RuntimeError("system swap usage increased during the benchmark")


def _development_time_limit_exceeded(benchmark_started_at: float) -> bool:
    return time.perf_counter() - benchmark_started_at > DEVELOPMENT_TIME_LIMIT_SECONDS


def _development_stop_reason() -> dict[str, str]:
    return {
        "type": "DevelopmentTimeLimitExceeded",
        "message": "candidate exceeded the 20-minute development limit",
    }
