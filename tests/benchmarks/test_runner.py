import json
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from protein_lm.benchmarks.config import BenchmarkConfig
from protein_lm.benchmarks.metrics import (
    ReadinessChecks,
    calculate_readiness_checks,
    swap_value_bytes,
)
import protein_lm.benchmarks.metrics as benchmark_metrics
import protein_lm.benchmarks.runner as benchmark_runner
from protein_lm.benchmarks.runner import (
    run_synthetic_benchmark,
    write_benchmark_result,
)
from protein_lm.benchmarks.workload import create_synthetic_token_tensors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMALL_CPU_CONFIG = BenchmarkConfig(
    identifier="cpu-test",
    batch_size=2,
    sequence_length=8,
    width=16,
    layers=1,
    heads=4,
    warmup_steps=1,
    measured_steps=2,
)


def test_synthetic_tokens_are_deterministic_canonical_residue_ids() -> None:
    first_tokens, first_targets = create_synthetic_token_tensors(
        SMALL_CPU_CONFIG,
        torch.device("cpu"),
    )
    second_tokens, second_targets = create_synthetic_token_tensors(
        SMALL_CPU_CONFIG,
        torch.device("cpu"),
    )

    assert first_tokens.shape == first_targets.shape == (2, 8)
    assert int(first_tokens.min()) >= 4
    assert int(first_tokens.max()) <= 23
    assert int(first_targets.min()) >= 4
    assert int(first_targets.max()) <= 23
    assert torch.equal(first_tokens, second_tokens)
    assert torch.equal(first_targets, second_targets)


def test_small_cpu_fixture_exercises_the_complete_training_shaped_workload() -> None:
    result = run_synthetic_benchmark(
        SMALL_CPU_CONFIG,
        device="cpu",
        project_root=PROJECT_ROOT,
    )

    assert result.status == "completed"
    assert result.parameter_count is not None and result.parameter_count > 0
    assert result.batch_shape == [2, 8]
    assert result.output_shape == [2, 8, 24]
    assert result.target_shape == [2, 8]
    assert len(result.measured_step_seconds) == 2
    assert result.mean_step_seconds is not None and result.mean_step_seconds > 0
    assert result.median_step_seconds is not None
    assert result.tokens_per_second is not None and result.tokens_per_second > 0
    assert result.projected_seconds is not None
    assert isinstance(result.readiness_checks, ReadinessChecks)
    assert result.finite_loss is True
    assert result.finite_gradients is True
    assert result.environment["backend"] == "cpu"
    assert result.maximum_sampled_mps_allocated_memory_bytes is None
    assert result.maximum_sampled_mps_driver_memory_bytes is None


def test_mps_request_records_failure_without_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    result = run_synthetic_benchmark(
        SMALL_CPU_CONFIG,
        device="mps",
        project_root=PROJECT_ROOT,
    )

    assert result.status == "failed"
    assert result.error is not None
    assert "CPU fallback is prohibited" in result.error["message"]
    assert result.batch_shape is None
    assert result.finite_loss is None
    assert result.finite_gradients is None
    assert result.readiness_checks == ReadinessChecks(False, False, False, False)


def test_result_writer_refuses_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    result = run_synthetic_benchmark(
        SMALL_CPU_CONFIG,
        device="cpu",
        project_root=PROJECT_ROOT,
    )
    result_path = tmp_path / "candidate.json"

    write_benchmark_result(result_path, result)

    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "completed"
    with pytest.raises(FileExistsError):
        write_benchmark_result(result_path, result)


def test_swap_parser_uses_bytes_for_macos_swapusage_output() -> None:
    raw = "total = 4096.00M  used = 530.50M  free = 3565.50M  (encrypted)"

    assert swap_value_bytes(raw, "total") == 4096 * 1024**2
    assert swap_value_bytes(raw, "used") == int(530.5 * 1024**2)


def test_macos_system_facts_record_host_model_cpu_and_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "hw.model": "Mac16,10\n",
        "machdep.cpu.brand_string": "Apple M4\n",
        "hw.memsize": "32000000000\n",
    }
    calls: list[tuple[list[str], object]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs["timeout"]))
        return SimpleNamespace(stdout=values[command[-1]])

    monkeypatch.setattr(benchmark_metrics.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(benchmark_metrics.subprocess, "run", fake_run)

    assert benchmark_metrics.macos_system_facts() == {
        "mac_model": "Mac16,10",
        "cpu_brand": "Apple M4",
        "physical_memory_bytes": 32_000_000_000,
    }
    assert calls == [
        (["sysctl", "-n", "hw.memsize"], 1),
        (["sysctl", "-n", "hw.model"], 1),
        (["sysctl", "-n", "machdep.cpu.brand_string"], 1),
    ]


def test_macos_system_facts_retain_null_when_sysctl_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*args: object, **kwargs: object) -> SimpleNamespace:
        raise subprocess.CalledProcessError(returncode=1, cmd="sysctl")

    monkeypatch.setattr(benchmark_metrics.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(benchmark_metrics.subprocess, "run", unavailable)

    assert benchmark_metrics.macos_system_facts() == {
        "mac_model": None,
        "cpu_brand": None,
        "physical_memory_bytes": None,
    }


def test_readiness_checks_apply_each_fixed_threshold_independently() -> None:
    assert calculate_readiness_checks(
        status="completed",
        tokens_per_second=10_000,
    ) == ReadinessChecks(True, True, True, True)
    assert calculate_readiness_checks(
        status="completed",
        tokens_per_second=5_000,
    ) == ReadinessChecks(False, True, False, False)
    assert calculate_readiness_checks(
        status="failed",
        tokens_per_second=10_000,
    ) == ReadinessChecks(False, False, False, False)
    assert calculate_readiness_checks(
        status="completed",
        tokens_per_second=None,
    ) == ReadinessChecks(False, False, False, False)


def test_warmup_that_exceeds_the_development_limit_stops_before_timing_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter((0.0, 1200.001))
    monkeypatch.setattr(
        benchmark_runner.time, "perf_counter", lambda: next(clock_values)
    )

    result = run_synthetic_benchmark(
        SMALL_CPU_CONFIG,
        device="cpu",
        project_root=PROJECT_ROOT,
    )

    assert result.status == "outside_normal_development_envelope"
    assert result.stop_reason is not None
    assert result.stop_reason["type"] == "DevelopmentTimeLimitExceeded"
    assert result.measured_step_seconds == []
    assert result.finite_loss is True
    assert result.finite_gradients is True
    assert result.readiness_checks == ReadinessChecks(False, False, False, False)
