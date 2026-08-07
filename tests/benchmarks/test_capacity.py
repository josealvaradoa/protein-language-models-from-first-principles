import json
from pathlib import Path

import pytest

from protein_lm.benchmarks.capacity import (
    CAPACITY_CANDIDATES,
    CAPACITY_CONTRACT_IDENTIFIER,
    SOFT_ESCALATION_DRIVER_MEMORY_BYTES,
    CapacityBenchmarkResult,
    MPSAllocatorConfigurationError,
    _continuation_decision,
    configure_mps_allocator,
    run_synthetic_capacity_benchmark,
    write_capacity_benchmark_result,
)
from protein_lm.benchmarks.metrics import ReadinessChecks, SwapState
from protein_lm.benchmarks.runner import BenchmarkResult
import protein_lm.benchmarks.capacity as capacity


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _base_result(
    *,
    status: str = "completed",
    driver_memory: int | None = 1,
    peak_memory: int | None = 1,
    finite_loss: bool | None = True,
    finite_gradients: bool | None = True,
    swap_after_bytes: int = 1,
    error: dict[str, str] | None = None,
    stop_reason: dict[str, str] | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        candidate=CAPACITY_CANDIDATES["E"].as_dict(),
        status=status,
        parameter_count=1,
        batch_shape=[2, 1024],
        output_shape=[2, 1024, 24],
        target_shape=[2, 1024],
        measured_step_seconds=[1.0],
        mean_step_seconds=1.0,
        median_step_seconds=1.0,
        tokens_per_second=2048.0,
        projected_seconds={"1000000": 1.0},
        readiness_checks=ReadinessChecks(False, False, False, False),
        maximum_sampled_mps_allocated_memory_bytes=1,
        maximum_sampled_mps_driver_memory_bytes=driver_memory,
        process_peak_resident_memory_bytes=peak_memory,
        swap_before=SwapState(raw=None, used_bytes=1, total_bytes=2),
        swap_after=SwapState(raw=None, used_bytes=swap_after_bytes, total_bytes=2),
        finite_loss=finite_loss,
        finite_gradients=finite_gradients,
        error=error,
        stop_reason=stop_reason,
        environment={},
    )


def test_capacity_candidates_match_the_frozen_staircase() -> None:
    expected = {
        "E": (2, 1024, 512, 8, 8),
        "F": (2, 1024, 768, 12, 12),
        "G": (2, 1024, 1024, 16, 16),
        "H": (2, 1024, 1280, 20, 20),
        "I": (2, 1024, 1408, 22, 22),
        "J": (2, 1024, 1536, 24, 24),
    }

    assert set(CAPACITY_CANDIDATES) == set(expected)
    for identifier, candidate in CAPACITY_CANDIDATES.items():
        assert (
            candidate.batch_size,
            candidate.sequence_length,
            candidate.width,
            candidate.layers,
            candidate.heads,
        ) == expected[identifier]
        assert candidate.values_per_head == 64
        assert candidate.warmup_steps == 1
        assert candidate.measured_steps == 3
        assert candidate.as_dict()["precision"] == "float32"


@pytest.mark.parametrize(
    ("recommended", "expected_ceiling", "expected_fraction"),
    [(30_000_000_000, 24_000_000_000, 0.8), (16_000_000_000, 16_000_000_000, 1.0)],
)
def test_allocator_cap_uses_the_lower_of_24gb_and_recommended_memory(
    monkeypatch: pytest.MonkeyPatch,
    recommended: int,
    expected_ceiling: int,
    expected_fraction: float,
) -> None:
    fractions: list[float] = []
    monkeypatch.setattr(
        capacity.torch.mps, "recommended_max_memory", lambda: recommended
    )
    monkeypatch.setattr(
        capacity.torch.mps,
        "set_per_process_memory_fraction",
        fractions.append,
    )

    configuration = configure_mps_allocator()

    assert configuration.recommended_max_memory_bytes == recommended
    assert configuration.configured_ceiling_bytes == expected_ceiling
    assert configuration.applied_fraction == expected_fraction
    assert fractions == [expected_fraction]


@pytest.mark.parametrize("recommended", [None, 0, -1])
def test_missing_or_invalid_recommended_memory_is_preserved_as_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
    recommended: int | None,
) -> None:
    monkeypatch.setattr(
        capacity.torch.mps, "recommended_max_memory", lambda: recommended
    )

    with pytest.raises(MPSAllocatorConfigurationError) as error:
        configure_mps_allocator()

    assert error.value.configuration.recommended_max_memory_bytes is None
    assert error.value.configuration.configured_ceiling_bytes is None
    assert error.value.configuration.applied_fraction is None


def test_set_fraction_failure_preserves_known_allocator_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        capacity.torch.mps, "recommended_max_memory", lambda: 30_000_000_000
    )

    def fail(_: float) -> None:
        raise RuntimeError("MPS allocator unavailable")

    monkeypatch.setattr(capacity.torch.mps, "set_per_process_memory_fraction", fail)

    with pytest.raises(MPSAllocatorConfigurationError) as error:
        configure_mps_allocator()

    assert error.value.configuration.recommended_max_memory_bytes == 30_000_000_000
    assert error.value.configuration.configured_ceiling_bytes == 24_000_000_000
    assert error.value.configuration.applied_fraction is None


def test_capacity_runner_preserves_preallocation_failure_in_base_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_base_runner(*args: object, **kwargs: object) -> BenchmarkResult:
        callback = kwargs["before_allocation"]
        assert callable(callback)
        try:
            callback()
        except MPSAllocatorConfigurationError as exception:
            captured["error"] = exception
        return _base_result(status="failed", finite_loss=None, finite_gradients=None)

    monkeypatch.setattr(capacity, "run_synthetic_benchmark", fake_base_runner)
    monkeypatch.setattr(capacity.torch.mps, "recommended_max_memory", lambda: 0)

    result = run_synthetic_capacity_benchmark(
        CAPACITY_CANDIDATES["E"], project_root=PROJECT_ROOT
    )

    assert "error" in captured
    assert result.status == "failed"
    assert result.base_benchmark_result.status == "failed"
    assert result.recommended_max_memory_bytes is None
    assert result.next_candidate_allowed is False


def test_continuation_boundary_allows_only_healthy_completed_results_below_20gb() -> (
    None
):
    assert _continuation_decision(_base_result(driver_memory=19_999_999_999))[0] is True
    assert (
        _continuation_decision(
            _base_result(driver_memory=SOFT_ESCALATION_DRIVER_MEMORY_BYTES)
        )[0]
        is False
    )
    assert _continuation_decision(_base_result(driver_memory=None))[0] is False
    assert _continuation_decision(_base_result(status="failed"))[0] is False
    assert _continuation_decision(_base_result(error={"type": "Error"}))[0] is False
    assert (
        _continuation_decision(_base_result(stop_reason={"type": "Stopped"}))[0]
        is False
    )
    assert _continuation_decision(_base_result(finite_gradients=False))[0] is False
    assert _continuation_decision(_base_result(swap_after_bytes=2))[0] is False
    assert (
        _continuation_decision(_base_result(driver_memory=24_000_000_001))[0] is False
    )
    assert _continuation_decision(_base_result(peak_memory=24_000_000_001))[0] is False


def test_capacity_result_writer_preserves_nested_base_record_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    result = CapacityBenchmarkResult(
        contract_identifier=CAPACITY_CONTRACT_IDENTIFIER,
        candidate=CAPACITY_CANDIDATES["E"].as_dict(),
        status="completed",
        recommended_max_memory_bytes=30_000_000_000,
        configured_allocator_ceiling_bytes=24_000_000_000,
        applied_allocator_fraction=0.8,
        base_benchmark_result=_base_result(),
        next_candidate_allowed=True,
        continuation_reason="healthy fixture",
    )
    result_path = tmp_path / "capacity-result.json"

    write_capacity_benchmark_result(result_path, result)

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["contract_identifier"] == CAPACITY_CONTRACT_IDENTIFIER
    assert payload["base_benchmark_result"]["status"] == "completed"
    assert payload["base_benchmark_result"]["candidate"]["identifier"] == "E"
    with pytest.raises(FileExistsError):
        write_capacity_benchmark_result(result_path, result)
