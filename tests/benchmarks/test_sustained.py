import hashlib
import json
from pathlib import Path

import pytest

from protein_lm.benchmarks.capacity import CAPACITY_CANDIDATES
from protein_lm.benchmarks.metrics import ReadinessChecks, SwapState
from protein_lm.benchmarks.runner import BenchmarkResult
from protein_lm.benchmarks.sustained import (
    SUSTAINED_H_CONFIG,
    SUSTAINED_MEASURED_STEPS,
    _confirmation_decision,
    _sustained_timing_metrics,
    run_sustained_h_confirmation,
    validate_capacity_h_source,
)
import protein_lm.benchmarks.sustained as sustained


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZERO_SWAP = SwapState(raw="total = 0.00M  used = 0.00M", used_bytes=0, total_bytes=0)


def _source_payload() -> dict[str, object]:
    candidate = CAPACITY_CANDIDATES["H"].as_dict()
    return {
        "contract_identifier": "2026-08-07-task-11a-2-mac-mini-capacity-v1",
        "candidate": candidate,
        "status": "completed",
        "next_candidate_allowed": True,
        "base_benchmark_result": {
            "candidate": candidate,
            "status": "completed",
            "parameter_count": 394_923_544,
            "maximum_sampled_mps_driver_memory_bytes": 14_062_288_896,
            "finite_loss": True,
            "finite_gradients": True,
            "error": None,
            "stop_reason": None,
            "swap_before": {"used_bytes": 0},
            "swap_after": {"used_bytes": 0},
        },
    }


def _write_source(path: Path, payload: dict[str, object] | None = None) -> bytes:
    raw = json.dumps(payload or _source_payload(), sort_keys=True).encode()
    path.write_bytes(raw)
    return raw


def _base_result(
    *,
    measurements: list[float] | None = None,
    status: str = "completed",
    driver_memory: int | None = 19_999_999_999,
    parameter_count: int | None = 394_923_544,
    swap_after: SwapState = ZERO_SWAP,
) -> BenchmarkResult:
    measurements = measurements or [1.0] * 30 + [1.2] * 30
    return BenchmarkResult(
        candidate=SUSTAINED_H_CONFIG.as_dict(),
        status=status,
        parameter_count=parameter_count,
        batch_shape=[2, 1024],
        output_shape=[2, 1024, 24],
        target_shape=[2, 1024],
        measured_step_seconds=measurements,
        mean_step_seconds=1.0,
        median_step_seconds=1.0,
        tokens_per_second=2048.0,
        projected_seconds={"1000000": 1.0},
        readiness_checks=ReadinessChecks(False, False, False, False),
        maximum_sampled_mps_allocated_memory_bytes=1,
        maximum_sampled_mps_driver_memory_bytes=driver_memory,
        process_peak_resident_memory_bytes=1,
        swap_before=ZERO_SWAP,
        swap_after=swap_after,
        finite_loss=True,
        finite_gradients=True,
        error=None,
        stop_reason=None,
        environment={},
    )


def test_source_validation_hashes_exact_h_evidence_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "capacity-h.json"
    raw = _write_source(source_path)

    assert validate_capacity_h_source(source_path) == hashlib.sha256(raw).hexdigest()

    tampered = _source_payload()
    tampered["candidate"] = CAPACITY_CANDIDATES["I"].as_dict()
    _write_source(source_path, tampered)
    with pytest.raises(ValueError, match="exact capacity H"):
        validate_capacity_h_source(source_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "failed", "source status"),
        ("next_candidate_allowed", False, "does not permit"),
    ],
)
def test_source_validation_rejects_ineligible_root_status(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    source_path = tmp_path / "capacity-h.json"
    payload = _source_payload()
    payload[field] = value
    _write_source(source_path, payload)

    with pytest.raises(ValueError, match=message):
        validate_capacity_h_source(source_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("parameter_count", 394_923_543, "parameter count"),
        ("parameter_count", 394_923_544.0, "parameter count"),
        ("maximum_sampled_mps_driver_memory_bytes", 20_000_000_000, "20 GB"),
        ("maximum_sampled_mps_driver_memory_bytes", True, "20 GB"),
        ("maximum_sampled_mps_driver_memory_bytes", None, "20 GB"),
    ],
)
def test_source_validation_rejects_tampered_parameter_or_driver_evidence(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    source_path = tmp_path / "capacity-h.json"
    payload = _source_payload()
    base = payload["base_benchmark_result"]
    assert isinstance(base, dict)
    base[field] = value
    _write_source(source_path, payload)

    with pytest.raises(ValueError, match=message):
        validate_capacity_h_source(source_path)


def test_readable_invalid_source_preserves_its_sha_before_mps_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "capacity-h.json"
    payload = _source_payload()
    base = payload["base_benchmark_result"]
    assert isinstance(base, dict)
    base["parameter_count"] = 1
    raw = _write_source(source_path, payload)
    monkeypatch.setattr(sustained, "read_swap_state", lambda: ZERO_SWAP)

    def must_not_run(*args: object, **kwargs: object) -> BenchmarkResult:
        raise AssertionError("invalid source evidence must not start MPS work")

    monkeypatch.setattr(sustained, "run_synthetic_benchmark", must_not_run)

    result = run_sustained_h_confirmation(source_path, project_root=PROJECT_ROOT)

    assert result.base_benchmark_result is None
    assert result.sustained_confirmation_passed is False
    assert result.source_capacity_result_sha256 == hashlib.sha256(raw).hexdigest()


def test_nonzero_preflight_swap_blocks_before_the_base_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "capacity-h.json"
    _write_source(source_path)
    monkeypatch.setattr(
        sustained,
        "read_swap_state",
        lambda: SwapState(raw="used = 1.00M", used_bytes=1, total_bytes=1),
    )

    def must_not_allocate(*args: object, **kwargs: object) -> BenchmarkResult:
        raise AssertionError("preflight failure must not start MPS work")

    monkeypatch.setattr(sustained, "run_synthetic_benchmark", must_not_allocate)

    result = run_sustained_h_confirmation(source_path, project_root=PROJECT_ROOT)

    assert result.status == "failed"
    assert result.base_benchmark_result is None
    assert result.sustained_confirmation_passed is False
    assert result.source_capacity_result_sha256 is not None
    assert "exactly zero" in result.reason


def test_sustained_run_uses_exact_h_with_one_warmup_and_sixty_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "capacity-h.json"
    _write_source(source_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(sustained, "read_swap_state", lambda: ZERO_SWAP)

    def fake_runner(*args: object, **kwargs: object) -> BenchmarkResult:
        config = args[0]
        captured["config"] = config
        observer = kwargs["measured_step_observer"]
        assert callable(observer)
        for index, seconds in enumerate([1.0] * 30 + [1.2] * 30):
            observer(seconds, 1_000 + index, 2_000 + index)
        return _base_result()

    monkeypatch.setattr(sustained, "run_synthetic_benchmark", fake_runner)
    monkeypatch.setattr(
        sustained,
        "configure_mps_allocator",
        lambda: sustained.MPSAllocatorConfiguration(1, 1, 1.0),
    )

    result = run_sustained_h_confirmation(source_path, project_root=PROJECT_ROOT)

    config = captured["config"]
    assert config == SUSTAINED_H_CONFIG
    assert config.identifier == "H"
    assert config.warmup_steps == 1
    assert config.measured_steps == SUSTAINED_MEASURED_STEPS
    assert result.sustained_confirmation_passed is True
    assert result.first_30_median_seconds == 1.0
    assert result.last_30_median_seconds == 1.2
    assert result.slowdown_fraction == pytest.approx(0.2)
    assert result.sampled_mps_driver_memory_growth_bytes == 59


def test_timing_boundary_and_missing_or_short_measurements_do_not_pass() -> None:
    first, last, slowdown = _sustained_timing_metrics([1.0] * 30 + [1.2] * 30)
    assert (first, last, slowdown) == pytest.approx((1.0, 1.2, 0.2))

    first, last, slowdown = _sustained_timing_metrics([1.0] * 30 + [1.21] * 30)
    assert slowdown is not None and slowdown > 0.2
    passed, _, _, _, _ = _confirmation_decision(
        _base_result(measurements=[1.0] * 30 + [1.21] * 30),
        ZERO_SWAP,
        [1] * 60,
        [1] * 60,
    )
    assert passed is False

    passed, reason, _, _, _ = _confirmation_decision(
        _base_result(measurements=[1.0] * 59),
        ZERO_SWAP,
        [1] * 59,
        [1] * 59,
    )
    assert passed is False
    assert "exactly 60" in reason


def test_missing_memory_samples_or_20gb_driver_boundary_do_not_pass() -> None:
    passed, reason, _, _, _ = _confirmation_decision(
        _base_result(), ZERO_SWAP, [1] * 60, [None] * 60
    )
    assert passed is False
    assert "unavailable" in reason

    passed, reason, _, _, _ = _confirmation_decision(
        _base_result(driver_memory=20_000_000_000),
        ZERO_SWAP,
        [1] * 60,
        [1] * 60,
    )
    assert passed is False
    assert "20 GB" in reason


def test_parameter_count_and_ending_swap_must_match_sustained_h_contract() -> None:
    passed, reason, _, _, _ = _confirmation_decision(
        _base_result(parameter_count=1),
        ZERO_SWAP,
        [1] * 60,
        [1] * 60,
    )
    assert passed is False
    assert "parameter count" in reason

    passed, reason, _, _, _ = _confirmation_decision(
        _base_result(swap_after=SwapState(raw=None, used_bytes=None, total_bytes=None)),
        ZERO_SWAP,
        [1] * 60,
        [1] * 60,
    )
    assert passed is False
    assert "ending system swap" in reason
