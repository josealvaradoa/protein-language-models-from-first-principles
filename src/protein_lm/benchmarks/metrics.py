"""Measurement, environment, and readiness helpers for device benchmarks."""

from __future__ import annotations

import platform
import resource
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


NORMAL_MEMORY_LIMIT_BYTES = 24_000_000_000
DEVELOPMENT_TIME_LIMIT_SECONDS = 20 * 60
PROJECTION_TOKEN_COUNTS = (1_000_000, 10_000_000, 100_000_000)
SYSTEM_FACT_QUERY_TIMEOUT_SECONDS = 1


@dataclass(frozen=True)
class SwapState:
    """A best-effort snapshot of macOS system swap usage."""

    raw: str | None
    used_bytes: int | None
    total_bytes: int | None


@dataclass(frozen=True)
class ReadinessChecks:
    """Independent fixed-budget checks, not an exclusive performance class."""

    publication_100m_within_3h: bool
    exceptional_100m_within_10h: bool
    development_10m_within_20m: bool
    smoke_1m_within_2m: bool

    def as_dict(self) -> dict[str, bool]:
        """Return named checks for concise command-line reporting."""
        return asdict(self)


def calculate_readiness_checks(
    *,
    status: str,
    tokens_per_second: float | None,
) -> ReadinessChecks:
    """Evaluate every frozen threshold when a complete run has a throughput."""
    if status != "completed" or tokens_per_second is None or tokens_per_second <= 0:
        return ReadinessChecks(False, False, False, False)

    projected = projected_seconds(tokens_per_second)
    assert projected is not None
    return ReadinessChecks(
        publication_100m_within_3h=projected["100000000"] <= 3 * 60 * 60,
        exceptional_100m_within_10h=projected["100000000"] <= 10 * 60 * 60,
        development_10m_within_20m=projected["10000000"] <= 20 * 60,
        smoke_1m_within_2m=projected["1000000"] <= 2 * 60,
    )


def projected_seconds(tokens_per_second: float | None) -> dict[str, float] | None:
    """Project fixed token totals from measured synthetic-work throughput."""
    if tokens_per_second is None or tokens_per_second <= 0:
        return None
    return {
        str(token_count): token_count / tokens_per_second
        for token_count in PROJECTION_TOKEN_COUNTS
    }


def read_swap_state() -> SwapState:
    """Read macOS swap state when sysctl exposes it, otherwise retain absence."""
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return SwapState(raw=None, used_bytes=None, total_bytes=None)

    raw = completed.stdout.strip()
    return SwapState(
        raw=raw,
        used_bytes=swap_value_bytes(raw, "used"),
        total_bytes=swap_value_bytes(raw, "total"),
    )


def process_peak_resident_memory_bytes() -> int | None:
    """Return the process peak resident memory in bytes where the OS exposes it."""
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None
    return int(peak if sys.platform == "darwin" else peak * 1024)


def collect_environment(
    device: torch.device,
    project_root: Path,
) -> dict[str, str | int | None]:
    """Collect the runtime facts needed to compare result records."""
    environment: dict[str, str | int | None] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "macos": platform.mac_ver()[0] or None,
        "architecture": platform.machine(),
        "backend": device.type,
        "precision": "float32",
        "git_revision": git_revision(project_root),
    }
    environment.update(macos_system_facts())
    return environment


def macos_system_facts() -> dict[str, str | int | None]:
    """Read descriptive Mac host facts without making them benchmark requirements."""
    unavailable = {
        "mac_model": None,
        "cpu_brand": None,
        "physical_memory_bytes": None,
    }
    if platform.system() != "Darwin":
        return unavailable

    memory_bytes = sysctl_text("hw.memsize")
    try:
        parsed_memory_bytes = int(memory_bytes) if memory_bytes is not None else None
    except ValueError:
        parsed_memory_bytes = None
    return {
        "mac_model": sysctl_text("hw.model"),
        "cpu_brand": sysctl_text("machdep.cpu.brand_string"),
        "physical_memory_bytes": parsed_memory_bytes,
    }


def sysctl_text(name: str) -> str | None:
    """Read one bounded macOS sysctl value, retaining absence on any failure."""
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
            timeout=SYSTEM_FACT_QUERY_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def synchronize(device: torch.device) -> None:
    """Wait for MPS work only when MPS was the explicit requested device."""
    if device.type == "mps":
        torch.mps.synchronize()


def sample_mps_memory(
    device: torch.device,
    maximum_allocated: int | None,
    maximum_driver: int | None,
) -> tuple[int | None, int | None]:
    """Return updated sampled MPS allocation and driver-memory maxima."""
    if device.type != "mps":
        return maximum_allocated, maximum_driver
    allocated = torch.mps.current_allocated_memory()
    driver = torch.mps.driver_allocated_memory()
    return max(maximum_allocated or 0, allocated), max(maximum_driver or 0, driver)


def memory_limit_exceeded(memory_bytes: int | None) -> bool:
    """Check the approved 24 GB normal memory limit."""
    return memory_bytes is not None and memory_bytes > NORMAL_MEMORY_LIMIT_BYTES


def swap_grew(before: SwapState, after: SwapState) -> bool:
    """Report measured system-swap growth when both snapshots are available."""
    return (
        before.used_bytes is not None
        and after.used_bytes is not None
        and after.used_bytes > before.used_bytes
    )


def mean_or_none(values: list[float]) -> float | None:
    """Return the arithmetic mean only when the benchmark timed a step."""
    return statistics.mean(values) if values else None


def error_details(exception: Exception) -> dict[str, str]:
    """Serialize an exception without hiding its device-specific message."""
    return {"type": type(exception).__name__, "message": str(exception)}


def swap_value_bytes(raw: str, label: str) -> int | None:
    """Parse one `sysctl vm.swapusage` size into bytes."""
    marker = f"{label} ="
    if marker not in raw:
        return None
    value = raw.split(marker, maxsplit=1)[1].lstrip().split(maxsplit=1)[0]
    if len(value) < 2:
        return None
    unit = value[-1].upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if unit not in multipliers:
        return None
    try:
        return int(float(value[:-1]) * multipliers[unit])
    except ValueError:
        return None


def git_revision(project_root: Path) -> str | None:
    """Read the checked-out revision, retaining absence if Git is unavailable."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None
