"""Synthetic hardware-envelope benchmarks with no protein data dependency."""

from protein_lm.benchmarks.config import BENCHMARK_CANDIDATES, BenchmarkConfig
from protein_lm.benchmarks.capacity import (
    CAPACITY_CANDIDATES,
    CapacityBenchmarkResult,
    run_synthetic_capacity_benchmark,
)
from protein_lm.benchmarks.metrics import ReadinessChecks
from protein_lm.benchmarks.runner import BenchmarkResult, run_synthetic_benchmark

__all__ = [
    "BENCHMARK_CANDIDATES",
    "CAPACITY_CANDIDATES",
    "BenchmarkConfig",
    "BenchmarkResult",
    "CapacityBenchmarkResult",
    "ReadinessChecks",
    "run_synthetic_benchmark",
    "run_synthetic_capacity_benchmark",
]
