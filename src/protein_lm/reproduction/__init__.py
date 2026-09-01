"""Shared, model-agnostic primitives for reproduction comparisons."""

from protein_lm.reproduction.comparison import (
    ComparisonIssue,
    ComparisonReport,
    ComparisonStage,
    CrossEntropyTolerances,
    MetricKey,
    MetricRecord,
    compare_metric_records,
)
from protein_lm.reproduction.run_bundle import RunBundle, RunBundleError, RunStatus

__all__ = [
    "ComparisonIssue",
    "ComparisonReport",
    "ComparisonStage",
    "CrossEntropyTolerances",
    "MetricKey",
    "MetricRecord",
    "compare_metric_records",
    "RunBundle",
    "RunBundleError",
    "RunStatus",
]
