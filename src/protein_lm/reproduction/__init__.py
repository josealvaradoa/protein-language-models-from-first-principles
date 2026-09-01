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

__all__ = [
    "ComparisonIssue",
    "ComparisonReport",
    "ComparisonStage",
    "CrossEntropyTolerances",
    "MetricKey",
    "MetricRecord",
    "compare_metric_records",
]
