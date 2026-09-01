"""Shared, model-agnostic primitives for reproduction comparisons."""

from protein_lm.reproduction.contract import (
    APPROVED_FOUNDATIONS_CONTRACT_SHA256,
    ComparisonClaim,
    FoundationsContract,
    ReproductionContractError,
    SourcePin,
    SourcePinVerification,
    SourcePinVerificationReport,
    load_foundations_contract,
    verify_source_pins,
)
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
    "APPROVED_FOUNDATIONS_CONTRACT_SHA256",
    "ComparisonClaim",
    "ComparisonIssue",
    "ComparisonReport",
    "ComparisonStage",
    "CrossEntropyTolerances",
    "FoundationsContract",
    "MetricKey",
    "MetricRecord",
    "ReproductionContractError",
    "SourcePin",
    "SourcePinVerification",
    "SourcePinVerificationReport",
    "compare_metric_records",
    "load_foundations_contract",
    "RunBundle",
    "RunBundleError",
    "RunStatus",
    "verify_source_pins",
]
