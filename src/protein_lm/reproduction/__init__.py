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
from protein_lm.reproduction.historical_evidence import (
    ClaimOutcome,
    HistoricalEvidenceError,
    HistoricalEvidenceReport,
    evaluate_historical_evidence,
)
from protein_lm.reproduction.run_bundle import RunBundle, RunBundleError, RunStatus

__all__ = [
    "APPROVED_FOUNDATIONS_CONTRACT_SHA256",
    "ComparisonClaim",
    "ClaimOutcome",
    "ComparisonIssue",
    "ComparisonReport",
    "ComparisonStage",
    "CrossEntropyTolerances",
    "FoundationsContract",
    "HistoricalEvidenceError",
    "HistoricalEvidenceReport",
    "MetricKey",
    "MetricRecord",
    "ReproductionContractError",
    "SourcePin",
    "SourcePinVerification",
    "SourcePinVerificationReport",
    "compare_metric_records",
    "evaluate_historical_evidence",
    "load_foundations_contract",
    "RunBundle",
    "RunBundleError",
    "RunStatus",
    "verify_source_pins",
]
