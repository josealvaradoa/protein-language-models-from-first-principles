"""Semantic error hierarchy for the fixed-budget similarity audit."""

from protein_lm.data.similarity_audit_policy import SimilarityAuditError

__all__ = [
    "SimilarityAuditError",
    "AuditConfigurationError",
    "SourceEvidenceError",
    "AuditExecutionError",
    "AuditValidationError",
    "AuditPublicationError",
]


class AuditConfigurationError(SimilarityAuditError):
    """Raised when fixed-budget audit configuration is invalid."""


class SourceEvidenceError(SimilarityAuditError):
    """Raised when frozen source evidence is missing or inconsistent."""


class AuditExecutionError(SimilarityAuditError):
    """Raised when a fixed-budget audit execution stage fails."""


class AuditValidationError(SimilarityAuditError):
    """Raised when fixed-budget audit evidence fails validation."""


class AuditPublicationError(SimilarityAuditError):
    """Raised when fixed-budget audit evidence cannot be published safely."""
