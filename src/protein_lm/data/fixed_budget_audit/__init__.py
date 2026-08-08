"""Public boundary for the fixed-budget similarity audit."""

from typing import TYPE_CHECKING

from protein_lm.data.fixed_budget_audit.errors import SimilarityAuditError

if TYPE_CHECKING:
    from protein_lm.data.fixed_budget_audit.workflow import (
        FixedBudgetAuditResult,
        run_fixed_budget_audit,
    )

__all__ = [
    "FixedBudgetAuditResult",
    "SimilarityAuditError",
    "run_fixed_budget_audit",
]


def __getattr__(name: str) -> object:
    """Load workflow-owned public names only when they are accessed."""

    if name not in {"FixedBudgetAuditResult", "run_fixed_budget_audit"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from protein_lm.data.fixed_budget_audit.workflow import (
        FixedBudgetAuditResult,
        run_fixed_budget_audit,
    )

    exports = {
        "FixedBudgetAuditResult": FixedBudgetAuditResult,
        "run_fixed_budget_audit": run_fixed_budget_audit,
    }
    globals().update(exports)
    return exports[name]
