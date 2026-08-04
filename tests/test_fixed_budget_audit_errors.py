"""Public contracts for the fixed-budget audit package and error hierarchy."""

import os
import subprocess
import sys
from pathlib import Path

from protein_lm.data import fixed_budget_audit
from protein_lm.data.fixed_budget_audit import errors
from protein_lm.data.similarity_audit_policy import (
    SimilarityAuditError as ExistingSimilarityAuditError,
)

SPECIFIC_ERRORS = (
    errors.AuditConfigurationError,
    errors.SourceEvidenceError,
    errors.AuditExecutionError,
    errors.AuditValidationError,
    errors.AuditPublicationError,
)


def test_package_reuses_the_existing_public_base_error() -> None:
    assert fixed_budget_audit.SimilarityAuditError is ExistingSimilarityAuditError
    assert errors.SimilarityAuditError is ExistingSimilarityAuditError
    assert issubclass(ExistingSimilarityAuditError, ValueError)
    assert ExistingSimilarityAuditError.__bases__ == (ValueError,)


def test_specific_errors_have_the_exact_approved_subclass_graph() -> None:
    assert tuple(error.__name__ for error in SPECIFIC_ERRORS) == (
        "AuditConfigurationError",
        "SourceEvidenceError",
        "AuditExecutionError",
        "AuditValidationError",
        "AuditPublicationError",
    )
    assert all(
        error.__bases__ == (ExistingSimilarityAuditError,) for error in SPECIFIC_ERRORS
    )


def test_package_exports_the_exact_lazy_public_contract() -> None:
    assert fixed_budget_audit.__all__ == [
        "FixedBudgetAuditResult",
        "SimilarityAuditError",
        "run_fixed_budget_audit",
    ]
    assert errors.__all__ == [
        "SimilarityAuditError",
        "AuditConfigurationError",
        "SourceEvidenceError",
        "AuditExecutionError",
        "AuditValidationError",
        "AuditPublicationError",
    ]


def test_package_workflow_exports_have_exact_owner_identities() -> None:
    from protein_lm.data.fixed_budget_audit.workflow import (
        FixedBudgetAuditResult,
        run_fixed_budget_audit,
    )

    assert fixed_budget_audit.FixedBudgetAuditResult is FixedBudgetAuditResult
    assert fixed_budget_audit.run_fixed_budget_audit is run_fixed_budget_audit
    assert not hasattr(fixed_budget_audit, "A004WorkflowResult")
    assert not hasattr(fixed_budget_audit, "run_a004_fixed_budget_audit")


def test_package_import_is_lazy_and_either_public_access_loads_workflow() -> None:
    project_root = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    package = "protein_lm.data.fixed_budget_audit"
    workflow = f"{package}.workflow"
    for public_name in ("FixedBudgetAuditResult", "run_fixed_budget_audit"):
        code = (
            "import importlib, sys; "
            f"package = importlib.import_module({package!r}); "
            f"workflow = {workflow!r}; "
            "assert workflow not in sys.modules; "
            f"getattr(package, {public_name!r}); "
            "assert workflow in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=project_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
