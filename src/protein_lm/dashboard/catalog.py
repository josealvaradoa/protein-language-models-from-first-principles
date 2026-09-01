"""Immutable, server-owned reproduction dashboard job definitions."""

from __future__ import annotations

from dataclasses import dataclass
import shlex
import sys


@dataclass(frozen=True, slots=True)
class JobDefinition:
    """A job the dashboard may describe or launch without client input."""

    job_id: str
    experiment_id: str
    stage: str
    title: str
    description: str
    argv: tuple[str, ...]
    availability: str
    reason: str | None = None

    @property
    def launchable(self) -> bool:
        return self.availability == "available"


JOB_CATALOG: tuple[JobDefinition, ...] = (
    JobDefinition(
        job_id="setup_check",
        experiment_id="workspace",
        stage="verify",
        title="Local setup check",
        description="Verify the local Python and library setup used by the workspace.",
        argv=(sys.executable, "scripts/check_setup.py"),
        availability="available",
    ),
    JobDefinition(
        job_id="week2_public_report_validation",
        experiment_id="week_02",
        stage="verify",
        title="Week 2 public-report validator",
        description="Read-only validation of the committed Week 2 public report.",
        argv=(sys.executable, "scripts/validate_week2_bigram_public_report.py"),
        availability="available",
    ),
    JobDefinition(
        job_id="week3_public_report_validation",
        experiment_id="week_03",
        stage="verify",
        title="Week 3 public-report validator",
        description="Read-only validation of the committed Week 3 public report.",
        argv=(sys.executable, "scripts/validate_week3_mlp_public_report.py"),
        availability="available",
    ),
    JobDefinition(
        job_id="week1_audit_reproduction",
        experiment_id="week_01",
        stage="reevaluate",
        title="Week 1 audit reproduction",
        description="Rerun the Week 1 split audit, not a model evaluation.",
        argv=(),
        availability="blocked",
        reason="reproduction_contract_pending",
    ),
    JobDefinition(
        job_id="week2_reevaluation",
        experiment_id="week_02",
        stage="reevaluate",
        title="Week 2 re-evaluation",
        description="Re-evaluate the approved Week 2 experiment contract.",
        argv=(),
        availability="blocked",
        reason="reproduction_contract_pending",
    ),
    JobDefinition(
        job_id="week2_retraining_refit",
        experiment_id="week_02",
        stage="retrain",
        title="Week 2 retraining/refit",
        description="Retrain or refit only after a new reproduction contract is approved.",
        argv=(),
        availability="blocked",
        reason="reproduction_contract_pending",
    ),
    JobDefinition(
        job_id="week3_reevaluation",
        experiment_id="week_03",
        stage="reevaluate",
        title="Week 3 re-evaluation",
        description="Re-evaluate the approved Week 3 experiment contract.",
        argv=(),
        availability="blocked",
        reason="reproduction_contract_pending",
    ),
    JobDefinition(
        job_id="week3_retraining",
        experiment_id="week_03",
        stage="retrain",
        title="Week 3 retraining",
        description="Retrain only after a new reproduction contract is approved.",
        argv=(),
        availability="blocked",
        reason="reproduction_contract_pending",
    ),
    JobDefinition(
        job_id="week1_retraining",
        experiment_id="week_01",
        stage="retrain",
        title="Week 1 retraining",
        description="Week 1 has no model-training stage to reproduce.",
        argv=(),
        availability="not_applicable",
        reason="no_retraining_model_stage",
    ),
)


HISTORICAL_RESULTS: tuple[dict[str, object], ...] = (
    {
        "label": "Week 1 historical audit result",
        "kind": "historical_display_only",
        "source": "reports/week_01/task_07_read_only_fixed_budget_audit_a004.md",
        "metrics": {
            "eligible_protein_count": 557718,
            "random_validation_strong_overlap_percent": 87.404979,
            "random_test_strong_overlap_percent": 87.559128,
            "group_aware_validation_strong_overlap_percent": 35.919440,
            "group_aware_test_strong_overlap_percent": 37.399853,
        },
        "historical_conclusion": (
            "The original hypothesis failed: balance failed and substantial "
            "overlap remained; neither assignment was approved for training."
        ),
    },
    {
        "label": "Week 2 family-aware native validation",
        "kind": "historical_display_only",
        "source": "reports/week_02/bigram_evaluation_v1.json",
        "metrics": {
            "unigram": {"cross_entropy": 2.906191441029875, "accuracy": 0.09607444315064043},
            "count_bigram": {
                "cross_entropy": 2.8915253371291687,
                "accuracy": 0.09885306773147293,
            },
            "neural_bigram": {
                "cross_entropy": 2.8985748162212692,
                "accuracy": 0.09885306773147293,
            },
        },
    },
    {
        "label": "Week 3 C20 final three-seed validation",
        "kind": "historical_display_only",
        "source": "reports/week_03/mlp_evaluation_v1.json",
        "metrics": {
            "mean_cross_entropy": 2.8636658562202886,
            "mean_accuracy": 0.1112489317787695,
            "cross_entropy_standard_deviation": 0.00001985865257320209,
            "parameter_count": 530293,
        },
    },
)


def catalog_payload() -> dict[str, object]:
    """Return only display data, never command arguments or runtime inputs."""

    return {
        "jobs": [
            {
                "job_id": job.job_id,
                "experiment_id": job.experiment_id,
                "stage": job.stage,
                "title": job.title,
                "description": job.description,
                "availability": job.availability,
                "reason": job.reason,
                "command_display": shlex.join(job.argv) if job.launchable else None,
            }
            for job in JOB_CATALOG
        ],
        "historical_results": list(HISTORICAL_RESULTS),
    }


def find_job(job_id: str) -> JobDefinition | None:
    """Find an exact server-declared job identifier."""

    return next((job for job in JOB_CATALOG if job.job_id == job_id), None)
