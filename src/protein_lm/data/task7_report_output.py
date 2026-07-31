"""Assemble and publish the aggregate-only Task 7 report."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from protein_lm.data.random_split import sha256_sidecar
from protein_lm.data.similarity_alignment import (
    CLOSEST_RESIDUAL_CATEGORIES,
    RESIDUAL_CATEGORIES,
)
from protein_lm.data.similarity_audit_policy import (
    APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
    SimilarityAuditError,
    SimilarityAuditPolicy,
)
from protein_lm.data.similarity_fastas import MaterializedInputs
from protein_lm.data.similarity_manifests import (
    PARTITIONS,
    STRATEGIES,
    StrategyManifest,
)
from protein_lm.data.task5_report import (
    CompletedPublicArtifact,
    render_completion_index,
)
from protein_lm.data.task7_report import RenderedTask7Report

OUTPUT_STEM = "task_07_diagnostic_similarity_audit"
REPORT_FILENAMES = (
    f"{OUTPUT_STEM}.json",
    f"{OUTPUT_STEM}.md",
    f"{OUTPUT_STEM}.sha256",
)
COMPLETION_FILENAME = f"{OUTPUT_STEM}.complete.json"
COMPLETION_SCOPE = "week_01_task_07_public_outputs"


def public_pass_evidence(marker: Mapping[str, object]) -> dict[str, object]:
    """Remove private query identifiers from one pass's public evidence."""

    convergence = dict(marker["convergence"])
    convergence.pop("escalated_query_ids", None)
    accepted = dict(marker["accepted"])
    return {
        "query_count": marker["query_count"],
        "stages": marker["stages"],
        "convergence": convergence,
        "accepted": accepted,
    }


def structural_membership_evidence(
    strategy: str,
    manifest: StrategyManifest,
    frozen_report: Mapping[str, object],
) -> dict[str, object]:
    """Build structural membership evidence for one split strategy."""

    evidence = asdict(manifest.structural_audit)
    evidence["record_retention_percent"] = "100.000000"
    evidence["residue_retention_percent"] = "100.000000"
    evidence["exclusion_reasons"] = {}
    if strategy == "group_aware":
        assignment_units = frozen_report.get("assignment_units")
        if not isinstance(assignment_units, dict):
            raise SimilarityAuditError("Task 6 assignment-unit evidence is malformed")
        evidence["largest_assignment_unit_records"] = assignment_units.get(
            "largest_unit_records"
        )
        evidence["largest_assignment_unit_residues"] = assignment_units.get(
            "largest_unit_residues"
        )
        evidence["largest_assignment_unit_original_groups"] = assignment_units.get(
            "largest_unit_original_groups"
        )
    return evidence


def overall_similarity(partitions: Mapping[str, object]) -> dict[str, object]:
    """Combine validation and test aggregates without exposing row-level data."""

    held_out = [partitions[name]["similarity"] for name in ("validation", "test")]
    numerator = sum(item["held_out_queries_with_prohibited_match"] for item in held_out)
    denominator = sum(item["held_out_query_count"] for item in held_out)
    attribution = Counter()
    closest_categories = Counter(
        {category: 0 for category in CLOSEST_RESIDUAL_CATEGORIES}
    )
    status_categories = Counter({category: 0 for category in RESIDUAL_CATEGORIES})
    for item in held_out:
        attribution.update(item["prohibited_pair_attribution"])
        closest_categories.update(item["closest_residual_categories"])
        status_categories.update(item["held_out_query_status_categories"])
    return {
        "held_out_queries_with_prohibited_match": numerator,
        "held_out_query_count": denominator,
        "prohibited_query_rate_percent": (
            f"{(Decimal(numerator) * 100 / Decimal(denominator)):.6f}"
        ),
        "unique_prohibited_pairs": sum(
            item["unique_prohibited_pairs"] for item in held_out
        ),
        "prohibited_pair_attribution": dict(attribution),
        "enforcement_returned_pairs": sum(
            item["enforcement_returned_pairs"] for item in held_out
        ),
        "residual_returned_pairs": sum(
            item["residual_returned_pairs"] for item in held_out
        ),
        "unique_returned_pair_union": sum(
            item["unique_returned_pair_union"] for item in held_out
        ),
        "closest_residual_categories": dict(closest_categories),
        "held_out_query_status_categories": dict(status_categories),
    }


def build_report(
    *,
    policy: SimilarityAuditPolicy,
    code_revision: str,
    fingerprint: str,
    mmseqs_version: str,
    started_at: datetime,
    completed_at: datetime,
    runtime_seconds: float,
    inputs: MaterializedInputs,
    manifests: Mapping[str, StrategyManifest],
    frozen_reports: Mapping[str, Mapping[str, object]],
    database_reports: Mapping[str, object],
    strategy_reports: Mapping[str, object],
) -> dict[str, object]:
    """Assemble the complete aggregate report before rendering validation."""

    command_count, command_runtime = _completed_command_runtime(
        database_reports,
        strategy_reports,
    )
    return {
        "schema_version": policy.schema_version,
        "scope": policy.scope,
        "adjustment_id": policy.adjustment_id,
        "diagnostic_only": True,
        "diagnostic_audit_authorized": policy.diagnostic_audit_authorized,
        "diagnostic_audit_completed": True,
        "candidate_status": policy.candidate_status,
        "repair_authorized": policy.repair_authorized,
        "repair_performed": policy.repair_performed,
        "selected_split_authorized": policy.selected_split_authorized,
        "task8_membership_use_authorized": policy.task8_membership_use_authorized,
        "model_use": policy.model_use,
        "post_audit_review_required": policy.post_audit_review_required,
        "network_requests_made": False,
        "code_revision": code_revision,
        "config_sha256": APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
        "run_fingerprint": fingerprint,
        "inputs": {
            "task4_catalog": asdict(inputs.catalog),
            "task5_public_manifest": asdict(manifests["random"].public_manifest),
            "task5_local_assignment": asdict(manifests["random"].local_assignment),
            "task5_report_sha256": policy.task5_report_sha256,
            "task6_public_manifest": asdict(
                manifests["group_aware"].public_manifest
            ),
            "task6_local_assignment": asdict(
                manifests["group_aware"].local_assignment
            ),
            "task6_report_sha256": policy.task6_report_sha256,
            "task6_repair_state_sha256": policy.task6_repair_state_sha256,
            "source_checksums": frozen_reports["random"]["sources"],
            "materialized_fastas": {
                strategy: {
                    partition: asdict(inputs.fastas[strategy][partition])
                    for partition in PARTITIONS
                }
                for strategy in STRATEGIES
            },
        },
        "procedure": {
            "mmseqs_executable": policy.mmseqs_executable,
            "mmseqs_version": mmseqs_version,
            "threads": policy.threads,
            "format_output": policy.format_output,
            "inclusive_prohibited_boundary": {
                "minimum_identity": policy.prohibited_min_sequence_identity,
                "minimum_query_coverage": policy.prohibited_min_query_coverage,
                "minimum_target_coverage": policy.prohibited_min_target_coverage,
            },
            "prohibited_pair_evidence": policy.prohibited_pair_evidence,
            "staged_caps": [
                policy.initial_cap,
                policy.comparison_cap,
                policy.escalation_cap,
            ],
            "boundary_fixtures_passed": True,
            "target_databases": database_reports,
        },
        "runtime": {
            "final_invocation_started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "final_invocation_wall_clock_seconds": f"{runtime_seconds:.3f}",
            "completed_mmseqs_command_count": command_count,
            "completed_mmseqs_command_runtime_seconds": command_runtime,
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
            },
            "workspace_byte_ceiling": policy.workspace_byte_ceiling,
            "free_space_reserve": policy.free_space_reserve,
        },
        "strategies": dict(strategy_reports),
        "limitations": [
            "Only validation-to-training and test-to-training were searched.",
            "Validation-to-test similarity was not measured.",
            "The heuristic search cannot prove the absence of homology.",
            "The failed-balance candidate has shorter held-out proteins on average.",
            "The diagnostic cannot repair, select, or authorize model use of a split.",
        ],
    }


def _completed_command_runtime(
    database_reports: Mapping[str, object],
    strategy_reports: Mapping[str, object],
) -> tuple[int, str]:
    durations = []
    for raw_database in database_reports.values():
        if not isinstance(raw_database, dict):
            raise SimilarityAuditError("database runtime evidence is malformed")
        durations.append(Decimal(str(raw_database.get("runtime_seconds"))))
    for raw_strategy in strategy_reports.values():
        if not isinstance(raw_strategy, dict):
            raise SimilarityAuditError("strategy runtime evidence is malformed")
        partitions = raw_strategy["partitions"]
        for partition in ("validation", "test"):
            for raw_pass in partitions[partition]["passes"].values():
                for raw_stage in raw_pass["stages"].values():
                    durations.append(Decimal(str(raw_stage["runtime_seconds"])))
    if any(not duration.is_finite() or duration < 0 for duration in durations):
        raise SimilarityAuditError("command runtime evidence is invalid")
    return len(durations), f"{sum(durations, Decimal('0')):.3f}"


def publish_report(
    rendered: RenderedTask7Report,
    workspace: Path,
    report_directory: Path,
) -> None:
    """Atomically publish the validated JSON, Markdown, digest, and completion marker."""

    staging = workspace / "public_report_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    outputs = {
        f"{OUTPUT_STEM}.json": rendered.json_text.encode("utf-8"),
        f"{OUTPUT_STEM}.md": rendered.markdown_text.encode("utf-8"),
        f"{OUTPUT_STEM}.sha256": sha256_sidecar(
            f"{OUTPUT_STEM}.json",
            rendered.json_sha256,
        ).encode("ascii"),
    }
    completed = []
    for filename, content in outputs.items():
        path = staging / filename
        path.write_bytes(content)
        completed.append(
            CompletedPublicArtifact(
                relative_path=f"reports/week_01/{filename}",
                byte_size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    completion_text = render_completion_index(
        tuple(completed),
        scope=COMPLETION_SCOPE,
    )
    (staging / COMPLETION_FILENAME).write_text(completion_text, encoding="utf-8")

    report_directory.mkdir(parents=True, exist_ok=True)
    completion_path = report_directory / COMPLETION_FILENAME
    completion_path.unlink(missing_ok=True)
    for filename in REPORT_FILENAMES:
        (staging / filename).replace(report_directory / filename)
    (staging / COMPLETION_FILENAME).replace(completion_path)
    staging.rmdir()
