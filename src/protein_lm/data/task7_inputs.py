"""Frozen-input validation and FASTA preparation for the Task 7 audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from protein_lm.data.similarity_audit_policy import (
    APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
    SimilarityAuditError,
    SimilarityAuditPolicy,
    load_similarity_audit_policy,
)
from protein_lm.data.similarity_fastas import (
    MaterializedInputs,
    materialize_strategy_fastas,
)
from protein_lm.data.similarity_manifests import (
    PARTITIONS,
    STRATEGIES,
    StrategyManifest,
    load_strategy_manifest,
)
from protein_lm.data.task7_checkpoints import (
    fasta_evidence_from,
    file_evidence_from,
    file_identity,
    read_json,
    require_marker_identity,
    verify_file,
    write_json_atomic,
)
from protein_lm.data.task7_execution import (
    git_output,
    require_committed_execution_code,
    verify_mmseqs,
)


def load_frozen_manifests(
    paths: Mapping[str, Path],
    policy: SimilarityAuditPolicy,
) -> dict[str, StrategyManifest]:
    """Load the exact Task 5 and Task 6 memberships authorized by A-003."""

    random_manifest = load_strategy_manifest(
        public_path=paths["task5_public"],
        local_path=paths["task5_local"],
        strategy="random",
        stage="diagnostic",
        expected_public_sha256=policy.task5_public_manifest_sha256,
        expected_local_sha256=policy.task5_local_assignment_sha256,
    )
    candidate_manifest = load_strategy_manifest(
        public_path=paths["task6_public"],
        local_path=paths["task6_local"],
        strategy="group_aware",
        stage="pre_repair",
        expected_public_sha256=policy.task6_public_manifest_sha256,
        expected_local_sha256=policy.task6_local_assignment_sha256,
    )
    return {"random": random_manifest, "group_aware": candidate_manifest}


def ensure_materialized_inputs(
    *,
    workspace: Path,
    catalog_path: Path,
    manifests: Mapping[str, StrategyManifest],
    policy: SimilarityAuditPolicy,
    fingerprint: str,
) -> MaterializedInputs:
    """Create or verify the six ignored FASTAs used by the audit."""

    fasta_directory = workspace / "fastas"
    marker_path = fasta_directory / "complete.json"
    if marker_path.exists():
        marker = read_json(marker_path)
        require_marker_identity(marker, fingerprint, "materialized_inputs")
        catalog_evidence = file_evidence_from(marker["catalog"])
        if (
            catalog_evidence.row_count != policy.task4_catalog_row_count
            or catalog_evidence.byte_size != policy.task4_catalog_byte_size
            or catalog_evidence.sha256 != policy.task4_catalog_sha256
        ):
            raise SimilarityAuditError("materialized-input catalog evidence drifted")
        verify_file(
            catalog_path,
            catalog_evidence.byte_size,
            catalog_evidence.sha256,
        )
        fastas = {
            strategy: {
                partition: fasta_evidence_from(marker["fastas"][strategy][partition])
                for partition in PARTITIONS
            }
            for strategy in STRATEGIES
        }
        for strategy in STRATEGIES:
            for partition in PARTITIONS:
                verify_file(
                    fasta_path(workspace, strategy, partition),
                    fastas[strategy][partition].byte_size,
                    fastas[strategy][partition].sha256,
                )
        return MaterializedInputs(
            catalog=catalog_evidence,
            fastas=fastas,
        )

    if fasta_directory.exists() and any(fasta_directory.iterdir()):
        raise SimilarityAuditError(
            "materialized-input FASTA directory lacks its completion marker"
        )

    print("materializing six pinned FASTA inputs...")
    inputs = materialize_strategy_fastas(
        catalog_path=catalog_path,
        manifests=manifests,
        output_directory=fasta_directory,
        policy=policy,
    )
    write_json_atomic(
        marker_path,
        {
            "schema_version": 1,
            "stage": "materialized_inputs",
            "fingerprint": fingerprint,
            "catalog": asdict(inputs.catalog),
            "fastas": {
                strategy: {
                    partition: asdict(inputs.fastas[strategy][partition])
                    for partition in PARTITIONS
                }
                for strategy in STRATEGIES
            },
        },
    )
    return inputs


def load_and_validate_frozen_reports(
    paths: Mapping[str, Path],
    policy: SimilarityAuditPolicy,
) -> dict[str, dict[str, object]]:
    """Load the pinned Task 5 and Task 6 reports and enforce their authority."""

    task5 = _load_pinned_json(paths["task5_report"], policy.task5_report_sha256)
    task6 = _load_pinned_json(paths["task6_report"], policy.task6_report_sha256)
    task5_expected = {
        "scope": "week_01_task_05_random_diagnostic",
        "strategy": "random",
        "stage": "diagnostic",
        "diagnostic_only": True,
        "selected_for_training": False,
        "model_use": "prohibited",
    }
    task6_expected = {
        "scope": "week_01_task_06_group_aware_pre_repair",
        "strategy": "group_aware",
        "stage": "pre_repair",
        "candidate_status": "failed_balance",
        "task6_gates_passed": False,
        "task7_authorized": False,
        "selected_for_training": False,
        "model_use": "prohibited",
    }
    _require_report_fields(task5, task5_expected, "Task 5")
    _require_report_fields(task6, task6_expected, "Task 6")
    if task5.get("sources") != task6.get("sources"):
        raise SimilarityAuditError("Task 5 and Task 6 source evidence differs")
    repair_state = task6.get("repair_state")
    if not isinstance(repair_state, dict) or repair_state.get("sha256") != (
        policy.task6_repair_state_sha256
    ):
        raise SimilarityAuditError("Task 6 repair-state-zero digest drifted")
    return {"random": task5, "group_aware": task6}


def validate_report_populations(
    reports: Mapping[str, Mapping[str, object]],
    manifests: Mapping[str, StrategyManifest],
    policy: SimilarityAuditPolicy,
) -> dict[str, dict[str, dict[str, object]]]:
    """Prove report populations agree with their frozen manifests."""

    balances = {}
    for strategy in STRATEGIES:
        report = reports[strategy]
        population = report.get("population")
        if not isinstance(population, dict) or (
            population.get("records"),
            population.get("residues"),
        ) != (policy.expected_eligible_records, policy.expected_eligible_residues):
            raise SimilarityAuditError(f"{strategy} report population drifted")
        report_partitions = report.get("partitions")
        if not isinstance(report_partitions, dict) or set(report_partitions) != set(
            PARTITIONS
        ):
            raise SimilarityAuditError(f"{strategy} report partitions drifted")
        strategy_balances = {}
        for partition in PARTITIONS:
            raw = report_partitions[partition]
            manifest_population = manifests[strategy].partitions[partition]
            if not isinstance(raw, dict) or (
                raw.get("records"),
                raw.get("residues"),
                raw.get("unique_groups"),
            ) != (
                manifest_population.records,
                manifest_population.residues,
                manifest_population.unique_groups,
            ):
                raise SimilarityAuditError(
                    f"{strategy} {partition} report and manifest disagree"
                )
            strategy_balances[partition] = {
                key: raw[key]
                for key in (
                    "target_numerator",
                    "target_denominator",
                    "target_share_percent",
                    "records",
                    "residues",
                    "unique_groups",
                    "record_share_percent",
                    "residue_share_percent",
                    "record_deviation_percentage_points",
                    "residue_deviation_percentage_points",
                )
            }
        balances[strategy] = strategy_balances
    return balances


def policy_paths(
    policy: SimilarityAuditPolicy,
    project_root: Path,
) -> dict[str, Path]:
    """Resolve every configured input under the repository root."""

    configured = {
        "workspace": policy.workspace_relative_path,
        "catalog": policy.task4_catalog_relative_path,
        "task5_public": policy.task5_public_manifest_relative_path,
        "task5_local": policy.task5_local_assignment_relative_path,
        "task5_report": policy.task5_report_relative_path,
        "task6_public": policy.task6_public_manifest_relative_path,
        "task6_local": policy.task6_local_assignment_relative_path,
        "task6_report": policy.task6_report_relative_path,
    }
    resolved = {}
    root = project_root.resolve()
    for name, relative in configured.items():
        path = (project_root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise SimilarityAuditError(
                f"configured {name} path leaves the repository"
            ) from error
        resolved[name] = path
    return resolved


def run_fingerprint(
    *,
    policy: SimilarityAuditPolicy,
    code_revision: str,
    mmseqs_version: str,
) -> str:
    """Identify the exact code, config, tool, and frozen inputs for this run."""

    payload = {
        "code_revision": code_revision,
        "config_sha256": APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
        "mmseqs_version": mmseqs_version,
        "task4_catalog_sha256": policy.task4_catalog_sha256,
        "task5_public_manifest_sha256": policy.task5_public_manifest_sha256,
        "task5_local_assignment_sha256": policy.task5_local_assignment_sha256,
        "task5_report_sha256": policy.task5_report_sha256,
        "task6_public_manifest_sha256": policy.task6_public_manifest_sha256,
        "task6_local_assignment_sha256": policy.task6_local_assignment_sha256,
        "task6_report_sha256": policy.task6_report_sha256,
        "task6_repair_state_sha256": policy.task6_repair_state_sha256,
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def reverify_frozen_run_state(
    *,
    paths: Mapping[str, Path],
    policy: SimilarityAuditPolicy,
    code_revision: str,
    mmseqs_version: str,
    config_path: Path,
    project_root: Path,
) -> None:
    """Repeat every mutable trust check immediately before publication."""

    expected_checksums = {
        "catalog": policy.task4_catalog_sha256,
        "task5_public": policy.task5_public_manifest_sha256,
        "task5_local": policy.task5_local_assignment_sha256,
        "task5_report": policy.task5_report_sha256,
        "task6_public": policy.task6_public_manifest_sha256,
        "task6_local": policy.task6_local_assignment_sha256,
        "task6_report": policy.task6_report_sha256,
    }
    for name, expected_sha256 in expected_checksums.items():
        if file_identity(paths[name])["sha256"] != expected_sha256:
            raise SimilarityAuditError(
                f"frozen {name} checksum changed during the audit"
            )
    load_similarity_audit_policy(config_path)
    require_committed_execution_code(project_root)
    if git_output(project_root, "rev-parse", "HEAD") != code_revision:
        raise SimilarityAuditError("code revision changed during the audit")
    if verify_mmseqs(policy, project_root) != mmseqs_version:
        raise SimilarityAuditError("MMseqs2 version changed during the audit")


def fasta_path(workspace: Path, strategy: str, partition: str) -> Path:
    """Return the deterministic path for one materialized FASTA."""

    return workspace / "fastas" / f"{strategy}_{partition}.fasta"


def _load_pinned_json(path: Path, expected_sha256: str) -> dict[str, object]:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise SimilarityAuditError(f"pinned report checksum drifted: {path.name}")
    try:
        parsed = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SimilarityAuditError(f"pinned report is malformed: {path.name}") from error
    if not isinstance(parsed, dict):
        raise SimilarityAuditError(f"pinned report root is not an object: {path.name}")
    return parsed


def _require_report_fields(
    report: Mapping[str, object],
    expected: Mapping[str, object],
    name: str,
) -> None:
    drift = [
        f"{key}: found {report.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if report.get(key) != value
    ]
    if drift:
        raise SimilarityAuditError(f"{name} authority drift: " + "; ".join(drift))
