"""Frozen inputs and read-only A-003 evidence imported by the A-004 audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath

from protein_lm.data.artifacts import (
    canonical_evidence_from,
    fasta_evidence_from,
    file_evidence_from,
    file_identity,
    read_json,
    require_marker_identity,
    verify_database_artifacts,
    verify_file,
    write_json_atomic,
)
from protein_lm.data.fixed_budget_audit.config import (
    A004Policy,
    FIXED_CAPS,
    resolve_a004_paths,
)
from protein_lm.data.fixed_budget_audit.errors import SourceEvidenceError
from protein_lm.data.fixed_budget_audit.execution import (
    git_output,
    require_committed_execution_code,
    verify_mmseqs,
)
from protein_lm.data.fixed_budget_audit.search import createdb_command, search_command
from protein_lm.data.similarity_audit_models import FileEvidence
from protein_lm.data.similarity_audit_policy import (
    APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256,
    SimilarityAuditPolicy,
    load_similarity_audit_policy,
)
from protein_lm.data.similarity_fastas import (
    FastaEvidence,
    MaterializedInputs,
    iter_one_line_fasta,
    materialize_strategy_fastas,
)
from protein_lm.data.similarity_manifests import (
    PARTITIONS,
    STRATEGIES,
    StrategyManifest,
    load_strategy_manifest,
)
from protein_lm.data.similarity_results import compare_canonical_results


# Public evidence records


@dataclass(frozen=True)
class MarkerEvidence:
    """Identity of one pinned historical completion marker."""

    byte_size: int
    sha256: str


@dataclass(frozen=True)
class DatabaseImport:
    """Verified A-003 target-database evidence."""

    marker: MarkerEvidence
    artifact_count: int


@dataclass(frozen=True)
class ImportedStage:
    """One verified A-003 residual search stage."""

    cap: int
    marker: MarkerEvidence
    query_fasta: FastaEvidence
    canonical: FileEvidence
    canonical_path: Path
    command: tuple[str, ...]
    runtime_seconds: str


@dataclass(frozen=True)
class A003Import:
    """Verified inputs and stages available to the A-004 workflow."""

    fingerprint: str
    fastas: Mapping[str, Mapping[str, FastaEvidence]]
    database: DatabaseImport
    stages: tuple[ImportedStage, ...]
    escalated_query_ids: tuple[str, ...]

    @property
    def training_fasta(self) -> FastaEvidence:
        """Return the imported random training FASTA for the reused database."""

        return self.fasta("random", "training")

    @property
    def validation_fasta(self) -> FastaEvidence:
        """Return the imported random validation FASTA for the reused stages."""

        return self.fasta("random", "validation")

    def fasta(self, strategy: str, partition: str) -> FastaEvidence:
        """Return one checksum-verified preserved Task 7 FASTA."""

        try:
            return self.fastas[strategy][partition]
        except KeyError as error:
            raise SourceEvidenceError(
                f"A-003 FASTA is unavailable: {strategy} {partition}"
            ) from error

    def stage(self, cap: int) -> ImportedStage:
        """Return one imported stage by its fixed cap."""

        try:
            return next(stage for stage in self.stages if stage.cap == cap)
        except StopIteration as error:
            raise SourceEvidenceError(
                f"cap was not imported from A-003: {cap}"
            ) from error


# Frozen Task 5 and Task 6 inputs


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
            raise SourceEvidenceError(
                f"configured {name} path leaves the repository"
            ) from error
        resolved[name] = path
    return resolved


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
        raise SourceEvidenceError("Task 5 and Task 6 source evidence differs")
    repair_state = task6.get("repair_state")
    if not isinstance(repair_state, dict) or repair_state.get("sha256") != (
        policy.task6_repair_state_sha256
    ):
        raise SourceEvidenceError("Task 6 repair-state-zero digest drifted")
    return {"random": task5, "group_aware": task6}


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
            raise SourceEvidenceError(f"{strategy} report population drifted")
        report_partitions = report.get("partitions")
        if not isinstance(report_partitions, dict) or set(report_partitions) != set(
            PARTITIONS
        ):
            raise SourceEvidenceError(f"{strategy} report partitions drifted")
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
                raise SourceEvidenceError(
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
            raise SourceEvidenceError("materialized-input catalog evidence drifted")
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
        raise SourceEvidenceError(
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


def fasta_path(workspace: Path, strategy: str, partition: str) -> Path:
    """Return the deterministic path for one materialized FASTA."""

    return workspace / "fastas" / f"{strategy}_{partition}.fasta"


def reverify_frozen_run_state(
    *,
    paths: Mapping[str, Path],
    policy: SimilarityAuditPolicy,
    code_revision: str,
    mmseqs_version: str,
    config_path: Path,
    project_root: Path,
) -> None:
    """Repeat every mutable frozen-source trust check at a workflow gate."""

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
            raise SourceEvidenceError(
                f"frozen {name} checksum changed during the audit"
            )
    try:
        source_policy_identity = file_identity(config_path)
    except OSError:
        load_similarity_audit_policy(config_path)
        raise
    if source_policy_identity["sha256"] != APPROVED_SIMILARITY_AUDIT_CONFIG_SHA256:
        raise SourceEvidenceError("A-003 source policy changed during the audit")
    load_similarity_audit_policy(config_path)
    require_committed_execution_code(project_root)
    if git_output(project_root, "rev-parse", "HEAD") != code_revision:
        raise SourceEvidenceError("code revision changed during the audit")
    if verify_mmseqs(policy, project_root) != mmseqs_version:
        raise SourceEvidenceError("MMseqs2 version changed during the audit")


# Read-only A-003 import


def verify_a003_residual_import(
    *,
    project_root: Path,
    policy: A004Policy,
) -> A003Import:
    """Verify, but never copy or modify, the reusable A-003 residual stages."""

    paths = resolve_a004_paths(policy, project_root)
    source_policy = _load_source_policy(paths["source_policy"], policy)
    source_workspace = paths["source_workspace"]
    expected_workspace = (
        project_root / source_policy.workspace_relative_path
    ).resolve()
    if source_workspace != expected_workspace:
        raise SourceEvidenceError("A-004 source workspace differs from A-003")

    fingerprint = run_fingerprint(
        policy=source_policy,
        code_revision=policy.source_code_revision,
        mmseqs_version=policy.source_mmseqs_version,
    )
    if fingerprint != policy.source_run_fingerprint:
        raise SourceEvidenceError("A-003 run fingerprint cannot be reconstructed")

    fastas_marker, _ = _read_pinned_marker(
        source_workspace / "fastas" / "complete.json",
        policy.source_fastas_marker_sha256,
    )
    require_marker_identity(fastas_marker, fingerprint, "materialized_inputs")
    fastas = _verify_materialized_fastas(fastas_marker, source_workspace)
    validation_path = (
        source_workspace
        / "fastas"
        / f"{policy.import_strategy.value}_{policy.import_partition.value}.fasta"
    )
    training = fastas[policy.import_strategy.value]["training"]
    validation = fastas[policy.import_strategy.value][policy.import_partition.value]

    database = _verify_database(
        source_workspace=source_workspace,
        source_policy=source_policy,
        policy=policy,
        fingerprint=fingerprint,
        training=training,
    )
    stages = tuple(
        _verify_stage(
            cap=cap,
            source_workspace=source_workspace,
            source_policy=source_policy,
            policy=policy,
            fingerprint=fingerprint,
            expected_query=(
                validation if cap != policy.staged_escalation_cap.value else None
            ),
        )
        for cap in FIXED_CAPS
    )
    escalated = _verify_escalation_membership(
        validation_path=validation_path,
        initial=stages[0],
        comparison=stages[1],
        escalation=stages[2],
        source_workspace=source_workspace,
        policy=policy,
    )
    return A003Import(
        fingerprint=fingerprint,
        fastas=fastas,
        database=database,
        stages=stages,
        escalated_query_ids=escalated,
    )


def _verify_database(
    *,
    source_workspace: Path,
    source_policy: SimilarityAuditPolicy,
    policy: A004Policy,
    fingerprint: str,
    training: FastaEvidence,
) -> DatabaseImport:
    """Verify the database marker, input, command, and current artifacts."""

    database_directory = source_workspace / "databases" / policy.import_strategy.value
    marker, marker_evidence = _read_pinned_marker(
        database_directory / "complete.json",
        policy.source_database_marker_sha256,
    )
    require_marker_identity(marker, fingerprint, "target_database")
    if marker.get("strategy") != policy.import_strategy.value:
        raise SourceEvidenceError("A-003 target-database strategy drifted")
    if marker.get("training_fasta") != asdict(training):
        raise SourceEvidenceError("A-003 target-database input drifted")

    command = _marker_command(marker)
    if len(command) < 4 or command != createdb_command(
        source_policy,
        training_fasta=Path(command[2]),
        database_prefix=Path(command[3]),
    ):
        raise SourceEvidenceError("A-003 target-database command drifted")
    _require_command_path(
        command[2],
        policy.source_workspace_relative_path,
        f"fastas/{policy.import_strategy.value}_training.fasta",
    )
    _require_command_path(
        command[3],
        policy.source_workspace_relative_path,
        f"databases/.{policy.import_strategy.value}.incomplete/target",
    )
    _runtime_seconds(marker)
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SourceEvidenceError("A-003 database artifact index is malformed")
    verify_database_artifacts(database_directory, artifacts)
    return DatabaseImport(marker=marker_evidence, artifact_count=len(artifacts))


def _verify_stage(
    *,
    cap: int,
    source_workspace: Path,
    source_policy: SimilarityAuditPolicy,
    policy: A004Policy,
    fingerprint: str,
    expected_query: FastaEvidence | None,
) -> ImportedStage:
    """Verify one canonical residual stage without modifying it."""

    relative_stage = (
        Path("tracks")
        / policy.import_strategy.value
        / policy.import_partition.value
        / policy.import_pass.value
        / f"cap_{cap}"
    )
    stage_directory = source_workspace / relative_stage
    marker, marker_evidence = _read_pinned_marker(
        stage_directory / "complete.json",
        policy.stage_marker_sha256(cap),
    )
    require_marker_identity(marker, fingerprint, "search_stage")
    if marker.get("cap") != cap:
        raise SourceEvidenceError(f"A-003 cap {cap} marker identity drifted")

    query = fasta_evidence_from(marker.get("query_fasta"))
    if marker.get("query_count") != query.record_count:
        raise SourceEvidenceError(f"A-003 cap {cap} query count drifted")
    if expected_query is not None and query != expected_query:
        raise SourceEvidenceError(f"A-003 cap {cap} query input drifted")
    query_path = _stage_query_path(cap, source_workspace, policy)
    verify_file(query_path, query.byte_size, query.sha256)

    command = _marker_command(marker)
    _verify_search_command(
        command=command,
        cap=cap,
        relative_stage=relative_stage,
        source_policy=source_policy,
        policy=policy,
    )
    evidence = canonical_evidence_from(marker.get("alignment_evidence"))
    canonical_path = stage_directory / "canonical.tsv"
    if (
        marker.get("raw_retained") is not False
        or (stage_directory / "raw.tsv").exists()
    ):
        raise SourceEvidenceError(f"A-003 cap {cap} raw-retention state drifted")
    verify_file(
        canonical_path,
        evidence.canonical.byte_size,
        evidence.canonical.sha256,
    )
    return ImportedStage(
        cap=cap,
        marker=marker_evidence,
        query_fasta=query,
        canonical=evidence.canonical,
        canonical_path=canonical_path,
        command=command,
        runtime_seconds=_runtime_seconds(marker),
    )


def _read_pinned_marker(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, object], MarkerEvidence]:
    """Read one marker only after its bytes match the A-004 pin."""

    if not path.is_file():
        raise SourceEvidenceError(f"A-003 completion marker is missing: {path}")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise SourceEvidenceError(
            f"could not read A-003 completion marker: {path}"
        ) from error
    calculated_sha256 = hashlib.sha256(content).hexdigest()
    if calculated_sha256 != expected_sha256:
        raise SourceEvidenceError(f"A-003 completion marker checksum drifted: {path}")
    try:
        marker = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceEvidenceError(
            f"A-003 completion marker is malformed: {path}"
        ) from error
    if not isinstance(marker, dict):
        raise SourceEvidenceError(f"A-003 completion marker root is malformed: {path}")
    evidence = MarkerEvidence(
        byte_size=len(content),
        sha256=calculated_sha256,
    )
    return marker, evidence


def _verify_search_command(
    *,
    command: tuple[str, ...],
    cap: int,
    relative_stage: Path,
    source_policy: SimilarityAuditPolicy,
    policy: A004Policy,
) -> None:
    """Match all options and the intended historical path suffixes."""

    if len(command) < 6 or command != search_command(
        source_policy,
        pass_name=policy.import_pass.value,
        cap=cap,
        query_fasta=Path(command[2]),
        target_database=Path(command[3]),
        raw_output=Path(command[4]),
        temp_directory=Path(command[5]),
    ):
        raise SourceEvidenceError(f"A-003 cap {cap} search command drifted")

    strategy = policy.import_strategy.value
    partition = policy.import_partition.value
    pass_name = policy.import_pass.value
    query_suffix = (
        f"tracks/{strategy}/{partition}/{pass_name}/escalated_queries.fasta"
        if cap == policy.staged_escalation_cap.value
        else f"fastas/{strategy}_{partition}.fasta"
    )
    _require_command_path(
        command[2], policy.source_workspace_relative_path, query_suffix
    )
    _require_command_path(
        command[3],
        policy.source_workspace_relative_path,
        f"databases/{strategy}/target",
    )
    stage = relative_stage.as_posix()
    _require_command_path(
        command[4], policy.source_workspace_relative_path, f"{stage}/raw.tsv"
    )
    _require_command_path(
        command[5], policy.source_workspace_relative_path, f"{stage}/mmseqs_tmp"
    )


def _stage_query_path(cap: int, source_workspace: Path, policy: A004Policy) -> Path:
    """Return the full-query or escalation FASTA for one cap."""

    if cap == policy.staged_escalation_cap.value:
        return (
            source_workspace
            / "tracks"
            / policy.import_strategy.value
            / policy.import_partition.value
            / policy.import_pass.value
            / "escalated_queries.fasta"
        )
    return (
        source_workspace
        / "fastas"
        / f"{policy.import_strategy.value}_{policy.import_partition.value}.fasta"
    )


def _marker_command(marker: dict[str, object]) -> tuple[str, ...]:
    """Return a marker command only when every argument is a string."""

    command = marker.get("command")
    if not isinstance(command, list) or any(
        not isinstance(value, str) for value in command
    ):
        raise SourceEvidenceError("A-003 MMseqs2 command is malformed")
    return tuple(command)


def _require_command_path(value: str, workspace: str, suffix: str) -> None:
    """Require an absolute command path ending in its pinned workspace path."""

    path = PurePosixPath(value)
    expected = PurePosixPath(workspace) / suffix
    if not path.is_absolute() or path.parts[-len(expected.parts) :] != expected.parts:
        raise SourceEvidenceError("A-003 MMseqs2 command path drifted")


def _runtime_seconds(marker: dict[str, object]) -> str:
    """Return a finite, nonnegative marker runtime without normalizing its text."""

    value = marker.get("runtime_seconds")
    if not isinstance(value, str):
        raise SourceEvidenceError("A-003 runtime is malformed")
    try:
        runtime = Decimal(value)
    except InvalidOperation as error:
        raise SourceEvidenceError("A-003 runtime is malformed") from error
    if not runtime.is_finite() or runtime < 0:
        raise SourceEvidenceError("A-003 runtime is malformed")
    return value


# Private source-evidence helpers


def _load_pinned_json(path: Path, expected_sha256: str) -> dict[str, object]:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise SourceEvidenceError(f"pinned report checksum drifted: {path.name}")
    try:
        parsed = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceEvidenceError(f"pinned report is malformed: {path.name}") from error
    if not isinstance(parsed, dict):
        raise SourceEvidenceError(f"pinned report root is not an object: {path.name}")
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
        raise SourceEvidenceError(f"{name} authority drift: " + "; ".join(drift))


def _load_source_policy(path: Path, policy: A004Policy) -> SimilarityAuditPolicy:
    if not path.is_file():
        raise SourceEvidenceError(f"A-003 source policy is missing: {path}")
    identity = file_identity(path)
    if identity["sha256"] != policy.source_policy_sha256:
        raise SourceEvidenceError("A-003 source policy checksum drifted")
    source = load_similarity_audit_policy(path)
    if (
        source.adjustment_id != policy.source_adjustment_id
        or source.mmseqs_version != policy.source_mmseqs_version
    ):
        raise SourceEvidenceError("A-003 source authority drifted")
    return source


def _verify_escalation_membership(
    *,
    validation_path: Path,
    initial: ImportedStage,
    comparison: ImportedStage,
    escalation: ImportedStage,
    source_workspace: Path,
    policy: A004Policy,
) -> tuple[str, ...]:
    validation_ids = tuple(
        accession for accession, _ in iter_one_line_fasta(validation_path)
    )
    if len(validation_ids) != len(set(validation_ids)):
        raise SourceEvidenceError(
            "A-003 validation FASTA contains duplicate accessions"
        )
    changed = compare_canonical_results(
        initial.canonical_path,
        comparison.canonical_path,
        expected_query_ids=validation_ids,
    )
    escalation_path = (
        source_workspace
        / "tracks"
        / policy.import_strategy.value
        / policy.import_partition.value
        / policy.import_pass.value
        / "escalated_queries.fasta"
    )
    escalated_ids = tuple(
        accession for accession, _ in iter_one_line_fasta(escalation_path)
    )
    if len(escalated_ids) != len(set(escalated_ids)):
        raise SourceEvidenceError(
            "A-003 escalation FASTA contains duplicate accessions"
        )
    if frozenset(escalated_ids) != frozenset(changed):
        raise SourceEvidenceError(
            "A-003 escalation membership differs from cap changes"
        )
    if escalation.query_fasta.record_count != len(escalated_ids):
        raise SourceEvidenceError("A-003 escalation query count drifted")
    return tuple(sorted(escalated_ids))


def _verify_materialized_fastas(
    marker: Mapping[str, object],
    source_workspace: Path,
) -> dict[str, dict[str, FastaEvidence]]:
    """Verify all six A-003 FASTAs before exposing any one to A-004."""

    fastas = marker.get("fastas")
    if not isinstance(fastas, dict) or set(fastas) != set(STRATEGIES):
        raise SourceEvidenceError("A-003 materialized FASTA marker is malformed")
    verified: dict[str, dict[str, FastaEvidence]] = {}
    for strategy in STRATEGIES:
        partitions = fastas.get(strategy)
        if not isinstance(partitions, dict) or set(partitions) != set(PARTITIONS):
            raise SourceEvidenceError("A-003 materialized FASTA marker is malformed")
        verified[strategy] = {}
        for partition in PARTITIONS:
            evidence = fasta_evidence_from(partitions[partition])
            path = source_workspace / "fastas" / f"{strategy}_{partition}.fasta"
            verify_file(path, evidence.byte_size, evidence.sha256)
            verified[strategy][partition] = evidence
    return verified
