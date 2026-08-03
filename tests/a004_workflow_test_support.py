"""Synthetic, MMseqs-free fixture for the top-level A-004 workflow tests."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import protein_lm.data.task7_a004_database as database_module
import protein_lm.data.task7_a004_workflow as workflow_module
import protein_lm.data.task7_fixed_budget_stages as stage_module
from protein_lm.data.similarity_audit_models import FileEvidence
from protein_lm.data.similarity_audit_policy import load_similarity_audit_policy
from protein_lm.data.similarity_fastas import FastaEvidence, MaterializedInputs
from protein_lm.data.similarity_manifests import (
    PartitionPopulation,
    StrategyManifest,
    StructuralMembershipAudit,
)
from protein_lm.data.similarity_results import canonicalize_mmseqs_tsv
from protein_lm.data.task7_a003_import import A003Import
from protein_lm.data.task7_a003_stages import (
    DatabaseImport,
    ImportedStage,
    MarkerEvidence,
)
from protein_lm.data.task7_a004_policy import load_a004_policy
from protein_lm.data.task7_a004_workflow import A004Configuration
from task7_test_support import alignment_tsv_row, metadata

REPOSITORY = Path(__file__).parents[1]
A004_CONFIG = REPOSITORY / "experiments/week_01/read_only_similarity_audit_a004.toml"
SOURCE_CONFIG = REPOSITORY / "experiments/week_01/diagnostic_similarity_audit.toml"


@dataclass
class SyntheticWorkflow:
    """Inputs and runner observations for one isolated A-004 invocation."""

    project_root: Path
    config_path: Path
    workspace: Path
    hardware: dict[str, object]
    database_calls: list[tuple[str, ...]]
    search_calls: list[tuple[str, ...]]
    database_runner: object
    search_runner: object


def install_synthetic_workflow(monkeypatch, tmp_path: Path) -> SyntheticWorkflow:
    """Patch only external/preflight seams while retaining orchestration and evidence."""

    project_root = tmp_path / "repo"
    workspace = project_root / "a004"
    source_workspace = project_root / "a003"
    project_root.mkdir()
    policy = load_a004_policy(A004_CONFIG)
    source_policy = load_similarity_audit_policy(SOURCE_CONFIG)
    manifests = _manifests()
    inputs = _materialized_inputs(workspace, manifests)
    imported = _imported_a003(source_workspace, manifests, inputs)
    source_paths = _frozen_source_paths(project_root)
    balances = {
        strategy: {
            partition: {"records": 1, "residues": 4, "unique_groups": 1}
            for partition in ("training", "validation", "test")
        }
        for strategy in ("random", "group_aware")
    }
    configuration = A004Configuration(
        policy=policy,
        source_policy=source_policy,
        paths={
            "workspace": workspace,
            "source_workspace": source_workspace,
            "source_policy": SOURCE_CONFIG,
        },
    )
    monkeypatch.setattr(
        workflow_module,
        "validate_a004_configuration",
        lambda **kwargs: configuration,
    )
    monkeypatch.setattr(workflow_module, "require_committed_execution_code", lambda *args: None)
    monkeypatch.setattr(workflow_module, "git_output", lambda *args: "b" * 40)
    monkeypatch.setattr(
        workflow_module,
        "verify_mmseqs",
        lambda *args: source_policy.mmseqs_version,
    )
    monkeypatch.setattr(workflow_module, "prove_path_is_ignored", lambda *args: None)
    monkeypatch.setattr(workflow_module, "require_disk_capacity", lambda *args: None)
    monkeypatch.setattr(stage_module, "require_disk_capacity", lambda *args: None)
    monkeypatch.setattr(workflow_module, "verify_boundary_fixtures", lambda: None)
    monkeypatch.setattr(workflow_module, "policy_paths", lambda *args: source_paths)
    monkeypatch.setattr(
        workflow_module,
        "load_and_validate_frozen_reports",
        lambda *args: {"random": {}, "group_aware": {}},
    )
    monkeypatch.setattr(workflow_module, "load_frozen_manifests", lambda *args: manifests)
    monkeypatch.setattr(
        workflow_module,
        "validate_report_populations",
        lambda *args: balances,
    )
    monkeypatch.setattr(
        workflow_module,
        "verify_a003_residual_import",
        lambda **kwargs: imported,
    )
    monkeypatch.setattr(
        workflow_module,
        "ensure_materialized_inputs",
        lambda **kwargs: inputs,
    )
    monkeypatch.setattr(workflow_module, "reverify_frozen_run_state", lambda **kwargs: None)
    monkeypatch.setattr(workflow_module, "load_a004_policy", lambda path: policy)

    def forbidden_runner(*args, **kwargs):
        raise AssertionError("a real MMseqs runner was invoked by a synthetic test")

    monkeypatch.setattr(database_module, "run_mmseqs_command", forbidden_runner)
    monkeypatch.setattr(stage_module, "run_mmseqs_command", forbidden_runner)

    database_calls: list[tuple[str, ...]] = []
    search_calls: list[tuple[str, ...]] = []

    def database_runner(command, *args):
        database_calls.append(tuple(command))
        Path(command[3]).write_bytes(b"synthetic database")
        return "0.01"

    def search_runner(command, *args):
        search_calls.append(tuple(command))
        query_ids = [
            line[1:]
            for line in Path(command[2]).read_text().splitlines()
            if line.startswith(">")
        ]
        rows = [
            _row(query, "R_TRAIN" if query.startswith("R_") else "G_TRAIN")
            for query in query_ids
        ]
        Path(command[4]).write_text("\n".join(rows) + "\n")
        return "0.02"

    return SyntheticWorkflow(
        project_root=project_root,
        config_path=A004_CONFIG,
        workspace=workspace,
        hardware={
            "platform": "synthetic-platform",
            "machine": "synthetic-machine",
            "processor": "synthetic-processor",
            "logical_cpu_count": 8,
        },
        database_calls=database_calls,
        search_calls=search_calls,
        database_runner=database_runner,
        search_runner=search_runner,
    )


def _manifests() -> dict[str, StrategyManifest]:
    manifests = {}
    for strategy, prefix in (("random", "R"), ("group_aware", "G")):
        records = {
            f"{prefix}_TRAIN": metadata(
                f"{prefix}-training", length=4, partition="training"
            ),
            f"{prefix}_VALID": metadata(
                f"{prefix}-validation", length=4, partition="validation"
            ),
            f"{prefix}_TEST": metadata(f"{prefix}-test", length=4, partition="test"),
        }
        manifests[strategy] = StrategyManifest(
            strategy=strategy,
            stage="synthetic",
            records=records,
            partitions={
                partition: PartitionPopulation(1, 4, 1)
                for partition in ("training", "validation", "test")
            },
            structural_audit=StructuralMembershipAudit(0, 0, 3, 12, 0, 0, 1, 4),
            public_manifest=FileEvidence(3, 1, "1" * 64),
            local_assignment=FileEvidence(3, 1, "2" * 64),
        )
    return manifests


def _materialized_inputs(
    workspace: Path, manifests: dict[str, StrategyManifest]
) -> MaterializedInputs:
    fastas = {}
    for strategy, manifest in manifests.items():
        fastas[strategy] = {}
        for accession, item in manifest.records.items():
            path = workspace / "fastas" / f"{strategy}_{item.partition}.fasta"
            fastas[strategy][item.partition] = _write_fasta(path, accession)
    return MaterializedInputs(
        catalog=FileEvidence(6, 1, "3" * 64),
        fastas=fastas,
    )


def _imported_a003(
    source_workspace: Path,
    manifests: dict[str, StrategyManifest],
    inputs: MaterializedInputs,
) -> A003Import:
    validation_fasta = source_workspace / "fastas/random_validation.fasta"
    validation = _write_fasta(validation_fasta, "R_VALID")
    pass_directory = source_workspace / "tracks/random/validation/residual"
    escalation = _write_fasta(pass_directory / "escalated_queries.fasta", "R_VALID")
    queries = {"R_VALID": manifests["random"].records["R_VALID"]}
    targets = {"R_TRAIN": manifests["random"].records["R_TRAIN"]}
    stages = []
    for cap, fident in ((1_000, "0.10"), (10_000, "0.60"), (100_000, "0.60")):
        stage_directory = pass_directory / f"cap_{cap}"
        stage_directory.mkdir(parents=True)
        raw_path = stage_directory / "raw.tsv"
        canonical_path = stage_directory / "canonical.tsv"
        raw_path.write_text(_row("R_VALID", "R_TRAIN", fident=fident) + "\n")
        evidence = canonicalize_mmseqs_tsv(
            raw_path,
            canonical_path,
            query_metadata=queries,
            target_metadata=targets,
            chunk_rows=2,
            delete_raw_after_parse=True,
        ).canonical
        marker_path = stage_directory / "complete.json"
        _write_json(
            marker_path,
            {
                "schema_version": 1,
                "stage": "search_stage",
                "fingerprint": "synthetic-a003-fingerprint",
                "cap": cap,
            },
        )
        marker_bytes = marker_path.read_bytes()
        stages.append(
            ImportedStage(
                cap=cap,
                marker=MarkerEvidence(
                    len(marker_bytes), hashlib.sha256(marker_bytes).hexdigest()
                ),
                query_fasta=escalation if cap == 100_000 else validation,
                canonical=evidence,
                canonical_path=canonical_path,
                command=("mmseqs", "easy-search", str(cap)),
                runtime_seconds="0.03",
            )
        )
    return A003Import(
        fingerprint="synthetic-a003-fingerprint",
        fastas=inputs.fastas,
        database=DatabaseImport(MarkerEvidence(1, "4" * 64), artifact_count=1),
        stages=tuple(stages),
        escalated_query_ids=("R_VALID",),
    )


def _frozen_source_paths(project_root: Path) -> dict[str, Path]:
    paths = {"catalog": project_root / "frozen/catalog.tsv"}
    for name in (
        "task5_public",
        "task5_local",
        "task5_report",
        "task6_public",
        "task6_local",
        "task6_report",
    ):
        paths[name] = project_root / "frozen" / f"{name}.json"
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n")
    return paths


def _write_fasta(path: Path, accession: str) -> FastaEvidence:
    content = f">{accession}\nAAAA\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    encoded = content.encode()
    return FastaEvidence(1, 4, len(encoded), hashlib.sha256(encoded).hexdigest())


def _row(query: str, target: str, *, fident: str = "0.60") -> str:
    return alignment_tsv_row(
        query,
        target,
        fident=fident,
        qcov="1.0",
        tcov="1.0",
        alnlen=4,
        qlen=4,
        tlen=4,
        qend=4,
        tend=4,
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
