"""Build the verified Week 1 Task 5 random diagnostic split."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from protein_lm.data.random_split import (
    DiagnosticSplitUseError,
    build_random_diagnostic,
    require_selected_training_manifest,
    sha256_sidecar,
    validate_task4_report,
)
from protein_lm.data.random_split_policy import (
    RandomSplitError,
    load_random_split_policy,
)
from protein_lm.data.task5_report import (
    CompletedPublicArtifact,
    DerivedArtifact,
    Task5Report,
    render_completion_index,
    render_task5_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SPLIT_CONFIG = PROJECT_ROOT / "experiments" / "week_01" / "random_split.toml"
TASK4_REPORT = PROJECT_ROOT / "reports" / "week_01" / "task_04_eligible_records.json"
TASK4_CATALOG = (
    PROJECT_ROOT / "data" / "processed" / "week_01" / "task_04_record_catalog.tsv"
)
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "week_01"
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "week_01"
OUTPUT_STEM = "task_05_random_diagnostic"
REPORT_FILENAMES = (
    f"{OUTPUT_STEM}.json",
    f"{OUTPUT_STEM}.md",
    f"{OUTPUT_STEM}.sha256",
)
COMPLETION_FILENAME = f"{OUTPUT_STEM}.complete.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Verify Task 4 and build the Task 5 random diagnostic manifests.")
    )
    parser.add_argument(
        "--repeat-check",
        action="store_true",
        required=True,
        help="required: build twice and require identical artifact evidence",
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        _require_committed_execution_code()
        config_bytes = RANDOM_SPLIT_CONFIG.read_bytes()
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        policy = load_random_split_policy(RANDOM_SPLIT_CONFIG)
        task4_report, task4_report_sha256 = _load_task4_report(
            policy.task4_report_sha256
        )
        sources = validate_task4_report(task4_report, policy)

        local_assignment_path = _repository_path(policy.local_assignment_relative_path)
        public_manifest_path = _repository_path(policy.public_manifest_relative_path)
        public_manifest_sidecar = public_manifest_path.with_name(
            f"{public_manifest_path.name}.sha256"
        )
        _prove_path_is_ignored(TASK4_CATALOG)
        _prove_path_is_ignored(local_assignment_path)
        for public_path in (
            public_manifest_path,
            public_manifest_sidecar,
            *(REPORT_DIRECTORY / filename for filename in REPORT_FILENAMES),
            REPORT_DIRECTORY / COMPLETION_FILENAME,
        ):
            _prove_path_is_public(public_path)
        code_revision = _git_output("rev-parse", "HEAD")

        PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
        public_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".task5-",
            dir=PROCESSED_DIRECTORY,
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            print("running random diagnostic pass 1...")
            started = time.perf_counter()
            first_build = build_random_diagnostic(
                catalog_path=TASK4_CATALOG,
                local_assignment_output_path=temporary_root / "pass1_assignments.tsv",
                public_manifest_output_path=temporary_root / "pass1_public.tsv",
                policy=policy,
                policy_sha256=config_sha256,
            )
            print(f"pass 1 completed in {time.perf_counter() - started:.1f} seconds")

            print("running random diagnostic pass 2...")
            started = time.perf_counter()
            second_build = build_random_diagnostic(
                catalog_path=TASK4_CATALOG,
                local_assignment_output_path=temporary_root / "pass2_assignments.tsv",
                public_manifest_output_path=temporary_root / "pass2_public.tsv",
                policy=policy,
                policy_sha256=config_sha256,
            )
            if first_build != second_build:
                raise RandomSplitError(
                    "repeated Task 5 runs produced different artifact evidence"
                )
            print(f"pass 2 completed in {time.perf_counter() - started:.1f} seconds")

            report = Task5Report(
                schema_version=policy.schema_version,
                scope=policy.scope,
                strategy=policy.strategy,
                stage=policy.stage,
                diagnostic_only=True,
                model_use=policy.model_use,
                selected_for_training=policy.selected_for_training,
                repeat_verified=True,
                verified_passes=2,
                seed=policy.seed,
                assignment_namespace=policy.assignment_namespace,
                hash_algorithm=policy.hash_algorithm,
                license_spdx=policy.license_spdx,
                code_revision=code_revision,
                config_sha256=config_sha256,
                task4_report_sha256=task4_report_sha256,
                task4_policy_sha256=policy.task4_policy_sha256,
                sources=sources,
                input_catalog=DerivedArtifact(
                    relative_path=("data/processed/week_01/task_04_record_catalog.tsv"),
                    row_count=policy.task4_catalog_row_count,
                    byte_size=policy.task4_catalog_byte_size,
                    sha256=policy.task4_catalog_sha256,
                ),
                population=first_build.population,
                partitions=first_build.partitions,
                local_assignments=first_build.local_assignments,
                public_manifest=first_build.public_manifest,
            )
            _prove_diagnostic_guard(report)
            rendered = render_task5_report(report)
            staged_outputs, completed_artifacts = _stage_public_outputs(
                temporary_root,
                rendered=rendered,
                manifest_relative_path=policy.public_manifest_relative_path,
                manifest_byte_size=first_build.public_manifest.byte_size,
                manifest_sha256=first_build.public_manifest.sha256,
            )

            (temporary_root / "pass1_assignments.tsv").replace(local_assignment_path)
            (temporary_root / "pass1_public.tsv").replace(public_manifest_path)
            staged_outputs[public_manifest_sidecar.name].replace(
                public_manifest_sidecar
            )
            for filename in REPORT_FILENAMES:
                staged_outputs[filename].replace(REPORT_DIRECTORY / filename)
            completion_path = REPORT_DIRECTORY / COMPLETION_FILENAME
            staged_outputs[COMPLETION_FILENAME].replace(completion_path)
            _verify_completed_public_outputs(
                completion_path,
                completed_artifacts,
            )
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"random diagnostic failed: {error}")
        return 1

    print(f"eligible records assigned: {report.population.records}")
    for partition in ("training", "validation", "test"):
        audit = report.partitions[partition]
        print(
            f"{partition}: {audit.records} records "
            f"({audit.record_share_percent}%), "
            f"{audit.residues} residues ({audit.residue_share_percent}%)"
        )
    print(f"public manifest SHA-256: {report.public_manifest.sha256}")
    print(f"report JSON SHA-256: {rendered.json_sha256}")
    print(f"public manifest: {public_manifest_path}")
    print(f"outputs: {REPORT_DIRECTORY}")
    print("network requests made: none")
    return 0


def _load_task4_report(
    expected_sha256: str,
) -> tuple[dict[str, object], str]:
    content = TASK4_REPORT.read_bytes()
    calculated = hashlib.sha256(content).hexdigest()
    if calculated != expected_sha256:
        raise RandomSplitError("Task 4 report bytes do not match the approved checksum")
    try:
        parsed = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RandomSplitError(f"Task 4 report is malformed: {error}") from error
    if not isinstance(parsed, dict):
        raise RandomSplitError("Task 4 report root must be an object")
    return parsed, calculated


def _prove_diagnostic_guard(report: Task5Report) -> None:
    try:
        require_selected_training_manifest(asdict(report))
    except DiagnosticSplitUseError:
        return
    raise RandomSplitError("random diagnostic passed the selected-training guard")


def _require_committed_execution_code() -> None:
    status = _git_output(
        "status",
        "--porcelain",
        "--",
        "src",
        "scripts",
        "experiments",
        "pyproject.toml",
        "uv.lock",
        ".gitignore",
        ".gitattributes",
    )
    if status:
        raise RandomSplitError(
            "execution code has uncommitted changes; review and commit it first"
        )


def _repository_path(relative_path: str) -> Path:
    candidate = PROJECT_ROOT / relative_path
    try:
        candidate.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise RandomSplitError(
            f"configured path leaves the repository: {relative_path}"
        ) from error
    return candidate


def _prove_path_is_ignored(path: Path) -> None:
    relative_path = path.resolve().relative_to(PROJECT_ROOT.resolve())
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative_path.as_posix()],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RandomSplitError(f"private path is not ignored by Git: {relative_path}")


def _prove_path_is_public(path: Path) -> None:
    relative_path = path.resolve().relative_to(PROJECT_ROOT.resolve())
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative_path.as_posix()],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode == 0:
        raise RandomSplitError(
            f"public path is unexpectedly ignored by Git: {relative_path}"
        )
    if result.returncode != 1:
        raise RandomSplitError(f"could not prove public Git status for {relative_path}")


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _stage_public_outputs(
    staging_directory: Path,
    *,
    rendered,
    manifest_relative_path: str,
    manifest_byte_size: int,
    manifest_sha256: str,
) -> tuple[dict[str, Path], tuple[CompletedPublicArtifact, ...]]:
    manifest_filename = Path(manifest_relative_path).name
    manifest_sidecar_filename = f"{manifest_filename}.sha256"
    outputs = {
        f"{OUTPUT_STEM}.json": rendered.json_text.encode("utf-8"),
        f"{OUTPUT_STEM}.md": rendered.markdown_text.encode("utf-8"),
        f"{OUTPUT_STEM}.sha256": (
            sha256_sidecar(
                f"{OUTPUT_STEM}.json",
                rendered.json_sha256,
            ).encode("ascii")
        ),
        manifest_sidecar_filename: (
            sha256_sidecar(
                manifest_filename,
                manifest_sha256,
            ).encode("ascii")
        ),
    }
    report_relative_directory = REPORT_DIRECTORY.relative_to(PROJECT_ROOT)
    manifest_sidecar_relative_path = Path(manifest_relative_path).with_name(
        manifest_sidecar_filename
    )
    completed_artifacts = [
        CompletedPublicArtifact(
            relative_path=manifest_relative_path,
            byte_size=manifest_byte_size,
            sha256=manifest_sha256,
        )
    ]
    for filename, content in outputs.items():
        if filename == manifest_sidecar_filename:
            relative_path = manifest_sidecar_relative_path.as_posix()
        else:
            relative_path = (report_relative_directory / filename).as_posix()
        completed_artifacts.append(
            CompletedPublicArtifact(
                relative_path=relative_path,
                byte_size=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    completion_text = render_completion_index(tuple(completed_artifacts))
    outputs[COMPLETION_FILENAME] = completion_text.encode("utf-8")

    staged = {}
    for filename, content in outputs.items():
        path = staging_directory / filename
        written = path.write_bytes(content)
        if written != len(content):
            raise RandomSplitError(f"staged output byte count changed for {filename}")
        staged[filename] = path

    report_bytes = staged[f"{OUTPUT_STEM}.json"].read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != rendered.json_sha256:
        raise RandomSplitError("staged report checksum does not match")
    return staged, tuple(completed_artifacts)


def _verify_completed_public_outputs(
    completion_path: Path,
    artifacts: tuple[CompletedPublicArtifact, ...],
) -> None:
    expected_index = render_completion_index(artifacts).encode("utf-8")
    if completion_path.read_bytes() != expected_index:
        raise RandomSplitError("public completion index changed after promotion")
    for artifact in artifacts:
        path = _repository_path(artifact.relative_path)
        if path.stat().st_size != artifact.byte_size:
            raise RandomSplitError(
                f"completed public byte size changed for {artifact.relative_path}"
            )
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != artifact.sha256:
            raise RandomSplitError(
                f"completed public checksum changed for {artifact.relative_path}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
