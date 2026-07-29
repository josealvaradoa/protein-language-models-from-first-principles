"""Run the verified Week 1 Task 4 eligible-record preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

from protein_lm.data.acquisition import (
    load_acquisition_contract,
    prove_heavy_paths_are_ignored,
    validate_release_metadata,
    verify_local_file,
)
from protein_lm.data.eligibility import build_task4_catalog
from protein_lm.data.eligibility_policy import (
    Task4PreparationError,
    load_eligibility_policy,
)
from protein_lm.data.proteingym import (
    PROTEINGYM_V1_3_PIN,
    verify_proteingym_source,
)
from protein_lm.data.task2_audit import SourceEvidence
from protein_lm.data.task4_report import (
    render_task4_report,
    validate_task2_anchors,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACQUISITION_CONFIG = PROJECT_ROOT / "experiments" / "week_01" / "acquisition.toml"
ELIGIBILITY_CONFIG = PROJECT_ROOT / "experiments" / "week_01" / "eligibility.toml"
RELEASE_METADATA = (
    PROJECT_ROOT / "data" / "raw" / "uniprot" / "2026_02" / "reldate.txt"
)
PROTEINGYM_METADATA = (
    PROJECT_ROOT / "data" / "raw" / "proteingym" / "v1.3" / "DMS_substitutions.csv"
)
TASK2_REPORT = PROJECT_ROOT / "reports" / "week_01" / "task_02_corpus_audit.json"
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "week_01"
CATALOG_RELATIVE_PATH = Path("data/processed/week_01/task_04_record_catalog.tsv")
RESERVED_FAMILIES_RELATIVE_PATH = Path(
    "data/processed/week_01/task_04_candidate_test_reserved_families.txt"
)
CATALOG_PATH = PROJECT_ROOT / CATALOG_RELATIVE_PATH
RESERVED_FAMILIES_PATH = PROJECT_ROOT / RESERVED_FAMILIES_RELATIVE_PATH
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "week_01"
OUTPUT_STEM = "task_04_eligible_records"
LOCAL_RETRIEVAL_DATE = "2026-07-28"
LOCAL_RETRIEVAL_METHOD = "manual_user_download"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify local sources and build the ignored Task 4 record catalog."
        )
    )
    parser.add_argument(
        "--repeat-check",
        action="store_true",
        required=True,
        help="required: run preparation twice and require byte-identical evidence",
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        _require_committed_execution_code()
        policy_bytes = ELIGIBILITY_CONFIG.read_bytes()
        policy_sha256 = hashlib.sha256(policy_bytes).hexdigest()
        policy = load_eligibility_policy(ELIGIBILITY_CONFIG)
        task2_data, task2_sha256 = _load_approved_task2_report(
            policy.approved_task2_report_sha256
        )

        contract = load_acquisition_contract(ACQUISITION_CONFIG)
        if contract.release_id != policy.source_release:
            raise Task4PreparationError(
                "acquisition and eligibility releases do not match"
            )
        if PROTEINGYM_V1_3_PIN.release != policy.proteingym_release:
            raise Task4PreparationError(
                "ProteinGym pin and eligibility policy do not match"
            )
        prove_heavy_paths_are_ignored(contract, PROJECT_ROOT)
        _prove_path_is_ignored(CATALOG_PATH)
        _prove_path_is_ignored(RESERVED_FAMILIES_PATH)
        validate_release_metadata(
            RELEASE_METADATA.read_text(encoding="utf-8"),
            contract,
        )
        _prove_path_is_ignored(PROTEINGYM_METADATA)

        source_by_role = {source.role: source for source in contract.sources}
        local_verifications = {}
        for role in ("swiss_prot_records", "uniref50_membership"):
            source = source_by_role[role]
            print(f"verifying {source.filename}...")
            local_path = PROJECT_ROOT / contract.local_path_for(source)
            local_verifications[role] = verify_local_file(local_path, source)

        print(f"verifying {PROTEINGYM_V1_3_PIN.filename}...")
        proteingym_verification = verify_proteingym_source(PROTEINGYM_METADATA)
        sources = _source_evidence(
            contract=contract,
            source_by_role=source_by_role,
            local_verifications=local_verifications,
            proteingym_verification=proteingym_verification,
        )
        code_revision = _git_output("rev-parse", "HEAD")

        PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
        REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".task4-", dir=PROCESSED_DIRECTORY
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            print("running eligible-record preparation pass 1...")
            started = time.perf_counter()
            first_report = _build_pass(
                temporary_root / "pass1_catalog.tsv",
                temporary_root / "pass1_reserved_families.txt",
                policy=policy,
                policy_sha256=policy_sha256,
                task2_sha256=task2_sha256,
                sources=sources,
                code_revision=code_revision,
                swiss_prot_path=local_verifications["swiss_prot_records"].path,
                uniref50_path=local_verifications["uniref50_membership"].path,
            )
            validate_task2_anchors(first_report, task2_data)
            first_rendered = render_task4_report(first_report)
            print(f"pass 1 completed in {time.perf_counter() - started:.1f} seconds")

            print("running eligible-record preparation pass 2...")
            started = time.perf_counter()
            second_report = _build_pass(
                temporary_root / "pass2_catalog.tsv",
                temporary_root / "pass2_reserved_families.txt",
                policy=policy,
                policy_sha256=policy_sha256,
                task2_sha256=task2_sha256,
                sources=sources,
                code_revision=code_revision,
                swiss_prot_path=local_verifications["swiss_prot_records"].path,
                uniref50_path=local_verifications["uniref50_membership"].path,
            )
            validate_task2_anchors(second_report, task2_data)
            second_rendered = render_task4_report(second_report)
            if first_report != second_report or first_rendered != second_rendered:
                raise Task4PreparationError(
                    "repeated Task 4 runs produced different evidence"
                )
            print(f"pass 2 completed in {time.perf_counter() - started:.1f} seconds")

            staged_outputs = _stage_outputs(temporary_root, first_rendered)
            (temporary_root / "pass1_catalog.tsv").replace(CATALOG_PATH)
            (temporary_root / "pass1_reserved_families.txt").replace(
                RESERVED_FAMILIES_PATH
            )
            _promote_outputs(staged_outputs, REPORT_DIRECTORY)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"eligible-record preparation failed: {error}")
        return 1

    print(f"eligible records: {first_report.population.eligible.records}")
    print(f"catalog SHA-256: {first_report.catalog.sha256}")
    print(
        "reserved-family SHA-256: "
        f"{first_report.reserved_families.sha256}"
    )
    print(f"report JSON SHA-256: {first_rendered.json_sha256}")
    print(f"catalog: {CATALOG_PATH}")
    print(f"outputs: {REPORT_DIRECTORY}")
    print("network requests made: none")
    return 0


def _build_pass(
    catalog_path: Path,
    reserved_families_path: Path,
    *,
    policy,
    policy_sha256,
    task2_sha256,
    sources,
    code_revision,
    swiss_prot_path,
    uniref50_path,
):
    return build_task4_catalog(
        swiss_prot_path=swiss_prot_path,
        uniref50_path=uniref50_path,
        proteingym_path=PROTEINGYM_METADATA,
        catalog_output_path=catalog_path,
        reserved_families_output_path=reserved_families_path,
        catalog_relative_path=CATALOG_RELATIVE_PATH.as_posix(),
        reserved_families_relative_path=(
            RESERVED_FAMILIES_RELATIVE_PATH.as_posix()
        ),
        policy=policy,
        policy_sha256=policy_sha256,
        task2_report_sha256=task2_sha256,
        sources=sources,
        code_revision=code_revision,
    )


def _load_approved_task2_report(
    expected_sha256: str,
) -> tuple[dict[str, object], str]:
    content = TASK2_REPORT.read_bytes()
    calculated = hashlib.sha256(content).hexdigest()
    if calculated != expected_sha256:
        raise Task4PreparationError(
            "Task 2 report bytes do not match the approved checksum"
        )
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise Task4PreparationError("Task 2 report root must be an object")
    return parsed, calculated


def _source_evidence(
    *,
    contract,
    source_by_role,
    local_verifications,
    proteingym_verification,
) -> dict[str, SourceEvidence]:
    evidence = {}
    for role in ("swiss_prot_records", "uniref50_membership"):
        source = source_by_role[role]
        verification = local_verifications[role]
        evidence[role] = SourceEvidence(
            release=contract.release_id,
            filename=source.filename,
            byte_size=verification.byte_size,
            sha256=verification.sha256,
            upstream_checksum_algorithm="md5",
            upstream_checksum=verification.md5,
            license_spdx=contract.license_spdx,
            retrieval_date=LOCAL_RETRIEVAL_DATE,
            retrieval_method=LOCAL_RETRIEVAL_METHOD,
        )
    evidence["proteingym_metadata"] = SourceEvidence(
        release=PROTEINGYM_V1_3_PIN.release,
        filename=PROTEINGYM_V1_3_PIN.filename,
        byte_size=proteingym_verification.byte_size,
        sha256=proteingym_verification.sha256,
        upstream_checksum_algorithm="git_blob_sha1",
        upstream_checksum=proteingym_verification.git_blob_sha1,
        license_spdx=PROTEINGYM_V1_3_PIN.license_spdx,
        retrieval_date=LOCAL_RETRIEVAL_DATE,
        retrieval_method=LOCAL_RETRIEVAL_METHOD,
    )
    return evidence


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
    )
    if status:
        raise Task4PreparationError(
            "execution code has uncommitted changes; review and commit it first"
        )


def _prove_path_is_ignored(path: Path) -> None:
    try:
        relative_path = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise Task4PreparationError(
            f"derived path must stay inside the repository: {path}"
        ) from error
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative_path.as_posix()],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise Task4PreparationError(
            f"derived path is not ignored by Git: {relative_path}"
        )


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _stage_outputs(
    staging_directory: Path,
    rendered,
) -> dict[str, Path]:
    outputs = {
        f"{OUTPUT_STEM}.json": rendered.json_text.encode("utf-8"),
        f"{OUTPUT_STEM}.md": rendered.markdown_text.encode("utf-8"),
        f"{OUTPUT_STEM}.sha256": (
            f"{rendered.json_sha256}  {OUTPUT_STEM}.json\n".encode("ascii")
        ),
    }
    staged = {}
    for filename, content in outputs.items():
        path = staging_directory / filename
        written = path.write_bytes(content)
        if written != len(content):
            raise Task4PreparationError(
                f"staged report byte count changed for {filename}"
            )
        staged[filename] = path

    staged_json = staged[f"{OUTPUT_STEM}.json"].read_bytes()
    if hashlib.sha256(staged_json).hexdigest() != rendered.json_sha256:
        raise Task4PreparationError("staged report checksum does not match")
    return staged


def _promote_outputs(
    staged_outputs: dict[str, Path],
    output_directory: Path,
) -> None:
    for filename, staged_path in staged_outputs.items():
        staged_path.replace(output_directory / filename)


if __name__ == "__main__":
    raise SystemExit(main())
