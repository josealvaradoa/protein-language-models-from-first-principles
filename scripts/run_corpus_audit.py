"""Run the verified Week 1 Task 2 aggregate corpus audit."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from protein_lm.data.acquisition import (
    load_acquisition_contract,
    prove_heavy_paths_are_ignored,
    validate_release_metadata,
    verify_local_file,
)
from protein_lm.data.proteingym import (
    PROTEINGYM_V1_3_PIN,
    verify_proteingym_source,
)
from protein_lm.data.task2_audit import (
    SourceEvidence,
    Task2AuditError,
    build_task2_audit,
    render_task2_audit,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "week_01" / "acquisition.toml"
DEFAULT_RELEASE_METADATA = (
    PROJECT_ROOT / "data" / "raw" / "uniprot" / "2026_02" / "reldate.txt"
)
DEFAULT_PROTEINGYM_METADATA = (
    PROJECT_ROOT / "data" / "raw" / "proteingym" / "v1.3" / "DMS_substitutions.csv"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "reports" / "week_01"
OUTPUT_STEM = "task_02_corpus_audit"
LOCAL_RETRIEVAL_DATE = "2026-07-28"
LOCAL_RETRIEVAL_METHOD = "manual_user_download"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify local sources and run the aggregate-only Task 2 audit."
    )
    parser.add_argument(
        "--release-metadata",
        type=Path,
        default=DEFAULT_RELEASE_METADATA,
    )
    parser.add_argument(
        "--proteingym-metadata",
        type=Path,
        default=DEFAULT_PROTEINGYM_METADATA,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--repeat-check",
        action="store_true",
        help="run the full audit twice and require byte-identical reports",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        _require_committed_execution_code()
        contract = load_acquisition_contract(DEFAULT_CONFIG)
        prove_heavy_paths_are_ignored(contract, PROJECT_ROOT)
        validate_release_metadata(
            args.release_metadata.read_text(encoding="utf-8"),
            contract,
        )
        _prove_path_is_ignored(args.proteingym_metadata)

        source_by_role = {source.role: source for source in contract.sources}
        local_verifications = {}
        for role in ("swiss_prot_records", "uniref50_membership"):
            source = source_by_role[role]
            print(f"verifying {source.filename}...")
            local_path = PROJECT_ROOT / contract.local_path_for(source)
            local_verifications[role] = verify_local_file(local_path, source)

        print(f"verifying {PROTEINGYM_V1_3_PIN.filename}...")
        proteingym_verification = verify_proteingym_source(args.proteingym_metadata)
        source_evidence = _source_evidence(
            contract=contract,
            source_by_role=source_by_role,
            local_verifications=local_verifications,
            proteingym_verification=proteingym_verification,
        )
        code_revision = _git_output("rev-parse", "HEAD")

        print("running aggregate audit pass 1...")
        started = time.perf_counter()
        first_report = build_task2_audit(
            swiss_prot_path=local_verifications["swiss_prot_records"].path,
            uniref50_path=local_verifications["uniref50_membership"].path,
            proteingym_path=args.proteingym_metadata,
            sources=source_evidence,
            code_revision=code_revision,
        )
        first_rendered = render_task2_audit(first_report)
        print(f"pass 1 completed in {time.perf_counter() - started:.1f} seconds")

        if args.repeat_check:
            print("running aggregate audit pass 2...")
            started = time.perf_counter()
            second_report = build_task2_audit(
                swiss_prot_path=local_verifications["swiss_prot_records"].path,
                uniref50_path=local_verifications["uniref50_membership"].path,
                proteingym_path=args.proteingym_metadata,
                sources=source_evidence,
                code_revision=code_revision,
            )
            second_rendered = render_task2_audit(second_report)
            if first_rendered != second_rendered:
                raise Task2AuditError(
                    "repeated audit runs produced different report bytes"
                )
            print(f"pass 2 completed in {time.perf_counter() - started:.1f} seconds")

        _write_outputs(args.output_directory, first_rendered)
    except (
        ValueError,
        OSError,
        subprocess.SubprocessError,
    ) as error:
        print(f"corpus audit failed: {error}")
        return 1

    print(f"report JSON SHA-256: {first_rendered.json_sha256}")
    print(f"outputs: {args.output_directory}")
    print("network requests made: none")
    return 0


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
        raise Task2AuditError(
            "execution code has uncommitted changes; review and commit it first"
        )


def _prove_path_is_ignored(path: Path) -> None:
    try:
        relative_path = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise Task2AuditError(
            "ProteinGym metadata must be stored inside the repository's ignored data/"
        ) from error
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative_path.as_posix()],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise Task2AuditError(
            f"ProteinGym metadata path is not ignored by Git: {relative_path}"
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


def _write_outputs(output_directory: Path, rendered) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        output_directory / f"{OUTPUT_STEM}.json": rendered.json_text,
        output_directory / f"{OUTPUT_STEM}.md": rendered.markdown_text,
        output_directory / f"{OUTPUT_STEM}.sha256": (
            f"{rendered.json_sha256}  {OUTPUT_STEM}.json\n"
        ),
    }
    for path, text in outputs.items():
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
