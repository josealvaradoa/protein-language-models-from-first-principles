"""Validate the Week 1 source contract without making network requests."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.data.acquisition import (
    AcquisitionValidationError,
    load_acquisition_contract,
    prove_heavy_paths_are_ignored,
    validate_release_metadata,
    verify_local_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "week_01" / "acquisition.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen acquisition config and ignored data paths. "
            "This command never accesses the network."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="path to the acquisition TOML file",
    )
    parser.add_argument(
        "--release-metadata",
        type=Path,
        help="optional local reldate.txt to verify",
    )
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="verify configured files already present under data/raw",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract = load_acquisition_contract(args.config)
        ignored_paths = prove_heavy_paths_are_ignored(contract, PROJECT_ROOT)

        release_status = "not checked because no local metadata was supplied"
        if args.release_metadata is not None:
            metadata = validate_release_metadata(
                args.release_metadata.read_text(encoding="utf-8"),
                contract,
            )
            release_status = (
                f"matched {metadata.release_id} ({metadata.release_date.isoformat()})"
            )

        verified_files = []
        if args.verify_files:
            for source in contract.sources:
                local_path = PROJECT_ROOT / contract.local_path_for(source)
                verified_files.append(verify_local_file(local_path, source))
    except (AcquisitionValidationError, OSError) as error:
        print(f"acquisition validation failed: {error}")
        return 1

    print("acquisition contract valid")
    print(f"release: {contract.release_id} ({contract.release_date.isoformat()})")
    print(f"release metadata: {release_status}")
    print(f"license: {contract.license_spdx}")
    for source, ignored_path in zip(contract.sources, ignored_paths, strict=True):
        print(
            f"source: {source.role} | {source.filename} | "
            f"{source.expected_bytes} bytes | Git ignored: {ignored_path}"
        )
    for result in verified_files:
        print(f"verified: {result.path.name} | sha256={result.sha256}")
    print("network requests made: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
