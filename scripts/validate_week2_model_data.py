"""Preflight or explicitly validate an existing Week 2 model-data candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.data.model_data.config import load_config
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.reporting import write_readiness_evidence
from protein_lm.data.model_data.validation import validate_candidate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "experiments/week_02/model_data_readiness.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Week 2 candidate readiness validation."
    )
    parser.add_argument(
        "--execute-readiness-validation",
        action="store_true",
        help="explicitly validate an existing candidate and write aggregate evidence",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        config = load_config(CONFIG_PATH)
        if not arguments.execute_readiness_validation:
            print(f"candidate identifier: {config.candidate_identifier}")
            print(
                "planned hard gates: accounting, partition isolation, source contract, evaluation bounds, checksums, deterministic regeneration"
            )
            print("public evidence is aggregate-only. MMseqs2 will not run.")
            print("network requests made: none")
            return 0
        candidate = PROJECT_ROOT / config.candidate_directory_relative_path
        if not candidate.is_dir():
            raise ModelDataError("candidate directory does not exist")
        result = validate_candidate(
            root=PROJECT_ROOT, candidate_directory=candidate, config=config
        )
        write_readiness_evidence(
            (
                PROJECT_ROOT / config.readiness_json_relative_path,
                PROJECT_ROOT / config.readiness_markdown_relative_path,
                PROJECT_ROOT / config.readiness_sha256_relative_path,
            ),
            result,
        )
    except ModelDataError as error:
        print(f"Week 2 readiness validation failed: {error}")
        return 1
    print(f"candidate status: {result.status}")
    print("network requests made: none")
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
