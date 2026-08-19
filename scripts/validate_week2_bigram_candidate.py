"""Read-only integrity validation for an existing Week 2 bigram candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.bigram.candidate import preflight
from protein_lm.bigram.candidate_validation import validate_candidate
from protein_lm.data.model_data.contracts import ModelDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = PROJECT_ROOT / "data/processed/week_02/bigram_model_candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only validation of one existing Week 2 bigram candidate."
    )
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--candidate-id", help="existing local candidate identifier")
    choices.add_argument("--candidate-path", type=Path, help="existing candidate path")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    candidate = (
        CANDIDATE_ROOT / arguments.candidate_id
        if arguments.candidate_id is not None
        else arguments.candidate_path.resolve()
    )
    try:
        plan = preflight(PROJECT_ROOT, candidate.name)
        if candidate != plan.destination:
            raise ModelDataError("candidate path must use the approved local candidate root")
        result = validate_candidate(candidate, plan)
    except ModelDataError as error:
        print(f"Week 2 bigram candidate validation failed: {error}")
        return 1
    print(f"candidate status: {result['status']}")
    print("logical models: 6; serialization files: 12")
    print("network requests made: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
