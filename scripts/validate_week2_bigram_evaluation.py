"""Read-only validation for an existing local Week 2 evaluation candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.bigram.evaluation import preflight
from protein_lm.bigram.evaluation_validation import validate_evaluation
from protein_lm.data.model_data.contracts import ModelDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one Week 2 bigram evaluation candidate."
    )
    parser.add_argument("--evaluation-id", required=True)
    arguments = parser.parse_args()
    try:
        plan = preflight(PROJECT_ROOT, arguments.evaluation_id)
        result = validate_evaluation(plan.destination, plan)
    except ModelDataError as error:
        print(f"Week 2 bigram evaluation validation failed: {error}")
        return 1
    print(f"evaluation status: {result['status']}")
    if result["status"] == "passed":
        print("principal records: 12")
    print("network requests made: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
