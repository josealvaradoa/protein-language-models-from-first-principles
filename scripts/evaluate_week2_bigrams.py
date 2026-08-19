"""Preflight or explicitly score the fixed Week 2 bigram evaluation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.bigram.evaluation import execute_evaluation, preflight
from protein_lm.data.model_data.contracts import ModelDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the pinned Week 2 bigram candidate without retraining or selection."
    )
    parser.add_argument("--execute-evaluation", action="store_true")
    parser.add_argument("--evaluation-id")
    arguments = parser.parse_args()
    if arguments.execute_evaluation != (arguments.evaluation_id is not None):
        parser.error(
            "--execute-evaluation and --evaluation-id must be provided together"
        )
    return arguments


def main() -> int:
    arguments = parse_args()
    try:
        plan = preflight(PROJECT_ROOT, arguments.evaluation_id or "preflight-only")
        if not arguments.execute_evaluation:
            print(
                "principal records: 12 (3 models x 2 arms x native/shared validation)"
            )
            print(
                "planned loads: random native, family-aware native, shared validation once"
            )
            print("sealed shared test: inaccessible")
            print("execution requires --execute-evaluation --evaluation-id NEW_ID")
            return 0
        destination = execute_evaluation(root=PROJECT_ROOT, plan=plan, progress=print)
    except ModelDataError as error:
        print(f"Week 2 bigram evaluation failed: {error}")
        return 1
    print(f"evaluation candidate created: {destination}")
    print("network requests made: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
