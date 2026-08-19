"""Preflight or explicitly publish aggregate-only Week 2 bigram evaluation evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.bigram.public_report_publication import execute_publication, preflight
from protein_lm.data.model_data.contracts import ModelDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish the pinned Week 2 bigram evaluation report."
    )
    parser.add_argument("--execute-publication", action="store_true")
    arguments = parser.parse_args()
    try:
        plan = preflight(PROJECT_ROOT)
        if not arguments.execute_publication:
            print("source evaluation: validated aggregate-only candidate")
            print("outputs: reports/week_02/bigram_evaluation_v1.{json,md,sha256}")
            print("execution requires --execute-publication")
            return 0
        execute_publication(PROJECT_ROOT, plan)
    except ModelDataError as error:
        print(f"Week 2 bigram evaluation publication failed: {error}")
        return 1
    print("public Week 2 bigram evaluation report created")
    print("network requests made: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
