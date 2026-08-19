"""Preflight or explicitly publish the pinned Week 2 sampling diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.bigram.sampling_publication import execute_publication, preflight
from protein_lm.data.model_data.contracts import ModelDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish pinned Week 2 bigram samples."
    )
    parser.add_argument("--execute-publication", action="store_true")
    arguments = parser.parse_args()
    try:
        plan = preflight(PROJECT_ROOT)
        if not arguments.execute_publication:
            print("source: validated passed candidate, two neural bigrams only")
            print("outputs: reports/week_02/bigram_sampling_v1.{json,md,sha256}")
            print("execution requires --execute-publication")
            return 0
        execute_publication(PROJECT_ROOT, plan)
    except ModelDataError as error:
        print(f"Week 2 bigram sampling publication failed: {error}")
        return 1
    print("synthetic non-functional Week 2 sampling diagnostic created")
    print("network requests made: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
