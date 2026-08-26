"""Preflight or operator-publish the byte-pinned Week 3 MLP aggregate report."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.publication_orchestration import execute_publication, preflight


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish pinned Week 3 MLP aggregate evidence.")
    parser.add_argument("--execute-publication", action="store_true")
    arguments = parser.parse_args()
    try:
        plan = preflight(PROJECT_ROOT)
        if not arguments.execute_publication:
            print("source evidence: validated byte-pinned local aggregates")
            print("planned outputs: reports/week_03/mlp_evaluation_v1.{json,md,sha256}")
            print("execution requires --execute-publication")
            return 0
        execute_publication(PROJECT_ROOT, plan)
    except ModelDataError as error:
        print(f"Week 3 MLP publication failed: {error}")
        return 1
    print("public Week 3 MLP report created")
    print("network requests made: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
