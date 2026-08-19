"""Preflight or explicitly create one local Week 2 bigram model candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.bigram.candidate import create_candidate, preflight
from protein_lm.data.model_data.contracts import ModelDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the frozen Week 2 bigram models into a named local candidate."
    )
    parser.add_argument(
        "--execute-candidate",
        action="store_true",
        help="load only the two approved training collections and create a new candidate",
    )
    parser.add_argument(
        "--candidate-id",
        help="new lowercase local candidate identifier; required with --execute-candidate",
    )
    arguments = parser.parse_args()
    if arguments.execute_candidate != (arguments.candidate_id is not None):
        parser.error("--execute-candidate and --candidate-id must be provided together")
    return arguments


def main() -> int:
    arguments = parse_args()
    candidate_id = arguments.candidate_id or "preflight-only"
    try:
        plan = preflight(PROJECT_ROOT, candidate_id)
        if not arguments.execute_candidate:
            _print_plan(plan)
            return 0
        destination = create_candidate(root=PROJECT_ROOT, plan=plan)
    except ModelDataError as error:
        print(f"Week 2 bigram candidate training failed: {error}")
        return 1
    print(f"candidate created: {destination}")
    print("network requests made: none")
    return 0


def _print_plan(plan) -> None:
    config = plan.training_config
    print(f"contract identifier: {config.contract_identifier}")
    print("collections: random_training, family_aware_training")
    print(f"prediction pairs per arm: {config.prediction_pair_budget}")
    print(
        f"batches per arm: {config.full_batches} full x {config.batch_size} + "
        f"{config.final_partial_batch_pairs} final partial"
    )
    print("planned output: six logical models, twelve JSON/Safetensors files")
    print("expected effect: local ignored candidate creation only")
    print("expected runtime category: long CPU training run")
    print("execution requires --execute-candidate --candidate-id NEW_ID")
    print("network requests made: none")


if __name__ == "__main__":
    raise SystemExit(main())
