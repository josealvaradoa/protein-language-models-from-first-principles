"""Preflight or explicitly create the local Week 2 model-data candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.data.model_data.config import load_config
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.workflow import create_candidate, preflight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "experiments/week_02/model_data_readiness.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen Week 2 model-data candidate."
    )
    parser.add_argument(
        "--execute-candidate",
        action="store_true",
        help="explicitly create the local Candidate v1",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        config = load_config(CONFIG_PATH)
        if not arguments.execute_candidate:
            plan = preflight(config, PROJECT_ROOT)
            print(f"candidate identifier: {plan['candidate_identifier']}")
            print(f"destination: {plan['destination']}")
            print("verified input identities:")
            for input_path in plan["verified_inputs"]:
                print(f"  - {input_path}")
            print("planned selection order: " + ", ".join(plan["selection_order"]))
            print("allocation namespaces:")
            for namespace in plan["allocation_namespaces"]:
                print(f"  - {namespace}")
            print("expected artifacts: " + ", ".join(plan["expected_artifacts"]))
            print("MMseqs2 will not run. Model code will not run.")
            print("network requests made: none")
            return 0
        destination = create_candidate(
            root=PROJECT_ROOT, config_path=CONFIG_PATH, config=config
        )
    except ModelDataError as error:
        print(f"Week 2 candidate preparation failed: {error}")
        return 1
    print(f"candidate created: {destination}")
    print("network requests made: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
