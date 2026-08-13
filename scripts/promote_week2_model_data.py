"""Read-only preflight or explicit promotion of approved Week 2 model data."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.data.model_data.config import load_config
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.promotion import preflight_promotion, promote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "experiments/week_02/model_data_readiness.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight or explicitly promote passing Week 2 model-data artifacts."
    )
    parser.add_argument(
        "--execute-promotion",
        action="store_true",
        help="copy the approved public artifacts into manifests/week_02",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    try:
        config = load_config(CONFIG_PATH)
        if arguments.execute_promotion:
            destination = promote(PROJECT_ROOT, config)
        else:
            plan = preflight_promotion(PROJECT_ROOT, config)
            print(f"destination: {plan['destination']}")
            print("promoted artifacts: " + ", ".join(plan["promoted_artifacts"]))
            print("sealed membership: not read")
            print("network requests made: none")
            return 0
    except ModelDataError as error:
        print(f"Week 2 model-data promotion failed: {error}")
        return 1
    print(f"promoted public manifests: {destination}")
    print("network requests made: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
