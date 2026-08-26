#!/usr/bin/env python3
"""Preflight or explicitly run the Week 3 no-training position diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.position_availability_diagnostic_orchestration import (
    execute_diagnostic,
    preflight,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-diagnostic", action="store_true", help="operator execution gate"
    )
    parser.add_argument("--run-id", help="new ignored local diagnostic identifier")
    parser.add_argument("--seed", type=int, help="one matched approved seed")
    parser.add_argument("--device", choices=("cpu",), help="must be cpu")
    args = parser.parse_args(argv)
    required = (args.run_id, args.seed, args.device)
    if args.execute_diagnostic and any(value is None for value in required):
        parser.error("--execute-diagnostic requires --run-id, --seed, and --device cpu")
    if not args.execute_diagnostic and any(value is not None for value in required):
        parser.error("diagnostic run options require --execute-diagnostic")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = preflight(PROJECT_ROOT, args.run_id or "preflight-only")
        print(
            f"position diagnostic contract identifier: {plan.config.contract_identifier}"
        )
        print(
            "descriptive only; CPU only; no gradients, training, selection, or report"
        )
        print("matched seeds: 20260821, 20260822, 20260823")
        print(
            "frozen C20-versus-E64 comparison is provenance only; selection remains closed"
        )
        print(
            "bins: 0..10, 11..19, and 20+ real residues before each target, including EOS"
        )
        print(
            "preflight reads only byte-pinned configs; it does not access sources, data, device, output, or git"
        )
        if not args.execute_diagnostic:
            return 0
        destination = execute_diagnostic(
            root=PROJECT_ROOT,
            plan=plan,
            seed=args.seed,
            device_name=args.device,
        )
    except ModelDataError as error:
        print(f"Week 3 position-availability diagnostic failed: {error}")
        return 1
    print(
        f"exploratory local position-availability diagnostic completed: {destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
