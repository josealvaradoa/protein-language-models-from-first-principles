#!/usr/bin/env python3
"""Preflight or explicitly execute the frozen Week 3 E=64 challenger."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.embedding64_challenger_orchestration import (
    execute_challenger,
    preflight,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-challenger",
        action="store_true",
        help="run one approved immutable-parent continuation",
    )
    parser.add_argument("--seed", type=int, help="approved parent seed")
    parser.add_argument("--run-id", help="new local run identifier")
    parser.add_argument("--device", choices=("cpu",), help="must be cpu")
    args = parser.parse_args(argv)
    required = (args.seed, args.run_id, args.device)
    if args.execute_challenger and any(value is None for value in required):
        parser.error("--execute-challenger requires --seed, --run-id, and --device cpu")
    if not args.execute_challenger and any(value is not None for value in required):
        parser.error("challenger run options require --execute-challenger")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = preflight(PROJECT_ROOT, args.run_id or "preflight-only")
        _print_plan(plan)
        if not args.execute_challenger:
            return 0
        destination = execute_challenger(
            root=PROJECT_ROOT,
            plan=plan,
            seed=args.seed,
            device_name=args.device,
            progress_callback=_progress,
        )
    except ModelDataError as error:
        print(f"Week 3 E=64 challenger failed: {error}")
        return 1
    print(f"exploratory local E=64 challenger completed: {destination}")
    return 0


def _print_plan(plan) -> None:
    config = plan.config
    print(f"embedding64 challenger contract identifier: {config.contract_identifier}")
    print("exploratory only; CPU only; non-resumable; no automatic decision or report")
    print("parents: exact E=64 25M runs only; no replay of predictions 0 through 25M")
    print(
        "final: 100,000,000 predictions / 97,660 updates; C=10, E=64, H=800, parameters=530,965"
    )
    print(
        "LR 0.1 before 90M and 0.01 from 90M; evaluates at 50M and 100M; saves at 50M, 90M, 100M"
    )
    print(
        "three-seed 100M native-CE interpretation is manual: "
        f"E64 mean >= {config.context20_materially_better_if_embedding64_mean_at_or_above} means C20 materially better; "
        f"E64 mean <= {config.embedding64_materially_better_if_embedding64_mean_at_or_below} means E64 materially better; "
        "strictly between is a practical tie"
    )
    print(
        "preflight reads only byte-pinned configs; it does not access parents, readiness, collections, device, output, or git"
    )


def _progress(event: str, payload: dict[str, object]) -> None:
    if event == "milestone":
        print(
            f"milestone predictions={payload['prediction_position']} native_CE={payload['cross_entropy']:.6f} accuracy={payload['accuracy']:.6f}"
        )
    elif event == "checkpoint":
        print(
            f"checkpoint predictions={payload['prediction_position']} path={payload['relative_path']}"
        )
    elif event == "completed":
        print(
            f"completed seed={payload['seed']} predictions={payload['prediction_position']} updates={payload['optimizer_steps']} native_CE={payload['native_cross_entropy']:.6f} runtime_seconds={payload['runtime_seconds']:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
