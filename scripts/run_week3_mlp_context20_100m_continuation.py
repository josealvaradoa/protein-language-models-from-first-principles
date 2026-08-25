"""Preflight or explicitly continue one approved C=20 Week 3 parent."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.context20_continuation_orchestration import (
    execute_continuation,
    preflight,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue one approved C=20 25M capacity-screen checkpoint to 100M."
    )
    parser.add_argument(
        "--execute-continuation", action="store_true", help="operator execution gate"
    )
    parser.add_argument("--run-id", help="new unique local continuation identifier")
    parser.add_argument("--seed", type=int, help="approved C=20 parent seed")
    parser.add_argument("--device", choices=("cpu",), help="explicit CPU execution")
    arguments = parser.parse_args(argv)
    required = (arguments.run_id, arguments.seed, arguments.device)
    if arguments.execute_continuation and any(value is None for value in required):
        parser.error("--execute-continuation requires --run-id --seed and --device cpu")
    if not arguments.execute_continuation and any(
        value is not None for value in required
    ):
        parser.error("continuation run options require --execute-continuation")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        plan = preflight(PROJECT_ROOT, arguments.run_id or "preflight-only")
        _print_plan(plan)
        if not arguments.execute_continuation:
            return 0
        destination = execute_continuation(
            root=PROJECT_ROOT,
            plan=plan,
            seed=arguments.seed,
            device_name=arguments.device,
            progress_callback=_progress,
        )
    except ModelDataError as error:
        print(f"Week 3 C=20 continuation failed: {error}")
        return 1
    print(f"exploratory local C=20 continuation completed: {destination}")
    return 0


def _print_plan(plan) -> None:
    config = plan.config
    print(f"context20 continuation contract identifier: {config.contract_identifier}")
    print("exploratory only; CPU only; non-resumable; no automatic decision or report")
    print("collections: family_aware_training, family_aware_native_validation only")
    print(
        "parent: 25,000,000 predictions / 24,416 updates; continuation: 75,000,000 predictions / 73,244 updates"
    )
    print(
        "final: 100,000,000 predictions / 97,660 updates; C=20, E=32, H=800, parameters=530,293"
    )
    print(
        "LR 0.1 before 90M and 0.01 from 90M; evaluates at 50M and 100M; saves at 50M, 90M, 100M"
    )
    print(
        f"three-seed-only 100M native CE threshold: <= {config.qualifying_mean_native_cross_entropy_at_most}; no per-seed selection"
    )
    print(
        "preflight reads only byte-pinned configs; it does not access parent runs, readiness, collections, device, output, or git"
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
