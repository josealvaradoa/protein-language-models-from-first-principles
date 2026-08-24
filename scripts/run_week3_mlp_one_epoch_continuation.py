"""Config-only preflight or explicit one-epoch continuation execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.one_epoch_orchestration import execute_continuation, preflight


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one exploratory Week 3 MLP continuation to the first epoch."
    )
    parser.add_argument(
        "--execute-continuation", action="store_true", help="operator execution gate"
    )
    parser.add_argument("--run-id", help="new unique local run identifier")
    parser.add_argument("--seed", type=int, help="approved 100M parent seed")
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
        print(f"Week 3 one-epoch continuation failed: {error}")
        return 1
    print(f"exploratory local continuation completed: {destination}")
    return 0


def _print_plan(plan) -> None:
    config = plan.config
    print(f"continuation contract identifier: {config.contract_identifier}")
    print("exploratory only; CPU only; non-resumable; fixed LR 0.01")
    print("collections: family_aware_training, family_aware_native_validation only")
    print(
        "parent: 100,000,000 predictions / 97,660 updates; "
        "continuation: 71,329,454 predictions / 69,658 updates; "
        "first epoch: 171,329,454 / 167,318"
    )
    print(
        "native-validation milestones: 124,999,936, 149,999,872, 171,329,454; "
        "only the final continuation batch is partial (686 predictions)"
    )
    print(
        "three-seed decision rule only: useful if mean native CE <= "
        f"{config.useful_three_seed_mean_native_cross_entropy_at_most}; "
        "no per-seed selection or automatic decision"
    )
    print(
        "preflight reads only byte-pinned configuration; it does not access a "
        "checkpoint, corpus, device, output, directory, readiness report, or git revision"
    )


def _progress(event: str, payload: dict[str, object]) -> None:
    if event == "milestone":
        print(
            "milestone "
            f"predictions={payload['prediction_position']} "
            f"native_CE={payload['cross_entropy']:.6f} "
            f"accuracy={payload['accuracy']:.6f}"
        )
    elif event == "completed":
        print(
            "completed "
            f"seed={payload['seed']} "
            f"predictions={payload['final_prediction_position']} "
            f"native_CE={payload['final_cross_entropy']:.6f} "
            f"accuracy={payload['final_accuracy']:.6f} "
            f"runtime_seconds={payload['runtime_seconds']:.3f} "
            f"continuation_updates={payload['continuation_optimizer_updates']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
