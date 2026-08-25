"""Config-only preflight or explicit execution for one Week 3 capacity arm."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.capacity_screen_orchestration import execute_screen, preflight


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one exploratory, non-resumable Week 3 MLP capacity-screen arm."
    )
    parser.add_argument(
        "--execute-screen", action="store_true", help="operator execution gate"
    )
    parser.add_argument("--run-id", help="new unique local run identifier")
    parser.add_argument("--arm", help="approved allocation arm")
    parser.add_argument("--seed", type=int, help="approved isolated seed")
    parser.add_argument("--device", choices=("cpu",), help="explicit CPU execution")
    arguments = parser.parse_args(argv)
    required = (arguments.run_id, arguments.arm, arguments.seed, arguments.device)
    if arguments.execute_screen and any(value is None for value in required):
        parser.error("--execute-screen requires --run-id --arm --seed and --device cpu")
    if not arguments.execute_screen and any(value is not None for value in required):
        parser.error("capacity-screen run options require --execute-screen")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        plan = preflight(
            PROJECT_ROOT,
            arguments.run_id or "preflight-only",
            arguments.arm or "context_20",
        )
        _print_plan(plan)
        if not arguments.execute_screen:
            return 0
        destination = execute_screen(
            root=PROJECT_ROOT,
            plan=plan,
            seed=arguments.seed,
            device_name=arguments.device,
            progress_callback=_progress,
        )
    except ModelDataError as error:
        print(f"Week 3 capacity screen failed: {error}")
        return 1
    print(f"exploratory local capacity-screen run completed: {destination}")
    return 0


def _print_plan(plan) -> None:
    config = plan.config
    print(f"capacity-screen contract identifier: {config.contract_identifier}")
    print("exploratory only; CPU only; non-resumable; no automatic selection or report")
    print("collections: family_aware_training, family_aware_native_validation only")
    print(
        f"arm: {plan.arm.name}; C={plan.arm.context_length}, "
        f"E={plan.arm.embedding_width}, H={plan.arm.hidden_width}, "
        f"parameters={plan.arm.parameter_count}"
    )
    print(
        "prediction budget: 25,000,000; optimizer updates: 24,416; "
        "fixed SGD LR: 0.1; batch size: 1,024"
    )
    print("native-validation events: 1,000,000, 5,000,000, 10,000,000, 25,000,000")
    print(
        "qualification uses only each arm's 25M three-seed mean native CE: "
        f"<= {config.qualifying_mean_native_cross_entropy_at_most}; "
        "no per-seed selection"
    )
    print(
        "preflight reads only byte-pinned configs; it does not access readiness, "
        "control files, collections, device, output, directories, or git"
    )


def _progress(event: str, payload: dict[str, object]) -> None:
    if event == "milestone":
        print(
            "milestone "
            f"predictions={payload['prediction_position']} "
            f"native_CE={payload['cross_entropy']:.6f} "
            f"accuracy={payload['accuracy']:.6f}"
        )
    elif event == "checkpoint":
        print(
            "checkpoint "
            f"predictions={payload['prediction_position']} "
            f"path={payload['relative_path']}"
        )
    elif event == "completed":
        print(
            "completed "
            f"seed={payload['seed']} predictions={payload['prediction_position']} "
            f"updates={payload['optimizer_steps']} "
            f"native_CE={payload['native_cross_entropy']:.6f} "
            f"accuracy={payload['native_accuracy']:.6f} "
            f"runtime_seconds={payload['runtime_seconds']:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
