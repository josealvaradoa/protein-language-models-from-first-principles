"""Preflight or operator-gated execution for one Week 3 scalable MLP seed."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.checkpoint import checkpoint_seed
from protein_lm.mlp.orchestration import execute_run, preflight


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or resume the frozen Week 3 MLP."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--new-run", action="store_true", help="operator-gated new local run"
    )
    mode.add_argument(
        "--resume-checkpoint", type=Path, help="operator-gated resume checkpoint"
    )
    parser.add_argument("--run-id", help="explicit local run identifier")
    parser.add_argument(
        "--seed", type=int, help="approved seed, required for a new run"
    )
    parser.add_argument(
        "--device", choices=("cpu", "mps"), help="explicit execution device"
    )
    arguments = parser.parse_args(argv)
    executing = arguments.new_run or arguments.resume_checkpoint is not None
    if arguments.new_run and (
        arguments.run_id is None or arguments.seed is None or arguments.device is None
    ):
        parser.error("--new-run requires --run-id --seed and --device")
    if arguments.resume_checkpoint is not None and (
        arguments.run_id is None or arguments.device is None
    ):
        parser.error("--resume-checkpoint requires --run-id and --device")
    if arguments.resume_checkpoint is not None and arguments.seed is not None:
        parser.error("resume seed is read from checkpoint; do not pass --seed")
    if not executing and any(
        value is not None
        for value in (arguments.run_id, arguments.seed, arguments.device)
    ):
        parser.error("run options require --new-run or --resume-checkpoint")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    run_id = arguments.run_id or "preflight-only"
    try:
        plan = preflight(PROJECT_ROOT, run_id)
        _print_plan(plan)
        if not arguments.new_run and arguments.resume_checkpoint is None:
            return 0
        seed = (
            arguments.seed
            if arguments.new_run
            else checkpoint_seed(arguments.resume_checkpoint)
        )
        destination = execute_run(
            root=PROJECT_ROOT,
            plan=plan,
            seed=seed,
            device_name=arguments.device,
            resume_checkpoint=arguments.resume_checkpoint,
            progress_callback=_progress,
        )
    except ModelDataError as error:
        print(f"Week 3 MLP training failed: {error}")
        return 1
    print(f"local run completed: {destination}")
    return 0


def _print_plan(plan) -> None:
    config = plan.config
    print(f"contract identifier: {config.contract_identifier}")
    print("collections: family_aware_training, family_aware_native_validation only")
    print(
        f"model: C={config.context_length}, E={config.embedding_width}, H={config.hidden_width}, parameters={config.parameter_count}"
    )
    print(
        f"prediction budget: {config.prediction_budget}; base batch size: {config.batch_size}"
    )
    print("events create short final batches where needed to preserve exact boundaries")
    print(
        f"milestones: {', '.join(str(value) for value in config.milestone_predictions)}"
    )
    print(
        "expected effect: ignored local checkpoints and a local run-status record only"
    )
    print("expected runtime category: long CPU or MPS training run")
    print(
        "preflight loads no collections, creates no directories, and does no device work"
    )


def _progress(event: str, payload: dict[str, object]) -> None:
    if event == "milestone":
        print(
            "milestone "
            f"{payload['predictions']}: CE={payload['cross_entropy']:.6f}, "
            f"accuracy={payload['accuracy']:.6f}"
        )
    elif event == "checkpoint":
        print(f"checkpoint {payload['predictions']}: {payload['path']}")


if __name__ == "__main__":
    raise SystemExit(main())
