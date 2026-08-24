"""Config-only preflight or explicit operator execution of exploratory MLP tails."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.tail_orchestration import execute_tail, preflight
from protein_lm.mlp.tail_training import schedule_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ARMS = ("staged_97m_003", "cosine_90m_100m_001")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one non-resumable Week 3 exploratory MLP learning-rate tail."
    )
    parser.add_argument("--new-tail", action="store_true", help="operator-gated tail")
    parser.add_argument("--run-id", help="new unique local run identifier")
    parser.add_argument("--seed", type=int, help="approved parent seed")
    parser.add_argument("--arm", choices=_ARMS, help="approved learning-rate arm")
    parser.add_argument("--device", choices=("cpu",), help="explicit CPU execution")
    arguments = parser.parse_args(argv)
    if arguments.new_tail and any(
        value is None
        for value in (arguments.run_id, arguments.seed, arguments.arm, arguments.device)
    ):
        parser.error("--new-tail requires --run-id --seed --arm and --device cpu")
    if not arguments.new_tail and any(
        value is not None
        for value in (arguments.run_id, arguments.seed, arguments.arm, arguments.device)
    ):
        parser.error("tail run options require --new-tail")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    arm = arguments.arm or "staged_97m_003"
    try:
        plan = preflight(PROJECT_ROOT, arguments.run_id or "preflight-only", arm)
        _print_plan(plan)
        if not arguments.new_tail:
            return 0
        destination = execute_tail(
            root=PROJECT_ROOT,
            plan=plan,
            seed=arguments.seed,
            device_name=arguments.device,
            progress_callback=_progress,
        )
    except ModelDataError as error:
        print(f"Week 3 MLP learning-rate tail failed: {error}")
        return 1
    print(f"exploratory local tail completed: {destination}")
    return 0


def _print_plan(plan) -> None:
    config = plan.config
    print(f"tail contract identifier: {config.contract_identifier}")
    print(f"arm: {plan.arm}; CPU only; non-resumable")
    print("collections: family_aware_training, family_aware_native_validation only")
    print(
        f"parent: 90,000,000 predictions / {config.parent_optimizer_steps} updates; "
        f"tail: {config.tail_optimizer_updates} updates to 100,000,000"
    )
    print("approved schedules:")
    for arm in config.approved_arms:
        schedule = schedule_provenance(arm, config)
        if arm == "staged_97m_003":
            print(
                "- staged_97m_003: LR 0.01 before 97,000,000, then 0.003; "
                f"first lower-LR inherited batch {schedule['effective_lower_lr_start_prediction']}"
            )
        else:
            print(
                "- cosine_90m_100m_001: cosine 0.01 to mathematical endpoint "
                "0.001 over 90M-100M; final applied LR is above the endpoint"
            )
    print(
        "preflight reads only byte-pinned configuration; it does not access a "
        "checkpoint, corpus, device, output, directory, or git revision"
    )


def _progress(event: str, payload: dict[str, object]) -> None:
    if event == "completed":
        print(
            "completed "
            f"arm={payload['arm']} seed={payload['seed']} "
            f"native_CE={payload['final_cross_entropy']:.6f} "
            f"accuracy={payload['final_accuracy']:.6f} "
            f"runtime_seconds={payload['runtime_seconds']:.3f} "
            f"tail_updates={payload['tail_optimizer_updates']}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
