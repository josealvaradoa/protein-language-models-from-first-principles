"""Run one approved MPS synthetic device-envelope candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.benchmarks import BENCHMARK_CANDIDATES, run_synthetic_benchmark
from protein_lm.benchmarks.runner import write_benchmark_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _SingleCandidateAction(argparse.Action):
    """Reject repeated candidate flags instead of silently keeping the last one."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error("--candidate may be provided exactly once")
        setattr(namespace, self.dest, values)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Require an intentional, single-candidate benchmark execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one approved synthetic Mac mini MPS device-envelope candidate. "
            "It never substitutes CPU for MPS."
        )
    )
    parser.add_argument(
        "--execute-mps-benchmark",
        action="store_true",
        required=True,
        help="explicitly permit this one MPS benchmark execution",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        choices=sorted(BENCHMARK_CANDIDATES),
        action=_SingleCandidateAction,
        help="run exactly one approved candidate",
    )
    parser.add_argument(
        "--result-path",
        required=True,
        type=Path,
        help="new JSON path for this candidate result",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the named candidate once and preserve its record."""
    args = parse_args(argv)
    if args.result_path.exists():
        print(f"refusing to overwrite existing result: {args.result_path}")
        return 2

    result = run_synthetic_benchmark(
        BENCHMARK_CANDIDATES[args.candidate],
        device="mps",
        project_root=PROJECT_ROOT,
    )
    try:
        write_benchmark_result(args.result_path, result)
    except FileExistsError:
        print(f"refusing to overwrite existing result: {args.result_path}")
        return 2

    print(f"candidate: {args.candidate}")
    print(f"status: {result.status}")
    print(f"readiness_checks: {result.readiness_checks.as_dict()}")
    print(f"result: {args.result_path}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
