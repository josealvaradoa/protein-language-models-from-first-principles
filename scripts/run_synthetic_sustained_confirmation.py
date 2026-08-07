"""Run the approved sustained MPS confirmation for Task 11A-2 capacity H."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.benchmarks.sustained import (
    run_sustained_h_confirmation,
    write_sustained_confirmation_result,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Require explicit execution and both immutable evidence paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the 60-step synthetic MPS sustained confirmation for capacity H. "
            "It never substitutes CPU for MPS."
        )
    )
    parser.add_argument(
        "--execute-mps-sustained-confirmation",
        action="store_true",
        required=True,
        help="explicitly permit this one MPS sustained confirmation",
    )
    parser.add_argument(
        "--source-capacity-result",
        required=True,
        type=Path,
        help="completed Task 11A-2 capacity-H JSON evidence",
    )
    parser.add_argument(
        "--result-path",
        required=True,
        type=Path,
        help="new JSON path for this sustained confirmation result",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute one confirmation and preserve a result for every outcome."""
    args = parse_args(argv)
    if args.result_path.exists():
        print(f"refusing to overwrite existing result: {args.result_path}")
        return 2

    result = run_sustained_h_confirmation(
        args.source_capacity_result,
        project_root=PROJECT_ROOT,
    )
    try:
        write_sustained_confirmation_result(args.result_path, result)
    except FileExistsError:
        print(f"refusing to overwrite existing result: {args.result_path}")
        return 2

    print(f"status: {result.status}")
    print(f"sustained_confirmation_passed: {result.sustained_confirmation_passed}")
    print(f"reason: {result.reason}")
    print(f"result: {args.result_path}")
    return 0 if result.sustained_confirmation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
