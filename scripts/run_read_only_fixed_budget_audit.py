"""Guarded entry point for the local A-004 fixed-budget similarity audit."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from protein_lm.data.task7_a004_plan import fixed_budget_stage_plan
from protein_lm.data.task7_a004_workflow import (
    run_a004_fixed_budget_audit,
    validate_a004_configuration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "read_only_similarity_audit_a004.toml"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the one explicit acknowledgement that permits expensive local work."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate or run the A-004 read-only fixed-budget MMseqs2 audit. "
            "Without the explicit flag it only validates byte-pinned configuration."
        )
    )
    parser.add_argument(
        "--execute-fixed-budget-audit",
        action="store_true",
        help="explicitly start the expensive local A-004 database and search stages",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print the immutable stage plan, executing only after explicit consent."""

    args = parse_args(argv)
    try:
        configuration = validate_a004_configuration(
            project_root=PROJECT_ROOT, config_path=CONFIG_PATH
        )
        _print_plan(configuration)
        if not args.execute_fixed_budget_audit:
            print("Configuration valid. No database, search, or evidence output was created.")
            print("Network requests made: none")
            return 0
        result = run_a004_fixed_budget_audit(
            project_root=PROJECT_ROOT, config_path=CONFIG_PATH
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"A-004 fixed-budget audit failed: {error}")
        return 1
    print(f"A-004 receipt: {result.receipt_path}")
    print(f"A-004 completion marker: {result.completion_path}")
    print("model use: prohibited")
    print("task8 membership use authorized: false")
    print("Network requests made: none")
    return 0


def _print_plan(configuration: object) -> None:
    print("A-004 fixed-budget stage plan:")
    for track in fixed_budget_stage_plan():
        print(
            f"- {track.origin}: {track.strategy}/{track.partition}/{track.pass_name} "
            "at all-query caps 1,000 and 10,000; 100,000 only for changed queries"
        )
    print(f"- source workspace: {getattr(configuration, 'paths')['source_workspace']}")
    print(f"- A-004 workspace: {getattr(configuration, 'paths')['workspace']}")


if __name__ == "__main__":
    raise SystemExit(main())
