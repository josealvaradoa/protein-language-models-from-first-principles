"""Run the A-003-authorized Week 1 Task 7 diagnostic similarity audit."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from protein_lm.data.task7_workflow import run_diagnostic_similarity_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "experiments" / "week_01" / "diagnostic_similarity_audit.toml"
)
REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "week_01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the A-003 diagnostic MMseqs2 audit. This command never "
            "repairs, selects, or trains on a split."
        )
    )
    parser.add_argument(
        "--execute-diagnostic-audit",
        action="store_true",
        required=True,
        help="required safety acknowledgement that starts the corpus searches",
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        rendered = run_diagnostic_similarity_audit(
            project_root=PROJECT_ROOT,
            config_path=CONFIG_PATH,
            report_directory=REPORT_DIRECTORY,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"diagnostic similarity audit failed: {error}")
        return 1

    print(f"report JSON SHA-256: {rendered.json_sha256}")
    print(f"outputs: {REPORT_DIRECTORY}")
    print("candidate status: failed_balance")
    print("repair performed: false")
    print("selected split authorized: false")
    print("model use: prohibited")
    print("post-audit review required: true")
    print("network requests made: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
