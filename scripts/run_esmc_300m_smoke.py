"""Run the approved local-only Task 11B ESMC-300M smoke once."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_lm.external.esmc_contract import load_esmc_contract
from protein_lm.external.esmc_result import result_reason, write_esmc_result
from protein_lm.external.esmc_smoke import run_esmc_smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "experiments" / "week_01" / "esmc_300m_smoke.toml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Require an intentional local run, device selection, and new evidence path."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline synthetic ESMC-300M smoke. It never downloads or "
            "substitutes CPU for an MPS request."
        )
    )
    parser.add_argument(
        "--execute-esmc-smoke",
        action="store_true",
        required=True,
        help="explicitly permit this one local inference smoke invocation",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="local directory containing the pinned model and tokenizer files",
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cpu"),
        required=True,
        help="explicit execution device; CPU is a separate invocation, not fallback",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        required=True,
        help="new JSON path for this immutable smoke record",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute once and preserve both completed and failed records."""
    args = parse_args(argv)
    if args.result_path.exists():
        print(f"refusing to overwrite existing result: {args.result_path}")
        return 2

    contract = load_esmc_contract(CONTRACT_PATH)
    result = run_esmc_smoke(
        contract,
        model_dir=args.model_dir,
        device=args.device,
        project_root=PROJECT_ROOT,
    )
    try:
        write_esmc_result(args.result_path, result)
    except FileExistsError:
        print(f"refusing to overwrite existing result: {args.result_path}")
        return 2

    print(f"device: {args.device}")
    print(f"status: {result['status']}")
    print(f"decision: {result['decision']}")
    print(f"reason: {result_reason(result)}")
    print(f"result: {args.result_path}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
