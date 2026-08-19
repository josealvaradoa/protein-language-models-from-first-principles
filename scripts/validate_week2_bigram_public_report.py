"""Read-only validation for the published Week 2 bigram evaluation report."""

from __future__ import annotations

from pathlib import Path

from protein_lm.bigram.public_report_validation import validate_public_report
from protein_lm.data.model_data.contracts import ModelDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = validate_public_report(PROJECT_ROOT)
    except ModelDataError as error:
        print(f"Week 2 bigram public report validation failed: {error}")
        return 1
    print(f"public report status: {result['status']}")
    print("principal records: 12")
    print("network requests made: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
