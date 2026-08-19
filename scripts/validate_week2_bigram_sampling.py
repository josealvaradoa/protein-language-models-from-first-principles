"""Read-only validation for the Week 2 bigram sampling diagnostic."""

from __future__ import annotations

from pathlib import Path

from protein_lm.bigram.sampling_validation import validate_sampling_diagnostic
from protein_lm.data.model_data.contracts import ModelDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = validate_sampling_diagnostic(PROJECT_ROOT)
    except ModelDataError as error:
        print(f"Week 2 bigram sampling validation failed: {error}")
        return 1
    print(f"sampling diagnostic status: {result['status']}")
    print(f"synthetic samples: {result['sample_count']}")
    print("network requests made: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
