"""Read-only validation for the published Week 3 MLP aggregate report."""

from __future__ import annotations

from pathlib import Path

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.mlp.publication_validation import validate_public_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        result = validate_public_report(PROJECT_ROOT)
    except ModelDataError as error:
        print(f"Week 3 MLP public report validation failed: {error}")
        return 1
    print(f"public report status: {result['status']}")
    print(f"PCA seeds: {result['pca_seed_count']}; residue pairs: {result['residue_pair_count']}")
    print("network requests made: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
