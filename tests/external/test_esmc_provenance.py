import json
from pathlib import Path

import pytest

from protein_lm.external.esmc_contract import ContractValidationError, load_esmc_contract
from protein_lm.external.esmc_provenance import validate_installed_package_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "experiments" / "week_01" / "esmc_300m_smoke.toml"


def test_installed_package_provenance_requires_frozen_versions_and_commits() -> None:
    contract = load_esmc_contract(CONTRACT_PATH)
    distributions = {
        "esm": _FakeDistribution("3.3.0", contract.code_revision),
        "transformers": _FakeDistribution("4.57.6", contract.transformers_revision),
    }

    assert validate_installed_package_provenance(
        contract, distribution_getter=distributions.__getitem__
    ) == {
        "esm": {"version": "3.3.0", "commit_id": contract.code_revision},
        "transformers": {
            "version": "4.57.6",
            "commit_id": contract.transformers_revision,
        },
    }

    distributions["transformers"] = _FakeDistribution("4.57.6", "wrong")
    with pytest.raises(ContractValidationError, match="Transformers direct_url"):
        validate_installed_package_provenance(
            contract, distribution_getter=distributions.__getitem__
        )


class _FakeDistribution:
    def __init__(self, version: str, commit_id: str) -> None:
        self.version = version
        self._direct_url = json.dumps({"vcs_info": {"commit_id": commit_id}})

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return self._direct_url
