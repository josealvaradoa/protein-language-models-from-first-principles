import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from protein_lm.external.esmc_contract import (
    ContractValidationError,
    load_esmc_contract,
    validate_local_model_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "experiments" / "week_01" / "esmc_300m_smoke.toml"


def test_public_contract_declares_only_two_expected_synthetic_fixtures() -> None:
    contract = load_esmc_contract(CONTRACT_PATH)

    assert [len(fixture.sequence) for fixture in contract.fixtures] == [32, 64]
    assert [fixture.mask_residue_index for fixture in contract.fixtures] == [7, 41]
    assert contract.expected_config.raw_fields() == {
        "architectures": ["ESMCForMaskedLM"],
        "model_type": "esmc",
        "vocab_size": 64,
        "d_model": 960,
        "n_layers": 30,
        "n_heads": 15,
        "mask_token_id": 32,
        "pad_token_id": 1,
        "dtype": "float32",
        "transformers_version": "4.57.6",
    }
    assert contract.published_context_limit == 2048


def test_local_preflight_rejects_config_or_weight_hash_mismatch(tmp_path: Path) -> None:
    contract = load_esmc_contract(CONTRACT_PATH)
    _write_local_files(tmp_path, weight=b"tiny fixture")

    with pytest.raises(ContractValidationError, match="SHA-256"):
        validate_local_model_dir(tmp_path, contract)

    (tmp_path / "model.safetensors").write_bytes(b"different fixture")
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    config["d_model"] = 959
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    matching_contract = contract.__class__(
        **{**contract.__dict__, "weight_sha256": _sha256(tmp_path / "model.safetensors")}
    )

    with pytest.raises(ContractValidationError, match="d_model"):
        validate_local_model_dir(tmp_path, matching_contract)


def test_local_preflight_requires_the_special_token_map(tmp_path: Path) -> None:
    contract = load_esmc_contract(CONTRACT_PATH)
    _write_local_files(tmp_path, weight=b"tiny fixture")
    (tmp_path / "special_tokens_map.json").unlink()

    with pytest.raises(ContractValidationError, match="special_tokens_map.json"):
        validate_local_model_dir(tmp_path, contract)


def test_contract_and_uv_override_pin_canonical_biohub_repositories() -> None:
    contract = load_esmc_contract(CONTRACT_PATH)
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    assert contract.code_repository == "https://github.com/Biohub/esm.git"
    assert contract.code_revision == "26b0bc2b771e3e419ea74f445a5f35cc094a1509"
    assert contract.transformers_repository == "https://github.com/Biohub/transformers.git"
    assert contract.transformers_revision == "ef32577f55da19a4989cd7b22e004dc43a4998cb"
    assert pyproject["dependency-groups"]["esmc"] == [
        "esm @ git+https://github.com/Biohub/esm.git@26b0bc2b771e3e419ea74f445a5f35cc094a1509"
    ]
    assert pyproject["tool"]["uv"]["override-dependencies"] == [
        "transformers @ git+https://github.com/Biohub/transformers.git@ef32577f55da19a4989cd7b22e004dc43a4998cb"
    ]


def _write_local_files(directory: Path, *, weight: bytes) -> None:
    config = {
        "architectures": ["ESMCForMaskedLM"],
        "model_type": "esmc",
        "vocab_size": 64,
        "d_model": 960,
        "n_layers": 30,
        "n_heads": 15,
        "mask_token_id": 32,
        "pad_token_id": 1,
        "dtype": "float32",
        "transformers_version": "4.57.6",
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    (directory / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(weight)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
