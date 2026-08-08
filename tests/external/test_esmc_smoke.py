import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import protein_lm.external.esmc_smoke as esmc_smoke
from protein_lm.external.esmc_contract import load_esmc_contract
from protein_lm.external.esmc_smoke import (
    _final_hidden_states,
    build_residue_mask,
    load_local_transformers,
    padding_aware_mean_pool,
    run_esmc_smoke,
)
from protein_lm.external.esmc_result import write_esmc_result
from protein_lm.benchmarks.metrics import SwapState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "experiments" / "week_01" / "esmc_300m_smoke.toml"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_esmc_300m_smoke.py"


def test_residue_pooling_excludes_special_tokens_and_pad_poison() -> None:
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    special = torch.tensor([[1, 0, 1, 0], [1, 0, 0, 1]])
    mask = build_residue_mask(attention, special)
    hidden = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
    poisoned = hidden.clone()
    poisoned[attention == 0] = 1_000_000.0

    assert mask.tolist() == [[False, True, False, False], [False, True, True, False]]
    assert torch.equal(
        padding_aware_mean_pool(hidden, mask),
        padding_aware_mean_pool(poisoned, mask),
    )


def test_mps_unavailable_is_preserved_without_loading_or_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract_for_temp_weight(tmp_path)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    def must_not_load(_: Path) -> tuple[object, object]:
        raise AssertionError("model loader must not run")

    result = run_esmc_smoke(
        contract,
        model_dir=tmp_path,
        device="mps",
        project_root=PROJECT_ROOT,
        loader=must_not_load,
        package_provenance_validator=_valid_package_provenance,
    )

    assert result["status"] == "failed"
    assert result["decision"] == "fail"
    assert "CPU fallback is prohibited" in result["error"]["message"]
    assert result["local_weight_sha256"] is None
    assert result["fallback"]["automatic_cpu_fallback"] == "prohibited"
    failure_path = tmp_path / "mps-unavailable.json"
    write_esmc_result(failure_path, result)
    assert json.loads(failure_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_cpu_fake_model_covers_both_paths_and_preserves_failure_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract_for_temp_weight(tmp_path)
    stable_swap = SwapState(raw="fixture", used_bytes=10, total_bytes=100)
    monkeypatch.setattr(esmc_smoke, "read_swap_state", lambda: stable_swap)
    model = _FakeModel()
    result = run_esmc_smoke(
        contract,
        model_dir=tmp_path,
        device="cpu",
        project_root=PROJECT_ROOT,
        loader=lambda _: (_FakeTokenizer(), model),
        package_provenance_validator=_valid_package_provenance,
    )

    assert result["status"] == "completed"
    assert result["decision"] == "pass"
    assert result["residue_counts"] == [32, 64]
    assert result["masked_residue_counts"] == [1, 1]
    assert result["finite_outputs"] is True
    assert result["padding_poison_invariant"] is True
    assert result["unmasked_hidden_state_shape"] == [2, 66, 960]
    assert result["masked_mlm_logit_shape"] == [2, 66, 64]
    assert result["installed_packages"] == _valid_package_provenance(contract)
    assert model.calls[0]["output_hidden_states"] is False

    failed_path = tmp_path / "failed.json"
    failed = {**result, "status": "failed", "error": {"type": "Example"}}
    write_esmc_result(failed_path, failed)
    assert json.loads(failed_path.read_text(encoding="utf-8"))["status"] == "failed"
    with pytest.raises(FileExistsError):
        write_esmc_result(failed_path, failed)


def test_lazy_loader_uses_offline_local_transformers_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer_calls: list[dict[str, object]] = []
    model_calls: list[dict[str, object]] = []

    class Tokenizer:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> str:
            tokenizer_calls.append({"path": path, **kwargs})
            return "tokenizer"

    class Model:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> str:
            model_calls.append({"path": path, **kwargs})
            return "model"

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=Tokenizer, AutoModelForMaskedLM=Model),
    )

    assert load_local_transformers(tmp_path) == ("tokenizer", "model")
    assert tokenizer_calls == [
        {"path": str(tmp_path), "local_files_only": True, "trust_remote_code": False}
    ]
    assert model_calls == [
        {
            "path": str(tmp_path),
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": torch.float32,
        }
    ]


def test_final_hidden_state_ignores_a_real_shape_multi_element_hidden_states_tensor() -> None:
    final_hidden_state = torch.ones((2, 66, 960), dtype=torch.float32)
    output = SimpleNamespace(
        last_hidden_state=final_hidden_state,
        hidden_states=torch.zeros((31, 2, 66, 960), dtype=torch.float32),
    )

    assert _final_hidden_states(output) is final_hidden_state


def test_cli_requires_explicit_arguments_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    with pytest.raises(SystemExit):
        module.parse_args(["--model-dir", "model", "--device", "mps", "--result-path", "x"])
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--execute-esmc-smoke",
                "--model-dir",
                "model",
                "--result-path",
                "x",
            ]
        )

    existing = tmp_path / "existing.json"
    existing.write_text("prior evidence", encoding="utf-8")
    monkeypatch.setattr(module, "run_esmc_smoke", lambda **_: pytest.fail("must not run"))
    assert (
        module.main(
            [
                "--execute-esmc-smoke",
                "--model-dir",
                "model",
                "--device",
                "mps",
                "--result-path",
                str(existing),
            ]
        )
        == 2
    )


def test_missing_swap_measurement_fails_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract_for_temp_weight(tmp_path)
    unknown_swap = SimpleNamespace(raw=None, used_bytes=None, total_bytes=None)
    monkeypatch.setattr(esmc_smoke, "read_swap_state", lambda: unknown_swap)

    result = run_esmc_smoke(
        contract,
        model_dir=tmp_path,
        device="cpu",
        project_root=PROJECT_ROOT,
        loader=lambda _: (_FakeTokenizer(), _FakeModel()),
        package_provenance_validator=_valid_package_provenance,
    )

    assert result["status"] == "failed"
    assert result["decision"] == "fail"
    assert result["swap_grew"] is None
    assert result["acceptance"]["swap_usage_measured_and_not_greater"] is False


def test_boolean_swap_measurement_fails_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract_for_temp_weight(tmp_path)
    boolean_swap = SimpleNamespace(raw="fixture", used_bytes=True, total_bytes=100)
    monkeypatch.setattr(esmc_smoke, "read_swap_state", lambda: boolean_swap)

    result = run_esmc_smoke(
        contract,
        model_dir=tmp_path,
        device="cpu",
        project_root=PROJECT_ROOT,
        loader=lambda _: (_FakeTokenizer(), _FakeModel()),
        package_provenance_validator=_valid_package_provenance,
    )

    assert result["decision"] == "fail"
    assert result["swap_grew"] is None
    assert result["acceptance"]["swap_usage_measured_and_not_greater"] is False


def test_tokenizer_identity_and_lockfile_presence_are_acceptance_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract_for_temp_weight(tmp_path)
    stable_swap = SwapState(raw="fixture", used_bytes=10, total_bytes=100)
    monkeypatch.setattr(esmc_smoke, "read_swap_state", lambda: stable_swap)

    bad_tokenizer = _FakeTokenizer()
    bad_tokenizer.mask_token_id = 63
    tokenizer_result = run_esmc_smoke(
        contract,
        model_dir=tmp_path,
        device="cpu",
        project_root=PROJECT_ROOT,
        loader=lambda _: (bad_tokenizer, _FakeModel()),
        package_provenance_validator=_valid_package_provenance,
    )
    assert tokenizer_result["decision"] == "fail"
    assert "mask_token_id=32" in tokenizer_result["error"]["message"]

    lockless_result = run_esmc_smoke(
        contract,
        model_dir=tmp_path,
        device="cpu",
        project_root=tmp_path / "no-lockfile",
        loader=lambda _: (_FakeTokenizer(), _FakeModel()),
        package_provenance_validator=_valid_package_provenance,
    )
    assert lockless_result["decision"] == "fail"
    assert lockless_result["acceptance"]["lockfile_sha256_present"] is False


def _contract_for_temp_weight(directory: Path):
    contract = load_esmc_contract(CONTRACT_PATH)
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
    weight = directory / "model.safetensors"
    weight.write_bytes(b"synthetic weight fixture")
    return replace(contract, weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest())


def _load_script_module() -> object:
    specification = importlib.util.spec_from_file_location("run_esmc_300m_smoke", SCRIPT_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class _FakeTokenizer:
    mask_token_id = 32
    pad_token_id = 1

    def __call__(self, sequences: list[str], **_: object) -> dict[str, torch.Tensor]:
        length = max(len(sequence) for sequence in sequences) + 2
        input_ids = torch.ones((len(sequences), length), dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        special = torch.ones_like(input_ids)
        for row, sequence in enumerate(sequences):
            item_length = len(sequence) + 2
            input_ids[row, 0] = 1
            input_ids[row, 1 : item_length - 1] = torch.arange(4, len(sequence) + 4)
            input_ids[row, item_length - 1] = 2
            attention[row, :item_length] = 1
            special[row, 1 : item_length - 1] = 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "special_tokens_mask": special,
        }


class _FakeModel:
    config = SimpleNamespace(
        architectures=["ESMCForMaskedLM"],
        model_type="esmc",
        vocab_size=64,
        d_model=960,
        n_layers=30,
        n_heads=15,
        mask_token_id=32,
        pad_token_id=1,
    )

    def __init__(self) -> None:
        self._parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.calls: list[dict[str, object]] = []

    def to(self, *_: object, **__: object) -> "_FakeModel":
        return self

    def eval(self) -> "_FakeModel":
        return self

    def parameters(self):
        return [self._parameter]

    def __call__(self, *, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        hidden = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 960)
        logits = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 64)
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=torch.stack((hidden, hidden)),
            logits=logits,
        )


def _valid_package_provenance(contract) -> dict[str, object]:
    return {
        "esm": {"version": contract.code_version, "commit_id": contract.code_revision},
        "transformers": {
            "version": "4.57.6",
            "commit_id": contract.transformers_revision,
        },
    }
