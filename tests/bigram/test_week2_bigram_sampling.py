"""Synthetic-only tests for the pinned Week 2 bigram sampling diagnostic."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from protein_lm.bigram.sampling import derived_seed, sample_neural_bigram
from protein_lm.bigram.sampling_contract import load_sampling_config
from protein_lm.bigram.sampling_io import write_evidence
from protein_lm.bigram.sampling_publication import (
    SamplingPlan,
    _payload as publication_payload,
    execute_publication,
)
from protein_lm.bigram.sampling_source import (
    RUN_GATES,
    neural_sources,
    validate_passed_source,
)
from protein_lm.bigram import sampling_validation
import protein_lm.bigram.sampling_io as sampling_io
import protein_lm.bigram.sampling_publication as sampling_publication
from protein_lm.data.model_data.contracts import ModelDataError


ROOT = Path(__file__).parents[2]
REVISION = "a" * 40
CONFIG_PATH = ROOT / "experiments/week_02/bigram_sampling_v1.toml"


def _payload() -> dict[str, object]:
    samples = []
    for arm, namespace in (
        ("random_training", "week2/sampling/random/v1"),
        ("family_aware_training", "week2/sampling/family-aware/v1"),
    ):
        for index in range(10):
            samples.append(
                {
                    "model_arm": arm,
                    "namespace": namespace,
                    "sample_index": index,
                    "seed": index + 1,
                    "sequence": "AC",
                    "residue_length": 2,
                    "termination_reason": "eos",
                }
            )
    return {
        "schema_version": 1,
        "scope": "week_02_bigram_sampling_diagnostic",
        "contract_identifier": "synthetic",
        "status": "passed",
        "hard_gates": {
            "validated_passed_candidate": True,
            "two_neural_artifacts_only": True,
            "twenty_samples_preserved": True,
            "synthetic_nonfunctional_educational_output": True,
            "no_selection_or_biological_claims": True,
            "no_network_requests": True,
        },
        "sampling_configuration_sha256": "",
        "publication_code_revision": REVISION,
        "runtime": {"uv_lock_sha256": "c" * 64, "torch_version": "synthetic"},
        "source": {
            "relative_path": "candidate",
            "candidate_registry_sha256": "a" * 64,
            "candidate_run_record_sha256": "b" * 64,
        },
        "sampling": {},
        "samples": samples,
        "network_requests_made": 0,
    }


def test_cpu_sampling_is_deterministic_and_maps_residue_target_to_context() -> None:
    logits = torch.full((21, 21), -1000.0, dtype=torch.float32)
    logits[0, 0] = 1000.0  # BOS -> A
    logits[1, 20] = 1000.0  # A context -> EOS
    first = sample_neural_bigram(
        logits, base_seed=20260812, namespace="unit", sample_index=0
    )
    second = sample_neural_bigram(
        logits, base_seed=20260812, namespace="unit", sample_index=0
    )
    assert first == second
    assert first["sequence"] == "A"
    assert first["termination_reason"] == "eos"
    assert derived_seed(20260812, "unit", 0) != derived_seed(20260812, "unit", 1)


def test_contract_pins_two_final_neural_models_and_public_output_names() -> None:
    config = load_sampling_config(CONFIG_PATH)
    assert config.candidate_id == "week2-bigram-v1-001"
    assert config.arms == ("random_training", "family_aware_training")
    assert (
        config.uv_lock_sha256
        == "0a273fe208a50476ef04af95f92e20dfa3ad71575e24c78d29435b9a3d607cc3"
    )
    assert config.torch_version == "2.13.0"
    assert config.output_paths == (
        "reports/week_02/bigram_sampling_v1.json",
        "reports/week_02/bigram_sampling_v1.md",
        "reports/week_02/bigram_sampling_v1.sha256",
    )


def test_sampling_stops_at_cap_when_eos_is_not_drawn() -> None:
    logits = torch.full((21, 21), -1000.0, dtype=torch.float32)
    logits[:, 0] = 1000.0
    sample = sample_neural_bigram(
        logits, base_seed=1, namespace="unit", sample_index=0, max_residues=3
    )
    assert sample["sequence"] == "AAA"
    assert sample["residue_length"] == 3
    assert sample["termination_reason"] == "max_residues"


def test_writer_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    paths = (
        tmp_path / "bigram_sampling_v1.json",
        tmp_path / "bigram_sampling_v1.md",
        tmp_path / "bigram_sampling_v1.sha256",
    )
    payload = _payload()
    write_evidence(paths, payload)
    assert all(path.is_file() for path in paths)
    with pytest.raises(ModelDataError, match="already exists"):
        write_evidence(paths, payload)


def test_writer_rolls_back_second_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = tuple(
        tmp_path / f"bigram_sampling_v1.{suffix}" for suffix in ("json", "md", "sha256")
    )
    real_link = sampling_io.os.link
    calls = 0

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic link failure")
        real_link(source, destination)

    monkeypatch.setattr(sampling_io.os, "link", fail_second)
    with pytest.raises(ModelDataError, match="could not install"):
        write_evidence(paths, _payload())
    assert not any(path.exists() for path in paths)


@pytest.mark.parametrize(
    "field, value",
    (("sequence", "X"), ("termination_reason", "max_residues")),
)
def test_validator_rejects_invalid_sample_without_candidate_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    paths = (
        tmp_path / "bigram_sampling_v1.json",
        tmp_path / "bigram_sampling_v1.md",
        tmp_path / "bigram_sampling_v1.sha256",
    )
    payload = _payload()
    config_path = tmp_path / "config.toml"
    config_path.write_text("synthetic\n", encoding="utf-8")
    import hashlib

    payload["sampling_configuration_sha256"] = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    payload["runtime"] = {"uv_lock_sha256": "c" * 64, "torch_version": "synthetic"}
    write_evidence(paths, payload)
    config = SimpleNamespace(
        contract_identifier="synthetic",
        arms=("random_training", "family_aware_training"),
        namespaces=("week2/sampling/random/v1", "week2/sampling/family-aware/v1"),
        max_residues=128,
        uv_lock_sha256="c" * 64,
        torch_version="synthetic",
    )
    plan = SamplingPlan(tmp_path, config_path, config, tmp_path / "candidate", paths)  # type: ignore[arg-type]
    monkeypatch.setattr(sampling_validation, "preflight", lambda _root: plan)
    monkeypatch.setattr(
        sampling_validation, "regenerate_payload", lambda _plan, _revision: payload
    )
    changed = json.loads(paths[0].read_text(encoding="utf-8"))
    changed["samples"][0][field] = value
    paths[0].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ModelDataError):
        sampling_validation.validate_sampling_diagnostic(tmp_path)


@pytest.mark.parametrize(
    "field, value",
    (("status", "failed"), ("network_requests_made", 1), ("failure_reason", "no")),
)
def test_source_record_validation_rejects_nonpassed_or_nonlocal_source(
    field: str, value: object
) -> None:
    config = load_sampling_config(CONFIG_PATH)
    candidate_plan = SimpleNamespace(
        training_config=SimpleNamespace(contract_identifier="training")
    )
    record = {
        "schema_version": 1,
        "scope": "week_02_bigram_model_candidate",
        "contract_identifier": "training",
        "candidate_id": config.candidate_id,
        "status": "passed",
        "hard_gates": {gate: True for gate in RUN_GATES},
        "training": {"base_seed": config.base_seed},
        "source_identity": {},
        "code_revision": config.candidate_code_revision,
        "arms": {},
        "runtime_seconds": 0.0,
        "network_requests_made": 0,
        "failure_reason": None,
    }
    artifacts = {}
    for arm in config.arms:
        for model in ("unigram", "count_bigram", "neural_bigram"):
            for format_name in ("json", "safetensors"):
                artifacts[f"{arm}__{model}.{format_name}"] = {
                    "arm": arm,
                    "model_type": model,
                    "format": format_name,
                    "byte_size": 1,
                    "sha256": "0" * 64,
                }
    for arm, _namespace, json_hash, safe_hash in neural_sources(config):
        artifacts[f"{arm}__neural_bigram.json"]["sha256"] = json_hash
        artifacts[f"{arm}__neural_bigram.safetensors"]["sha256"] = safe_hash
    registry = {
        "schema_version": 1,
        "scope": "week_02_bigram_model_candidate_registry",
        "candidate_id": config.candidate_id,
        "logical_model_count": 6,
        "serialization_file_count": 12,
        "artifacts": artifacts,
    }
    record[field] = value
    with pytest.raises(ModelDataError):
        validate_passed_source(record, registry, candidate_plan, config)  # type: ignore[arg-type]


def test_source_registry_validation_rejects_wrong_neural_entry() -> None:
    config = load_sampling_config(CONFIG_PATH)
    candidate_plan = SimpleNamespace(
        training_config=SimpleNamespace(contract_identifier="training")
    )
    record = {
        "schema_version": 1,
        "scope": "week_02_bigram_model_candidate",
        "contract_identifier": "training",
        "candidate_id": config.candidate_id,
        "status": "passed",
        "hard_gates": {gate: True for gate in RUN_GATES},
        "training": {"base_seed": config.base_seed},
        "source_identity": {},
        "code_revision": config.candidate_code_revision,
        "arms": {},
        "runtime_seconds": 0.0,
        "network_requests_made": 0,
        "failure_reason": None,
    }
    artifacts = {
        f"{arm}__{model}.{format_name}": {
            "arm": arm,
            "model_type": model,
            "format": format_name,
            "byte_size": 1,
            "sha256": "0" * 64,
        }
        for arm in config.arms
        for model in ("unigram", "count_bigram", "neural_bigram")
        for format_name in ("json", "safetensors")
    }
    for arm, _namespace, json_hash, safe_hash in neural_sources(config):
        artifacts[f"{arm}__neural_bigram.json"]["sha256"] = json_hash
        artifacts[f"{arm}__neural_bigram.safetensors"]["sha256"] = safe_hash
    registry = {
        "schema_version": 1,
        "scope": "week_02_bigram_model_candidate_registry",
        "candidate_id": config.candidate_id,
        "logical_model_count": 6,
        "serialization_file_count": 12,
        "artifacts": artifacts,
    }
    artifacts["random_training__neural_bigram.json"]["byte_size"] = 0
    with pytest.raises(ModelDataError, match="registry entry"):
        validate_passed_source(record, registry, candidate_plan, config)  # type: ignore[arg-type]


def test_payload_opens_only_two_neural_model_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_sampling_config(CONFIG_PATH)
    plan = SamplingPlan(
        tmp_path,
        CONFIG_PATH,
        config,
        tmp_path / "candidate",
        tuple(
            tmp_path / name for name in ("report.json", "report.md", "report.sha256")
        ),
    )
    logits = torch.zeros((21, 21), dtype=torch.float32)
    metadata = {
        "arm": "",
        "model_type": "neural_bigram",
        "code_revision": config.candidate_code_revision,
        "seed": config.base_seed,
        "context_roles": ["BOS", *"ACDEFGHIKLMNPQRSTVWY"],
        "target_roles": [*"ACDEFGHIKLMNPQRSTVWY", "EOS"],
    }
    opened = []

    def fake_loader(*, json_path: Path, safetensors_path: Path):
        opened.append((json_path.name, safetensors_path.name))
        arm = json_path.name.split("__", maxsplit=1)[0]
        return "neural_bigram", logits, {**metadata, "arm": arm}

    monkeypatch.setattr(sampling_publication, "load_model_artifacts", fake_loader)
    payload = publication_payload(plan, REVISION)
    assert len(opened) == 2
    assert {pair[0] for pair in opened} == {
        "random_training__neural_bigram.json",
        "family_aware_training__neural_bigram.json",
    }
    assert len(payload["samples"]) == 20


def test_execute_captures_clean_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_sampling_config(CONFIG_PATH)
    paths = tuple(
        tmp_path / f"bigram_sampling_v1.{suffix}" for suffix in ("json", "md", "sha256")
    )
    plan = SamplingPlan(tmp_path, CONFIG_PATH, config, tmp_path / "candidate", paths)
    monkeypatch.setattr(sampling_publication, "preflight", lambda _root: plan)
    monkeypatch.setattr(sampling_publication, "_clean_revision", lambda _root: REVISION)
    monkeypatch.setattr(
        sampling_publication,
        "_payload",
        lambda _plan, revision: {
            **_payload_for_writer(),
            "publication_code_revision": revision,
        },
    )
    execute_publication(tmp_path, plan)
    assert json.loads(paths[0].read_text())["publication_code_revision"] == REVISION


def _payload_for_writer() -> dict[str, object]:
    payload = _payload()
    payload["sampling_configuration_sha256"] = "synthetic"
    return payload


def test_no_flag_cli_does_not_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = ROOT / "scripts/publish_week2_bigram_sampling.py"
    spec = importlib.util.spec_from_file_location("sampling_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = object()
    monkeypatch.setattr(module, "preflight", lambda _root: plan)
    called = []
    monkeypatch.setattr(
        module, "execute_publication", lambda *_args: called.append(True)
    )
    monkeypatch.setattr(sys, "argv", [str(script)])
    assert module.main() == 0
    assert called == []
    monkeypatch.setattr(sys, "argv", [str(script), "--execute-publication"])
    assert module.main() == 0
    assert called == [True]


def test_sampling_modules_do_not_import_data_or_evaluation_loaders() -> None:
    modules = (
        "sampling.py",
        "sampling_contract.py",
        "sampling_io.py",
        "sampling_render.py",
        "sampling_source.py",
        "sampling_publication.py",
        "sampling_validation.py",
    )
    forbidden = (
        "load_collection",
        "ModelDataCollection",
        "evaluation_execution",
        "evaluate_",
    )
    assert all(
        not any(
            token in (ROOT / "src/protein_lm/bigram" / name).read_text(encoding="utf-8")
            for token in forbidden
        )
        for name in modules
    )
