"""Synthetic contract tests for the local Week 2 bigram candidate entrypoints."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from protein_lm.bigram import candidate
from protein_lm.bigram import candidate_contract
from protein_lm.bigram.candidate import AuditedArm, CandidatePlan, create_candidate
from protein_lm.bigram.candidate_fitting import _ObservedStream
from protein_lm.bigram.candidate_validation import validate_candidate
from protein_lm.bigram.config import load_config as load_stream_config
from protein_lm.bigram.serialization import load_model_artifacts, write_model_artifacts
from protein_lm.bigram.stream import audit_stream, iter_pair_batches
from protein_lm.bigram.training import TrainingSettings
from protein_lm.bigram.training_config import load_training_config
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ModelDataCollection, ProteinSequence


ROOT = Path(__file__).parents[2]
REVISION = "a" * 40


def protein(sequence: str, accession: str) -> ProteinSequence:
    return ProteinSequence(
        primary_accession=accession,
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        biological_length=len(sequence),
        length_bucket="synthetic",
        uniref50_group=f"group-{accession}",
    )


def synthetic_plan(tmp_path: Path) -> tuple[CandidatePlan, dict[str, tuple[ProteinSequence, ...]]]:
    training = replace(
        load_training_config(ROOT / "experiments/week_02/bigram_training_v1.toml"),
        batch_size=2,
        prediction_pair_budget=4,
        full_batches=1,
        final_partial_batch_pairs=2,
        total_optimizer_steps=2,
    )
    stream = load_stream_config(ROOT / training.stream_config_relative_path)
    collections = {
        "random_training": (protein("A", "R1"), protein("C", "R2")),
        "family_aware_training": (protein("D", "F1"), protein("E", "F2")),
    }
    arms = []
    for collection, namespace in zip(
        stream.training_collections, stream.training_namespaces, strict=True
    ):
        audit = audit_stream(
            collections[collection],
            namespace=namespace,
            base_seed=training.base_seed,
            pair_budget=training.prediction_pair_budget,
            hash_domain=stream.stream_hash_domain,
            batch_size=training.batch_size,
        )
        arms.append(
            AuditedArm(
                collection=collection,
                namespace=namespace,
                stream_sha256=audit.stream_sha256,
                context_counts=audit.context_counts,
                target_counts=audit.target_counts,
                proteins_started=audit.proteins_started,
                proteins_completed=audit.proteins_completed,
                final_protein_partial=audit.final_protein_partial,
            )
        )
    plan = CandidatePlan(
        candidate_id="synthetic-run-1",
        destination=tmp_path / "data/processed/week_02/bigram_model_candidates/synthetic-run-1",
        training_config=training,
        training_config_sha256="b" * 64,
        stream_config=stream,
        stream_report_sha256="c" * 64,
        source_identity={
            "training_config": "b" * 64,
            "stream_config": training.stream_config_sha256,
            "stream_report": "c" * 64,
            "model_data_registry": "d" * 64,
        },
        arms=tuple(arms),
    )
    return plan, collections


def test_observed_fitting_stream_matches_audit_stream_hash_header() -> None:
    proteins = (protein("A", "R1"), protein("C", "R2"))
    namespace = "synthetic/fitting-header/v1"
    domain = "synthetic/hash-domain/v1"
    audit = audit_stream(
        proteins,
        namespace=namespace,
        base_seed=7,
        pair_budget=4,
        hash_domain=domain,
        batch_size=2,
    )
    observed = _ObservedStream(namespace=namespace, hash_domain=domain, base_seed=7)
    for batch in iter_pair_batches(
        proteins, namespace=namespace, base_seed=7, pair_budget=4, batch_size=2
    ):
        observed.update(batch)
    assert observed.finish() == audit


def loader_for(collections, calls: list[ModelDataCollection]):
    def load(_root: Path, collection: ModelDataCollection):
        calls.append(collection)
        return collections[collection.value]

    return load


def test_one_pass_six_model_candidate_is_aggregate_only_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, collections = synthetic_plan(tmp_path)
    calls: list[ModelDataCollection] = []
    monkeypatch.setattr(candidate, "_require_ignored", lambda _root, _path: None)
    destination = create_candidate(
        root=tmp_path,
        plan=plan,
        loader=loader_for(collections, calls),
        code_revision=REVISION,
        settings=TrainingSettings(batch_size=2, prediction_pair_budget=4),
    )
    assert calls == [
        ModelDataCollection.RANDOM_TRAINING,
        ModelDataCollection.FAMILY_AWARE_TRAINING,
    ]
    record = json.loads((destination / "run_record.json").read_text())
    assert record["status"] == "passed"
    assert record["network_requests_made"] == 0
    assert set(record["arms"]) == {"random_training", "family_aware_training"}
    assert "R1" not in json.dumps(record)
    assert "group-" not in json.dumps(record)
    assert validate_candidate(destination, plan)["logical_model_count"] == 6
    registry = json.loads((destination / "candidate_registry.json").read_text())
    assert registry["logical_model_count"] == 6
    assert registry["serialization_file_count"] == 12
    assert len(registry["artifacts"]) == 12


def test_stream_mismatch_preserves_failed_candidate_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, collections = synthetic_plan(tmp_path)
    bad_arm = replace(plan.arms[0], stream_sha256="e" * 64)
    plan = replace(plan, arms=(bad_arm, plan.arms[1]))
    monkeypatch.setattr(candidate, "_require_ignored", lambda _root, _path: None)
    with pytest.raises(ModelDataError, match="does not equal"):
        create_candidate(
            root=tmp_path,
            plan=plan,
            loader=loader_for(collections, []),
            code_revision=REVISION,
            settings=TrainingSettings(batch_size=2, prediction_pair_budget=4),
        )
    failed = json.loads((plan.destination / "run_record.json").read_text())
    assert failed["status"] == "failed"
    assert "does not equal" in failed["failure_reason"]
    with pytest.raises(ModelDataError, match="already exists"):
        create_candidate(
            root=tmp_path,
            plan=plan,
            loader=loader_for(collections, []),
            code_revision=REVISION,
            settings=TrainingSettings(batch_size=2, prediction_pair_budget=4),
        )


@pytest.mark.parametrize(
    "settings",
    (
        TrainingSettings(batch_size=1, prediction_pair_budget=4),
        TrainingSettings(batch_size=2, prediction_pair_budget=3),
        TrainingSettings(batch_size=2, prediction_pair_budget=4, learning_rate=0.5),
        TrainingSettings(batch_size=2, prediction_pair_budget=4, momentum=0.1),
        TrainingSettings(batch_size=2, prediction_pair_budget=4, weight_decay=0.1),
    ),
)
def test_injected_settings_must_match_every_frozen_optimizer_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: TrainingSettings
) -> None:
    plan, collections = synthetic_plan(tmp_path)
    monkeypatch.setattr(candidate, "_require_ignored", lambda _root, _path: None)
    with pytest.raises(ModelDataError, match="settings disagree"):
        create_candidate(
            root=tmp_path,
            plan=plan,
            loader=loader_for(collections, []),
            code_revision=REVISION,
            settings=settings,
        )
    assert not plan.destination.exists()


@pytest.mark.parametrize("tamper", ("remove", "modify"))
def test_validator_rejects_missing_or_tampered_artifact_without_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    plan, collections = synthetic_plan(tmp_path)
    monkeypatch.setattr(candidate, "_require_ignored", lambda _root, _path: None)
    destination = create_candidate(
        root=tmp_path,
        plan=plan,
        loader=loader_for(collections, []),
        code_revision=REVISION,
        settings=TrainingSettings(batch_size=2, prediction_pair_budget=4),
    )
    artifact = destination / "random_training__unigram.json"
    if tamper == "remove":
        artifact.unlink()
    else:
        artifact.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ModelDataError):
        validate_candidate(destination, plan)


def test_validator_rejects_count_bigram_with_wrong_audited_margins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, collections = synthetic_plan(tmp_path)
    monkeypatch.setattr(candidate, "_require_ignored", lambda _root, _path: None)
    destination = create_candidate(
        root=tmp_path,
        plan=plan,
        loader=loader_for(collections, []),
        code_revision=REVISION,
        settings=TrainingSettings(batch_size=2, prediction_pair_budget=4),
    )
    json_path = destination / "random_training__count_bigram.json"
    safe_path = destination / "random_training__count_bigram.safetensors"
    model_type, tensor, metadata = load_model_artifacts(
        json_path=json_path, safetensors_path=safe_path
    )
    location = torch.nonzero(tensor, as_tuple=False)[0]
    changed = tensor.clone()
    changed[location[0], location[1]] -= 1
    changed[location[0], (location[1] + 1) % 21] += 1
    json_path.unlink()
    safe_path.unlink()
    write_model_artifacts(
        json_path=json_path,
        safetensors_path=safe_path,
        model_type=model_type,
        tensor=changed,
        metadata=metadata,
    )
    _refresh_registry(destination)
    with pytest.raises(ModelDataError, match="margins"):
        validate_candidate(destination, plan)


@pytest.mark.parametrize(
    "tamper",
    (
        "empty_run_gates",
        "extra_run_gate",
        "empty_arm_gates",
        "extra_arm_gate",
        "bad_runtime",
        "nonfinite_runtime",
    ),
)
def test_validator_rejects_non_strict_gates_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    plan, collections = synthetic_plan(tmp_path)
    monkeypatch.setattr(candidate, "_require_ignored", lambda _root, _path: None)
    destination = create_candidate(
        root=tmp_path,
        plan=plan,
        loader=loader_for(collections, []),
        code_revision=REVISION,
        settings=TrainingSettings(batch_size=2, prediction_pair_budget=4),
    )
    record_path = destination / "run_record.json"
    record = json.loads(record_path.read_text())
    if tamper == "empty_run_gates":
        record["hard_gates"] = {}
    elif tamper == "extra_run_gate":
        record["hard_gates"]["unexpected"] = True
    elif tamper == "empty_arm_gates":
        record["arms"]["random_training"]["hard_gates"] = {}
    elif tamper == "extra_arm_gate":
        record["arms"]["random_training"]["hard_gates"]["unexpected"] = True
    elif tamper == "nonfinite_runtime":
        record["runtime_seconds"] = float("nan")
    else:
        record["runtime_seconds"] = True
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ModelDataError):
        validate_candidate(destination, plan)


def test_preflight_converts_missing_audited_report_to_domain_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = load_training_config(ROOT / "experiments/week_02/bigram_training_v1.toml")
    stream = load_stream_config(ROOT / training.stream_config_relative_path)
    stream_path = tmp_path / training.stream_config_relative_path
    stream_path.parent.mkdir(parents=True)
    stream_path.write_bytes((ROOT / training.stream_config_relative_path).read_bytes())
    monkeypatch.setattr(candidate_contract, "load_training_config", lambda _path: training)
    monkeypatch.setattr(candidate_contract, "load_stream_config", lambda _path: stream)
    with pytest.raises(ModelDataError, match="report is unavailable"):
        candidate_contract.preflight(tmp_path, "synthetic-run-1")


def test_preflight_rejects_report_rewritten_with_matching_companion_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = load_training_config(ROOT / "experiments/week_02/bigram_training_v1.toml")
    stream = load_stream_config(ROOT / training.stream_config_relative_path)
    stream_path = tmp_path / training.stream_config_relative_path
    stream_path.parent.mkdir(parents=True)
    stream_path.write_bytes((ROOT / training.stream_config_relative_path).read_bytes())
    report_path = tmp_path / training.stream_report_relative_path
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes((ROOT / training.stream_report_relative_path).read_bytes() + b"\n")
    report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    checksum_path = tmp_path / stream.report_sha256_relative_path
    checksum_path.write_text(f"{report_digest}  {report_path.name}\n", encoding="utf-8")
    monkeypatch.setattr(candidate_contract, "load_training_config", lambda _path: training)
    monkeypatch.setattr(candidate_contract, "load_stream_config", lambda _path: stream)
    with pytest.raises(ModelDataError, match="bytes do not match approval"):
        candidate_contract.preflight(tmp_path, "synthetic-run-1")


def test_train_cli_no_flag_does_not_execute_loader_or_create_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script_module("train_week2_bigrams")
    calls = []
    monkeypatch.setattr(module, "preflight", lambda root, ident: calls.append((root, ident)) or _cli_plan())
    monkeypatch.setattr(sys, "argv", ["train_week2_bigrams.py"])
    assert module.main() == 0
    assert calls and "execution requires" in capsys.readouterr().out


def _cli_plan():
    plan, _ = synthetic_plan(Path("/tmp"))
    return plan


def _script_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refresh_registry(destination: Path) -> None:
    path = destination / "candidate_registry.json"
    registry = json.loads(path.read_text())
    for name, entry in registry["artifacts"].items():
        content = (destination / name).read_bytes()
        entry["byte_size"] = len(content)
        entry["sha256"] = hashlib.sha256(content).hexdigest()
    path.write_text(json.dumps(registry), encoding="utf-8")
