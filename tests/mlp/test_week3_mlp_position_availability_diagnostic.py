"""Focused synthetic checks for the Week 3 position-availability diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ProteinSequence
from protein_lm.mlp.position_availability_diagnostic import (
    BIN_NAMES,
    BinMetrics,
    evaluate_position_availability,
    iter_position_availability_batches,
    overall_metrics,
    position_availability_bin,
)
from protein_lm.mlp.position_availability_diagnostic_config import (
    FinalRun,
    load_position_availability_diagnostic_config,
)
from protein_lm.mlp.position_availability_diagnostic_orchestration import (
    _results_payload,
    _validate_arm_result,
    _validate_paired_training_configs,
    _verify_source_artifact,
    _verify_readiness,
    preflight,
)
from protein_lm.mlp import (
    position_availability_diagnostic_orchestration as orchestration,
)
from protein_lm.data.model_data.loaders import ModelDataCollection


ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "experiments/week_03/mlp_position_availability_diagnostic_v1.toml"
SCRIPT_PATH = ROOT / "scripts/run_week3_mlp_position_availability_diagnostic.py"


def _protein(sequence: str, accession: str = "P00001") -> ProteinSequence:
    return ProteinSequence(
        accession,
        sequence,
        hashlib.sha256(sequence.encode()).hexdigest(),
        len(sequence),
        "synthetic",
        "UniRef50_SYNTHETIC",
    )


class _ConstantModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        return torch.zeros((contexts.shape[0], 21), device=contexts.device)


def test_exact_bin_boundaries_and_eos_are_assigned_once() -> None:
    assert [position_availability_bin(value) for value in (0, 10, 11, 19, 20)] == [
        BIN_NAMES[0],
        BIN_NAMES[0],
        BIN_NAMES[1],
        BIN_NAMES[1],
        BIN_NAMES[2],
    ]
    proteins = (_protein("A" * 20),)
    batches = tuple(
        iter_position_availability_batches(
            proteins,
            namespace="synthetic",
            base_seed=1,
            batch_size=64,
            context_length=2,
        )
    )
    assert batches[0].context_batch.targets[-1].item() == 20
    assert batches[0].prior_residue_counts[-1].item() == 20
    assert batches[0].prior_residue_counts.tolist() == list(range(21))
    metrics = evaluate_position_availability(_ConstantModel(), batches)
    assert [metrics[name].token_count for name in BIN_NAMES] == [11, 9, 1]
    assert overall_metrics(metrics).token_count == 21


def test_evaluation_restores_mode_and_conserves_nll_and_counts() -> None:
    model = _ConstantModel()
    model.train()
    batches = iter_position_availability_batches(
        (_protein("A" * 20),),
        namespace="synthetic",
        base_seed=1,
        batch_size=3,
        context_length=3,
    )
    metrics = evaluate_position_availability(model, batches)
    assert model.training is True
    overall = overall_metrics(metrics)
    assert overall.token_count == 21
    assert overall.correct_predictions == 20
    assert overall.nll_numerator > 0
    short = evaluate_position_availability(
        _ConstantModel(),
        iter_position_availability_batches(
            (_protein("ACD"),),
            namespace="synthetic",
            base_seed=1,
            batch_size=3,
            context_length=3,
        ),
    )
    with pytest.raises(ModelDataError, match="bins are incomplete"):
        overall_metrics(short)


def test_byte_pinned_preflight_and_safe_no_flag_cli() -> None:
    plan = preflight(ROOT, "safe-preflight")
    assert (
        plan.destination
        == ROOT
        / "data/processed/week_03/mlp_position_availability_diagnostic_runs/safe-preflight"
    )
    assert plan.config.bins == BIN_NAMES
    assert plan.config.frozen_category == "context20_materially_better"
    assert plan.config.frozen_comparison_scope.endswith("cannot_reopen_selection")
    spec = importlib.util.spec_from_file_location(
        "position_diagnostic_script", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main([]) == 0


def test_config_type_and_source_tamper_reject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "batch_size = 1024", 'batch_size = "1024"'
    )
    path = tmp_path / "bad.toml"
    path.write_text(changed, encoding="utf-8")
    monkeypatch.setattr(
        "protein_lm.mlp.position_availability_diagnostic_config.APPROVED_POSITION_AVAILABILITY_DIAGNOSTIC_CONFIG_SHA256",
        hashlib.sha256(changed.encode()).hexdigest(),
    )
    with pytest.raises(ModelDataError, match="batch_size must be an integer"):
        load_position_availability_diagnostic_config(path)

    status = tmp_path / "run_status.json"
    checkpoint = tmp_path / "checkpoint-100000000"
    checkpoint.mkdir()
    status_payload = {
        "status": "passed",
        "seed": 20260821,
        "run_id": "synthetic-source",
        "derived_code_revision": "a" * 40,
    }
    status.write_text(json.dumps(status_payload), encoding="utf-8")
    (checkpoint / "checkpoint.json").write_bytes(b"metadata")
    (checkpoint / "model.safetensors").write_bytes(b"tensors")
    run = FinalRun(
        20260821,
        "synthetic-source",
        hashlib.sha256(status.read_bytes()).hexdigest(),
        hashlib.sha256(b"metadata").hexdigest(),
        hashlib.sha256(b"tensors").hexdigest(),
        1.0,
        0.1,
        1.0,
        1,
    )
    _verify_source_artifact(status, checkpoint, run, "synthetic", "a" * 40)
    (checkpoint / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ModelDataError, match="bytes do not match"):
        _verify_source_artifact(status, checkpoint, run, "synthetic", "a" * 40)


def test_results_have_signed_differences_for_every_bin_and_overall() -> None:
    context = {name: BinMetrics(2, 4.0, 1) for name in BIN_NAMES}
    embedding = {name: BinMetrics(2, 5.0, 0) for name in BIN_NAMES}
    results = _results_payload(context, embedding)
    assert results["overall"]["embedding64_minus_context20_cross_entropy"] > 0
    assert results["overall"]["context20_minus_embedding64_accuracy"] > 0
    assert set(results["bins"]) == set(BIN_NAMES)


def test_paired_invariant_and_expected_overall_mismatch_reject() -> None:
    plan = preflight(ROOT, "pairing-guard")
    with pytest.raises(ModelDataError, match="paired model invariant"):
        _validate_paired_training_configs(
            plan.context20_training_config,
            replace(plan.embedding64_training_config, training_namespace="drift"),
        )
    config = replace(plan.config, native_validation_prediction_tokens=6)
    bins = {name: BinMetrics(2, 2.0, 1) for name in BIN_NAMES}
    expected = replace(
        plan.config.run("context20", 20260821),
        native_cross_entropy=1.0,
        native_accuracy=0.5,
        native_nll_numerator=6.0,
        native_correct_predictions=3,
    )
    _validate_arm_result(bins, expected, config)
    with pytest.raises(ModelDataError, match="overall metrics"):
        _validate_arm_result(
            bins,
            replace(expected, native_correct_predictions=2),
            config,
        )


def test_readiness_bytes_and_semantics_reject(tmp_path: Path) -> None:
    base_plan = preflight(ROOT, "readiness-guard")
    report = {
        "scope": "week_02_model_data_readiness",
        "candidate_status": "passed",
        "network_requests_made": 0,
        "collection_aggregates": {
            "family_aware_native_validation": {"prediction_tokens": 8, "records": 2}
        },
    }
    report_content = json.dumps(report, sort_keys=True).encode()
    report_path = tmp_path / "reports/week_02/readiness.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes(report_content)
    registry_path = tmp_path / "manifests/registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_content = json.dumps(
        {
            "readiness": {
                "relative_path": "reports/week_02/readiness.json",
                "sha256": hashlib.sha256(report_content).hexdigest(),
            }
        },
        sort_keys=True,
    ).encode()
    registry_path.write_bytes(registry_content)
    config = replace(
        base_plan.config,
        readiness_report_relative_path="reports/week_02/readiness.json",
        readiness_report_sha256=hashlib.sha256(report_content).hexdigest(),
        native_validation_prediction_tokens=8,
        native_validation_records=2,
    )
    training = replace(
        base_plan.context20_training_config,
        model_data_registry_relative_path="manifests/registry.json",
        model_data_registry_sha256=hashlib.sha256(registry_content).hexdigest(),
    )
    plan = replace(base_plan, config=config, context20_training_config=training)
    _verify_readiness(tmp_path, plan)
    report_path.write_bytes(report_content + b"tampered")
    with pytest.raises(ModelDataError, match="does not match approval"):
        _verify_readiness(tmp_path, plan)
    failed = {**report, "candidate_status": "failed"}
    failed_content = json.dumps(failed, sort_keys=True).encode()
    report_path.write_bytes(failed_content)
    semantic_config = replace(
        config, readiness_report_sha256=hashlib.sha256(failed_content).hexdigest()
    )
    semantic_registry = json.dumps(
        {
            "readiness": {
                "relative_path": "reports/week_02/readiness.json",
                "sha256": semantic_config.readiness_report_sha256,
            }
        },
        sort_keys=True,
    ).encode()
    registry_path.write_bytes(semantic_registry)
    semantic_plan = replace(
        plan,
        config=semantic_config,
        context20_training_config=replace(
            training,
            model_data_registry_sha256=hashlib.sha256(semantic_registry).hexdigest(),
        ),
    )
    with pytest.raises(ModelDataError, match="does not match approval"):
        _verify_readiness(tmp_path, semantic_plan)


def test_execution_rejects_a_redirected_plan_before_operational_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = preflight(ROOT, "canonical-guard")
    monkeypatch.setattr(orchestration, "preflight", lambda *_: plan)
    with pytest.raises(ModelDataError, match="execution plan differs"):
        orchestration.execute_diagnostic(
            root=ROOT,
            plan=replace(plan, destination=ROOT / "unexpected"),
            seed=20260821,
            device_name="cpu",
            code_revision="a" * 40,
        )


def test_execution_loads_only_native_validation_and_writes_failure_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_plan = preflight(ROOT, "synthetic-position-diagnostic")
    plan = replace(
        base_plan,
        config=replace(base_plan.config, native_validation_records=1),
        destination=tmp_path / "synthetic-position-diagnostic",
    )
    monkeypatch.setattr(orchestration, "preflight", lambda *_: plan)
    monkeypatch.setattr(orchestration, "_require_revision", lambda *_: "a" * 40)
    monkeypatch.setattr(orchestration, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(orchestration, "_verify_readiness", lambda *_: None)
    monkeypatch.setattr(orchestration, "_require_ignored", lambda *_: None)
    monkeypatch.setattr(orchestration, "_verify_source_artifact", lambda *_: None)
    monkeypatch.setattr(orchestration, "_target_order_digest", lambda *_: "b" * 64)

    loads: list[ModelDataCollection] = []

    def loader(_root: Path, collection: ModelDataCollection):
        loads.append(collection)
        return (_protein("ACDEFGHIJKLMNOPQRST".replace("J", "A").replace("O", "A")),)

    monkeypatch.setattr(
        orchestration,
        "_evaluate_arm",
        lambda *_: (_ for _ in ()).throw(ModelDataError("synthetic arm failure")),
    )
    with pytest.raises(ModelDataError, match="synthetic arm failure"):
        orchestration.execute_diagnostic(
            root=tmp_path,
            plan=plan,
            seed=20260821,
            device_name="cpu",
            loader=loader,
            code_revision="a" * 40,
        )
    assert loads == [ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION]
    status = json.loads((plan.destination / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert status["no_training"] is True
    assert status["training_predictions"] == 0
    assert status["optimizer_steps"] == 0
    assert status["backward_passes"] == 0
    assert status["sealed_test_accessed"] is False
    assert status["models_evaluated_sequentially"] is True
    assert not list(plan.destination.glob("checkpoint-*"))


def test_successful_execution_records_ordered_arms_and_frozen_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_plan = preflight(ROOT, "synthetic-position-success")
    plan = replace(
        base_plan,
        config=replace(base_plan.config, native_validation_records=1),
        destination=tmp_path / "synthetic-position-success",
    )
    monkeypatch.setattr(orchestration, "preflight", lambda *_: plan)
    monkeypatch.setattr(orchestration, "_require_revision", lambda *_: "a" * 40)
    monkeypatch.setattr(orchestration, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(orchestration, "_verify_readiness", lambda *_: None)
    monkeypatch.setattr(orchestration, "_require_ignored", lambda *_: None)
    monkeypatch.setattr(orchestration, "_verify_source_artifact", lambda *_: None)
    monkeypatch.setattr(orchestration, "_target_order_digest", lambda *_: "b" * 64)
    calls: list[str] = []
    bins = {name: BinMetrics(1, 1.0, 1) for name in BIN_NAMES}

    def evaluate(_plan, _run, arm, *_args):
        calls.append(arm)
        return bins

    monkeypatch.setattr(orchestration, "_evaluate_arm", evaluate)
    loads: list[ModelDataCollection] = []

    def loader(_root: Path, collection: ModelDataCollection):
        loads.append(collection)
        return (_protein("ACDEFGHIJKLMNOPQRST".replace("J", "A").replace("O", "A")),)

    destination = orchestration.execute_diagnostic(
        root=tmp_path,
        plan=plan,
        seed=20260821,
        device_name="cpu",
        loader=loader,
        code_revision="a" * 40,
    )
    status = json.loads((destination / "run_status.json").read_text())
    assert calls == ["context20", "embedding64"]
    assert loads == [ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION]
    assert status["status"] == "passed"
    assert status["results"] is not None
    assert status["frozen_comparison_provenance"] == {
        "scope": "provenance_only_diagnostic_cannot_reopen_selection",
        "context20_mean_native_cross_entropy": 2.863665856220289,
        "context20_sample_standard_deviation": 0.00001985865257320209,
        "embedding64_mean_native_cross_entropy": 2.8708249214089068,
        "embedding64_sample_standard_deviation": 0.000007648534920387316,
        "embedding64_minus_context20_mean_native_cross_entropy": 0.0071590651886177525,
        "material_gap": 0.001,
        "frozen_category": "context20_materially_better",
        "selection_reopened": False,
    }
    assert not list(destination.glob("checkpoint-*"))
