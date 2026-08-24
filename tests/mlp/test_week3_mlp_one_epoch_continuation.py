"""Synthetic checks for the Week 3 first-epoch continuation diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ModelDataCollection, ProteinSequence
from protein_lm.mlp import one_epoch_orchestration
from protein_lm.mlp.checkpoint import save_checkpoint
from protein_lm.mlp.config import load_config
from protein_lm.mlp.model import ContextMLP, resolve_device
from protein_lm.mlp.one_epoch_config import ParentPin, load_one_epoch_config
from protein_lm.mlp.one_epoch_orchestration import (
    OneEpochContinuationPlan,
    _validate_native_metric,
    _verify_readiness_aggregate,
    execute_continuation,
    preflight,
)
from protein_lm.mlp.one_epoch_training import train_continuation_batch
from protein_lm.mlp.stream import iter_context_batches
from protein_lm.mlp.training import TrainingState, new_optimizer, train_batch
from protein_lm.mlp.metrics import NativeMetrics


ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "experiments/week_03/mlp_one_epoch_continuation_v1.toml"
BASE_CONFIG_PATH = ROOT / "experiments/week_03/mlp_training_v1.toml"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_week3_mlp_one_epoch_continuation",
    ROOT / "scripts/run_week3_mlp_one_epoch_continuation.py",
)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
SCRIPT = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT)


def protein(sequence: str, accession: str) -> ProteinSequence:
    return ProteinSequence(
        accession,
        sequence,
        hashlib.sha256(sequence.encode()).hexdigest(),
        len(sequence),
        "synthetic",
        "UniRef50_SYNTHETIC",
    )


def _synthetic_plan(
    tmp_path: Path,
) -> tuple[OneEpochContinuationPlan, tuple[ProteinSequence, ...]]:
    base = replace(
        load_config(BASE_CONFIG_PATH),
        prediction_budget=8,
        batch_size=4,
        milestone_predictions=(8,),
        checkpoint_predictions=(4, 8),
        learning_rate_boundary_predictions=4,
    )
    proteins = (protein("ACD", "P00001"), protein("ACE", "P00002"))
    parent_model = ContextMLP(base, 20260821, resolve_device("cpu"))
    optimizer = new_optimizer(parent_model, base)
    parent_state = TrainingState()
    first_batch = next(
        iter_context_batches(
            proteins,
            namespace=base.training_namespace,
            base_seed=base.stream_base_seed,
            prediction_budget=8,
            batch_size=4,
            event_predictions=(8,),
        )
    )
    train_batch(parent_model, optimizer, first_batch, parent_state, base)
    parent_path = (
        tmp_path / base.output_relative_root / "synthetic-parent" / "checkpoint-4"
    )
    save_checkpoint(
        parent_path,
        model=parent_model,
        optimizer=optimizer,
        state=parent_state,
        config=base,
        config_path=BASE_CONFIG_PATH,
        seed=20260821,
        run_id="synthetic-parent",
        device_name="cpu",
        code_revision="d" * 40,
    )
    pin = ParentPin(
        seed=20260821,
        run_id="synthetic-parent",
        metadata_sha256=hashlib.sha256(
            (parent_path / "checkpoint.json").read_bytes()
        ).hexdigest(),
        tensor_sha256=hashlib.sha256(
            (parent_path / "model.safetensors").read_bytes()
        ).hexdigest(),
    )
    config = replace(
        load_one_epoch_config(CONFIG_PATH),
        parent_code_revision="d" * 40,
        training_prediction_tokens=8,
        training_records=2,
        native_validation_prediction_tokens=8,
        native_validation_records=2,
        parent_prediction_position=4,
        parent_optimizer_steps=1,
        parent_cursor_prediction_index=parent_state.cursor.prediction_index,
        parent_cursor_protein_index=parent_state.cursor.protein_index,
        parent_cursor_within_protein_target_offset=parent_state.cursor.within_protein_target_offset,
        final_prediction_position=8,
        final_optimizer_steps=2,
        continuation_optimizer_updates=1,
        batch_size=4,
        milestone_predictions=(8,),
        output_relative_root="data/processed/week_03/synthetic_one_epoch_runs",
        parent_pins=(pin,),
    )
    return (
        OneEpochContinuationPlan(
            config=config,
            config_path=CONFIG_PATH,
            base_config=base,
            base_config_path=BASE_CONFIG_PATH,
            run_id="synthetic-one-epoch",
            destination=tmp_path / config.output_relative_root / "synthetic-one-epoch",
        ),
        proteins,
    )


def test_config_pins_types_values_and_exact_arithmetic(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_one_epoch_config(CONFIG_PATH)
    assert config.continuation_predictions == 71_329_454
    assert config.final_partial_batch_predictions == 686
    assert config.milestone_predictions == (124_999_936, 149_999_872, 171_329_454)
    assert (
        config.parent_optimizer_steps + config.continuation_optimizer_updates == 167_318
    )
    assert config.parent_pin(20260823).run_id == "week3-mlp-seed-20260823-cpu"
    assert (
        preflight(ROOT, "synthetic-one-epoch").destination.name == "synthetic-one-epoch"
    )

    text = CONFIG_PATH.read_text(encoding="utf-8")
    typed = tmp_path / "typed.toml"
    altered = text.replace("batch_size = 1024", 'batch_size = "1024"')
    typed.write_text(altered, encoding="utf-8")
    monkeypatch.setattr(
        "protein_lm.mlp.one_epoch_config.APPROVED_ONE_EPOCH_CONFIG_SHA256",
        hashlib.sha256(altered.encode()).hexdigest(),
    )
    with pytest.raises(ModelDataError, match="batch_size must be an integer"):
        load_one_epoch_config(typed)

    value = tmp_path / "value.toml"
    changed = text.replace("fixed_learning_rate = 0.01", "fixed_learning_rate = 0.02")
    value.write_text(changed, encoding="utf-8")
    monkeypatch.setattr(
        "protein_lm.mlp.one_epoch_config.APPROVED_ONE_EPOCH_CONFIG_SHA256",
        hashlib.sha256(changed.encode()).hexdigest(),
    )
    with pytest.raises(ModelDataError, match="values are not approved"):
        load_one_epoch_config(value)


def test_readiness_aggregate_and_native_metric_guards_are_synthetic(
    tmp_path: Path,
) -> None:
    base = load_config(BASE_CONFIG_PATH)
    report_path = tmp_path / "reports/week_02/readiness.json"
    report_path.parent.mkdir(parents=True)
    payload = {
        "scope": "week_02_model_data_readiness",
        "candidate_status": "passed",
        "network_requests_made": 0,
        "collection_aggregates": {
            "family_aware_training": {"prediction_tokens": 8, "records": 2},
            "family_aware_native_validation": {
                "prediction_tokens": 3,
                "records": 1,
            },
        },
    }
    content = json.dumps(payload, sort_keys=True).encode()
    report_path.write_bytes(content)
    config = replace(
        load_one_epoch_config(CONFIG_PATH),
        readiness_report_relative_path="reports/week_02/readiness.json",
        readiness_report_sha256=hashlib.sha256(content).hexdigest(),
        training_prediction_tokens=8,
        training_records=2,
        native_validation_prediction_tokens=3,
        native_validation_records=1,
    )
    registry_path = tmp_path / base.model_data_registry_relative_path
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "readiness": {
                    "relative_path": config.readiness_report_relative_path,
                    "sha256": config.readiness_report_sha256,
                }
            }
        ),
        encoding="utf-8",
    )
    _verify_readiness_aggregate(tmp_path, config, base)

    report_path.write_bytes(content + b"tamper")
    with pytest.raises(ModelDataError, match="bytes do not match"):
        _verify_readiness_aggregate(tmp_path, config, base)

    mismatch = {
        **payload,
        "collection_aggregates": {
            **payload["collection_aggregates"],
            "family_aware_native_validation": {
                "prediction_tokens": 4,
                "records": 1,
            },
        },
    }
    mismatch_content = json.dumps(mismatch, sort_keys=True).encode()
    mismatch_config = replace(
        config,
        readiness_report_sha256=hashlib.sha256(mismatch_content).hexdigest(),
    )
    report_path.write_bytes(mismatch_content)
    registry_path.write_text(
        json.dumps(
            {
                "readiness": {
                    "relative_path": mismatch_config.readiness_report_relative_path,
                    "sha256": mismatch_config.readiness_report_sha256,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ModelDataError, match="does not match approval"):
        _verify_readiness_aggregate(tmp_path, mismatch_config, base)

    _validate_native_metric(NativeMetrics(3, 4.0, 1), config)
    with pytest.raises(ModelDataError, match="token count"):
        _validate_native_metric(NativeMetrics(2, 4.0, 1), config)


def test_fixed_lr_matches_primary_mechanics_and_exact_stream(tmp_path: Path) -> None:
    plan, proteins = _synthetic_plan(tmp_path)
    parent_path = (
        tmp_path
        / plan.base_config.output_relative_root
        / "synthetic-parent"
        / "checkpoint-4"
    )
    from protein_lm.mlp.checkpoint import load_checkpoint

    primary_model = ContextMLP(plan.base_config, 20260821, resolve_device("cpu"))
    continuation_model = ContextMLP(plan.base_config, 20260821, resolve_device("cpu"))
    primary_optimizer = new_optimizer(primary_model, plan.base_config)
    continuation_optimizer = new_optimizer(continuation_model, plan.base_config)
    primary_state = load_checkpoint(
        parent_path,
        model=primary_model,
        optimizer=primary_optimizer,
        config=plan.base_config,
        config_path=BASE_CONFIG_PATH,
        seed=20260821,
        run_id="synthetic-parent",
        device_name="cpu",
        code_revision="d" * 40,
    )
    continuation_state = load_checkpoint(
        parent_path,
        model=continuation_model,
        optimizer=continuation_optimizer,
        config=plan.base_config,
        config_path=BASE_CONFIG_PATH,
        seed=20260821,
        run_id="synthetic-parent",
        device_name="cpu",
        code_revision="d" * 40,
    )
    primary_batch = next(
        iter_context_batches(
            proteins,
            namespace=plan.base_config.training_namespace,
            base_seed=plan.base_config.stream_base_seed,
            prediction_budget=8,
            batch_size=4,
            event_predictions=(8,),
            cursor=primary_state.cursor,
        )
    )
    continuation_batch = next(
        iter_context_batches(
            proteins,
            namespace=plan.base_config.training_namespace,
            base_seed=plan.base_config.stream_base_seed,
            prediction_budget=8,
            batch_size=4,
            event_predictions=(8,),
            cursor=continuation_state.cursor,
        )
    )
    assert torch.equal(primary_batch.contexts, continuation_batch.contexts)
    assert torch.equal(primary_batch.targets, continuation_batch.targets)
    train_batch(
        primary_model, primary_optimizer, primary_batch, primary_state, plan.base_config
    )
    continuation_state.training_loss_numerator = 0.0
    train_continuation_batch(
        continuation_model,
        continuation_optimizer,
        continuation_batch,
        continuation_state,
        plan.config,
    )
    assert primary_optimizer.param_groups[0]["lr"] == 0.01
    assert continuation_optimizer.param_groups[0]["lr"] == 0.01
    assert all(
        torch.equal(primary, continuation)
        for primary, continuation in zip(
            primary_model.parameters(), continuation_model.parameters(), strict=True
        )
    )
    assert continuation_state.cursor.prediction_index == 8
    assert continuation_state.cursor.protein_index == 2
    assert continuation_state.cursor.within_protein_target_offset == 0


def test_synthetic_execution_is_atomic_nonresumable_and_collection_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, proteins = _synthetic_plan(tmp_path)
    parent_path = (
        tmp_path
        / plan.base_config.output_relative_root
        / "synthetic-parent"
        / "checkpoint-4"
    )
    parent_before = {
        item.name: item.read_bytes()
        for item in (parent_path / "checkpoint.json", parent_path / "model.safetensors")
    }
    monkeypatch.setattr(one_epoch_orchestration, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(
        one_epoch_orchestration, "_verify_readiness_aggregate", lambda *_: None
    )
    monkeypatch.setattr(one_epoch_orchestration, "_require_ignored", lambda *_: None)
    loaded: list[ModelDataCollection] = []

    def loader(_root: Path, collection: ModelDataCollection):
        loaded.append(collection)
        assert collection in {
            ModelDataCollection.FAMILY_AWARE_TRAINING,
            ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION,
        }
        return proteins

    milestones: list[int] = []
    destination = execute_continuation(
        root=tmp_path,
        plan=plan,
        seed=20260821,
        device_name="cpu",
        loader=loader,
        code_revision="a" * 40,
        progress_callback=lambda event, payload: (
            milestones.append(payload["prediction_position"])
            if event == "milestone"
            else None
        ),
    )
    assert destination == plan.destination
    assert loaded == [
        ModelDataCollection.FAMILY_AWARE_TRAINING,
        ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION,
    ]
    assert milestones == [8]
    assert parent_before == {
        item.name: item.read_bytes()
        for item in (parent_path / "checkpoint.json", parent_path / "model.safetensors")
    }
    status = json.loads((destination / "run_status.json").read_text())
    assert status["status"] == "passed"
    assert status["network_requests_made"] == 0
    assert status["continuation_online_training"]["token_count"] == 4
    assert status["native_validation_milestones"][0]["prediction_position"] == 8
    assert (
        status["final_model_native_validation"]
        == status["native_validation_milestones"][0]
    )
    tensors = load_file(str(destination / "final_model.safetensors"), device="cpu")
    assert tensors
    assert (
        status["final_model_sha256"]
        == hashlib.sha256(
            (destination / "final_model.safetensors").read_bytes()
        ).hexdigest()
    )
    with pytest.raises(ModelDataError, match="cannot resume"):
        execute_continuation(
            root=tmp_path,
            plan=plan,
            seed=20260821,
            device_name="cpu",
            loader=loader,
            code_revision="a" * 40,
        )


def test_rejects_parent_tampering_cpu_seed_and_default_preflight_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plan, proteins = _synthetic_plan(tmp_path)
    parent_tensor = (
        tmp_path
        / plan.base_config.output_relative_root
        / "synthetic-parent"
        / "checkpoint-4"
        / "model.safetensors"
    )
    parent_tensor.write_bytes(parent_tensor.read_bytes() + b"tamper")
    monkeypatch.setattr(one_epoch_orchestration, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(
        one_epoch_orchestration, "_verify_readiness_aggregate", lambda *_: None
    )
    monkeypatch.setattr(one_epoch_orchestration, "_require_ignored", lambda *_: None)
    with pytest.raises(ModelDataError, match="bytes do not match"):
        execute_continuation(
            root=tmp_path,
            plan=plan,
            seed=20260821,
            device_name="cpu",
            loader=lambda *_: proteins,
            code_revision="a" * 40,
        )
    assert (
        json.loads((plan.destination / "run_status.json").read_text())["status"]
        == "failed"
    )

    metadata_plan, metadata_proteins = _synthetic_plan(tmp_path / "metadata")
    metadata_path = (
        tmp_path
        / "metadata"
        / metadata_plan.base_config.output_relative_root
        / "synthetic-parent"
        / "checkpoint-4"
        / "checkpoint.json"
    )
    metadata_path.write_bytes(metadata_path.read_bytes() + b"tamper")
    with pytest.raises(ModelDataError, match="bytes do not match"):
        execute_continuation(
            root=tmp_path / "metadata",
            plan=metadata_plan,
            seed=20260821,
            device_name="cpu",
            loader=lambda *_: metadata_proteins,
            code_revision="a" * 40,
        )

    cpu_plan, cpu_proteins = _synthetic_plan(tmp_path / "cpu")
    with pytest.raises(ModelDataError, match="CPU"):
        execute_continuation(
            root=tmp_path / "cpu",
            plan=cpu_plan,
            seed=20260821,
            device_name="mps",
            loader=lambda *_: cpu_proteins,
            code_revision="a" * 40,
        )
    with pytest.raises(ModelDataError, match="parent pin"):
        execute_continuation(
            root=tmp_path / "cpu",
            plan=cpu_plan,
            seed=20260822,
            device_name="cpu",
            loader=lambda *_: cpu_proteins,
            code_revision="a" * 40,
        )

    revision_plan, revision_proteins = _synthetic_plan(tmp_path / "revision")
    with pytest.raises(ModelDataError, match="injected code revision"):
        execute_continuation(
            root=tmp_path / "revision",
            plan=revision_plan,
            seed=20260821,
            device_name="cpu",
            loader=lambda *_: revision_proteins,
            code_revision="not-a-revision",
        )

    state_plan, state_proteins = _synthetic_plan(tmp_path / "state")
    state_plan = replace(
        state_plan,
        config=replace(state_plan.config, parent_optimizer_steps=2),
    )
    with pytest.raises(ModelDataError, match="parent identity"):
        execute_continuation(
            root=tmp_path / "state",
            plan=state_plan,
            seed=20260821,
            device_name="cpu",
            loader=lambda *_: state_proteins,
            code_revision="a" * 40,
        )

    def prohibited(*_args, **_kwargs):
        raise AssertionError("default preflight must not inspect execution state")

    monkeypatch.setattr(SCRIPT, "execute_continuation", prohibited)
    for name in (
        "load_checkpoint",
        "load_collection",
        "resolve_device",
        "_require_ignored",
        "_require_revision",
        "_verify_source_pins",
        "_verify_readiness_aggregate",
    ):
        monkeypatch.setattr(one_epoch_orchestration, name, prohibited)
    assert SCRIPT.main([]) == 0
    assert "does not access a checkpoint" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        SCRIPT.parse_args(["--execute-continuation"])

    def completed(**kwargs):
        callback = kwargs["progress_callback"]
        callback(
            "milestone",
            {
                "prediction_position": 124_999_936,
                "cross_entropy": 1.25,
                "accuracy": 0.5,
            },
        )
        callback(
            "completed",
            {
                "seed": 20260821,
                "final_prediction_position": 171_329_454,
                "final_cross_entropy": 1.2,
                "final_accuracy": 0.55,
                "runtime_seconds": 2.0,
                "continuation_optimizer_updates": 69_658,
            },
        )
        return Path("/tmp/synthetic-one-epoch")

    monkeypatch.setattr(SCRIPT, "execute_continuation", completed)
    assert (
        SCRIPT.main(
            [
                "--execute-continuation",
                "--run-id",
                "synthetic-one-epoch",
                "--seed",
                "20260821",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "milestone predictions=124999936 native_CE=1.250000" in output
    assert "completed seed=20260821 predictions=171329454" in output
