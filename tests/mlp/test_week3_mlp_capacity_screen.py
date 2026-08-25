"""Focused synthetic coverage for the 25M Week 3 capacity allocation screen."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ModelDataCollection, ProteinSequence
from protein_lm.mlp import capacity_screen_orchestration as screen_orchestration
from protein_lm.mlp.capacity_screen_config import (
    CapacityArm,
    load_capacity_screen_config,
)
from protein_lm.mlp.capacity_screen_orchestration import (
    CapacityScreenPlan,
    _arm_training_config,
    _completion_payload,
    _validate_final_state,
    _validate_native_metric,
    _verify_readiness_aggregate,
    execute_screen,
    preflight,
)
from protein_lm.mlp.checkpoint import load_checkpoint
from protein_lm.mlp.config import load_config
from protein_lm.mlp.metrics import NativeMetrics
from protein_lm.mlp.model import ContextMLP, resolve_device
from protein_lm.mlp.stream import StreamCursor, iter_context_batches
from protein_lm.mlp.training import TrainingState, new_optimizer


ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "experiments/week_03/mlp_capacity_screen_v1.toml"
BASE_CONFIG_PATH = ROOT / "experiments/week_03/mlp_training_v1.toml"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_week3_mlp_capacity_screen", ROOT / "scripts/run_week3_mlp_capacity_screen.py"
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


def test_config_is_exact_and_arms_change_one_axis_only(
    tmp_path: Path, monkeypatch
) -> None:
    config = load_capacity_screen_config(CONFIG_PATH)
    assert config.event_predictions == (1_000_000, 5_000_000, 10_000_000, 25_000_000)
    assert config.expected_optimizer_steps(25_000_000) == 24_416
    final_cursor = StreamCursor(25_000_000, 67_233, 99)
    _validate_final_state(
        TrainingState(25_000_000, 24_416, 0.0, final_cursor),
        [{"prediction_position": position} for position in config.event_predictions],
        [{"prediction_position": position} for position in config.event_predictions],
        config,
    )
    assert [arm.parameter_count for arm in config.arms] == [530_293, 530_965, 547_893]
    assert config.qualifying_mean_native_cross_entropy_at_most == 2.870545191729816
    base = load_config(BASE_CONFIG_PATH)
    for arm in config.arms:
        derived = _arm_training_config(base, config, arm)
        assert derived.parameter_count == arm.parameter_count
        assert derived.event_predictions == config.event_predictions
        assert derived.base_learning_rate == derived.post_boundary_learning_rate == 0.1
        first = ContextMLP(derived, 20260821, resolve_device("cpu"))
        second = ContextMLP(derived, 20260821, resolve_device("cpu"))
        assert first.w1.shape == (
            arm.context_length * arm.embedding_width,
            arm.hidden_width,
        )
        assert all(
            torch.equal(left, right)
            for left, right in zip(first.parameters(), second.parameters(), strict=True)
        )
        assert (
            sum(
                left != right
                for left, right in zip(
                    (arm.context_length, arm.embedding_width, arm.hidden_width),
                    (base.context_length, base.embedding_width, base.hidden_width),
                    strict=True,
                )
            )
            == 1
        )
    with pytest.raises(ModelDataError, match="arm is not approved"):
        preflight(ROOT, "safe-preflight", "not-an-arm")

    content = CONFIG_PATH.read_text(encoding="utf-8")
    typed = tmp_path / "typed.toml"
    changed = content.replace(
        "prediction_budget = 25000000", 'prediction_budget = "25000000"'
    )
    typed.write_text(changed, encoding="utf-8")
    monkeypatch.setattr(
        "protein_lm.mlp.capacity_screen_config.APPROVED_CAPACITY_SCREEN_CONFIG_SHA256",
        hashlib.sha256(changed.encode()).hexdigest(),
    )
    with pytest.raises(ModelDataError, match="prediction_budget must be an integer"):
        load_capacity_screen_config(typed)


def test_preflight_and_cli_default_are_config_only(monkeypatch, capsys) -> None:
    touched: list[str] = []
    monkeypatch.setattr(
        screen_orchestration,
        "_verify_readiness_aggregate",
        lambda *_: touched.append("readiness"),
    )
    monkeypatch.setattr(
        screen_orchestration,
        "_verify_source_pins",
        lambda *_: touched.append("source"),
    )
    monkeypatch.setattr(
        screen_orchestration,
        "_require_revision",
        lambda *_: touched.append("git"),
    )
    monkeypatch.setattr(
        screen_orchestration,
        "resolve_device",
        lambda *_: touched.append("device"),
    )
    monkeypatch.setattr(
        screen_orchestration,
        "load_collection",
        lambda *_: touched.append("collection"),
    )
    monkeypatch.setattr(
        screen_orchestration,
        "_require_ignored",
        lambda *_: touched.append("output"),
    )
    plan = preflight(ROOT, "safe-preflight", "context_20")
    assert plan.destination.name == "safe-preflight"
    assert touched == []
    assert SCRIPT.main([]) == 0
    assert "preflight reads only byte-pinned configs" in capsys.readouterr().out


def test_readiness_and_native_count_guards_are_exact(tmp_path: Path) -> None:
    base = load_config(BASE_CONFIG_PATH)
    report_path = tmp_path / "reports/week_02/readiness.json"
    report_path.parent.mkdir(parents=True)
    report = {
        "scope": "week_02_model_data_readiness",
        "candidate_status": "passed",
        "network_requests_made": 0,
        "collection_aggregates": {
            "family_aware_training": {"prediction_tokens": 8, "records": 2},
            "family_aware_native_validation": {"prediction_tokens": 8, "records": 2},
        },
    }
    content = json.dumps(report, sort_keys=True).encode()
    report_path.write_bytes(content)
    screen = replace(
        load_capacity_screen_config(CONFIG_PATH),
        readiness_report_relative_path="reports/week_02/readiness.json",
        readiness_report_sha256=hashlib.sha256(content).hexdigest(),
        training_prediction_tokens=8,
        training_records=2,
        native_validation_prediction_tokens=8,
        native_validation_records=2,
    )
    registry_path = tmp_path / base.model_data_registry_relative_path
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "readiness": {
                    "relative_path": screen.readiness_report_relative_path,
                    "sha256": screen.readiness_report_sha256,
                }
            }
        ),
        encoding="utf-8",
    )
    _verify_readiness_aggregate(tmp_path, screen, base)
    _validate_native_metric(NativeMetrics(8, 10.0, 2), screen)
    with pytest.raises(ModelDataError, match="token count"):
        _validate_native_metric(NativeMetrics(7, 10.0, 2), screen)
    report_path.write_bytes(content + b"tampered")
    with pytest.raises(ModelDataError, match="bytes do not match"):
        _verify_readiness_aggregate(tmp_path, screen, base)
    malformed = b"[]"
    report_path.write_bytes(malformed)
    malformed_config = replace(
        screen,
        readiness_report_sha256=hashlib.sha256(malformed).hexdigest(),
    )
    with pytest.raises(ModelDataError, match="not an object"):
        _verify_readiness_aggregate(tmp_path, malformed_config, base)


def test_each_arm_executes_synthetically_and_checkpoints_strictly_load(
    tmp_path: Path, monkeypatch
) -> None:
    base = replace(load_config(BASE_CONFIG_PATH), batch_size=4)
    original = load_capacity_screen_config(CONFIG_PATH)
    proteins = (protein("ACD", "P00001"), protein("ACE", "P00002"))
    tiny_arms = (
        CapacityArm("context_20", 20, 1, 1, 84),
        CapacityArm("embedding_64", 10, 2, 1, 105),
        CapacityArm("hidden_1600", 10, 1, 2, 106),
    )
    control_runs = tuple(
        replace(
            run,
            optimizer_steps=2,
            cursor_prediction_index=8,
            cursor_protein_index=2,
            cursor_within_protein_target_offset=0,
        )
        for run in original.control_runs
    )
    screen = replace(
        original,
        prediction_budget=8,
        batch_size=4,
        event_predictions=(4, 8),
        training_prediction_tokens=8,
        training_records=2,
        native_validation_prediction_tokens=8,
        native_validation_records=2,
        arms=tiny_arms,
        control_runs=control_runs,
        output_relative_root="data/processed/week_03/synthetic_capacity_screen_runs",
    )
    monkeypatch.setattr(screen_orchestration, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(
        screen_orchestration, "_verify_readiness_aggregate", lambda *_: None
    )
    monkeypatch.setattr(screen_orchestration, "_require_ignored", lambda *_: None)
    collection_loads: list[ModelDataCollection] = []

    def loader(_root: Path, collection: ModelDataCollection):
        assert collection in {
            ModelDataCollection.FAMILY_AWARE_TRAINING,
            ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION,
        }
        collection_loads.append(collection)
        return proteins

    for arm in tiny_arms:
        collection_loads.clear()
        plan = CapacityScreenPlan(
            screen,
            CONFIG_PATH,
            base,
            BASE_CONFIG_PATH,
            arm,
            f"synthetic-{arm.name}",
            tmp_path / arm.name,
        )
        destination = execute_screen(
            root=tmp_path,
            plan=plan,
            seed=20260821,
            device_name="cpu",
            loader=loader,
            code_revision="a" * 40,
        )
        status = json.loads((destination / "run_status.json").read_text())
        assert status["status"] == "passed"
        assert status["arm"] == arm.name
        assert status["cumulative_online_training"]["prediction_count"] == 8
        assert [
            item["prediction_position"]
            for item in status["native_validation_milestones"]
        ] == [4, 8]
        assert status["checkpoints"][-1]["prediction_position"] == 8
        assert collection_loads == [
            ModelDataCollection.FAMILY_AWARE_TRAINING,
            ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION,
        ]
        for checkpoint in status["checkpoints"]:
            checkpoint_path = destination / checkpoint["relative_path"]
            assert (
                checkpoint["metadata_sha256"]
                == hashlib.sha256(
                    (checkpoint_path / "checkpoint.json").read_bytes()
                ).hexdigest()
            )
            assert (
                checkpoint["tensor_sha256"]
                == hashlib.sha256(
                    (checkpoint_path / "model.safetensors").read_bytes()
                ).hexdigest()
            )
        model = ContextMLP(plan.training_config, 20260821, resolve_device("cpu"))
        optimizer = new_optimizer(model, plan.training_config)
        state = load_checkpoint(
            destination / "checkpoint-8",
            model=model,
            optimizer=optimizer,
            config=plan.training_config,
            config_path=CONFIG_PATH,
            seed=20260821,
            run_id=plan.run_id,
            device_name="cpu",
            code_revision="a" * 40,
        )
        assert (state.predictions_seen, state.optimizer_steps) == (8, 2)
        with pytest.raises(ModelDataError, match="already exists"):
            execute_screen(
                root=tmp_path,
                plan=plan,
                seed=20260821,
                device_name="cpu",
                loader=loader,
                code_revision="a" * 40,
            )


def test_execution_rejects_invalid_operating_inputs_before_data_access(
    tmp_path: Path,
) -> None:
    config = load_capacity_screen_config(CONFIG_PATH)
    base = load_config(BASE_CONFIG_PATH)
    plan = CapacityScreenPlan(
        config,
        CONFIG_PATH,
        base,
        BASE_CONFIG_PATH,
        config.arm("context_20"),
        "input-guard-run",
        tmp_path / "input-guard-run",
    )
    with pytest.raises(ModelDataError, match="CPU"):
        execute_screen(
            root=tmp_path,
            plan=plan,
            seed=20260821,
            device_name="mps",
            code_revision="a" * 40,
        )
    with pytest.raises(ModelDataError, match="seed is not approved"):
        execute_screen(
            root=tmp_path,
            plan=plan,
            seed=1,
            device_name="cpu",
            code_revision="a" * 40,
        )
    with pytest.raises(ModelDataError, match="injected code revision is invalid"):
        execute_screen(
            root=tmp_path,
            plan=plan,
            seed=20260821,
            device_name="cpu",
            code_revision="invalid",
        )
    injected_arm_plan = replace(
        plan,
        arm=replace(plan.arm, context_length=99),
    )
    with pytest.raises(ModelDataError, match="differs from strict approval"):
        execute_screen(
            root=tmp_path,
            plan=injected_arm_plan,
            seed=20260821,
            device_name="cpu",
            code_revision="a" * 40,
        )


def test_completion_runtime_is_validated(monkeypatch) -> None:
    state = TrainingState(8, 2, 0.0, StreamCursor(8, 2, 0))
    metric = {"cross_entropy": 2.0, "accuracy": 0.1}
    monkeypatch.setattr(screen_orchestration.time, "perf_counter", lambda: 11.0)
    assert _completion_payload(20260821, state, metric, 10.0)["runtime_seconds"] == 1.0
    monkeypatch.setattr(screen_orchestration.time, "perf_counter", lambda: float("nan"))
    with pytest.raises(ModelDataError, match="completion runtime"):
        _completion_payload(20260821, state, metric, 10.0)
    monkeypatch.setattr(screen_orchestration.time, "perf_counter", lambda: 9.0)
    with pytest.raises(ModelDataError, match="completion runtime"):
        _completion_payload(20260821, state, metric, 10.0)


def test_identical_stream_is_independent_of_arm_shape() -> None:
    proteins = (protein("ACDE", "P00001"), protein("KLMN", "P00002"))
    config = load_capacity_screen_config(CONFIG_PATH)
    base = replace(load_config(BASE_CONFIG_PATH), prediction_budget=10, batch_size=4)
    first = _arm_training_config(base, config, config.arm("context_20"))
    second = _arm_training_config(base, config, config.arm("hidden_1600"))
    left = tuple(
        iter_context_batches(
            proteins,
            namespace=first.training_namespace,
            base_seed=first.stream_base_seed,
            prediction_budget=10,
            batch_size=4,
            event_predictions=(4, 10),
            context_length=first.context_length,
        )
    )
    right = tuple(
        iter_context_batches(
            proteins,
            namespace=second.training_namespace,
            base_seed=second.stream_base_seed,
            prediction_budget=10,
            batch_size=4,
            event_predictions=(4, 10),
            context_length=second.context_length,
        )
    )
    assert [batch.targets.tolist() for batch in left] == [
        batch.targets.tolist() for batch in right
    ]
    assert [batch.end_cursor for batch in left] == [batch.end_cursor for batch in right]
