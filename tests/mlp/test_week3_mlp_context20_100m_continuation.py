"""Synthetic checks for the C=20 25M-to-100M continuation contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ModelDataCollection, ProteinSequence
from protein_lm.mlp import context20_continuation_orchestration as orchestration
from protein_lm.mlp.checkpoint import save_checkpoint
from protein_lm.mlp.config import load_config
from protein_lm.mlp.context20_continuation_config import (
    ParentRun,
    load_context20_continuation_config,
)
from protein_lm.mlp.context20_continuation_orchestration import (
    Context20ContinuationPlan,
    _status_payload,
    _validate_final_state,
    _validate_native_metric,
    _verify_parent_checkpoint_bytes,
    _verify_parent_status,
    _verify_readiness_aggregate,
    execute_continuation,
    preflight,
)
from protein_lm.mlp.metrics import NativeMetrics
from protein_lm.mlp.model import ContextMLP, resolve_device
from protein_lm.mlp.stream import StreamCursor, iter_context_batches
from protein_lm.mlp.training import (
    TrainingState,
    learning_rate_for,
    new_optimizer,
    train_batch,
)


ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "experiments/week_03/mlp_context20_100m_continuation_v1.toml"
BASE_CONFIG_PATH = ROOT / "experiments/week_03/mlp_training_v1.toml"
CAPACITY_CONFIG_PATH = ROOT / "experiments/week_03/mlp_capacity_screen_v1.toml"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_week3_mlp_context20_100m_continuation",
    ROOT / "scripts/run_week3_mlp_context20_100m_continuation.py",
)
assert SCRIPT_SPEC and SCRIPT_SPEC.loader
SCRIPT = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT)


def _protein(sequence: str, accession: str) -> ProteinSequence:
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
) -> tuple[Context20ContinuationPlan, tuple[ProteinSequence, ...]]:
    base = replace(
        load_config(BASE_CONFIG_PATH),
        context_length=1,
        embedding_width=1,
        hidden_width=1,
        prediction_budget=8,
        batch_size=4,
        milestone_predictions=(4, 8),
        checkpoint_predictions=(4, 6, 8),
        learning_rate_boundary_predictions=6,
    )
    parent_config = replace(
        base,
        contract_identifier="2026-08-24-week-03-mlp-capacity-screen-v1",
        prediction_budget=4,
        milestone_predictions=(4,),
        checkpoint_predictions=(4,),
        learning_rate_boundary_predictions=4,
        post_boundary_learning_rate=0.1,
    )
    proteins = (_protein("ACD", "P00001"), _protein("ACE", "P00002"))
    model = ContextMLP(parent_config, 20260821, resolve_device("cpu"))
    optimizer = new_optimizer(model, parent_config)
    state = TrainingState()
    batch = next(
        iter_context_batches(
            proteins,
            namespace=base.training_namespace,
            base_seed=base.stream_base_seed,
            prediction_budget=4,
            batch_size=4,
            event_predictions=(4,),
            context_length=1,
        )
    )
    train_batch(model, optimizer, batch, state, parent_config)
    parent_path = (
        tmp_path
        / parent_config.output_relative_root
        / "synthetic-parent"
        / "checkpoint-4"
    )
    save_checkpoint(
        parent_path,
        model=model,
        optimizer=optimizer,
        state=state,
        config=parent_config,
        config_path=CAPACITY_CONFIG_PATH,
        seed=20260821,
        run_id="synthetic-parent",
        device_name="cpu",
        code_revision="d" * 40,
    )
    pin = ParentRun(
        seed=20260821,
        run_id="synthetic-parent",
        run_status_sha256="0" * 64,
        metadata_sha256=hashlib.sha256(
            (parent_path / "checkpoint.json").read_bytes()
        ).hexdigest(),
        tensor_sha256=hashlib.sha256(
            (parent_path / "model.safetensors").read_bytes()
        ).hexdigest(),
        training_loss_numerator=state.training_loss_numerator,
    )
    config = replace(
        load_context20_continuation_config(CONFIG_PATH),
        parent_code_revision="d" * 40,
        training_prediction_tokens=8,
        training_records=2,
        native_validation_prediction_tokens=8,
        native_validation_records=2,
        context_length=1,
        embedding_width=1,
        hidden_width=1,
        parameter_count=base.parameter_count,
        parent_prediction_position=4,
        parent_optimizer_steps=1,
        parent_cursor_prediction_index=state.cursor.prediction_index,
        parent_cursor_protein_index=state.cursor.protein_index,
        parent_cursor_within_protein_target_offset=state.cursor.within_protein_target_offset,
        final_prediction_position=8,
        final_optimizer_steps=3,
        continuation_optimizer_updates=2,
        final_cursor_prediction_index=8,
        final_cursor_protein_index=2,
        final_cursor_within_protein_target_offset=0,
        batch_size=4,
        learning_rate_boundary_predictions=6,
        historical_milestone_predictions=(4, 8),
        historical_checkpoint_predictions=(4, 6, 8),
        continuation_evaluation_predictions=(8,),
        continuation_checkpoint_predictions=(6, 8),
        output_relative_root="data/processed/week_03/synthetic_context20_continuation_runs",
        parent_runs=(pin,),
    )
    return Context20ContinuationPlan(
        config,
        CONFIG_PATH,
        base,
        BASE_CONFIG_PATH,
        parent_config,
        CAPACITY_CONFIG_PATH,
        "synthetic-context20",
        tmp_path / config.output_relative_root / "synthetic-context20",
    ), proteins


def test_config_types_values_and_event_arithmetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_context20_continuation_config(CONFIG_PATH)
    assert config.continuation_predictions == 75_000_000
    assert config.event_predictions == (
        1_000_000,
        5_000_000,
        10_000_000,
        25_000_000,
        50_000_000,
        90_000_000,
        100_000_000,
    )
    plan = preflight(ROOT, "safe-preflight")
    assert [
        plan.training_config.expected_optimizer_steps(position)
        for position in (25_000_000, 50_000_000, 90_000_000, 100_000_000)
    ] == [24_416, 48_831, 87_894, 97_660]
    altered = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "batch_size = 1024", 'batch_size = "1024"'
    )
    typed = tmp_path / "typed.toml"
    typed.write_text(altered, encoding="utf-8")
    monkeypatch.setattr(
        "protein_lm.mlp.context20_continuation_config.APPROVED_CONTEXT20_CONTINUATION_CONFIG_SHA256",
        hashlib.sha256(altered.encode()).hexdigest(),
    )
    with pytest.raises(ModelDataError, match="batch_size must be an integer"):
        load_context20_continuation_config(typed)
    changed = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "base_learning_rate = 0.1", "base_learning_rate = 0.2"
    )
    values = tmp_path / "values.toml"
    values.write_text(changed, encoding="utf-8")
    monkeypatch.setattr(
        "protein_lm.mlp.context20_continuation_config.APPROVED_CONTEXT20_CONTINUATION_CONFIG_SHA256",
        hashlib.sha256(changed.encode()).hexdigest(),
    )
    with pytest.raises(ModelDataError, match="schedule is not approved"):
        load_context20_continuation_config(values)


def test_preflight_cli_and_parent_checkpoint_guards_are_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    touched: list[str] = []
    for name in (
        "_verify_readiness_aggregate",
        "_verify_source_pins",
        "_require_revision",
        "_require_ignored",
        "load_collection",
    ):
        monkeypatch.setattr(
            orchestration,
            name,
            lambda *_args, _name=name, **_kwargs: touched.append(_name),
        )
    assert preflight(ROOT, "safe-preflight").destination.name == "safe-preflight"
    assert touched == []
    assert SCRIPT.main([]) == 0
    assert "preflight reads only byte-pinned configs" in capsys.readouterr().out
    checkpoint = tmp_path / "checkpoint-4"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_bytes(b"metadata")
    (checkpoint / "model.safetensors").write_bytes(b"tensors")
    pin = ParentRun(
        20260821,
        "parent",
        "0" * 64,
        hashlib.sha256(b"metadata").hexdigest(),
        hashlib.sha256(b"tensors").hexdigest(),
        1.0,
    )
    _verify_parent_checkpoint_bytes(checkpoint, pin)
    (checkpoint / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ModelDataError, match="bytes do not match"):
        _verify_parent_checkpoint_bytes(checkpoint, pin)
    plan, _ = _synthetic_plan(tmp_path / "status")
    tampered_status = tmp_path / "tampered-status.json"
    tampered_status.write_bytes(b"tampered")
    with pytest.raises(ModelDataError, match="status bytes do not match"):
        _verify_parent_status(
            tampered_status,
            load_context20_continuation_config(CONFIG_PATH).parent_run(20260821),
            plan,
        )


def test_synthetic_readiness_guard_and_schedule_boundary(tmp_path: Path) -> None:
    plan, _ = _synthetic_plan(tmp_path)
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
    report_path = tmp_path / "reports/week_02/readiness.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes(content)
    config = replace(
        plan.config,
        readiness_report_relative_path="reports/week_02/readiness.json",
        readiness_report_sha256=hashlib.sha256(content).hexdigest(),
    )
    registry_path = tmp_path / plan.base_config.model_data_registry_relative_path
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
    _verify_readiness_aggregate(tmp_path, config, plan.base_config)
    report_path.write_bytes(content + b"tampered")
    with pytest.raises(ModelDataError, match="bytes do not match"):
        _verify_readiness_aggregate(tmp_path, config, plan.base_config)
    production = preflight(ROOT, "boundary-check").training_config
    assert learning_rate_for(89_999_999, production) == 0.1
    assert learning_rate_for(90_000_000, production) == 0.01


def test_synthetic_execution_preserves_parent_and_records_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, proteins = _synthetic_plan(tmp_path)
    parent_path = (
        tmp_path
        / plan.parent_training_config.output_relative_root
        / "synthetic-parent"
        / "checkpoint-4"
    )
    before = {item.name: item.read_bytes() for item in parent_path.iterdir()}
    monkeypatch.setattr(orchestration, "preflight", lambda *_: plan)
    monkeypatch.setattr(orchestration, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(orchestration, "_verify_readiness_aggregate", lambda *_: None)
    monkeypatch.setattr(orchestration, "_require_ignored", lambda *_: None)
    monkeypatch.setattr(orchestration, "_verify_parent_status", lambda *_: None)
    loads: list[ModelDataCollection] = []

    def loader(_root: Path, collection: ModelDataCollection):
        loads.append(collection)
        return proteins

    destination = execute_continuation(
        root=tmp_path,
        plan=plan,
        seed=20260821,
        device_name="cpu",
        loader=loader,
        code_revision="a" * 40,
    )
    status = json.loads((destination / "run_status.json").read_text())
    assert loads == [
        ModelDataCollection.FAMILY_AWARE_TRAINING,
        ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION,
    ]
    assert status["status"] == "passed"
    assert status["training_loss_lineage"]["total_prediction_count"] == 8
    assert status["training_loss_lineage"]["continuation_prediction_count"] == 4
    assert [item["prediction_position"] for item in status["checkpoints"]] == [6, 8]
    assert status["automatic_selection_generated"] is False
    assert status["automatic_report_generated"] is False
    assert before == {item.name: item.read_bytes() for item in parent_path.iterdir()}
    with pytest.raises(ModelDataError, match="cannot resume"):
        execute_continuation(
            root=tmp_path,
            plan=plan,
            seed=20260821,
            device_name="cpu",
            loader=loader,
            code_revision="a" * 40,
        )


def test_canonical_plan_guard_rejects_tampering_before_operational_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, proteins = _synthetic_plan(tmp_path)
    touches: list[str] = []
    loader_calls: list[object] = []
    monkeypatch.setattr(orchestration, "preflight", lambda *_: plan)
    for name in (
        "_require_revision",
        "_verify_source_pins",
        "_verify_readiness_aggregate",
        "_require_ignored",
    ):
        monkeypatch.setattr(
            orchestration,
            name,
            lambda *_args, _name=name, **_kwargs: touches.append(_name),
        )
    for tampered in (
        replace(plan, destination=tmp_path / "unexpected-destination"),
        replace(plan, config=replace(plan.config, output_relative_root="unexpected")),
    ):
        with pytest.raises(ModelDataError, match="execution plan differs"):
            execute_continuation(
                root=tmp_path,
                plan=tampered,
                seed=20260821,
                device_name="cpu",
                loader=lambda *_: loader_calls.append("loader") or proteins,
                code_revision="a" * 40,
            )
    assert touches == []
    assert loader_calls == []
    assert not (tmp_path / "unexpected-destination").exists()


def test_status_lr_uses_the_current_continuation_position() -> None:
    plan = preflight(ROOT, "status-lr")
    parent = plan.config.parent_run(20260821)
    common = {
        "plan": plan,
        "seed": 20260821,
        "parent": parent,
        "parent_status": Path("parent-status.json"),
        "parent_checkpoint": Path("parent-checkpoint"),
        "derived_revision": "a" * 40,
        "status": "running",
        "metrics": [],
        "checkpoints": [],
        "failure_reason": None,
        "started": time.perf_counter(),
    }
    before_boundary = _status_payload(
        state=TrainingState(
            50_000_000,
            48_831,
            parent.training_loss_numerator,
            StreamCursor(50_000_000, 0, 0),
        ),
        **common,
    )
    at_boundary = _status_payload(
        state=TrainingState(
            90_000_000,
            87_894,
            parent.training_loss_numerator,
            StreamCursor(90_000_000, 0, 0),
        ),
        **common,
    )
    assert (
        before_boundary["final_cursor_step_accounting"]["active_learning_rate"] == 0.1
    )
    assert at_boundary["final_cursor_step_accounting"]["active_learning_rate"] == 0.01


def test_input_state_metric_and_event_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = _synthetic_plan(tmp_path)
    monkeypatch.setattr(orchestration, "preflight", lambda *_: plan)
    with pytest.raises(ModelDataError, match="CPU"):
        execute_continuation(
            root=tmp_path,
            plan=plan,
            seed=20260821,
            device_name="mps",
            code_revision="a" * 40,
        )
    with pytest.raises(ModelDataError, match="seed is not approved"):
        execute_continuation(
            root=tmp_path, plan=plan, seed=1, device_name="cpu", code_revision="a" * 40
        )
    with pytest.raises(ModelDataError, match="injected code revision is invalid"):
        execute_continuation(
            root=tmp_path,
            plan=plan,
            seed=20260821,
            device_name="cpu",
            code_revision="invalid",
        )
    _validate_native_metric(NativeMetrics(8, 1.0, 1), plan.config)
    with pytest.raises(ModelDataError, match="token count"):
        _validate_native_metric(NativeMetrics(7, 1.0, 1), plan.config)
    state = TrainingState(8, 3, 3.0, StreamCursor(8, 2, 0))
    _validate_final_state(
        state,
        [{"prediction_position": 8}],
        [{"prediction_position": 6}, {"prediction_position": 8}],
        plan.config,
    )
    with pytest.raises(ModelDataError, match="accounting"):
        _validate_final_state(
            state,
            [],
            [{"prediction_position": 6}, {"prediction_position": 8}],
            plan.config,
        )
