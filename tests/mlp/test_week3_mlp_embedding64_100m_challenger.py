"""Focused synthetic checks for the fixed Week 3 E=64 challenger contract."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ModelDataCollection, ProteinSequence
from protein_lm.mlp import embedding64_challenger_orchestration as orchestration
from protein_lm.mlp.checkpoint import save_checkpoint
from protein_lm.mlp.config import load_config
from protein_lm.mlp.embedding64_challenger_config import (
    ParentRun,
    load_embedding64_challenger_config,
)
from protein_lm.mlp.embedding64_challenger_orchestration import (
    Embedding64ChallengerPlan,
    _status_payload,
    _validate_final_state,
    _validate_native_metric,
    _verify_parent_checkpoint_bytes,
    _verify_parent_status,
    _verify_readiness_aggregate,
    execute_challenger,
    preflight,
)
from protein_lm.mlp.metrics import NativeMetrics
from protein_lm.mlp.model import ContextMLP, resolve_device
from protein_lm.mlp.stream import StreamCursor, iter_context_batches
from protein_lm.mlp.training import TrainingState, new_optimizer, train_batch


ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "experiments/week_03/mlp_embedding64_100m_challenger_v1.toml"
BASE_CONFIG_PATH = ROOT / "experiments/week_03/mlp_training_v1.toml"
CAPACITY_CONFIG_PATH = ROOT / "experiments/week_03/mlp_capacity_screen_v1.toml"


def _protein(sequence: str, accession: str) -> ProteinSequence:
    return ProteinSequence(
        accession,
        sequence,
        hashlib.sha256(sequence.encode()).hexdigest(),
        len(sequence),
        "synthetic",
        "UniRef50_SYNTHETIC",
    )


def test_config_is_byte_pinned_and_has_symmetric_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_embedding64_challenger_config(CONFIG_PATH)
    assert config.continuation_predictions == 75_000_000
    assert config.parameter_count == 530_965
    assert config.reference_context20_parameter_count == 530_293
    assert not hasattr(config, "context20_materially_better_at_or_above")
    assert not hasattr(config, "embedding64_materially_better_at_or_below")
    assert (
        config.context20_materially_better_if_embedding64_mean_at_or_above
        == 2.864665856220289
    )
    assert (
        config.embedding64_materially_better_if_embedding64_mean_at_or_below
        == 2.862665856220289
    )
    changed = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "batch_size = 1024", 'batch_size = "1024"'
    )
    path = tmp_path / "bad-type.toml"
    path.write_text(changed, encoding="utf-8")
    monkeypatch.setattr(
        "protein_lm.mlp.embedding64_challenger_config.APPROVED_EMBEDDING64_CHALLENGER_CONFIG_SHA256",
        hashlib.sha256(changed.encode()).hexdigest(),
    )
    with pytest.raises(ModelDataError, match="batch_size must be an integer"):
        load_embedding64_challenger_config(path)
    ambiguous = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "context20_materially_better_if_embedding64_mean_at_or_above",
        "context20_materially_better_at_or_above",
    )
    ambiguous_path = tmp_path / "ambiguous.toml"
    ambiguous_path.write_text(ambiguous, encoding="utf-8")
    monkeypatch.setattr(
        "protein_lm.mlp.embedding64_challenger_config.APPROVED_EMBEDDING64_CHALLENGER_CONFIG_SHA256",
        hashlib.sha256(ambiguous.encode()).hexdigest(),
    )
    with pytest.raises(ModelDataError, match="keys differ from schema"):
        load_embedding64_challenger_config(ambiguous_path)


def test_safe_preflight_has_canonical_destination_and_schedule() -> None:
    plan = preflight(ROOT, "safe-preflight")
    assert (
        plan.destination
        == ROOT
        / "data/processed/week_03/mlp_embedding64_100m_challenger_runs/safe-preflight"
    )
    assert [
        plan.training_config.expected_optimizer_steps(position)
        for position in (25_000_000, 50_000_000, 90_000_000, 100_000_000)
    ] == [24_416, 48_831, 87_894, 97_660]
    assert plan.config.continuation_evaluation_predictions == (50_000_000, 100_000_000)
    assert plan.config.continuation_checkpoint_predictions == (
        50_000_000,
        90_000_000,
        100_000_000,
    )


def test_parent_checkpoint_bytes_fail_closed_when_tampered(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-25000000"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_bytes(b"metadata")
    (checkpoint / "model.safetensors").write_bytes(b"tensors")
    parent = ParentRun(
        20260821,
        "synthetic-parent",
        "0" * 64,
        hashlib.sha256(b"metadata").hexdigest(),
        hashlib.sha256(b"tensors").hexdigest(),
        1.0,
    )
    _verify_parent_checkpoint_bytes(checkpoint, parent)
    (checkpoint / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ModelDataError, match="bytes do not match"):
        _verify_parent_checkpoint_bytes(checkpoint, parent)


def test_parent_status_and_readiness_bytes_fail_closed(tmp_path: Path) -> None:
    plan = preflight(ROOT, "readiness-guard")
    parent = plan.config.parent_run(20260821)
    status = tmp_path / "run_status.json"
    status.write_bytes(b"tampered")
    with pytest.raises(ModelDataError, match="status bytes do not match"):
        _verify_parent_status(status, parent, plan)
    non_object = b"[]"
    status.write_bytes(non_object)
    hash_matched_parent = replace(
        parent, run_status_sha256=hashlib.sha256(non_object).hexdigest()
    )
    with pytest.raises(ModelDataError, match="not an object"):
        _verify_parent_status(status, hash_matched_parent, plan)
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
        training_prediction_tokens=8,
        training_records=2,
        native_validation_prediction_tokens=8,
        native_validation_records=2,
        readiness_report_relative_path="reports/week_02/readiness.json",
        readiness_report_sha256=hashlib.sha256(content).hexdigest(),
    )
    registry = tmp_path / plan.base_config.model_data_registry_relative_path
    registry.parent.mkdir(parents=True)
    registry.write_text(
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
    with pytest.raises(ModelDataError, match="does not match approval"):
        _verify_readiness_aggregate(tmp_path, config, plan.base_config)


def test_status_lr_final_state_and_canonical_plan_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = preflight(ROOT, "status-lr")
    parent = plan.config.parent_run(20260821)
    common = dict(
        plan=plan,
        seed=20260821,
        parent=parent,
        parent_status=ROOT
        / "data/processed/week_03/mlp_capacity_screen_runs/parent/run_status.json",
        parent_checkpoint=ROOT
        / "data/processed/week_03/mlp_capacity_screen_runs/parent/checkpoint-25000000",
        revision="a" * 40,
        status="running",
        metrics=[],
        checkpoints=[],
        failure_reason=None,
        started=time.perf_counter(),
    )
    before = _status_payload(
        state=TrainingState(
            50_000_000,
            48_831,
            parent.training_loss_numerator,
            StreamCursor(50_000_000, 0, 0),
        ),
        **common,
    )
    at = _status_payload(
        state=TrainingState(
            90_000_000,
            87_894,
            parent.training_loss_numerator,
            StreamCursor(90_000_000, 0, 0),
        ),
        **common,
    )
    assert before["final_cursor_step_accounting"]["active_learning_rate"] == 0.1
    assert at["final_cursor_step_accounting"]["active_learning_rate"] == 0.01
    assert before["schedule"]["continuation_optimizer_updates"] == 73_244
    assert before["schedule"]["repeat_or_wraparound"] is False
    assert before["challenger_selection_provenance"] == {
        "scope": "post_screen_adversarial_challenger_not_open_model_selection",
        "basis": "lowest_25m_mean_native_cross_entropy_non_context_arm",
        "open_model_selection": False,
        "embedding64_25m_three_seed_mean_native_cross_entropy": 2.8713892507036705,
        "hidden1600_25m_three_seed_mean_native_cross_entropy": 2.871764147540805,
    }
    assert before["three_seed_interpretation_rule"] == {
        "metric": "100M three-seed mean native cross entropy",
        "delta_definition": "embedding64_mean_native_cross_entropy - context20_mean_native_cross_entropy",
        "material_gap": 0.001,
        "context20_materially_better_if_embedding64_mean_at_or_above": 2.864665856220289,
        "embedding64_materially_better_if_embedding64_mean_at_or_below": 2.862665856220289,
        "context20_materially_better_boundary_inclusive": True,
        "embedding64_materially_better_boundary_inclusive": True,
        "practical_tie_interval": {
            "lower_exclusive": 2.862665856220289,
            "upper_exclusive": 2.864665856220289,
        },
        "categories": {
            "context20_materially_better": {
                "embedding64_mean_at_or_above": 2.864665856220289,
                "boundary_inclusive": True,
            },
            "practical_tie": {
                "lower_exclusive": 2.862665856220289,
                "upper_exclusive": 2.864665856220289,
            },
            "embedding64_materially_better": {
                "embedding64_mean_at_or_below": 2.862665856220289,
                "boundary_inclusive": True,
            },
        },
        "per_seed_interpretation_generated": False,
        "automatic_decision_generated": False,
        "automatic_report_generated": False,
    }
    _validate_native_metric(
        NativeMetrics(plan.config.native_validation_prediction_tokens, 1.0, 1),
        plan.config,
    )
    with pytest.raises(ModelDataError, match="token count"):
        _validate_native_metric(
            NativeMetrics(plan.config.native_validation_prediction_tokens - 1, 1.0, 1),
            plan.config,
        )
    final = TrainingState(
        100_000_000,
        97_660,
        parent.training_loss_numerator,
        StreamCursor(100_000_000, 269_057, 270),
    )
    _validate_final_state(
        final,
        [{"prediction_position": 50_000_000}, {"prediction_position": 100_000_000}],
        [
            {"prediction_position": 50_000_000},
            {"prediction_position": 90_000_000},
            {"prediction_position": 100_000_000},
        ],
        plan.config,
    )
    monkeypatch.setattr(orchestration, "preflight", lambda *_: plan)
    with pytest.raises(ModelDataError, match="execution plan differs"):
        execute_challenger(
            root=ROOT,
            plan=replace(plan, destination=ROOT / "unexpected"),
            seed=20260821,
            device_name="cpu",
            code_revision="a" * 40,
        )


def test_synthetic_execution_loads_parent_and_never_replays_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    train_batch(
        model,
        optimizer,
        next(
            iter_context_batches(
                proteins,
                namespace=base.training_namespace,
                base_seed=base.stream_base_seed,
                prediction_budget=4,
                batch_size=4,
                event_predictions=(4,),
                context_length=1,
            )
        ),
        state,
        parent_config,
    )
    parent_path = (
        tmp_path / parent_config.output_relative_root / "parent" / "checkpoint-4"
    )
    save_checkpoint(
        parent_path,
        model=model,
        optimizer=optimizer,
        state=state,
        config=parent_config,
        config_path=CAPACITY_CONFIG_PATH,
        seed=20260821,
        run_id="parent",
        device_name="cpu",
        code_revision="d" * 40,
    )
    pin = ParentRun(
        20260821,
        "parent",
        "0" * 64,
        hashlib.sha256((parent_path / "checkpoint.json").read_bytes()).hexdigest(),
        hashlib.sha256((parent_path / "model.safetensors").read_bytes()).hexdigest(),
        state.training_loss_numerator,
    )
    config = replace(
        load_embedding64_challenger_config(CONFIG_PATH),
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
        output_relative_root="data/processed/week_03/synthetic_embedding64_challenger_runs",
        parent_runs=(pin,),
    )
    plan = Embedding64ChallengerPlan(
        config,
        CONFIG_PATH,
        base,
        BASE_CONFIG_PATH,
        parent_config,
        CAPACITY_CONFIG_PATH,
        "synthetic-embedding64",
        tmp_path / config.output_relative_root / "synthetic-embedding64",
    )
    before = {item.name: item.read_bytes() for item in parent_path.iterdir()}
    monkeypatch.setattr(orchestration, "preflight", lambda *_: plan)
    monkeypatch.setattr(orchestration, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(orchestration, "_verify_readiness_aggregate", lambda *_: None)
    monkeypatch.setattr(orchestration, "_require_ignored", lambda *_: None)
    monkeypatch.setattr(orchestration, "_verify_parent_status", lambda *_: None)
    monkeypatch.setattr(orchestration, "_validate_parent_state", lambda *_: None)
    loads: list[ModelDataCollection] = []

    def loader(_root: Path, collection: ModelDataCollection):
        loads.append(collection)
        return proteins

    destination = execute_challenger(
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
    assert status["automatic_selection_generated"] is False
    assert status["automatic_report_generated"] is False
    assert status["training_loss_lineage"]["parent_prediction_count"] == 4
    assert status["training_loss_lineage"]["continuation_prediction_count"] == 4
    assert [item["prediction_position"] for item in status["checkpoints"]] == [6, 8]
    assert before == {item.name: item.read_bytes() for item in parent_path.iterdir()}
