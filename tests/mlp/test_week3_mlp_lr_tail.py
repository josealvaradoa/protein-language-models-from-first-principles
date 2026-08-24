"""Synthetic checks for the non-primary Week 3 learning-rate tail runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ModelDataCollection, ProteinSequence
from protein_lm.mlp import tail_orchestration
from protein_lm.mlp.checkpoint import load_checkpoint, save_checkpoint
from protein_lm.mlp.config import load_config
from protein_lm.mlp.model import ContextMLP, resolve_device
from protein_lm.mlp.stream import iter_context_batches
from protein_lm.mlp.tail_config import ParentPin, load_tail_config
from protein_lm.mlp.tail_orchestration import MLPTailPlan, execute_tail, preflight
from protein_lm.mlp.tail_training import (
    schedule_provenance,
    staged_effective_boundary,
    tail_last_applied_learning_rate,
    tail_learning_rate,
    train_tail_batch,
)
from protein_lm.mlp.training import TrainingState, new_optimizer, train_batch


ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "experiments/week_03/mlp_lr_tail_v1.toml"
BASE_CONFIG_PATH = ROOT / "experiments/week_03/mlp_training_v1.toml"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_week3_mlp_lr_tail", ROOT / "scripts/run_week3_mlp_lr_tail.py"
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


def test_config_tamper_type_value_and_preflight_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_tail_config(CONFIG_PATH)
    assert config.parent_pin(20260822).run_id == "week3-mlp-seed-20260822-cpu"
    assert (
        preflight(ROOT, "synthetic-tail", "staged_97m_003").destination.name
        == "synthetic-tail"
    )
    text = CONFIG_PATH.read_text(encoding="utf-8")
    tampered = tmp_path / "tampered.toml"
    tampered.write_text(text.replace("batch_size = 1024", "batch_size = 8"))
    with pytest.raises(ModelDataError, match="bytes do not match"):
        load_tail_config(tampered)
    typed = tmp_path / "typed.toml"
    typed_text = text.replace(
        "tail_optimizer_updates = 9766", 'tail_optimizer_updates = "9766"'
    )
    typed.write_text(typed_text)
    monkeypatch.setattr(
        "protein_lm.mlp.tail_config.APPROVED_MLP_LR_TAIL_CONFIG_SHA256",
        hashlib.sha256(typed_text.encode()).hexdigest(),
    )
    with pytest.raises(
        ModelDataError, match="tail_optimizer_updates must be an integer"
    ):
        load_tail_config(typed)
    pinned_value = tmp_path / "pinned-value.toml"
    altered = text.replace("week3-mlp-seed-20260821-cpu", "wrong-parent-run")
    pinned_value.write_text(altered)
    monkeypatch.setattr(
        "protein_lm.mlp.tail_config.APPROVED_MLP_LR_TAIL_CONFIG_SHA256",
        hashlib.sha256(altered.encode()).hexdigest(),
    )
    with pytest.raises(ModelDataError, match="values are not approved"):
        load_tail_config(pinned_value)


def test_schedule_is_exact_without_an_interior_batch_event() -> None:
    config = load_tail_config(CONFIG_PATH)
    starts = tuple(
        range(
            config.parent_prediction_position,
            config.final_prediction_position,
            config.batch_size,
        )
    )
    assert len(starts) == config.tail_optimizer_updates
    assert starts[-1] == 99_999_360
    sizes = (config.batch_size,) * (config.tail_optimizer_updates - 1) + (640,)
    assert sum(sizes) == 10_000_000
    assert staged_effective_boundary(config) == 97_000_064
    assert starts[starts.index(staged_effective_boundary(config)) - 1] == 96_999_040
    assert tail_learning_rate("staged_97m_003", 96_999_040, config) == 0.01
    assert tail_learning_rate("staged_97m_003", 97_000_064, config) == 0.003
    assert tail_learning_rate("cosine_90m_100m_001", 90_000_000, config) == 0.01
    representative = 94_999_168
    progress = (representative - 90_000_000) / 10_000_000
    expected = 0.001 + 0.5 * (0.01 - 0.001) * (1 + math.cos(math.pi * progress))
    assert tail_learning_rate("cosine_90m_100m_001", representative, config) == expected
    with pytest.raises(ModelDataError, match="inherited batch boundary"):
        tail_learning_rate("cosine_90m_100m_001", 95_000_000, config)
    last = tail_last_applied_learning_rate("cosine_90m_100m_001", config)
    assert 0.001 < last < 0.01
    assert config.cosine_endpoint_learning_rate == 0.001
    staged = schedule_provenance("staged_97m_003", config)
    assert set(staged) == {
        "arm",
        "declared_boundary_prediction",
        "effective_lower_lr_start_prediction",
        "initial_learning_rate",
        "lower_learning_rate",
        "last_applied_learning_rate",
    }
    cosine = schedule_provenance("cosine_90m_100m_001", config)
    assert set(cosine) == {
        "arm",
        "formula_identifier",
        "start_prediction",
        "final_prediction",
        "start_learning_rate",
        "mathematical_endpoint_learning_rate",
        "last_applied_learning_rate",
    }


def _synthetic_plan(tmp_path: Path) -> tuple[MLPTailPlan, tuple[ProteinSequence, ...]]:
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
    batches = tuple(
        iter_context_batches(
            proteins,
            namespace=base.training_namespace,
            base_seed=base.stream_base_seed,
            prediction_budget=8,
            batch_size=4,
            event_predictions=(8,),
        )
    )
    parent_state = TrainingState()
    train_batch(parent_model, optimizer, batches[0], parent_state, base)
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
        load_tail_config(CONFIG_PATH),
        parent_code_revision="d" * 40,
        parent_prediction_position=4,
        parent_optimizer_steps=1,
        parent_cursor_prediction_index=parent_state.cursor.prediction_index,
        parent_cursor_protein_index=parent_state.cursor.protein_index,
        parent_cursor_within_protein_target_offset=parent_state.cursor.within_protein_target_offset,
        final_prediction_position=8,
        final_optimizer_steps=2,
        tail_optimizer_updates=1,
        batch_size=4,
        staged_boundary_prediction=6,
        parent_pins=(pin,),
        output_relative_root="data/processed/week_03/synthetic_tail_runs",
    )
    plan = MLPTailPlan(
        config=config,
        config_path=CONFIG_PATH,
        base_config=base,
        base_config_path=BASE_CONFIG_PATH,
        run_id="synthetic-tail",
        arm="cosine_90m_100m_001",
        destination=tmp_path / config.output_relative_root / "synthetic-tail",
    )
    return plan, proteins


def test_tail_mechanics_match_primary_control_and_arm_streams(tmp_path: Path) -> None:
    plan, proteins = _synthetic_plan(tmp_path)
    parent_path = (
        tmp_path
        / plan.base_config.output_relative_root
        / "synthetic-parent"
        / "checkpoint-4"
    )

    def parent_state_for(model: ContextMLP, optimizer) -> TrainingState:
        return load_checkpoint(
            parent_path,
            model=model,
            optimizer=optimizer,
            config=plan.base_config,
            config_path=BASE_CONFIG_PATH,
            seed=20260821,
            run_id="synthetic-parent",
            device_name="cpu",
            code_revision="d" * 40,
        )

    primary_model = ContextMLP(plan.base_config, 20260821, resolve_device("cpu"))
    tail_model = ContextMLP(plan.base_config, 20260821, resolve_device("cpu"))
    primary_optimizer = new_optimizer(primary_model, plan.base_config)
    tail_optimizer = new_optimizer(tail_model, plan.base_config)
    primary_state = parent_state_for(primary_model, primary_optimizer)
    tail_state = parent_state_for(tail_model, tail_optimizer)
    primary_batches = tuple(
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
    staged_batches = tuple(
        iter_context_batches(
            proteins,
            namespace=plan.base_config.training_namespace,
            base_seed=plan.base_config.stream_base_seed,
            prediction_budget=8,
            batch_size=4,
            event_predictions=(8,),
            cursor=tail_state.cursor,
        )
    )
    cosine_batches = tuple(
        iter_context_batches(
            proteins,
            namespace=plan.base_config.training_namespace,
            base_seed=plan.base_config.stream_base_seed,
            prediction_budget=8,
            batch_size=4,
            event_predictions=(8,),
            cursor=tail_state.cursor,
        )
    )
    assert len(primary_batches) == len(staged_batches) == len(cosine_batches) == 1
    for staged, cosine in zip(staged_batches, cosine_batches, strict=True):
        assert staged.start_prediction == cosine.start_prediction
        assert staged.predictions == cosine.predictions
        assert torch.equal(staged.contexts, cosine.contexts)
        assert torch.equal(staged.targets, cosine.targets)
    train_batch(
        primary_model,
        primary_optimizer,
        primary_batches[0],
        primary_state,
        plan.base_config,
    )
    train_tail_batch(
        tail_model,
        tail_optimizer,
        staged_batches[0],
        tail_state,
        plan.config,
        "staged_97m_003",
    )
    assert (
        primary_optimizer.param_groups[0]["lr"]
        == tail_optimizer.param_groups[0]["lr"]
        == 0.01
    )
    assert primary_state == tail_state
    assert all(
        torch.equal(primary, tail)
        for primary, tail in zip(
            primary_model.parameters(), tail_model.parameters(), strict=True
        )
    )


def test_synthetic_tail_is_nonresumable_atomic_and_preserves_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, proteins = _synthetic_plan(tmp_path)
    parent_path = (
        tmp_path
        / plan.base_config.output_relative_root
        / "synthetic-parent"
        / "checkpoint-4"
    )
    before = {
        path.name: path.read_bytes()
        for path in (parent_path / "checkpoint.json", parent_path / "model.safetensors")
    }
    monkeypatch.setattr(
        "protein_lm.mlp.tail_orchestration._verify_source_pins", lambda *_: None
    )
    monkeypatch.setattr(
        "protein_lm.mlp.tail_orchestration._require_ignored", lambda *_: None
    )
    loaded: list[ModelDataCollection] = []

    def loader(_root: Path, collection: ModelDataCollection):
        loaded.append(collection)
        assert collection in {
            ModelDataCollection.FAMILY_AWARE_TRAINING,
            ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION,
        }
        return proteins

    destination = execute_tail(
        root=tmp_path,
        plan=plan,
        seed=20260821,
        device_name="cpu",
        loader=loader,
        code_revision="a" * 40,
    )
    assert destination == plan.destination
    assert loaded == [
        ModelDataCollection.FAMILY_AWARE_TRAINING,
        ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION,
    ]
    assert before == {
        path.name: path.read_bytes()
        for path in (parent_path / "checkpoint.json", parent_path / "model.safetensors")
    }
    status = json.loads((destination / "run_status.json").read_text())
    assert status["status"] == "passed"
    assert status["tail_optimizer_updates"] == 1
    assert status["tail_online_loss_numerator"] > 0
    assert status["final_native_validation"]["token_count"] == 8
    assert (
        status["tail_online_cross_entropy"] == status["tail_online_loss_numerator"] / 4
    )
    native = status["final_native_validation"]
    assert native["cross_entropy"] == native["nll_numerator"] / native["token_count"]
    assert native["accuracy"] == native["correct_predictions"] / native["token_count"]
    assert status["schedule"]["last_applied_learning_rate"] > 0.001
    assert "declared_boundary_prediction" not in status["schedule"]
    assert status["network_requests_made"] == 0
    assert all(
        status[name]
        for name in ("python_version", "torch_version", "platform", "machine")
    )
    model_path = destination / "final_model.safetensors"
    assert (
        hashlib.sha256(model_path.read_bytes()).hexdigest()
        == status["final_model_sha256"]
    )
    assert set(load_file(str(model_path))) == {"embedding", "w1", "b1", "w2", "b2"}
    with pytest.raises(ModelDataError, match="already exists"):
        execute_tail(
            root=tmp_path,
            plan=plan,
            seed=20260821,
            device_name="cpu",
            loader=loader,
            code_revision="a" * 40,
        )


def test_parent_byte_pin_rejection_and_default_cli_does_not_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plan, proteins = _synthetic_plan(tmp_path)
    parent_tensors = (
        tmp_path
        / plan.base_config.output_relative_root
        / "synthetic-parent"
        / "checkpoint-4"
        / "model.safetensors"
    )
    parent_tensors.write_bytes(parent_tensors.read_bytes() + b"tamper")
    monkeypatch.setattr(
        "protein_lm.mlp.tail_orchestration._verify_source_pins", lambda *_: None
    )
    monkeypatch.setattr(
        "protein_lm.mlp.tail_orchestration._require_ignored", lambda *_: None
    )
    with pytest.raises(ModelDataError, match="bytes do not match"):
        execute_tail(
            root=tmp_path,
            plan=plan,
            seed=20260821,
            device_name="cpu",
            loader=lambda *_: proteins,
            code_revision="a" * 40,
        )
    failed = json.loads((plan.destination / "run_status.json").read_text())
    assert failed["status"] == "failed"

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
        execute_tail(
            root=tmp_path / "metadata",
            plan=metadata_plan,
            seed=20260821,
            device_name="cpu",
            loader=lambda *_: metadata_proteins,
            code_revision="a" * 40,
        )

    def prohibited(*_args, **_kwargs):
        raise AssertionError("default preflight must not execute a tail")

    monkeypatch.setattr(SCRIPT, "execute_tail", prohibited)
    for name in (
        "load_checkpoint",
        "load_collection",
        "resolve_device",
        "_require_ignored",
        "_require_revision",
    ):
        monkeypatch.setattr(tail_orchestration, name, prohibited)
    assert SCRIPT.main([]) == 0
    output = capsys.readouterr().out
    assert "staged_97m_003" in output
    assert "cosine_90m_100m_001" in output
    assert "does not access a checkpoint" in output

    def completed(**kwargs):
        kwargs["progress_callback"](
            "completed",
            {
                "arm": "staged_97m_003",
                "seed": 20260821,
                "final_cross_entropy": 1.25,
                "final_accuracy": 0.5,
                "runtime_seconds": 2.0,
                "tail_optimizer_updates": 9766,
            },
        )
        return Path("/tmp/synthetic-tail")

    monkeypatch.setattr(SCRIPT, "execute_tail", completed)
    assert (
        SCRIPT.main(
            [
                "--new-tail",
                "--run-id",
                "synthetic-tail",
                "--seed",
                "20260821",
                "--arm",
                "staged_97m_003",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    completion = capsys.readouterr().out
    assert "native_CE=1.250000" in completion
    assert "tail_updates=9766" in completion


def test_tail_rejects_cpu_seed_revision_and_parent_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "protein_lm.mlp.tail_orchestration._verify_source_pins", lambda *_: None
    )
    monkeypatch.setattr(
        "protein_lm.mlp.tail_orchestration._require_ignored", lambda *_: None
    )
    plan, proteins = _synthetic_plan(tmp_path / "cpu")
    with pytest.raises(ModelDataError, match="CPU"):
        execute_tail(
            root=tmp_path / "cpu",
            plan=plan,
            seed=20260821,
            device_name="mps",
            loader=lambda *_: proteins,
            code_revision="a" * 40,
        )
    with pytest.raises(ModelDataError, match="parent pin"):
        execute_tail(
            root=tmp_path / "cpu",
            plan=plan,
            seed=20260822,
            device_name="cpu",
            loader=lambda *_: proteins,
            code_revision="a" * 40,
        )

    revision_plan, revision_proteins = _synthetic_plan(tmp_path / "revision")
    revision_plan = replace(
        revision_plan,
        config=replace(revision_plan.config, parent_code_revision="a" * 40),
    )
    with pytest.raises(ModelDataError, match="identity"):
        execute_tail(
            root=tmp_path / "revision",
            plan=revision_plan,
            seed=20260821,
            device_name="cpu",
            loader=lambda *_: revision_proteins,
            code_revision="a" * 40,
        )

    state_plan, state_proteins = _synthetic_plan(tmp_path / "state")
    state_plan = replace(
        state_plan,
        config=replace(state_plan.config, parent_optimizer_steps=2),
    )
    with pytest.raises(ModelDataError, match="tail cursor"):
        execute_tail(
            root=tmp_path / "state",
            plan=state_plan,
            seed=20260821,
            device_name="cpu",
            loader=lambda *_: state_proteins,
            code_revision="a" * 40,
        )
