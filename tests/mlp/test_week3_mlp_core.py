"""Synthetic checks for Week 3's streaming MLP mechanics."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ProteinSequence
from protein_lm.mlp.checkpoint import load_checkpoint, save_checkpoint
from protein_lm.mlp import config as mlp_config
from protein_lm.mlp.config import load_config
from protein_lm.mlp.metrics import evaluate_native
from protein_lm.mlp.model import ContextMLP, parameter_count, resolve_device
from protein_lm.mlp import stream as mlp_stream
from protein_lm.mlp.stream import (
    StreamCursor,
    iter_context_batches,
    iter_native_context_batches,
    protein_context_pairs,
)
from protein_lm.mlp.training import (
    TrainingState,
    learning_rate_for,
    new_optimizer,
    train_batch,
)


ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "experiments/week_03/mlp_training_v1.toml"


def protein(sequence: str, accession: str = "P00001") -> ProteinSequence:
    return ProteinSequence(
        accession,
        sequence,
        hashlib.sha256(sequence.encode()).hexdigest(),
        len(sequence),
        "synthetic",
        "UniRef50_SYNTHETIC",
    )


def tiny_config():
    config = load_config(CONFIG_PATH)
    return replace(
        config,
        prediction_budget=8,
        batch_size=4,
        milestone_predictions=(4, 8),
        checkpoint_predictions=(4, 8),
    )


def test_config_and_model_contract_are_exact() -> None:
    config = load_config(CONFIG_PATH)
    assert config.parameter_count == 274_293
    assert config.batch_size == 1_024
    assert [
        config.expected_optimizer_steps(position)
        for position in config.checkpoint_predictions
    ] == [977, 4_884, 9_767, 24_416, 48_831, 87_894, 97_660]
    assert config.event_predictions == (
        1_000_000,
        5_000_000,
        10_000_000,
        25_000_000,
        50_000_000,
        90_000_000,
        100_000_000,
    )
    model = ContextMLP(config, 20260821, resolve_device("cpu"))
    assert parameter_count(model) == 274_293
    assert model(torch.zeros((2, 10), dtype=torch.int64)).shape == (2, 21)
    same = ContextMLP(config, 20260821, resolve_device("cpu"))
    different = ContextMLP(config, 20260822, resolve_device("cpu"))
    assert all(
        torch.equal(first, second)
        for first, second in zip(model.parameters(), same.parameters(), strict=True)
    )
    assert any(
        not torch.equal(first, second)
        for first, second in zip(
            model.parameters(), different.parameters(), strict=True
        )
    )
    if not torch.backends.mps.is_available():
        with pytest.raises(ModelDataError, match="MPS"):
            resolve_device("mps")


def test_initialization_matches_the_frozen_pilot_draw_order() -> None:
    config = load_config(CONFIG_PATH)
    model = ContextMLP(config, 20260821, resolve_device("cpu"))
    generator = torch.Generator(device="cpu").manual_seed(20260821)
    embedding = torch.randn((21, 32), generator=generator) * 32**-0.5
    w1 = torch.randn((320, 800), generator=generator) * 320**-0.5
    w2 = torch.randn((800, 21), generator=generator) * 800**-0.5
    assert torch.equal(model.embedding, embedding)
    assert torch.equal(model.w1, w1)
    assert torch.equal(model.b1, torch.zeros(800))
    assert torch.equal(model.w2, w2)
    assert torch.equal(model.b2, torch.zeros(21))
    contexts = torch.tensor([[0] * 10, list(range(1, 11))], dtype=torch.int64)
    reference = torch.tanh(embedding[contexts].flatten(1) @ w1) @ w2
    assert torch.equal(model(contexts), reference)


def test_parameters_stay_float32_when_the_global_default_changes() -> None:
    baseline = ContextMLP(load_config(CONFIG_PATH), 20260821, resolve_device("cpu"))
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = ContextMLP(load_config(CONFIG_PATH), 20260821, resolve_device("cpu"))
    finally:
        torch.set_default_dtype(previous)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(
            baseline.parameters(), model.parameters(), strict=True
        )
    )


def test_config_tamper_and_type_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = CONFIG_PATH.read_text(encoding="utf-8")
    tampered = tmp_path / "tampered.toml"
    tampered.write_text(
        original.replace("batch_size = 1024", "batch_size = 8"), encoding="utf-8"
    )
    with pytest.raises(ModelDataError, match="bytes do not match"):
        load_config(tampered)
    wrong_type = original.replace("context_length = 10", 'context_length = "10"')
    typed = tmp_path / "typed.toml"
    typed.write_text(wrong_type, encoding="utf-8")
    monkeypatch.setattr(
        mlp_config,
        "APPROVED_MLP_TRAINING_CONFIG_SHA256",
        hashlib.sha256(wrong_type.encode()).hexdigest(),
    )
    with pytest.raises(ModelDataError, match="context_length must be an integer"):
        load_config(typed)


def test_c10_coordinates_event_batches_and_resume_suffix() -> None:
    contexts, targets = protein_context_pairs("ACD")
    assert contexts[:10] == (0,) * 10
    assert contexts[10:20] == (0,) * 9 + (1,)
    assert contexts[30:40] == (0,) * 7 + (1, 2, 3)
    assert targets == (0, 1, 2, 20)
    values = tuple(
        iter_context_batches(
            (protein("ACD"), protein("A", "P00002")),
            namespace="synthetic/week3",
            base_seed=7,
            prediction_budget=6,
            batch_size=4,
            event_predictions=(3, 6),
        )
    )
    assert [batch.predictions for batch in values] == [3, 3]
    suffix = tuple(
        iter_context_batches(
            (protein("ACD"), protein("A", "P00002")),
            namespace="synthetic/week3",
            base_seed=7,
            prediction_budget=6,
            batch_size=4,
            event_predictions=(3, 6),
            cursor=values[0].end_cursor,
        )
    )
    assert torch.equal(values[1].contexts, suffix[0].contexts)
    assert torch.equal(values[1].targets, suffix[0].targets)
    assert values[0].end_cursor == StreamCursor(
        3,
        values[0].end_cursor.protein_index,
        values[0].end_cursor.within_protein_target_offset,
    )
    with pytest.raises(ModelDataError, match="cannot satisfy"):
        tuple(
            iter_context_batches(
                (protein("A"),),
                namespace="x",
                base_seed=1,
                prediction_budget=3,
                batch_size=2,
            )
        )


def test_cursor_restarts_inside_and_at_a_protein_boundary_without_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proteins = (protein("ACDEFG"), protein("KLMNPQ", "P00002"))
    ordered = mlp_stream.ordered_proteins(proteins, "synthetic/cursor", 7)
    first = ordered[0]
    inside = StreamCursor(2, 0, 2)
    called: list[tuple[str, int]] = []
    original = mlp_stream._iter_protein_context_pairs

    def recording(sequence: str, context_length: int, offset: int):
        called.append((sequence, offset))
        yield from original(sequence, context_length, offset)

    monkeypatch.setattr(mlp_stream, "_iter_protein_context_pairs", recording)
    suffix = tuple(
        iter_context_batches(
            ordered,
            namespace="synthetic/cursor",
            base_seed=7,
            prediction_budget=14,
            batch_size=4,
            cursor=inside,
        )
    )
    assert called[0] == (first.sequence, 2)
    assert suffix[0].start_cursor == inside
    boundary = StreamCursor(first.biological_length + 1, 1, 0)
    called.clear()
    tuple(
        iter_context_batches(
            ordered,
            namespace="synthetic/cursor",
            base_seed=7,
            prediction_budget=14,
            batch_size=4,
            cursor=boundary,
        )
    )
    assert all(sequence != first.sequence for sequence, _ in called)


def test_lr_metrics_and_nonpickle_checkpoint_resume(tmp_path: Path) -> None:
    config = tiny_config()
    assert learning_rate_for(89_999_999, load_config(CONFIG_PATH)) == 0.1
    assert learning_rate_for(90_000_000, load_config(CONFIG_PATH)) == 0.01
    proteins = (protein("ACD"), protein("ACDE", "P00002"))
    uninterrupted = ContextMLP(config, 20260821, resolve_device("cpu"))
    optimizer = new_optimizer(uninterrupted, config)
    state = TrainingState()
    batches = tuple(
        iter_context_batches(
            proteins,
            namespace="synthetic/resume",
            base_seed=7,
            prediction_budget=8,
            batch_size=4,
            event_predictions=config.event_predictions,
        )
    )
    train_batch(uninterrupted, optimizer, batches[0], state, config)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint-4",
        model=uninterrupted,
        optimizer=optimizer,
        state=state,
        config=config,
        config_path=CONFIG_PATH,
        seed=20260821,
        run_id="synthetic-run",
        device_name="cpu",
        code_revision="d" * 40,
    )
    train_batch(uninterrupted, optimizer, batches[1], state, config)
    resumed = ContextMLP(config, 20260821, resolve_device("cpu"))
    resumed_optimizer = new_optimizer(resumed, config)
    with pytest.raises(ModelDataError, match="identity"):
        load_checkpoint(
            checkpoint,
            model=resumed,
            optimizer=resumed_optimizer,
            config=config,
            config_path=CONFIG_PATH,
            seed=20260821,
            run_id="synthetic-run",
            device_name="cpu",
            code_revision="e" * 40,
        )
    resumed_state = load_checkpoint(
        checkpoint,
        model=resumed,
        optimizer=resumed_optimizer,
        config=config,
        config_path=CONFIG_PATH,
        seed=20260821,
        run_id="synthetic-run",
        device_name="cpu",
        code_revision="d" * 40,
    )
    suffix = tuple(
        iter_context_batches(
            proteins,
            namespace="synthetic/resume",
            base_seed=7,
            prediction_budget=8,
            batch_size=4,
            event_predictions=config.event_predictions,
            cursor=resumed_state.cursor,
        )
    )
    train_batch(resumed, resumed_optimizer, suffix[0], resumed_state, config)
    assert resumed_state == state
    assert all(
        torch.equal(first, second)
        for first, second in zip(
            uninterrupted.parameters(), resumed.parameters(), strict=True
        )
    )
    with pytest.raises(ModelDataError, match="already exists"):
        save_checkpoint(
            checkpoint,
            model=resumed,
            optimizer=resumed_optimizer,
            state=resumed_state,
            config=config,
            config_path=CONFIG_PATH,
            seed=20260821,
            run_id="synthetic-run",
            device_name="cpu",
            code_revision="d" * 40,
        )
    tensor_path = checkpoint / "model.safetensors"
    renamed = checkpoint.with_name("checkpoint-5")
    checkpoint.rename(renamed)
    with pytest.raises(ModelDataError, match="identity"):
        load_checkpoint(
            renamed,
            model=resumed,
            optimizer=resumed_optimizer,
            config=config,
            config_path=CONFIG_PATH,
            seed=20260821,
            run_id="synthetic-run",
            device_name="cpu",
            code_revision="d" * 40,
        )
    renamed.rename(checkpoint)
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")
    with pytest.raises(ModelDataError, match="integrity"):
        load_checkpoint(
            checkpoint,
            model=resumed,
            optimizer=resumed_optimizer,
            config=config,
            config_path=CONFIG_PATH,
            seed=20260821,
            run_id="synthetic-run",
            device_name="cpu",
            code_revision="d" * 40,
        )
    native = evaluate_native(
        resumed,
        iter_native_context_batches(
            proteins, namespace="synthetic/native", base_seed=7, batch_size=3
        ),
    )
    assert native.predictions == 9
    assert native.nll_numerator > 0 and 0 <= native.accuracy <= 1
