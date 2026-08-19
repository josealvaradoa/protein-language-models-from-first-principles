"""Synthetic checks for Week 2 bigram fitting and dual artifact formats."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from protein_lm.bigram import serialization, stream, training_config
from protein_lm.bigram.serialization import load_model_artifacts, write_model_artifacts
from protein_lm.bigram.stream import PairBatch, audit_stream, iter_pair_batches
from protein_lm.bigram.training import (
    TrainingSettings,
    all_zero_weights_sha256,
    fit_batches,
    new_training_state,
)
from protein_lm.bigram.training_config import load_training_config
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ProteinSequence


ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "experiments/week_02/bigram_training_v1.toml"
DOMAIN = "protein-lm/week2/bigram-transition-stream/v1"
CONTEXT_ROLES = [
    "BOS",
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]
TARGET_ROLES = CONTEXT_ROLES[1:] + ["EOS"]
UNIGRAM_COUNTS = torch.tensor([2, 1, 1] + [0] * 17 + [2], dtype=torch.int64)
COUNT_BIGRAM_COUNTS = torch.zeros((21, 21), dtype=torch.int64)
COUNT_BIGRAM_COUNTS[0, 0] = 2
COUNT_BIGRAM_COUNTS[1, 1] = 1
COUNT_BIGRAM_COUNTS[2, 2] = 1
COUNT_BIGRAM_COUNTS[3, 20] = 1
COUNT_BIGRAM_COUNTS[1, 20] = 1


def protein(sequence: str, accession: str = "P00001") -> ProteinSequence:
    """Build a loader-shaped synthetic protein without reading a manifest."""

    return ProteinSequence(
        primary_accession=accession,
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
        biological_length=len(sequence),
        length_bucket="synthetic",
        uniref50_group="UniRef50_SYNTHETIC",
    )


def metadata(model_type: str) -> dict[str, object]:
    """Return complete synthetic provenance required by model artifacts."""

    neural = model_type == "neural_bigram"
    return {
        "arm": "synthetic",
        "model_type": model_type,
        "context_roles": CONTEXT_ROLES,
        "target_roles": TARGET_ROLES,
        "stream_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "source_identity": {"manifest_sha256": "c" * 64},
        "code_revision": "d" * 40,
        "seed": 7,
        "prediction_pair_budget": 6,
        "batch_size": 4,
        "batches_consumed": 2,
        "optimizer_steps": 2 if neural else 0,
        "smoothing_alpha": None if neural else 1,
        "initial_weights_sha256": all_zero_weights_sha256() if neural else None,
        "optimizer": (
            {
                "name": "SGD",
                "learning_rate": 1.0,
                "momentum": 0.0,
                "weight_decay": 0.0,
            }
            if neural
            else None
        ),
    }


def synthetic_batches() -> tuple[PairBatch, ...]:
    """Return the frozen small full-plus-final schedule for ACD and A."""

    return tuple(
        iter_pair_batches(
            (protein("ACD"), protein("A", "P00002")),
            namespace="synthetic/training/v1",
            base_seed=7,
            pair_budget=6,
            batch_size=4,
        )
    )


def test_public_training_config_is_pinned_and_freezes_numerical_choices() -> None:
    config = load_training_config(CONFIG_PATH)
    assert (
        config.batch_size * config.full_batches + config.final_partial_batch_pairs
        == 100_000_000
    )
    assert config.total_optimizer_steps == config.full_batches + 1 == 1_526
    assert config.count_smoothing_alpha == 1
    assert config.role_space_size == 21
    assert config.unigram_tensor_dtype == config.count_bigram_tensor_dtype == "int64"
    assert config.neural_tensor_dtype == "float32"
    assert config.neural_tensor_shape == (21, 21)
    assert not config.neural_bias
    assert config.neural_loss == "cross_entropy"
    assert config.neural_loss_reduction == "mean"
    assert config.neural_zero_grad_set_to_none
    assert config.initial_weights_sha256 == all_zero_weights_sha256()
    assert config.stream_report_relative_path == (
        "reports/week_02/bigram_training_streams_v1.json"
    )
    assert config.stream_report_sha256 == (
        "914bcae29e989d550b2db2cc16fe2245821caa496de68e779e69b102439386b8"
    )
    assert config.serialization_formats == ("json", "safetensors")


def test_training_config_rejects_tampered_bytes(tmp_path: Path) -> None:
    path = tmp_path / "training.toml"
    path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace(
            "batch_size = 65536", "batch_size = 8"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ModelDataError, match="bytes do not match"):
        load_training_config(path)


def test_training_config_checks_types_after_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "neural_bias = false", 'neural_bias = "no"'
    )
    path = tmp_path / "wrong-type.toml"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        training_config,
        "APPROVED_BIGRAM_TRAINING_CONFIG_SHA256",
        hashlib.sha256(content.encode()).hexdigest(),
    )
    with pytest.raises(ModelDataError, match="neural_bias must be a boolean"):
        load_training_config(path)


def test_acd_shared_batches_update_counts_and_neural_with_exact_accounting() -> None:
    batches = synthetic_batches()
    assert [len(batch.contexts) for batch in batches] == [4, 2]
    assert sum(batch.proteins_started for batch in batches) == 2
    assert sum(batch.proteins_completed for batch in batches) == 2
    settings = TrainingSettings(batch_size=4, prediction_pair_budget=6)
    state = new_training_state(settings)
    initial = state.neural_weights.detach().clone()
    losses = fit_batches(state, batches, settings)
    assert len(losses) == 2
    assert state.pairs_seen == 6
    assert state.optimizer_steps == 2
    assert state.unigram_counts[:4].tolist() == [2, 1, 1, 0]
    assert state.unigram_counts[20].item() == 2
    assert state.count_bigram_counts[0, 0].item() == 2
    assert state.count_bigram_counts[1, 1].item() == 1
    assert state.count_bigram_counts[2, 2].item() == 1
    assert state.count_bigram_counts[3, 20].item() == 1
    assert not torch.equal(initial, state.neural_weights.detach())


def test_fit_rejects_undersized_and_reused_state() -> None:
    settings = TrainingSettings(batch_size=4, prediction_pair_budget=6)
    undersized = PairBatch(b"\0\0\0", b"\0\0\0", 1, 0, False)
    with pytest.raises(ModelDataError, match="fixed schedule"):
        fit_batches(new_training_state(settings), (undersized, undersized), settings)
    state = new_training_state(settings)
    fit_batches(state, synthetic_batches(), settings)
    with pytest.raises(ModelDataError, match="fresh all-zero"):
        fit_batches(state, synthetic_batches(), settings)


def test_fit_accepts_a_full_final_batch_when_budget_is_divisible() -> None:
    settings = TrainingSettings(batch_size=4, prediction_pair_budget=8)
    batches = tuple(
        iter_pair_batches(
            (protein("ACD"), protein("CDE", "P00002")),
            namespace="synthetic/divisible/v1",
            base_seed=7,
            pair_budget=8,
            batch_size=4,
        )
    )
    state = new_training_state(settings)
    fit_batches(state, batches, settings)
    assert [len(batch.contexts) for batch in batches] == [4, 4]
    assert state.optimizer_steps == 2


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_training_settings_reject_nonfinite_optimizer_values(value: float) -> None:
    with pytest.raises(ModelDataError, match="optimizer settings"):
        TrainingSettings(batch_size=1, prediction_pair_budget=1, learning_rate=value)


def test_zero_neural_initialization_is_uniform_before_sgd() -> None:
    batches = tuple(
        iter_pair_batches(
            (protein("ACD"),),
            namespace="synthetic/uniform/v1",
            base_seed=7,
            pair_budget=4,
            batch_size=4,
        )
    )
    settings = TrainingSettings(batch_size=4, prediction_pair_budget=4)
    state = new_training_state(settings)
    probabilities = torch.softmax(state.neural_weights[0], dim=0)
    assert torch.allclose(probabilities, torch.full((21,), 1 / 21))
    losses = fit_batches(state, batches, settings)
    assert losses[0] == pytest.approx(math.log(21))
    assert not torch.equal(state.neural_weights.detach(), torch.zeros((21, 21)))


def test_training_batches_and_audit_share_pair_bytes_and_boundary_behavior() -> None:
    proteins = (protein("A"), protein("C", "P00002"))
    batches = tuple(
        iter_pair_batches(
            proteins,
            namespace="synthetic/boundary/v1",
            base_seed=7,
            pair_budget=4,
            batch_size=3,
        )
    )
    audit = audit_stream(
        proteins,
        namespace="synthetic/boundary/v1",
        base_seed=7,
        pair_budget=4,
        hash_domain=DOMAIN,
    )
    hasher = hashlib.sha256()
    hasher.update(DOMAIN.encode())
    hasher.update(b"\0synthetic/boundary/v1\0" + b"7\0")
    for batch in batches:
        hasher.update(batch.pair_bytes)
    assert audit.stream_sha256 == hasher.hexdigest()
    assert b"\x01\x01" not in b"".join(batch.pair_bytes for batch in batches)


def test_audit_uses_bounded_default_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    actual = stream.iter_pair_batches
    sizes: list[int] = []

    def recording_batches(*args, **kwargs):
        sizes.append(kwargs["batch_size"])
        yield from actual(*args, **kwargs)

    monkeypatch.setattr(stream, "iter_pair_batches", recording_batches)
    audit_stream(
        (protein("A"),),
        namespace="synthetic/bounded-audit/v1",
        base_seed=7,
        pair_budget=2,
        hash_domain=DOMAIN,
    )
    assert sizes == [65_536]


@pytest.mark.parametrize(
    ("model_type", "tensor"),
    (
        ("unigram", UNIGRAM_COUNTS),
        ("count_bigram", COUNT_BIGRAM_COUNTS),
        (
            "neural_bigram",
            torch.arange(441, dtype=torch.float32).reshape(21, 21) / 10,
        ),
    ),
)
def test_json_and_safetensors_are_exactly_equivalent(
    tmp_path: Path, model_type: str, tensor: torch.Tensor
) -> None:
    json_path = tmp_path / f"{model_type}.json"
    safetensors_path = tmp_path / f"{model_type}.safetensors"
    write_model_artifacts(
        json_path=json_path,
        safetensors_path=safetensors_path,
        model_type=model_type,  # type: ignore[arg-type]
        tensor=tensor,
        metadata=metadata(model_type),
    )
    loaded_type, loaded_tensor, loaded_metadata = load_model_artifacts(
        json_path=json_path,
        safetensors_path=safetensors_path,
    )
    assert loaded_type == model_type
    assert loaded_tensor.dtype == tensor.dtype
    assert torch.equal(loaded_tensor, tensor)
    assert loaded_metadata == metadata(model_type)
    with pytest.raises(ModelDataError, match="already exists"):
        write_model_artifacts(
            json_path=json_path,
            safetensors_path=safetensors_path,
            model_type=model_type,  # type: ignore[arg-type]
            tensor=tensor,
            metadata=metadata(model_type),
        )


def test_writer_verifies_staged_pair_before_installing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / "model.json"
    safetensors_path = tmp_path / "model.safetensors"
    monkeypatch.setattr(
        serialization,
        "load_model_artifacts",
        lambda **_: (_ for _ in ()).throw(ModelDataError("staged failure")),
    )
    with pytest.raises(ModelDataError, match="staged failure"):
        write_model_artifacts(
            json_path=json_path,
            safetensors_path=safetensors_path,
            model_type="unigram",
            tensor=UNIGRAM_COUNTS,
            metadata=metadata("unigram"),
        )
    assert not json_path.exists()
    assert not safetensors_path.exists()


def test_malformed_or_mismatched_model_artifacts_fail_closed(tmp_path: Path) -> None:
    json_path = tmp_path / "model.json"
    safetensors_path = tmp_path / "model.safetensors"
    tensor = torch.zeros((21, 21), dtype=torch.float32)
    write_model_artifacts(
        json_path=json_path,
        safetensors_path=safetensors_path,
        model_type="neural_bigram",
        tensor=tensor,
        metadata=metadata("neural_bigram"),
    )
    json_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ModelDataError, match="schema"):
        load_model_artifacts(json_path=json_path, safetensors_path=safetensors_path)
    other_json = tmp_path / "other.json"
    other_safetensors = tmp_path / "other.safetensors"
    write_model_artifacts(
        json_path=other_json,
        safetensors_path=other_safetensors,
        model_type="neural_bigram",
        tensor=tensor,
        metadata=metadata("neural_bigram"),
    )
    save_file(
        {"parameters": torch.ones((21, 21), dtype=torch.float32)},
        other_safetensors,
        metadata={
            "schema_version": "1",
            "model_type": "neural_bigram",
            "tensor_name": "parameters",
            "dtype": "float32",
            "shape": "21,21",
        },
    )
    with pytest.raises(ModelDataError, match="values disagree"):
        load_model_artifacts(
            json_path=other_json,
            safetensors_path=other_safetensors,
        )


def test_artifact_provenance_and_nonfinite_neural_values_fail_closed(
    tmp_path: Path,
) -> None:
    invalid_metadata = metadata("neural_bigram")
    invalid_metadata["stream_sha256"] = "not-a-checksum"
    with pytest.raises(ModelDataError, match="identity"):
        write_model_artifacts(
            json_path=tmp_path / "bad.json",
            safetensors_path=tmp_path / "bad.safetensors",
            model_type="neural_bigram",
            tensor=torch.zeros((21, 21), dtype=torch.float32),
            metadata=invalid_metadata,
        )
    with pytest.raises(ModelDataError, match="nonfinite"):
        write_model_artifacts(
            json_path=tmp_path / "nan.json",
            safetensors_path=tmp_path / "nan.safetensors",
            model_type="neural_bigram",
            tensor=torch.full((21, 21), float("nan")),
            metadata=metadata("neural_bigram"),
        )


def test_artifact_roles_counts_and_batch_accounting_fail_closed(tmp_path: Path) -> None:
    wrong_roles = metadata("unigram")
    wrong_roles["context_roles"] = ["BOS"] + ["X"] * 20
    with pytest.raises(ModelDataError, match="roles"):
        write_model_artifacts(
            json_path=tmp_path / "roles.json",
            safetensors_path=tmp_path / "roles.safetensors",
            model_type="unigram",
            tensor=UNIGRAM_COUNTS,
            metadata=wrong_roles,
        )
    wrong_counts = metadata("count_bigram")
    with pytest.raises(ModelDataError, match="tensor accounting"):
        write_model_artifacts(
            json_path=tmp_path / "counts.json",
            safetensors_path=tmp_path / "counts.safetensors",
            model_type="count_bigram",
            tensor=torch.zeros((21, 21), dtype=torch.int64),
            metadata=wrong_counts,
        )
    wrong_batches = metadata("unigram")
    wrong_batches["batches_consumed"] = 1
    with pytest.raises(ModelDataError, match="batch accounting"):
        write_model_artifacts(
            json_path=tmp_path / "batches.json",
            safetensors_path=tmp_path / "batches.safetensors",
            model_type="unigram",
            tensor=UNIGRAM_COUNTS,
            metadata=wrong_batches,
        )
    wrong_steps = metadata("neural_bigram")
    wrong_steps["optimizer_steps"] = 1
    with pytest.raises(ModelDataError, match="optimizer metadata"):
        write_model_artifacts(
            json_path=tmp_path / "steps.json",
            safetensors_path=tmp_path / "steps.safetensors",
            model_type="neural_bigram",
            tensor=torch.zeros((21, 21), dtype=torch.float32),
            metadata=wrong_steps,
        )
