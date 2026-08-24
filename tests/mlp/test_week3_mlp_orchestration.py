"""Synthetic durable-run checks without loading project collections."""

from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from protein_lm.data.model_data.loaders import ModelDataCollection
from protein_lm.data.model_data.loaders import ProteinSequence
from protein_lm.mlp import orchestration
from protein_lm.mlp.config import load_config
from protein_lm.mlp.orchestration import MLPPlan, execute_run


CONFIG_PATH = Path(__file__).parents[2] / "experiments/week_03/mlp_training_v1.toml"


def protein(sequence: str, accession: str = "P00001") -> ProteinSequence:
    return ProteinSequence(
        accession,
        sequence,
        hashlib.sha256(sequence.encode()).hexdigest(),
        len(sequence),
        "synthetic",
        "UniRef50_SYNTHETIC",
    )


def test_interrupted_run_retains_history_and_latest_checkpoint_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        load_config(CONFIG_PATH),
        prediction_budget=8,
        batch_size=4,
        milestone_predictions=(8,),
        checkpoint_predictions=(4, 8),
    )
    plan = MLPPlan(config, CONFIG_PATH, "synthetic-run", tmp_path / "synthetic-run")
    proteins = (protein("ACD"), protein("ACDE", "P00002"))

    def loader(_root: Path, collection: ModelDataCollection):
        assert collection in {
            ModelDataCollection.FAMILY_AWARE_TRAINING,
            ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION,
        }
        return proteins

    monkeypatch.setattr(orchestration, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(orchestration, "_require_ignored", lambda *_: None)

    native_calls = 0
    original_evaluate = orchestration.evaluate_native

    def counting_evaluate(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(orchestration, "evaluate_native", counting_evaluate)

    def interrupt(event: str, payload: dict[str, object]) -> None:
        if event == "milestone" and payload["predictions"] == 8:
            raise RuntimeError("synthetic interruption")

    with pytest.raises(Exception, match="synthetic interruption"):
        execute_run(
            root=tmp_path,
            plan=plan,
            seed=20260821,
            device_name="cpu",
            loader=loader,
            code_revision="a" * 40,
            progress_callback=interrupt,
        )
    status_path = plan.destination / "run_status.json"
    failed = json.loads(status_path.read_text())
    assert failed["status"] == "failed"
    assert failed["predictions_seen"] == 8
    assert [item["predictions"] for item in failed["native_validation"]] == [8]
    checkpoint = plan.destination / "checkpoint-4"
    execute_run(
        root=tmp_path,
        plan=plan,
        seed=20260821,
        device_name="cpu",
        loader=loader,
        resume_checkpoint=checkpoint,
        code_revision="a" * 40,
    )
    passed = json.loads(status_path.read_text())
    assert passed["status"] == "passed"
    assert [item["predictions"] for item in passed["native_validation"]] == [8]
    assert native_calls == 2
    with pytest.raises(Exception, match="not resumable"):
        execute_run(
            root=tmp_path,
            plan=plan,
            seed=20260821,
            device_name="cpu",
            loader=loader,
            resume_checkpoint=plan.destination / "checkpoint-8",
            code_revision="a" * 40,
        )
