"""Synthetic contract checks for the frozen Week 2 bigram evaluation boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from protein_lm.bigram import (
    evaluation_execution,
    evaluation_plan,
    evaluation_validation,
)
from protein_lm.bigram.evaluation_execution import execute_evaluation
from protein_lm.bigram.evaluation_contract import config_sha256, load_evaluation_config
from protein_lm.bigram.evaluation_metrics import (
    _median,
    _median_bounds,
    score_collection,
)
from protein_lm.bigram.evaluation_models import log_probabilities
from protein_lm.bigram.evaluation_plan import EvaluationPlan, validate_plan
from protein_lm.bigram.evaluation_reporting import write_run_record
from protein_lm.bigram.evaluation_validation import validate_evaluation
from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import ModelDataCollection, ProteinSequence


ROOT = Path(__file__).parents[2]
REVISION = "a" * 40


def protein(sequence: str, accession: str, bucket: str) -> ProteinSequence:
    return ProteinSequence(
        primary_accession=accession,
        sequence=sequence,
        sequence_sha256=hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        biological_length=len(sequence),
        length_bucket=bucket,
        uniref50_group=f"group-{accession}",
    )


def test_probability_models_use_float64_smoothing_and_frozen_neural_logits() -> None:
    unigram = torch.zeros(21, dtype=torch.int64)
    unigram[0] = 2
    unigram_table = log_probabilities("unigram", unigram)
    assert unigram_table.dtype == torch.float64
    assert torch.exp(unigram_table[0, 0]).item() == pytest.approx(3 / 23)

    counts = torch.zeros((21, 21), dtype=torch.int64)
    counts[0, 0] = 2
    bigram_table = log_probabilities("count_bigram", counts)
    assert torch.exp(bigram_table[0, 0]).item() == pytest.approx(3 / 23)
    assert torch.exp(bigram_table[1, 0]).item() == pytest.approx(1 / 21)

    weights = torch.zeros((21, 21), dtype=torch.float32)
    weights[0, 3] = 2
    neural_table = log_probabilities("neural_bigram", weights)
    assert neural_table.dtype == torch.float64
    assert torch.exp(neural_table[0]).sum().item() == pytest.approx(1.0)
    assert weights[0, 3].item() == 2.0


def test_contract_pins_full_roles_and_candidate_model_and_data_identities() -> None:
    config = load_evaluation_config(
        ROOT / "experiments/week_02/bigram_evaluation_v1.toml"
    )
    assert config.valid_context_roles == (
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
    )
    assert config.valid_target_roles[-2:] == ("Y", "EOS")
    assert config.model_candidate_registry_sha256 == (
        "18bca1ec67b639d0ae68ee022a18cb2cdd9e9f571c7b305e5c9960c3a8257e5f"
    )
    assert config.model_data_registry_sha256 == (
        "13b8e1b3bb371df46f6d363b20882b91a06dde51c64d39b4e5406e0dc44efb5c"
    )


def test_boundary_pairs_tie_rule_and_odd_even_median_bounds() -> None:
    table = log_probabilities(
        "neural_bigram", torch.zeros((21, 21), dtype=torch.float32)
    )
    scored = score_collection([protein("A", "P1", "32-127")], table, ("32-127",))
    assert scored.overall.token_count == 2  # BOS->A and A->EOS
    assert (
        scored.overall.correct_tokens == 1
    )  # lowest target ID wins the zero-logit tie
    assert (
        scored.overall.median_lower_per_protein_nll
        == scored.overall.median_upper_per_protein_nll
    )
    assert _median_bounds([1.0, 3.0]) == (1.0, 3.0)
    assert _median_bounds([1.0, 2.0, 3.0]) == (2.0, 2.0)
    assert _median([1.0, 3.0]) == 2.0


def test_empty_frozen_bucket_is_rejected() -> None:
    table = log_probabilities(
        "neural_bigram", torch.zeros((21, 21), dtype=torch.float32)
    )
    with pytest.raises(ModelDataError, match="accounting"):
        score_collection([protein("A", "P1", "32-127")], table, ("32-127", "128-255"))


def _plan(tmp_path: Path) -> EvaluationPlan:
    config_path = ROOT / "experiments/week_02/bigram_evaluation_v1.toml"
    config = load_evaluation_config(config_path)
    return EvaluationPlan(
        root=tmp_path,
        evaluation_id="synthetic-evaluation-1",
        destination=tmp_path
        / "data/processed/week_02/bigram_evaluation_candidates/synthetic-evaluation-1",
        config=config,
        config_sha256=config_sha256(config_path),
        model_candidate=tmp_path / "model-candidate",
        model_data_registry=tmp_path / "model-data-registry.json",
    )


def _tables() -> dict[tuple[str, str], torch.Tensor]:
    table = log_probabilities(
        "neural_bigram", torch.zeros((21, 21), dtype=torch.float32)
    )
    return {
        (arm, model): table
        for arm in ("random_training", "family_aware_training")
        for model in ("unigram", "count_bigram", "neural_bigram")
    }


def _patch_execution(monkeypatch: pytest.MonkeyPatch, plan: EvaluationPlan) -> None:
    monkeypatch.setattr(
        evaluation_execution, "validate_plan", lambda _root, _plan: None
    )
    monkeypatch.setattr(
        evaluation_execution, "require_ignored", lambda _root, _path: None
    )
    monkeypatch.setattr(
        evaluation_execution, "verify_candidate_provenance", lambda _plan: None
    )
    monkeypatch.setattr(
        evaluation_execution,
        "model_candidate_preflight",
        lambda _root, _id: SimpleNamespace(destination=plan.model_candidate),
    )
    monkeypatch.setattr(evaluation_execution, "validate_candidate", lambda *_args: {})
    monkeypatch.setattr(
        evaluation_execution, "_load_models", lambda _candidate: _tables()
    )


def _five_buckets(prefix: str) -> tuple[ProteinSequence, ...]:
    return tuple(
        protein(sequence, f"{prefix}{index}", bucket)
        for index, (sequence, bucket) in enumerate(
            zip(
                "ACDEF",
                load_evaluation_config(
                    ROOT / "experiments/week_02/bigram_evaluation_v1.toml"
                ).length_buckets,
                strict=True,
            )
        )
    )


def test_execution_loads_only_three_allowed_collections_and_writes_twelve_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    _patch_execution(monkeypatch, plan)
    calls: list[ModelDataCollection] = []
    collections = {
        ModelDataCollection.RANDOM_NATIVE_VALIDATION: _five_buckets("R"),
        ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION: _five_buckets("F"),
        ModelDataCollection.SHARED_VALIDATION: _five_buckets("S"),
    }

    def loader(_root: Path, collection: ModelDataCollection):
        calls.append(collection)
        return collections[collection]

    destination = execute_evaluation(
        root=tmp_path, plan=plan, loader=loader, code_revision=REVISION
    )
    assert calls == [
        ModelDataCollection.RANDOM_NATIVE_VALIDATION,
        ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION,
        ModelDataCollection.SHARED_VALIDATION,
    ]
    assert not hasattr(ModelDataCollection, "SHARED_SEALED_TEST")
    result = json.loads((destination / "evaluation.json").read_text())
    assert len(result["records"]) == 12
    record = json.loads((destination / "run_record.json").read_text())
    assert record["collection_loads"] == {
        "random_native_validation": 1,
        "family_aware_native_validation": 1,
        "shared_validation": 1,
        "shared_sealed_test": 0,
    }
    assert all(record["hard_gates"].values())


def test_strict_plan_rejects_redirected_write_target() -> None:
    config_path = ROOT / "experiments/week_02/bigram_evaluation_v1.toml"
    config = load_evaluation_config(config_path)
    plan = EvaluationPlan(
        root=ROOT,
        evaluation_id="strict-plan-1",
        destination=ROOT / config.output_root_relative_path / "strict-plan-1",
        config=config,
        config_sha256=config_sha256(config_path),
        model_candidate=ROOT / config.model_candidate_relative_path,
        model_data_registry=ROOT / config.model_data_registry_relative_path,
    )
    validate_plan(ROOT, plan)
    with pytest.raises(ModelDataError, match="escapes"):
        validate_plan(ROOT, replace(plan, destination=ROOT / "wrong"))
    with pytest.raises(ModelDataError, match="targets"):
        validate_plan(ROOT, replace(plan, config_sha256="0" * 64))
    with pytest.raises(ModelDataError, match="identifier"):
        validate_plan(
            ROOT,
            replace(
                plan,
                evaluation_id="../escape",
                destination=ROOT / config.output_root_relative_path / "../escape",
            ),
        )
    with pytest.raises(ModelDataError, match="escapes"):
        validate_plan(
            ROOT,
            replace(
                plan, destination=ROOT / config.output_root_relative_path / "../escape"
            ),
        )


def test_overwrite_clean_revision_and_failed_validation_preserve_a_failed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    _patch_execution(monkeypatch, plan)
    plan.destination.mkdir(parents=True)
    with pytest.raises(ModelDataError, match="already exists"):
        execute_evaluation(root=tmp_path, plan=plan, loader=lambda *_args: ())

    clean_plan = _plan(tmp_path / "clean")
    _patch_execution(monkeypatch, clean_plan)
    monkeypatch.setattr(
        evaluation_execution,
        "clean_revision",
        lambda _root: (_ for _ in ()).throw(
            ModelDataError("evaluation requires a clean committed revision")
        ),
    )
    with pytest.raises(ModelDataError, match="clean committed"):
        execute_evaluation(root=tmp_path, plan=clean_plan, loader=lambda *_args: ())
    assert not clean_plan.destination.exists()

    failed_plan = _plan(tmp_path / "failed")
    _patch_execution(monkeypatch, failed_plan)
    monkeypatch.setattr(
        evaluation_execution,
        "validate_candidate",
        lambda *_args: (_ for _ in ()).throw(ModelDataError("tampered model")),
    )
    with pytest.raises(ModelDataError, match="tampered model"):
        execute_evaluation(
            root=tmp_path,
            plan=failed_plan,
            loader=lambda *_args: (_ for _ in ()).throw(
                AssertionError("loader called")
            ),
            code_revision=REVISION,
        )
    record = json.loads((failed_plan.destination / "run_record.json").read_text())
    assert record["status"] == "failed"
    assert record["failure_reason"] == "tampered model"
    assert record["hard_gates"]["candidate_validation"] is False


def test_candidate_data_registry_provenance_rejects_drift(tmp_path: Path) -> None:
    config = load_evaluation_config(
        ROOT / "experiments/week_02/bigram_evaluation_v1.toml"
    )
    record = tmp_path / "run_record.json"
    record.write_text(
        json.dumps({"source_identity": {"model_data_registry": "0" * 64}})
    )
    with pytest.raises(ModelDataError, match="data registry"):
        evaluation_plan._verify_candidate_data_provenance(record, config)


def _completed_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> EvaluationPlan:
    plan = _plan(tmp_path)
    _patch_execution(monkeypatch, plan)
    collections = {
        ModelDataCollection.RANDOM_NATIVE_VALIDATION: _five_buckets("R"),
        ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION: _five_buckets("F"),
        ModelDataCollection.SHARED_VALIDATION: _five_buckets("S"),
    }
    execute_evaluation(
        root=tmp_path,
        plan=plan,
        loader=lambda _root, collection: collections[collection],
        code_revision=REVISION,
    )
    monkeypatch.setattr(
        evaluation_validation, "verify_candidate_provenance", lambda _plan: None
    )
    return plan


@pytest.mark.parametrize(
    "tamper",
    (
        "bool",
        "nonfinite",
        "arithmetic",
        "median",
        "odd_midpoint",
        "population",
        "registry",
    ),
)
def test_read_only_validator_rejects_tampering_without_a_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    plan = _completed_candidate(tmp_path, monkeypatch)
    result_path = plan.destination / "evaluation.json"
    if tamper == "nonfinite":
        result_path.write_text('{"total": NaN}\n', encoding="utf-8")
    else:
        payload = json.loads(result_path.read_text())
        overall = payload["records"][0]["metrics"]["overall"]
        if tamper == "bool":
            overall["token_count"] = True
        elif tamper == "arithmetic":
            overall["cross_entropy"] = 9.0
        elif tamper == "median":
            overall["median_upper_per_protein_nll"] += 1.0
        elif tamper == "odd_midpoint":
            midpoint = overall["median_per_protein_nll"]
            overall["median_lower_per_protein_nll"] = midpoint - 0.1
            overall["median_upper_per_protein_nll"] = midpoint + 0.1
        elif tamper == "population":
            changed = payload["records"][0]["metrics"]
            nll = 3.044522437723423
            for aggregate in [
                changed["overall"],
                *changed["length_buckets"].values(),
            ]:
                aggregate["token_count"] = 3
                aggregate["total_nll"] = nll * 3
                aggregate["correct_tokens"] = 0
                aggregate["cross_entropy"] = nll
                aggregate["accuracy"] = 0.0
            changed["overall"]["token_count"] = 15
            changed["overall"]["total_nll"] = nll * 15
            changed["overall"]["correct_tokens"] = 0
        elif tamper == "registry":
            result_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with pytest.raises(ModelDataError, match="checksum"):
                validate_evaluation(plan.destination, plan)
            return
        result_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    expected = "population" if tamper == "population" else None
    with pytest.raises(ModelDataError, match=expected):
        validate_evaluation(plan.destination, plan)


def test_run_record_replacement_is_atomic_and_preserves_prior_state_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run_record.json"
    write_run_record(path, {"status": "running"})
    write_run_record(path, {"status": "passed"})
    assert json.loads(path.read_text()) == {"status": "passed"}
    monkeypatch.setattr(
        "protein_lm.bigram.evaluation_reporting.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(ModelDataError, match="replace"):
        write_run_record(path, {"status": "failed"})
    assert json.loads(path.read_text()) == {"status": "passed"}


def test_no_flag_cli_only_preflights_without_writes_or_loaders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = ROOT / "scripts/evaluate_week2_bigrams.py"
    spec = importlib.util.spec_from_file_location("week2_evaluation_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = _plan(tmp_path)
    monkeypatch.setattr(module, "preflight", lambda *_args: plan)
    monkeypatch.setattr(sys, "argv", [str(script)])
    assert module.main() == 0
    assert not plan.destination.exists()
