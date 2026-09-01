"""Synthetic checks for the frozen Week 4 reproduction contract."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "experiments/week_04/foundations_reproduction_v1.toml"


def _contract() -> dict[str, object]:
    return tomllib.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _load_script(name: str):
    path = ROOT / "scripts" / name
    specification = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_contract_schema_locked_values_and_source_bytes() -> None:
    contract = _contract()
    assert set(contract) == {
        "schema_version",
        "schema_identifier",
        "contract_identifier",
        "scope",
        "status",
        "evidence_scope",
        "tolerances",
        "identity",
        "operator_boundary",
        "run_bundle",
        "week_01_identity",
        "week_02_identity",
        "week_03_identity",
        "comparison_scope",
        "source_pins",
        "stages",
        "models",
        "metric_targets",
        "comparison_claims",
        "exclusions",
    }
    assert contract["schema_version"] == 1
    assert contract["schema_identifier"] == "protein_lm.foundations_reproduction_contract"
    assert contract["contract_identifier"] == "2026-09-01-week-04-foundations-reproduction-v1"
    assert contract["status"] == "frozen_contract_not_yet_executable"

    evidence = contract["evidence_scope"]
    assert evidence == {
        "week_01": "complete_closed_a004_aggregate_read_only_audit",
        "week_02": "complete_final_bigram_evaluation_report",
        "week_03": "final_c10_c20_e64_three_seed_comparison",
        "historical_evidence_mutable": False,
        "public_scope": "aggregate_only_no_sequences_no_accessions_no_family_ids_no_raw_tensors",
    }
    tolerances = contract["tolerances"]
    assert tolerances["reevaluation_cross_entropy_absolute"] == 0.000001
    assert tolerances["retraining_cross_entropy_absolute"] == 0.0001
    assert tolerances["material_cross_entropy_gap"] == 0.001
    assert tolerances["comparison_operator"] == "absolute_delta_lte_tolerance"

    pins = contract["source_pins"]
    assert len(pins) == 15
    paths = [pin["relative_path"] for pin in pins]
    assert len(paths) == len(set(paths))
    assert str(CONTRACT_PATH.relative_to(ROOT)) not in paths
    for pin in pins:
        relative_path = Path(pin["relative_path"])
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        assert (ROOT / relative_path).is_file()
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == pin["sha256"]
    assert next(
        pin
        for pin in pins
        if pin["relative_path"] == "experiments/week_03/mlp_capacity_screen_v1.toml"
    )["sha256"] == "78e52264ed569a4f5b4592cab7dada6dc7e71ded1d40ab507909bc2598ee459b"
    assert next(pin for pin in pins if pin["relative_path"] == "pyproject.toml") == {
        "kind": "project_dependency_manifest",
        "relative_path": "pyproject.toml",
        "sha256": "ce77f471c92733c7278ba9a1efea276eda5da6d6d2aaa22bf5dc2fed7e9cc825",
    }
    assert next(pin for pin in pins if pin["relative_path"] == "uv.lock") == {
        "kind": "project_lockfile",
        "relative_path": "uv.lock",
        "sha256": "0a273fe208a50476ef04af95f92e20dfa3ad71575e24c78d29435b9a3d607cc3",
    }


def test_contract_locks_identity_models_stages_and_bundle_layout() -> None:
    contract = _contract()
    identity = contract["identity"]
    assert identity["sealed_test_access"] == "denied"
    assert identity["network_requests_made"] == 0
    assert identity["configuration_checksums"] == "exact"
    assert identity["manifest_memberships"] == "exact"
    assert identity["dataset_and_token_counts"] == "exact"
    assert identity["parameter_counts"] == "exact"

    assert contract["week_01_identity"]["model_use"] == "prohibited"
    assert contract["week_01_identity"]["task8_membership_use_authorized"] is False
    assert contract["week_02_identity"]["primary_record_count"] == 12
    assert contract["week_03_identity"]["run_seeds"] == [20260821, 20260822, 20260823]

    models = {model["model_id"]: model for model in contract["models"]}
    assert set(models) == {
        "week1_a004_read_only_audit",
        "week2_random_unigram",
        "week2_random_count_bigram",
        "week2_random_neural_bigram",
        "week2_family_aware_unigram",
        "week2_family_aware_count_bigram",
        "week2_family_aware_neural_bigram",
        "c10",
        "c20",
        "e64",
    }
    assert models["week1_a004_read_only_audit"]["training_applicable"] is False
    assert models["c10"]["parameter_count"] == 274293
    assert models["c20"]["parameter_count"] == 530293
    assert models["e64"]["parameter_count"] == 530965

    targets = contract["metric_targets"]
    assert len(targets) == 21
    week2_targets = [target for target in targets if target["model_id"].startswith("week2_")]
    assert len(week2_targets) == 12
    assert all("seed" not in target for target in week2_targets)
    assert {(target["model_id"], target["collection"]) for target in week2_targets} == {
        (model_id, collection)
        for model_id in (
            "week2_random_unigram",
            "week2_random_count_bigram",
            "week2_random_neural_bigram",
            "week2_family_aware_unigram",
            "week2_family_aware_count_bigram",
            "week2_family_aware_neural_bigram",
        )
        for collection in (
            "random_native_validation"
            if model_id.startswith("week2_random_")
            else "family_aware_native_validation",
            "shared_validation",
        )
    }
    week3_targets = [target for target in targets if target["model_id"] in {"c10", "c20", "e64"}]
    assert {(target["model_id"], target["seed"]) for target in week3_targets} == {
        (model_id, seed)
        for model_id in ("c10", "c20", "e64")
        for seed in (20260821, 20260822, 20260823)
    }
    assert next(
        target for target in targets if target.get("model_id") == "c20" and target.get("seed") == 20260822
    ) == {
        "model_id": "c20",
        "seed": 20260822,
        "collection": "family_aware_native_validation",
        "cross_entropy": 2.8636887412663112,
        "correct_predictions": 111289,
        "token_count": 1000495,
    }

    claims = {claim["claim_id"]: claim for claim in contract["comparison_claims"]}
    assert set(claims) == {
        "week1_group_aware_lower_detected_strong_overlap",
        "week2_prospective_comparison_supported",
        "c20_beats_week2_family_aware_neural_bigram",
        "c20_beats_c10",
        "c20_beats_e64",
    }
    assert all(
        claim["minimum_cross_entropy_gap"] == 0.001
        for claim_id, claim in claims.items()
        if claim_id.startswith("c20_beats_")
    )
    assert claims["week2_prospective_comparison_supported"] == {
        "claim_id": "week2_prospective_comparison_supported",
        "rule": "random_neural_shared_ce_minus_random_neural_native_ce_gt_family_aware_neural_shared_ce_minus_family_aware_neural_native_ce",
        "historical_random_neural_optimism_gap": 0.007410326406358969,
        "historical_family_aware_neural_optimism_gap": 0.0029218877435779333,
        "minimum_cross_entropy_gap": 0.0,
    }

    comparison_scope = contract["comparison_scope"]
    assert comparison_scope["week_01_scientific_aggregate_fields"] == (
        "all_result_staged_result_balance_context_and_interpretation_aggregate_fields"
    )
    assert comparison_scope["week_01_exact_provenance_identity_fields"] == [
        "a004_fingerprint",
        "configuration_identity",
        "source_input_identities",
        "assignment_identities",
        "mmseqs2_version",
    ]
    assert comparison_scope["week_01_variable_comparison_exclusions"] == [
        "reproduction_code_revision",
        "runtime",
        "hardware",
        "local_evidence_paths",
        "machine_specific_paths",
    ]
    assert comparison_scope["week_02_overall_record_count"] == 12
    assert comparison_scope["week_02_family_aware_shared_validation_length_bucket_record_count"] == 15
    assert comparison_scope["week_02_derived_comparisons_payload"] == (
        "complete_payload_and_field_coverage_exact_numeric_values_follow_stage_tolerance_or_claim_rules"
    )
    assert comparison_scope["week_02_secondary_coverage"] == (
        "required_exact_coverage_not_independent_pass_gate"
    )

    stages = {stage["stage_id"]: stage for stage in contract["stages"]}
    assert stages["week_02_reevaluate"]["depends_on"] == ["week_02_verify"]
    assert stages["week_02_retrain"]["depends_on"] == ["week_02_verify", "week_02_reevaluate"]
    assert stages["week_03_retrain"]["depends_on"] == ["week_03_verify", "week_03_reevaluate"]
    assert stages["final_compare"]["depends_on"] == [
        "week_01_reevaluate",
        "week_02_retrain",
        "week_03_retrain",
    ]
    assert contract["run_bundle"]["required_files"] == [
        "contract.toml",
        "run.json",
        "log.txt",
        "metrics.json",
        "comparison.json",
        "provenance.json",
    ]
    assert contract["operator_boundary"]["dashboard_jobs_remain_blocked"] is True
    assert all(value is False for value in contract["exclusions"].values())


def test_week1_audit_cli_default_is_synthetic_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("run_read_only_fixed_budget_audit.py")
    calls: list[object] = []
    monkeypatch.setattr(
        module,
        "validate_a004_configuration",
        lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(paths={"source_workspace": "a003", "workspace": "a004"}),
    )
    monkeypatch.setattr(
        module,
        "run_fixed_budget_audit",
        lambda **kwargs: pytest.fail("default Week 1 CLI must not execute the audit"),
    )

    assert module.main([]) == 0
    assert len(calls) == 1
    assert module.parse_args([]).execute_searches is False
    assert module.parse_args(["--execute-searches"]).execute_searches is True


def test_week2_cli_defaults_do_not_fit_or_evaluate(monkeypatch: pytest.MonkeyPatch) -> None:
    training = _load_script("train_week2_bigrams.py")
    evaluation = _load_script("evaluate_week2_bigrams.py")
    training_calls: list[tuple[object, str]] = []
    evaluation_calls: list[tuple[object, str]] = []
    monkeypatch.setattr(
        training,
        "preflight",
        lambda root, candidate_id: training_calls.append((root, candidate_id))
        or SimpleNamespace(),
    )
    monkeypatch.setattr(training, "_print_plan", lambda _plan: None)
    monkeypatch.setattr(
        training,
        "create_candidate",
        lambda **kwargs: pytest.fail("default Week 2 training CLI must not fit"),
    )
    monkeypatch.setattr(
        evaluation,
        "preflight",
        lambda root, evaluation_id: evaluation_calls.append((root, evaluation_id))
        or SimpleNamespace(),
    )
    monkeypatch.setattr(
        evaluation,
        "execute_evaluation",
        lambda **kwargs: pytest.fail("default Week 2 evaluation CLI must not score"),
    )
    monkeypatch.setattr("sys.argv", ["train_week2_bigrams.py"])
    assert training.main() == 0
    monkeypatch.setattr("sys.argv", ["evaluate_week2_bigrams.py"])
    assert evaluation.main() == 0
    assert training_calls[0][1] == "preflight-only"
    assert evaluation_calls[0][1] == "preflight-only"
    monkeypatch.setattr("sys.argv", ["train_week2_bigrams.py", "--execute-candidate"])
    with pytest.raises(SystemExit):
        training.parse_args()
    monkeypatch.setattr(
        "sys.argv", ["evaluate_week2_bigrams.py", "--execute-evaluation"]
    )
    with pytest.raises(SystemExit):
        evaluation.parse_args()


def test_week3_mlp_cli_default_does_not_train(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("train_week3_mlp.py")
    calls: list[tuple[object, str]] = []
    monkeypatch.setattr(
        module,
        "preflight",
        lambda root, run_id: calls.append((root, run_id)) or SimpleNamespace(),
    )
    monkeypatch.setattr(module, "_print_plan", lambda _plan: None)
    monkeypatch.setattr(
        module,
        "execute_run",
        lambda **kwargs: pytest.fail("default Week 3 CLI must not train"),
    )

    assert module.main([]) == 0
    assert calls[0][1] == "preflight-only"
    with pytest.raises(SystemExit):
        module.parse_args(["--new-run"])
    with pytest.raises(SystemExit):
        module.parse_args(["--run-id", "candidate"])
