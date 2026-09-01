"""Focused synthetic checks for the read-only Week 2/3 evidence adapter."""

from __future__ import annotations

import copy
import json
import shutil
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from protein_lm.reproduction import (
    HistoricalEvidenceError,
    evaluate_historical_evidence,
)
from protein_lm.reproduction.contract import load_foundations_contract
from protein_lm.reproduction.historical_evidence import (
    _normalize_week2,
    _normalize_week3,
    _order_by_contract_targets,
    _validate_source_identities,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "experiments/week_04/foundations_reproduction_v1.toml"
WEEK2_PATH = ROOT / "reports/week_02/bigram_evaluation_v1.json"
WEEK3_PATH = ROOT / "reports/week_03/mlp_evaluation_v1.json"
C10_PATH = ROOT / "experiments/week_03/mlp_context20_100m_continuation_v1.toml"


@pytest.fixture
def contract():
    return load_foundations_contract(CONTRACT_PATH)


def _week2() -> dict[str, object]:
    return json.loads(WEEK2_PATH.read_text(encoding="utf-8"))


def _week3() -> dict[str, object]:
    return json.loads(WEEK3_PATH.read_text(encoding="utf-8"))


def _c10() -> dict[str, object]:
    return tomllib.loads(C10_PATH.read_text(encoding="utf-8"))


def test_pinned_historical_evidence_matches_all_contract_targets_in_contract_order(contract) -> None:
    result = evaluate_historical_evidence(ROOT, contract)

    assert result.source_verification.passed
    assert result.observed_records == contract.metric_targets
    assert len(result.observed_records) == 21
    assert result.metric_comparison.passed
    assert [(outcome.claim_id, outcome.evaluated, outcome.passed) for outcome in result.claim_outcomes] == [
        ("week1_group_aware_lower_detected_strong_overlap", False, None),
        ("week2_prospective_comparison_supported", True, True),
        ("c20_beats_week2_family_aware_neural_bigram", True, True),
        ("c20_beats_c10", True, True),
        ("c20_beats_e64", True, True),
    ]
    assert result.claim_outcomes[0].reason == "not_evaluated_by_historical_evidence_pipeline"
    assert result.to_json() == evaluate_historical_evidence(ROOT, contract).to_json()
    assert json.loads(result.to_json()) == result.to_dict()


def test_source_pin_byte_drift_fails_closed_before_historical_sources_are_parsed(
    contract, tmp_path: Path
) -> None:
    copied_contract_path = tmp_path / CONTRACT_PATH.relative_to(ROOT)
    copied_contract_path.parent.mkdir(parents=True)
    shutil.copyfile(CONTRACT_PATH, copied_contract_path)
    for pin in contract.source_pins:
        target = tmp_path / pin.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / pin.relative_path, target)
    drifted_source = tmp_path / "reports/week_02/bigram_evaluation_v1.json"
    drifted_source.write_bytes(drifted_source.read_bytes() + b"\n")
    copied_contract = load_foundations_contract(copied_contract_path)

    with pytest.raises(HistoricalEvidenceError, match="source pins"):
        evaluate_historical_evidence(tmp_path, copied_contract)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda contract: replace(
            contract,
            metric_targets=(
                replace(contract.metric_targets[0], cross_entropy=0.0),
                *contract.metric_targets[1:],
            ),
        ),
        lambda contract: replace(
            contract,
            comparison_claims=(
                replace(contract.comparison_claims[0], minimum_cross_entropy_gap=1.0),
                *contract.comparison_claims[1:],
            ),
        ),
    ],
)
def test_replaced_metric_or_claim_policy_is_rejected_against_canonical_contract(
    contract, mutate
) -> None:
    with pytest.raises(HistoricalEvidenceError, match="canonical frozen contract"):
        evaluate_historical_evidence(ROOT, mutate(contract))


def test_normalized_records_are_ordered_by_contract_not_report_row_order(contract) -> None:
    observed = _normalize_week2(_week2()) + _normalize_week3(_c10(), _week3())

    ordered = _order_by_contract_targets(tuple(reversed(observed)), contract.metric_targets)

    assert ordered == observed == contract.metric_targets


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update(records={}),
        lambda raw: raw["records"].__setitem__(1, copy.deepcopy(raw["records"][0])),
    ],
)
def test_week2_malformed_duplicate_or_missing_identities_fail_closed(mutate) -> None:
    raw = _week2()
    mutate(raw)

    with pytest.raises(HistoricalEvidenceError):
        _normalize_week2(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["records"][0]["metrics"]["overall"].update(cross_entropy=float("nan")),
        lambda raw: raw["records"][0]["metrics"]["overall"].update(correct_tokens=-1),
        lambda raw: raw["records"][0]["metrics"]["overall"].update(total_nll=1.0),
    ],
)
def test_week2_nonfinite_invalid_counts_and_arithmetic_fail_closed(mutate) -> None:
    raw = _week2()
    mutate(raw)

    with pytest.raises(HistoricalEvidenceError):
        _normalize_week2(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c10, report: c10["control_runs"].__setitem__(1, copy.deepcopy(c10["control_runs"][0])),
        lambda c10, report: report["final_three_seed_comparison"]["context20"]["records"][0].update(nll_numerator=1.0),
        lambda c10, report: report["final_three_seed_comparison"]["embedding64_challenger"]["records"][0].update(correct_predictions=-1),
    ],
)
def test_week3_duplicate_identity_arithmetic_and_invalid_counts_fail_closed(mutate) -> None:
    c10 = _c10()
    report = _week3()
    mutate(c10, report)

    with pytest.raises(HistoricalEvidenceError):
        _normalize_week3(c10, report)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda week2, week3, c10: week2.update(status="failed"),
        lambda week2, week3, c10: week3["hard_gates"].update(no_network_requests=False),
        lambda week2, week3, c10: c10.update(context_length=19),
    ],
)
def test_historical_source_identities_fail_closed(contract, mutate) -> None:
    week2 = _week2()
    week3 = _week3()
    c10 = _c10()
    mutate(week2, week3, c10)

    with pytest.raises(HistoricalEvidenceError):
        _validate_source_identities(week2, week3, c10, contract.metric_targets)
