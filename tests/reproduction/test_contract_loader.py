"""Focused tests for the strict frozen Week 4 contract boundary."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from protein_lm.reproduction.contract import (
    APPROVED_FOUNDATIONS_CONTRACT_SHA256,
    ReproductionContractError,
    SourcePin,
    load_foundations_contract,
    verify_source_pins,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "experiments/week_04/foundations_reproduction_v1.toml"


def test_loads_the_exact_frozen_contract_inventory_and_layout() -> None:
    contract = load_foundations_contract(CONTRACT_PATH)

    assert contract.contract_sha256 == APPROVED_FOUNDATIONS_CONTRACT_SHA256
    assert contract.contract_identifier == "2026-09-01-week-04-foundations-reproduction-v1"
    assert contract.status == "frozen_contract_not_yet_executable"
    assert contract.tolerances.reevaluation == 0.000001
    assert contract.tolerances.retraining == 0.0001
    assert len(contract.source_pins) == 15
    assert len(contract.metric_targets) == 21
    assert len(contract.comparison_claims) == 5
    assert contract.required_run_bundle_files == (
        "contract.toml",
        "run.json",
        "log.txt",
        "metrics.json",
        "comparison.json",
        "provenance.json",
    )
    assert contract.terminal_states == (
        "completed",
        "failed",
        "cancelled",
        "runner_restarted",
    )
    assert sum(record.key.seed is None for record in contract.metric_targets) == 12
    assert sum(record.key.seed is not None for record in contract.metric_targets) == 9


def test_actual_project_source_pins_pass_read_only_verification() -> None:
    report = verify_source_pins(
        ROOT, load_foundations_contract(CONTRACT_PATH)
    )

    assert report.passed is True
    assert len(report.results) == 15
    assert all(result.passed for result in report.results)
    assert json.loads(report.to_json()) == report.to_dict()


def test_changed_contract_bytes_are_rejected_before_toml_is_accepted(
    tmp_path: Path,
) -> None:
    changed = tmp_path / "foundations.toml"
    changed.write_bytes(CONTRACT_PATH.read_bytes() + b"\n")

    with pytest.raises(ReproductionContractError, match="bytes do not match"):
        load_foundations_contract(changed)


def test_value_objects_are_frozen_and_invalid_contract_object_fails_closed() -> None:
    contract = load_foundations_contract(CONTRACT_PATH)
    with pytest.raises(FrozenInstanceError):
        contract.source_pins[0].kind = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        contract.status = "other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        contract.comparison_claims[0].rule = "other"  # type: ignore[misc]
    with pytest.raises(ReproductionContractError, match="requires FoundationsContract"):
        verify_source_pins(ROOT, object())  # type: ignore[arg-type]
    with pytest.raises(ReproductionContractError, match="contract is invalid"):
        verify_source_pins(
            ROOT,
            replace(contract, tolerances="not-a-tolerance-policy"),  # type: ignore[arg-type]
        )
    with pytest.raises(ReproductionContractError, match="contract is invalid"):
        verify_source_pins(ROOT, replace(contract, source_pins=contract.source_pins[:14]))
    with pytest.raises(ReproductionContractError, match="contract is invalid"):
        verify_source_pins(ROOT, replace(contract, contract_sha256="1" * 64))
    with pytest.raises(ReproductionContractError, match="safe relative path"):
        verify_source_pins(
            ROOT,
            replace(
                contract,
                source_pins=(
                    SourcePin("fixture", "../outside.txt", "0" * 64),
                    *contract.source_pins[1:],
                ),
            ),
        )


@pytest.mark.parametrize(
    ("pin", "expected_code"),
    [
        (SourcePin("fixture", "missing.txt", "0" * 64), "MISSING_FILE"),
        (SourcePin("fixture", "drift.txt", "0" * 64), "SHA256_MISMATCH"),
        (SourcePin("fixture", "link.txt", "0" * 64), "SYMLINK_PATH"),
    ],
)
def test_synthetic_source_pin_failures_are_structured_and_stable(
    tmp_path: Path, pin: SourcePin, expected_code: str
) -> None:
    loaded = load_foundations_contract(CONTRACT_PATH)
    contract = replace(loaded, source_pins=(pin, *loaded.source_pins[1:]))
    (tmp_path / "drift.txt").write_text("changed", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(tmp_path / "outside.txt")

    first = verify_source_pins(tmp_path, contract)
    second = verify_source_pins(tmp_path, contract)

    assert first.passed is False
    assert first == second
    assert len(first.results) == 15
    assert first.results[0].issue_code == expected_code
    assert all(result.issue_code == "MISSING_FILE" for result in first.results[1:])
    assert first.to_json() == second.to_json()
