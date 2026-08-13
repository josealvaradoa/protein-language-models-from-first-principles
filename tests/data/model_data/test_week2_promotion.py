"""Synthetic tests for the Week 2 public-manifest promotion boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.promotion import PROMOTION_CONTRACT, preflight_promotion
from week2_promotion_loader_test_support import (
    build_synthetic_environment,
    promote_synthetic,
)


def test_synthetic_promotion_is_atomic_and_preserves_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, contract = build_synthetic_environment(tmp_path)
    assert not (root / "manifests/week_02").exists()
    assert preflight_promotion(root, config, contract=contract)["sealed_membership"] == "not read"
    assert not (root / "manifests/week_02").exists()
    destination = promote_synthetic(root, config, contract, monkeypatch)
    assert destination.is_dir()
    for artifact in contract.artifacts:
        assert (destination / artifact.filename).read_bytes() == (
            root / config.candidate_directory_relative_path / artifact.filename
        ).read_bytes()
    assert set(path.name for path in destination.iterdir()) == {
        "shared_validation.tsv",
        "random_arm.tsv",
        "family_aware_arm.tsv",
        "model_data_v1.json",
    }
    with pytest.raises(ModelDataError, match="already exists"):
        promote_synthetic(root, config, contract, monkeypatch)


def test_promotion_rejects_tampered_readiness_without_creating_output(
    tmp_path: Path,
) -> None:
    root, config, contract = build_synthetic_environment(tmp_path)
    (root / config.readiness_json_relative_path).write_text(
        "tampered\n", encoding="utf-8"
    )
    with pytest.raises(ModelDataError, match="readiness JSON identity"):
        preflight_promotion(root, config, contract=contract)
    assert not (root / "manifests/week_02").exists()


def test_preflight_never_opens_the_sealed_candidate_path(tmp_path: Path) -> None:
    root, config, contract = build_synthetic_environment(tmp_path)
    sealed = root / config.candidate_directory_relative_path / "shared_sealed_test.tsv"
    sealed.unlink()
    sealed.mkdir()
    assert preflight_promotion(root, config, contract=contract)["sealed_membership"] == "not read"


def test_promotion_rejects_tampered_public_source_without_output(tmp_path: Path) -> None:
    root, config, contract = build_synthetic_environment(tmp_path)
    source = root / config.candidate_directory_relative_path / "random_arm.tsv"
    source.write_bytes(source.read_bytes() + b"tampered\n")
    with pytest.raises(ModelDataError, match="identity drifted"):
        preflight_promotion(root, config, contract=contract)
    assert not (root / "manifests/week_02").exists()


def test_production_contract_matches_tracked_readiness_commitments() -> None:
    root = Path(__file__).parents[3]
    readiness = root / "reports/week_02/model_data_readiness_v1.json"
    content = readiness.read_bytes()
    payload = json.loads(content)
    assert hashlib.sha256(content).hexdigest() == PROMOTION_CONTRACT.readiness_json_sha256
    artifacts = payload["artifacts"]
    assert {
        item.filename: {
            "row_count": item.row_count,
            "byte_size": item.byte_size,
            "sha256": item.sha256,
        }
        for item in PROMOTION_CONTRACT.artifacts
    } == {
        name: artifacts[name]
        for name in ("shared_validation.tsv", "random_arm.tsv", "family_aware_arm.tsv")
    }
    assert artifacts["shared_sealed_test.tsv"] == {
        "row_count": PROMOTION_CONTRACT.sealed_rows,
        "byte_size": PROMOTION_CONTRACT.sealed_bytes,
        "sha256": PROMOTION_CONTRACT.sealed_sha256,
    }
