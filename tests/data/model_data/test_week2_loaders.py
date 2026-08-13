"""Synthetic tests for the approved Week 2 sequence-loader boundary."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from protein_lm.data.model_data.contracts import ModelDataError
from protein_lm.data.model_data.loaders import (
    ModelDataCollection,
    _load_collection,
    load_collection,
)
from protein_lm.data.model_data.promotion import registry_payload
from week2_promotion_loader_test_support import (
    build_synthetic_environment,
    membership_bytes,
    promote_synthetic,
    replace_public_artifact,
)


def test_loader_only_accepts_collections_and_proves_catalog_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, contract = build_synthetic_environment(tmp_path)
    assert {item.value for item in ModelDataCollection} == {
        "random_training",
        "random_native_validation",
        "family_aware_training",
        "family_aware_native_validation",
        "shared_validation",
    }
    promote_synthetic(root, config, contract, monkeypatch)
    proteins = _load_collection(
        root, ModelDataCollection.RANDOM_TRAINING, config, contract
    )
    assert [(protein.primary_accession, protein.sequence) for protein in proteins] == [
        ("P00001", "A" * 32)
    ]
    with pytest.raises(ModelDataError, match="approved ModelDataCollection"):
        _load_collection(root, "shared_sealed_test", config, contract)  # type: ignore[arg-type]
    source = root / "manifests/week_02/random_arm.tsv"
    content = source.read_text(encoding="utf-8").replace(
        "UniRef50_G1", "UniRef50_WRONG"
    )
    changed_contract = replace_public_artifact(
        root, config, contract, "random_arm.tsv", content.encode("utf-8")
    )
    with pytest.raises(ModelDataError, match="membership disagrees"):
        _load_collection(
            root, ModelDataCollection.RANDOM_TRAINING, config, changed_contract
        )


@pytest.mark.parametrize(
    ("collection", "accession"),
    (
        (ModelDataCollection.RANDOM_TRAINING, "P00001"),
        (ModelDataCollection.RANDOM_NATIVE_VALIDATION, "P00003"),
        (ModelDataCollection.FAMILY_AWARE_TRAINING, "P00001"),
        (ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION, "P00003"),
        (ModelDataCollection.SHARED_VALIDATION, "P00002"),
    ),
)
def test_each_collection_returns_only_its_selected_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collection: ModelDataCollection,
    accession: str,
) -> None:
    root, config, contract = build_synthetic_environment(tmp_path)
    promote_synthetic(root, config, contract, monkeypatch)
    proteins = _load_collection(root, collection, config, contract)
    assert [protein.primary_accession for protein in proteins] == [accession]


def test_loader_rejects_registry_and_pinned_catalog_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config, contract = build_synthetic_environment(tmp_path)
    promote_synthetic(root, config, contract, monkeypatch)
    registry = root / "manifests/week_02/model_data_v1.json"
    registry.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ModelDataError, match="registry does not match"):
        _load_collection(root, ModelDataCollection.SHARED_VALIDATION, config, contract)
    registry.write_text(
        json.dumps(registry_payload(config, contract), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    catalog = root / config.task4_catalog_relative_path
    catalog.write_bytes(catalog.read_bytes() + b"drift\n")
    with pytest.raises(ModelDataError):
        _load_collection(root, ModelDataCollection.SHARED_VALIDATION, config, contract)


@pytest.mark.parametrize(
    "content",
    (
        membership_bytes(
            [
                ("P00001", "training", "A" * 32, "UniRef50_G1"),
                ("P00003", "not_a_partition", "D" * 34, "UniRef50_G3"),
            ]
        ),
        membership_bytes(
            [
                ("P00003", "native_validation", "D" * 34, "UniRef50_G3"),
                ("P00001", "training", "A" * 32, "UniRef50_G1"),
            ]
        ),
    ),
)
def test_loader_rejects_bad_arm_partition_and_order_with_matching_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    root, config, contract = build_synthetic_environment(tmp_path)
    promote_synthetic(root, config, contract, monkeypatch)
    changed_contract = replace_public_artifact(
        root, config, contract, "random_arm.tsv", content
    )
    with pytest.raises(ModelDataError, match="partition or ordering"):
        _load_collection(
            root, ModelDataCollection.RANDOM_TRAINING, config, changed_contract
        )


def test_public_loader_has_no_manifest_path_and_registry_has_no_sealed_path(
    tmp_path: Path,
) -> None:
    _, config, contract = build_synthetic_environment(tmp_path)
    assert tuple(inspect.signature(load_collection).parameters) == ("root", "collection")
    registry = json.dumps(registry_payload(config, contract), sort_keys=True)
    assert "shared_sealed_test.tsv" not in registry
    assert "sealed_test_commitment" in registry
