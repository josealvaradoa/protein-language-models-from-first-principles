"""Boundary-safe sequence loaders for the promoted Week 2 model-data registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from protein_lm.data.model_data.catalog import load_catalog, load_reserved_families
from protein_lm.data.model_data.config import load_config
from protein_lm.data.model_data.contracts import (
    CatalogRecord,
    ModelDataConfig,
    ModelDataError,
)
from protein_lm.data.model_data.manifests import MEMBERSHIP_HEADER
from protein_lm.data.model_data.promotion import PROMOTION_CONTRACT, PromotionContract, registry_payload


class ModelDataCollection(StrEnum):
    """The complete set of public collections permitted to model code."""

    RANDOM_TRAINING = "random_training"
    RANDOM_NATIVE_VALIDATION = "random_native_validation"
    FAMILY_AWARE_TRAINING = "family_aware_training"
    FAMILY_AWARE_NATIVE_VALIDATION = "family_aware_native_validation"
    SHARED_VALIDATION = "shared_validation"


@dataclass(frozen=True)
class ProteinSequence:
    """One catalog-proved member of the requested public collection."""

    primary_accession: str
    sequence: str
    sequence_sha256: str
    biological_length: int
    length_bucket: str
    uniref50_group: str


_COLLECTION_SOURCES = {
    ModelDataCollection.RANDOM_TRAINING: ("random_arm.tsv", "training"),
    ModelDataCollection.RANDOM_NATIVE_VALIDATION: (
        "random_arm.tsv",
        "native_validation",
    ),
    ModelDataCollection.FAMILY_AWARE_TRAINING: ("family_aware_arm.tsv", "training"),
    ModelDataCollection.FAMILY_AWARE_NATIVE_VALIDATION: (
        "family_aware_arm.tsv",
        "native_validation",
    ),
    ModelDataCollection.SHARED_VALIDATION: ("shared_validation.tsv", "shared_validation"),
}


def load_collection(
    root: Path,
    collection: ModelDataCollection,
) -> tuple[ProteinSequence, ...]:
    """Load exactly one permitted collection and resolve it against the pinned catalog."""

    if not isinstance(collection, ModelDataCollection):
        raise ModelDataError("collection must be an approved ModelDataCollection value")
    config = load_config(root / "experiments/week_02/model_data_readiness.toml")
    return _load_collection(root, collection, config, PROMOTION_CONTRACT)


def _load_collection(
    root: Path,
    collection: ModelDataCollection,
    config: ModelDataConfig,
    contract: PromotionContract,
) -> tuple[ProteinSequence, ...]:
    """Load one collection with an injected contract for synthetic tests."""

    if not isinstance(collection, ModelDataCollection):
        raise ModelDataError("collection must be an approved ModelDataCollection value")
    registry = root / "manifests/week_02/model_data_v1.json"
    _verify_registry(registry, config, contract)
    filename, selected_partition = _COLLECTION_SOURCES[collection]
    rows = _read_manifest(
        root / "manifests/week_02" / filename,
        filename,
        selected_partition,
        contract,
    )
    reserved = load_reserved_families(
        root / config.reserved_families_relative_path, config
    )
    catalog = load_catalog(root / config.task4_catalog_relative_path, config, reserved)
    by_accession = {record.primary_accession: record for record in catalog}
    return tuple(_resolve(row, by_accession, config) for row in rows)


def _verify_registry(
    path: Path, config: ModelDataConfig, contract: PromotionContract
) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError("public Week 2 registry is malformed") from error
    if value != registry_payload(config, contract):
        raise ModelDataError("public Week 2 registry does not match the frozen contract")


def _read_manifest(
    path: Path,
    filename: str,
    selected_partition: str,
    contract: PromotionContract,
) -> list[tuple[str, str, str, int, str, str]]:
    artifact = next(item for item in contract.artifacts if item.filename == filename)
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ModelDataError(f"public manifest cannot be read: {filename}") from error
    if len(content) != artifact.byte_size or hashlib.sha256(content).hexdigest() != artifact.sha256:
        raise ModelDataError(f"public manifest identity drifted: {filename}")
    expected_header = "\t".join(MEMBERSHIP_HEADER).encode("utf-8") + b"\n"
    if not content.startswith(expected_header) or b"\r" in content:
        raise ModelDataError(f"public manifest header is invalid: {filename}")
    lines = content.splitlines()
    if len(lines) - 1 != artifact.row_count or not content.endswith(b"\n"):
        raise ModelDataError(f"public manifest row count is invalid: {filename}")
    permitted = {selected_partition}
    if filename != "shared_validation.tsv":
        permitted = {"training", "native_validation"}
    rows = []
    previous_accession = ""
    for raw in lines[1:]:
        try:
            accession, partition, digest, raw_length, bucket, group = raw.decode("utf-8").split("\t")
            length = int(raw_length)
        except (UnicodeDecodeError, ValueError) as error:
            raise ModelDataError(f"public manifest row is malformed: {filename}") from error
        if partition not in permitted or accession <= previous_accession:
            raise ModelDataError(f"public manifest partition or ordering is invalid: {filename}")
        previous_accession = accession
        if partition == selected_partition:
            rows.append((accession, partition, digest, length, bucket, group))
    return rows


def _resolve(
    row: tuple[str, str, str, int, str, str],
    catalog: dict[str, CatalogRecord],
    config: ModelDataConfig,
) -> ProteinSequence:
    accession, _, digest, length, bucket, group = row
    record = catalog.get(accession)
    if record is None:
        raise ModelDataError("public manifest accession is absent from the pinned catalog")
    sequence = record.sequence
    expected_bucket = next(
        (
            item.name
            for item in config.length_buckets
            if item.minimum <= record.biological_length <= item.maximum
        ),
        None,
    )
    if (
        record.primary_accession != accession
        or record.sequence_sha256 != digest
        or record.biological_length != length
        or expected_bucket != bucket
        or record.uniref50_group != group
        or hashlib.sha256(sequence.encode("ascii")).hexdigest() != digest
    ):
        raise ModelDataError("public manifest membership disagrees with the pinned catalog")
    return ProteinSequence(accession, sequence, digest, length, bucket, group)
