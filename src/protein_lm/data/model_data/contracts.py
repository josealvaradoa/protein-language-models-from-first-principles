"""Small value objects and invariants for the Week 2 candidate."""

from __future__ import annotations

from dataclasses import dataclass


class ModelDataError(ValueError):
    """Raised when frozen Week 2 data contracts cannot be proved."""


@dataclass(frozen=True)
class LengthBucket:
    name: str
    minimum: int
    maximum: int


@dataclass(frozen=True)
class ModelDataConfig:
    schema_version: int
    scope: str
    candidate_identifier: str
    source_release: str
    proteingym_release: str
    license_spdx: str
    task4_catalog_relative_path: str
    task4_catalog_sha256: str
    task4_catalog_byte_size: int
    task4_catalog_row_count: int
    reserved_families_relative_path: str
    reserved_families_sha256: str
    reserved_family_count: int
    task4_report_sha256: str
    task4_report_relative_path: str
    task4_eligibility_policy_sha256: str
    task4_eligibility_policy_relative_path: str
    canonical_amino_acids: str
    minimum_length: int
    maximum_length: int
    sequence_hash: str
    prediction_token_target: int
    minimum_evaluation_predictions: int
    maximum_evaluation_predictions: int
    minimum_bucket_predictions: int
    base_seed: int
    hash_algorithm: str
    allocation_namespaces: tuple[str, ...]
    length_buckets: tuple[LengthBucket, ...]
    candidate_directory_relative_path: str
    readiness_json_relative_path: str
    readiness_markdown_relative_path: str
    readiness_sha256_relative_path: str
    mmseqs2_status: str
    model_use: str


@dataclass(frozen=True)
class CatalogRecord:
    primary_accession: str
    sequence: str
    sequence_sha256: str
    biological_length: int
    uniref50_group: str


@dataclass(frozen=True)
class CandidateRecord:
    primary_accession: str
    sequence_sha256: str
    biological_length: int
    length_bucket: str
    uniref50_group: str

    @property
    def prediction_tokens(self) -> int:
        return self.biological_length + 1


@dataclass(frozen=True)
class Alias:
    sequence_sha256: str
    canonical_accession: str
    alias_accession: str


@dataclass(frozen=True)
class AllocationUnit:
    identifier: str
    records: tuple[CandidateRecord, ...]

    @property
    def prediction_tokens(self) -> int:
        return sum(record.prediction_tokens for record in self.records)

    def bucket_tokens(self, bucket: str) -> int:
        return sum(
            record.prediction_tokens
            for record in self.records
            if record.length_bucket == bucket
        )
