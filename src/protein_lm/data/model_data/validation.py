"""Hard-gate validation for an already-created Week 2 candidate."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from protein_lm.data.model_data.aggregates import collection_aggregate
from protein_lm.data.model_data.candidate_provenance import (
    preparation_record_matches,
    regeneration_is_identical,
)
from protein_lm.data.model_data.contracts import (
    CandidateRecord,
    CatalogRecord,
    ModelDataConfig,
    ModelDataError,
)
from protein_lm.data.model_data.deduplication import prepare_population
from protein_lm.data.model_data.manifests import (
    MEMBERSHIP_HEADER,
    read_and_verify_checksums,
)
from protein_lm.data.model_data.workflow import verify_frozen_inputs


@dataclass(frozen=True)
class ValidationResult:
    gates: dict[str, bool]
    random_group_crossings: int
    artifacts: dict[str, dict[str, object]]
    collection_aggregates: dict[str, dict[str, object]]

    @property
    def status(self) -> str:
        return "passed" if all(self.gates.values()) else "failed"


def validate_candidate(
    *,
    root: Path,
    candidate_directory: Path,
    config: ModelDataConfig,
    allow_synthetic_revision: bool = False,
) -> ValidationResult:
    """Validate immutable candidate artifacts against frozen source identities."""

    evidence = read_and_verify_checksums(candidate_directory)
    catalog, reserved_groups = verify_frozen_inputs(root, config)
    population, aliases, reserved_records = prepare_population(
        catalog, reserved_groups, config
    )
    expected = {record.primary_accession: record for record in population}
    shared = _read_membership(
        candidate_directory / "shared_validation.tsv", "shared_validation"
    )
    sealed = _read_membership(
        candidate_directory / "shared_sealed_test.tsv", "shared_sealed_test"
    )
    random_rows = _read_membership(candidate_directory / "random_arm.tsv", None)
    family_rows = _read_membership(candidate_directory / "family_aware_arm.tsv", None)
    all_rows = {
        "shared": shared,
        "sealed": sealed,
        "random": random_rows,
        "family": family_rows,
    }
    aggregates = _collection_aggregates(
        shared, sealed, random_rows, family_rows, config
    )
    population_counts = {
        "eligible_catalog_records": len(catalog),
        "protein_gym_reserved_records": reserved_records,
        "deduplicated_unreserved_records": len(population),
        "collapsed_aliases": len(aliases),
    }
    reserved_ids = _reserved_identities(catalog, reserved_groups)
    gates = {
        "complete_per_arm_population_accounting": _complete_arms(
            shared, sealed, random_rows, family_rows, expected
        ),
        "no_accession_or_hash_crossings_within_arm": _no_partition_crossings(
            shared + sealed + random_rows
        )
        and _no_partition_crossings(shared + sealed + family_rows),
        "shared_and_sealed_groups_are_isolated": _groups(shared).isdisjoint(
            _groups(sealed)
        )
        and _groups(shared).isdisjoint(_groups(random_rows + family_rows))
        and _groups(sealed).isdisjoint(_groups(random_rows + family_rows)),
        "family_aware_groups_do_not_cross": _family_groups_do_not_cross(family_rows),
        "reserved_members_are_absent": _reserved_ids_absent(all_rows, reserved_ids),
        "source_sequence_contract": _source_contract(all_rows, expected),
        "evaluation_token_bounds": all(
            _token_bounds(rows, config)
            for rows in (
                shared,
                sealed,
                _partition(random_rows, "native_validation"),
                _partition(family_rows, "native_validation"),
            )
        ),
        "evaluation_bucket_bounds": all(
            _bucket_bounds(rows, config)
            for rows in (
                shared,
                sealed,
                _partition(random_rows, "native_validation"),
                _partition(family_rows, "native_validation"),
            )
        ),
        "artifact_and_provenance_checksums": preparation_record_matches(
            root=root,
            directory=candidate_directory,
            config=config,
            evidence=evidence,
            population=population_counts,
            collection_aggregates=aggregates,
            allow_synthetic_revision=allow_synthetic_revision,
        ),
        "deterministic_regeneration": regeneration_is_identical(
            root, candidate_directory, config
        ),
    }
    return ValidationResult(
        gates=gates,
        random_group_crossings=_crossing_groups(random_rows),
        artifacts={
            name: {
                "row_count": item.row_count,
                "byte_size": item.byte_size,
                "sha256": item.sha256,
            }
            for name, item in evidence.items()
        },
        collection_aggregates=aggregates,
    )


def _read_membership(
    path: Path, expected_partition: str | None
) -> list[tuple[CandidateRecord, str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ModelDataError(
            f"candidate manifest cannot be read: {path.name}"
        ) from error
    if not lines or tuple(lines[0].split("\t")) != MEMBERSHIP_HEADER:
        raise ModelDataError(f"candidate manifest header is invalid: {path.name}")
    rows = []
    previous_accession = ""
    for line in lines[1:]:
        values = line.split("\t")
        if len(values) != len(MEMBERSHIP_HEADER):
            raise ModelDataError(f"candidate manifest row is malformed: {path.name}")
        accession, partition, digest, raw_length, bucket, group = values
        if expected_partition is not None and partition != expected_partition:
            raise ModelDataError(
                f"candidate manifest partition is invalid: {path.name}"
            )
        if expected_partition is None and partition not in {
            "training",
            "native_validation",
        }:
            raise ModelDataError(f"candidate arm partition is invalid: {path.name}")
        if accession <= previous_accession:
            raise ModelDataError(f"candidate manifest is not sorted: {path.name}")
        previous_accession = accession
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ModelDataError(
                f"candidate manifest length is invalid: {path.name}"
            ) from error
        rows.append(
            (CandidateRecord(accession, digest, length, bucket, group), partition)
        )
    return rows


def _complete_arms(
    shared: list[tuple[CandidateRecord, str]],
    sealed: list[tuple[CandidateRecord, str]],
    random_rows: list[tuple[CandidateRecord, str]],
    family_rows: list[tuple[CandidateRecord, str]],
    expected: dict[str, CandidateRecord],
) -> bool:
    return _matches_population(
        shared + sealed + random_rows, expected
    ) and _matches_population(shared + sealed + family_rows, expected)


def _matches_population(
    rows: list[tuple[CandidateRecord, str]], expected: dict[str, CandidateRecord]
) -> bool:
    accessions = [record.primary_accession for record, _ in rows]
    hashes = [record.sequence_sha256 for record, _ in rows]
    return (
        len(rows) == len(expected)
        and len(set(accessions)) == len(rows)
        and len(set(hashes)) == len(rows)
        and set(accessions) == set(expected)
    )


def _no_partition_crossings(rows: list[tuple[CandidateRecord, str]]) -> bool:
    by_accession: dict[str, str] = {}
    by_hash: dict[str, str] = {}
    for record, partition in rows:
        if (
            by_accession.setdefault(record.primary_accession, partition) != partition
            or by_hash.setdefault(record.sequence_sha256, partition) != partition
        ):
            return False
    return True


def _family_groups_do_not_cross(rows: list[tuple[CandidateRecord, str]]) -> bool:
    partitions: dict[str, str] = {}
    return all(
        partitions.setdefault(record.uniref50_group, partition) == partition
        for record, partition in rows
    )


def _reserved_identities(
    catalog: list[CatalogRecord], reserved_groups: frozenset[str]
) -> tuple[set[str], set[str], set[str]]:
    reserved = [
        record for record in catalog if record.uniref50_group in reserved_groups
    ]
    return (
        {record.primary_accession for record in reserved},
        {record.sequence_sha256 for record in reserved},
        set(reserved_groups),
    )


def _reserved_ids_absent(
    collections: dict[str, list[tuple[CandidateRecord, str]]],
    reserved_ids: tuple[set[str], set[str], set[str]],
) -> bool:
    accessions, hashes, groups = reserved_ids
    return all(
        record.primary_accession not in accessions
        and record.sequence_sha256 not in hashes
        and record.uniref50_group not in groups
        for rows in collections.values()
        for record, _ in rows
    )


def _crossing_groups(rows: list[tuple[CandidateRecord, str]]) -> int:
    partitions: dict[str, set[str]] = defaultdict(set)
    for record, partition in rows:
        partitions[record.uniref50_group].add(partition)
    return sum(len(value) > 1 for value in partitions.values())


def _groups(rows: list[tuple[CandidateRecord, str]]) -> set[str]:
    return {record.uniref50_group for record, _ in rows}


def _partition(
    rows: list[tuple[CandidateRecord, str]], partition: str
) -> list[tuple[CandidateRecord, str]]:
    return [row for row in rows if row[1] == partition]


def _source_contract(
    collections: dict[str, list[tuple[CandidateRecord, str]]],
    expected: dict[str, CandidateRecord],
) -> bool:
    return all(
        expected.get(record.primary_accession) == record
        for rows in collections.values()
        for record, _ in rows
    )


def _token_bounds(
    rows: list[tuple[CandidateRecord, str]], config: ModelDataConfig
) -> bool:
    tokens = sum(record.prediction_tokens for record, _ in rows)
    return (
        config.minimum_evaluation_predictions
        <= tokens
        <= config.maximum_evaluation_predictions
    )


def _bucket_bounds(
    rows: list[tuple[CandidateRecord, str]], config: ModelDataConfig
) -> bool:
    tokens = defaultdict(int)
    for record, _ in rows:
        tokens[record.length_bucket] += record.prediction_tokens
    return all(
        tokens[bucket.name] >= config.minimum_bucket_predictions
        for bucket in config.length_buckets
    )


def _collection_aggregates(
    shared: list[tuple[CandidateRecord, str]],
    sealed: list[tuple[CandidateRecord, str]],
    random_rows: list[tuple[CandidateRecord, str]],
    family_rows: list[tuple[CandidateRecord, str]],
    config: ModelDataConfig,
) -> dict[str, dict[str, object]]:
    return {
        "shared_validation": collection_aggregate(_records(shared), config),
        "shared_sealed_test": collection_aggregate(_records(sealed), config),
        "random_training": collection_aggregate(
            _records(_partition(random_rows, "training")), config
        ),
        "random_native_validation": collection_aggregate(
            _records(_partition(random_rows, "native_validation")), config
        ),
        "family_aware_training": collection_aggregate(
            _records(_partition(family_rows, "training")), config
        ),
        "family_aware_native_validation": collection_aggregate(
            _records(_partition(family_rows, "native_validation")), config
        ),
    }


def _records(rows: list[tuple[CandidateRecord, str]]) -> list[CandidateRecord]:
    return [record for record, _ in rows]
