"""Deterministic closest-to-target selection for the four Week 2 collections."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

from protein_lm.data.model_data.contracts import (
    AllocationUnit,
    CandidateRecord,
    ModelDataConfig,
    ModelDataError,
)


@dataclass(frozen=True)
class CandidateAllocation:
    shared_validation: tuple[CandidateRecord, ...]
    shared_sealed_test: tuple[CandidateRecord, ...]
    random_training: tuple[CandidateRecord, ...]
    random_native_validation: tuple[CandidateRecord, ...]
    family_training: tuple[CandidateRecord, ...]
    family_native_validation: tuple[CandidateRecord, ...]


def assignment_digest(namespace: str, base_seed: int, identifier: str) -> bytes:
    """Return the explicitly domain-separated digest for one assignment unit."""

    if not namespace or not identifier or "\x00" in namespace or "\x00" in identifier:
        raise ModelDataError(
            "allocation namespace and identifier must not be empty or contain nulls"
        )
    return hashlib.sha256(
        namespace.encode("utf-8")
        + b"\x00"
        + str(base_seed).encode("ascii")
        + b"\x00"
        + identifier.encode("utf-8")
    ).digest()


def ordered_units(
    records: tuple[CandidateRecord, ...],
    *,
    namespace: str,
    config: ModelDataConfig,
    grouped_by_family: bool,
) -> tuple[AllocationUnit, ...]:
    """Build complete-family or individual-sequence allocation units."""

    grouped: dict[str, list[CandidateRecord]] = defaultdict(list)
    for record in records:
        identifier = (
            record.uniref50_group if grouped_by_family else record.sequence_sha256
        )
        grouped[identifier].append(record)
    units = tuple(
        AllocationUnit(
            identifier,
            tuple(sorted(members, key=lambda record: record.primary_accession)),
        )
        for identifier, members in grouped.items()
    )
    if len({unit.identifier for unit in units}) != len(units):
        raise ModelDataError("allocation unit identifiers are not unique")
    return tuple(
        sorted(
            units,
            key=lambda unit: (
                assignment_digest(namespace, config.base_seed, unit.identifier),
                unit.identifier,
            ),
        )
    )


def select_closest_to_target(
    units: tuple[AllocationUnit, ...], config: ModelDataConfig
) -> tuple[CandidateRecord, ...]:
    """Keep a complete unit exactly when it gets the running total no farther away."""

    current_tokens = 0
    selected = []
    for unit in units:
        with_unit = current_tokens + unit.prediction_tokens
        if abs(with_unit - config.prediction_token_target) <= abs(
            current_tokens - config.prediction_token_target
        ):
            selected.extend(unit.records)
            current_tokens = with_unit
    return tuple(sorted(selected, key=lambda record: record.primary_accession))


def allocate(
    records: tuple[CandidateRecord, ...], config: ModelDataConfig
) -> CandidateAllocation:
    """Reserve shared sets first, then create the two independent native arms."""

    shared_validation = select_closest_to_target(
        ordered_units(
            records,
            namespace=config.allocation_namespaces[0],
            config=config,
            grouped_by_family=True,
        ),
        config,
    )
    shared_groups = {record.uniref50_group for record in shared_validation}
    after_shared = tuple(
        record for record in records if record.uniref50_group not in shared_groups
    )
    shared_sealed_test = select_closest_to_target(
        ordered_units(
            after_shared,
            namespace=config.allocation_namespaces[1],
            config=config,
            grouped_by_family=True,
        ),
        config,
    )
    sealed_groups = {record.uniref50_group for record in shared_sealed_test}
    development = tuple(
        record for record in after_shared if record.uniref50_group not in sealed_groups
    )
    random_native_validation = select_closest_to_target(
        ordered_units(
            development,
            namespace=config.allocation_namespaces[2],
            config=config,
            grouped_by_family=False,
        ),
        config,
    )
    random_hashes = {record.sequence_sha256 for record in random_native_validation}
    family_native_validation = select_closest_to_target(
        ordered_units(
            development,
            namespace=config.allocation_namespaces[3],
            config=config,
            grouped_by_family=True,
        ),
        config,
    )
    family_groups = {record.uniref50_group for record in family_native_validation}
    return CandidateAllocation(
        shared_validation=shared_validation,
        shared_sealed_test=shared_sealed_test,
        random_training=tuple(
            record
            for record in development
            if record.sequence_sha256 not in random_hashes
        ),
        random_native_validation=random_native_validation,
        family_training=tuple(
            record
            for record in development
            if record.uniref50_group not in family_groups
        ),
        family_native_validation=family_native_validation,
    )
