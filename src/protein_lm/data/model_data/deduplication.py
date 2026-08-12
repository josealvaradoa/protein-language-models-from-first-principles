"""ProteinGym exclusion and exact-sequence representative selection."""

from __future__ import annotations

from collections import defaultdict

from protein_lm.data.model_data.contracts import (
    Alias,
    CandidateRecord,
    CatalogRecord,
    ModelDataConfig,
    ModelDataError,
)


def prepare_population(
    records: list[CatalogRecord],
    reserved_families: frozenset[str],
    config: ModelDataConfig,
) -> tuple[tuple[CandidateRecord, ...], tuple[Alias, ...], int]:
    """Exclude reserved groups and collapse exact duplicate sequence hashes."""

    unreserved = [
        record for record in records if record.uniref50_group not in reserved_families
    ]
    grouped: dict[str, list[CatalogRecord]] = defaultdict(list)
    for record in unreserved:
        grouped[record.sequence_sha256].append(record)

    candidates = []
    aliases = []
    for sequence_digest, duplicates in grouped.items():
        sequences = {record.sequence for record in duplicates}
        groups = {record.uniref50_group for record in duplicates}
        if len(sequences) != 1 or len(groups) != 1:
            raise ModelDataError(
                "exact duplicate sequence hashes disagree on sequence or UniRef50 group"
            )
        canonical, *other_records = sorted(
            duplicates, key=lambda record: record.primary_accession
        )
        candidates.append(
            CandidateRecord(
                primary_accession=canonical.primary_accession,
                sequence_sha256=sequence_digest,
                biological_length=canonical.biological_length,
                length_bucket=length_bucket(canonical.biological_length, config),
                uniref50_group=canonical.uniref50_group,
            )
        )
        aliases.extend(
            Alias(
                sequence_digest, canonical.primary_accession, record.primary_accession
            )
            for record in other_records
        )
    candidates.sort(key=lambda record: record.primary_accession)
    aliases.sort(
        key=lambda alias: (
            alias.sequence_sha256,
            alias.canonical_accession,
            alias.alias_accession,
        )
    )
    return tuple(candidates), tuple(aliases), len(records) - len(unreserved)


def length_bucket(length: int, config: ModelDataConfig) -> str:
    """Return the one frozen bucket that contains an approved biological length."""

    for bucket in config.length_buckets:
        if bucket.minimum <= length <= bucket.maximum:
            return bucket.name
    raise ModelDataError("record length does not belong to a frozen bucket")
