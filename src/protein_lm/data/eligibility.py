"""Build the frozen Week 1 eligible-record catalog."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from protein_lm.data.eligibility_policy import (
    APPROVED_ELIGIBILITY_POLICY,
    APPROVED_ELIGIBILITY_POLICY_SHA256,
    EXCLUSION_FLAGS,
    TASK4_SCHEMA_VERSION,
    TASK4_SCOPE,
    EligibilityPolicy,
    Task4PreparationError,
)
from protein_lm.data.proteingym import ProteinGymScan, scan_proteingym_metadata
from protein_lm.data.task2_audit import SourceEvidence
from protein_lm.data.task4_report import (
    DerivedArtifact,
    DuplicateAudit,
    GroupAudit,
    PopulationAudit,
    ProteinGymReservationAudit,
    RecordResidueCount,
    Task4Report,
)
from protein_lm.data.uniprot import SwissProtRecord, parse_swiss_prot
from protein_lm.data.uniref import (
    UniRef50Audit,
    UniRef50Scan,
    scan_uniref50_membership,
)
CATALOG_COLUMNS = (
    "primary_accession",
    "sequence",
    "sequence_sha256",
    "biological_length",
    *EXCLUSION_FLAGS,
    "eligible",
    "primary_exclusion_reason",
    "uniref50_group",
    "proteingym_candidate_test_reserved",
)
_SOURCE_ROLES = frozenset(
    {"swiss_prot_records", "uniref50_membership", "proteingym_metadata"}
)
_ASCII_LETTERS_PATTERN = re.compile(r"^[A-Za-z]+$")


@dataclass(frozen=True)
class EligibilityFlags:
    """Every independent reason a source record can fail eligibility."""

    noncanonical_residue: bool
    fragment: bool
    below_min_length: bool
    above_max_length: bool
    blank_uniref50_mapping: bool


@dataclass(frozen=True)
class EligibleRecord:
    """One ignored local catalog row."""

    primary_accession: str
    sequence: str
    sequence_sha256: str
    biological_length: int
    flags: EligibilityFlags
    eligible: bool
    primary_exclusion_reason: str | None
    uniref50_group: str
    proteingym_candidate_test_reserved: bool


@dataclass(frozen=True)
class _ReservationResolution:
    families: frozenset[str]
    resolvable_target_count: int
    resolvable_assay_count: int
    family_set_sha256: str


def normalize_sequence(sequence: str) -> str:
    """Uppercase an already de-formatted ASCII amino-acid sequence."""

    if _ASCII_LETTERS_PATTERN.fullmatch(sequence) is None:
        raise Task4PreparationError("sequence must contain only ASCII letters")
    return sequence.upper()


def sequence_sha256(sequence: str) -> str:
    """Return the approved content identifier for one normalized sequence."""

    normalized = normalize_sequence(sequence)
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def classify_record(
    record: SwissProtRecord,
    *,
    uniref50_group: str,
    reserved_families: frozenset[str],
    policy: EligibilityPolicy = APPROVED_ELIGIBILITY_POLICY,
) -> EligibleRecord:
    """Apply every frozen flag before selecting one primary exclusion."""

    sequence = normalize_sequence(record.sequence)
    if len(sequence) != record.declared_length:
        raise Task4PreparationError(
            f"{record.primary_accession}: normalized sequence length changed"
        )

    canonical = frozenset(policy.canonical_amino_acids)
    flags = EligibilityFlags(
        noncanonical_residue=any(residue not in canonical for residue in sequence),
        fragment=record.is_fragment,
        below_min_length=record.declared_length < policy.minimum_length,
        above_max_length=record.declared_length > policy.maximum_length,
        blank_uniref50_mapping=not uniref50_group,
    )
    matched = {
        name for name in EXCLUSION_FLAGS if getattr(flags, name)
    }
    primary_reason = next(
        (name for name in policy.primary_exclusion_precedence if name in matched),
        None,
    )
    return EligibleRecord(
        primary_accession=record.primary_accession,
        sequence=sequence,
        sequence_sha256=sequence_sha256(sequence),
        biological_length=record.declared_length,
        flags=flags,
        eligible=primary_reason is None,
        primary_exclusion_reason=primary_reason,
        uniref50_group=uniref50_group,
        proteingym_candidate_test_reserved=(
            bool(uniref50_group) and uniref50_group in reserved_families
        ),
    )


def build_task4_catalog(
    *,
    swiss_prot_path: Path,
    uniref50_path: Path,
    proteingym_path: Path,
    catalog_output_path: Path,
    reserved_families_output_path: Path,
    catalog_relative_path: str,
    reserved_families_relative_path: str,
    policy: EligibilityPolicy,
    policy_sha256: str,
    task2_report_sha256: str,
    sources: Mapping[str, SourceEvidence],
    code_revision: str,
) -> Task4Report:
    """Build staged ignored artifacts and return only aggregate public evidence.

    The caller supplies staging paths and promotes them only after this function
    returns successfully.
    """

    _validate_build_inputs(
        policy=policy,
        policy_sha256=policy_sha256,
        task2_report_sha256=task2_report_sha256,
        sources=sources,
        code_revision=code_revision,
    )
    proteingym_scan = scan_proteingym_metadata(proteingym_path)
    accessions, matched_records, first_pass_residues = _index_swiss_prot(
        swiss_prot_path,
        frozenset(proteingym_scan.target_entry_names),
    )
    uniref50_scan = scan_uniref50_membership(
        uniref50_path,
        accessions,
        target_entry_names=proteingym_scan.target_entry_names,
    )
    _require_usable_mappings(uniref50_scan)
    resolution = _resolve_proteingym_families(
        proteingym_scan,
        matched_records=matched_records,
        uniref50_scan=uniref50_scan,
        policy=policy,
    )
    family_artifact = _write_reserved_families(
        reserved_families_output_path,
        relative_path=reserved_families_relative_path,
        families=resolution.families,
    )

    counters = _CatalogCounters()
    catalog_hasher = hashlib.sha256()
    byte_size = 0
    catalog_output_path.parent.mkdir(parents=True, exist_ok=True)
    with catalog_output_path.open("wb") as output:
        header = ("\t".join(CATALOG_COLUMNS) + "\n").encode()
        output.write(header)
        catalog_hasher.update(header)
        byte_size += len(header)

        for index, source_record in enumerate(parse_swiss_prot(swiss_prot_path)):
            if index >= len(accessions) or (
                source_record.primary_accession != accessions[index]
            ):
                raise Task4PreparationError(
                    "Swiss-Prot record order changed between preparation passes"
                )
            group = _group_for(source_record.primary_accession, uniref50_scan)
            catalog_record = classify_record(
                source_record,
                uniref50_group=group,
                reserved_families=resolution.families,
                policy=policy,
            )
            counters.observe(catalog_record)
            row = (_catalog_row(catalog_record) + "\n").encode()
            output.write(row)
            catalog_hasher.update(row)
            byte_size += len(row)

    if counters.source.records != len(accessions):
        raise Task4PreparationError(
            "Swiss-Prot record count changed between preparation passes"
        )
    if counters.source.residues != first_pass_residues:
        raise Task4PreparationError(
            "Swiss-Prot residue count changed between preparation passes"
        )
    if catalog_output_path.stat().st_size != byte_size:
        raise Task4PreparationError("catalog byte count changed after writing")
    population = counters.population_audit()
    source_duplicates = _duplicate_audit(counters.source_hash_counts)
    eligible_duplicates = _duplicate_audit(counters.eligible_hash_counts)
    groups = counters.group_audit(uniref50_scan.audit)
    reservation = counters.reservation_audit(resolution)
    _validate_reconciliations(
        population=population,
        mapping=uniref50_scan.audit,
        source_duplicates=source_duplicates,
        eligible_duplicates=eligible_duplicates,
        groups=groups,
        reservation=reservation,
        family_artifact=family_artifact,
    )

    catalog_artifact = DerivedArtifact(
        relative_path=catalog_relative_path,
        row_count=counters.source.records,
        byte_size=byte_size,
        sha256=catalog_hasher.hexdigest(),
    )
    return Task4Report(
        schema_version=TASK4_SCHEMA_VERSION,
        scope=TASK4_SCOPE,
        code_revision=code_revision,
        policy_sha256=policy_sha256,
        approved_task2_report_sha256=task2_report_sha256,
        sources=dict(sorted(sources.items())),
        population=population,
        mapping=uniref50_scan.audit,
        source_duplicates=source_duplicates,
        eligible_duplicates=eligible_duplicates,
        groups=groups,
        proteingym_reservation=reservation,
        catalog=catalog_artifact,
        reserved_families=family_artifact,
    )


class _CatalogCounters:
    def __init__(self) -> None:
        self.source = RecordResidueCount(0, 0)
        self.eligible = RecordResidueCount(0, 0)
        self.flag_counts = {name: [0, 0] for name in EXCLUSION_FLAGS}
        self.primary_counts = {name: [0, 0] for name in EXCLUSION_FLAGS}
        self.source_hash_counts: Counter[str] = Counter()
        self.eligible_hash_counts: Counter[str] = Counter()
        self.hash_sequences: dict[str, str] = {}
        self.eligible_hash_first_group: dict[str, str] = {}
        self.cross_group_hashes: set[str] = set()
        self.eligible_group_sizes: Counter[str] = Counter()
        self.source_reserved_families: set[str] = set()
        self.eligible_reserved_families: set[str] = set()
        self.source_reserved = [0, 0]
        self.eligible_reserved = [0, 0]

    def observe(self, record: EligibleRecord) -> None:
        self.source = _add(self.source, record.biological_length)
        self.source_hash_counts[record.sequence_sha256] += 1
        previous = self.hash_sequences.setdefault(
            record.sequence_sha256, record.sequence
        )
        if previous != record.sequence:
            raise Task4PreparationError(
                "one SHA-256 digest was associated with different sequences"
            )

        for name in EXCLUSION_FLAGS:
            if getattr(record.flags, name):
                self.flag_counts[name][0] += 1
                self.flag_counts[name][1] += record.biological_length
        if record.primary_exclusion_reason is not None:
            primary = self.primary_counts[record.primary_exclusion_reason]
            primary[0] += 1
            primary[1] += record.biological_length

        if record.proteingym_candidate_test_reserved:
            self.source_reserved_families.add(record.uniref50_group)
            self.source_reserved[0] += 1
            self.source_reserved[1] += record.biological_length

        if not record.eligible:
            return
        if record.primary_exclusion_reason is not None:
            raise RuntimeError("eligible record has an exclusion reason")

        self.eligible = _add(self.eligible, record.biological_length)
        self.eligible_hash_counts[record.sequence_sha256] += 1
        self.eligible_group_sizes[record.uniref50_group] += 1
        first_group = self.eligible_hash_first_group.setdefault(
            record.sequence_sha256, record.uniref50_group
        )
        if first_group != record.uniref50_group:
            self.cross_group_hashes.add(record.sequence_sha256)
        if record.proteingym_candidate_test_reserved:
            self.eligible_reserved_families.add(record.uniref50_group)
            self.eligible_reserved[0] += 1
            self.eligible_reserved[1] += record.biological_length

    def population_audit(self) -> PopulationAudit:
        excluded = RecordResidueCount(
            records=self.source.records - self.eligible.records,
            residues=self.source.residues - self.eligible.residues,
        )
        return PopulationAudit(
            source=self.source,
            eligible=self.eligible,
            excluded=excluded,
            matched_flags={
                name: RecordResidueCount(*self.flag_counts[name])
                for name in EXCLUSION_FLAGS
            },
            primary_exclusions={
                name: RecordResidueCount(*self.primary_counts[name])
                for name in EXCLUSION_FLAGS
            },
        )

    def group_audit(self, mapping: UniRef50Audit) -> GroupAudit:
        return GroupAudit(
            source_unique_group_count=mapping.unique_group_count,
            eligible_unique_group_count=len(self.eligible_group_sizes),
            maximum_source_group_size=mapping.maximum_group_size,
            maximum_eligible_group_size=max(
                self.eligible_group_sizes.values(), default=0
            ),
            eligible_duplicate_hashes_across_groups=len(self.cross_group_hashes),
            eligible_records_in_cross_group_duplicate_hashes=sum(
                self.eligible_hash_counts[digest]
                for digest in self.cross_group_hashes
            ),
        )

    def reservation_audit(
        self, resolution: _ReservationResolution
    ) -> ProteinGymReservationAudit:
        return ProteinGymReservationAudit(
            resolvable_target_count=resolution.resolvable_target_count,
            resolvable_assay_count=resolution.resolvable_assay_count,
            reserved_family_count=len(resolution.families),
            reserved_family_set_sha256=resolution.family_set_sha256,
            source_represented_family_count=len(self.source_reserved_families),
            source_records=self.source_reserved[0],
            source_residues=self.source_reserved[1],
            eligible_represented_family_count=len(self.eligible_reserved_families),
            eligible_records=self.eligible_reserved[0],
            eligible_residues=self.eligible_reserved[1],
        )


def _index_swiss_prot(
    path: Path, target_entry_names: frozenset[str]
) -> tuple[list[str], dict[str, SwissProtRecord], int]:
    accessions: list[str] = []
    matched_records: dict[str, SwissProtRecord] = {}
    residue_count = 0
    for record in parse_swiss_prot(path):
        accessions.append(record.primary_accession)
        residue_count += record.declared_length
        if record.entry_name in target_entry_names:
            if record.entry_name in matched_records:
                raise Task4PreparationError(
                    f"duplicate Swiss-Prot entry name {record.entry_name!r}"
                )
            matched_records[record.entry_name] = record
    if not accessions:
        raise Task4PreparationError("Swiss-Prot source contains no records")
    return accessions, matched_records, residue_count


def _require_usable_mappings(scan: UniRef50Scan) -> None:
    failures = {
        "missing": scan.missing_accessions,
        "duplicate": scan.duplicate_accessions,
        "conflicting": scan.conflicting_accessions,
    }
    found = [
        f"{name}={len(values)}" for name, values in failures.items() if values
    ]
    if found:
        raise Task4PreparationError(
            "fatal UniRef50 accession mapping evidence: " + ", ".join(found)
        )


def _resolve_proteingym_families(
    scan: ProteinGymScan,
    *,
    matched_records: Mapping[str, SwissProtRecord],
    uniref50_scan: UniRef50Scan,
    policy: EligibilityPolicy,
) -> _ReservationResolution:
    target_names = frozenset(scan.target_entry_names)
    duplicate_targets = target_names.intersection(
        uniref50_scan.duplicate_entry_names
    )
    if duplicate_targets:
        raise Task4PreparationError(
            "ProteinGym target entry names have duplicate UniRef50 rows"
        )

    family_by_target: dict[str, str] = {}
    for entry_name in scan.target_entry_names:
        swiss_record = matched_records.get(entry_name)
        family = _proteingym_family(
            entry_name=entry_name,
            swiss_record=swiss_record,
            uniref50_scan=uniref50_scan,
        )
        if family is not None:
            family_by_target[entry_name] = family

    families = frozenset(family_by_target.values())
    resolvable_assays = sum(
        assay.entry_name in family_by_target for assay in scan.assays
    )
    actual = (
        len(family_by_target),
        resolvable_assays,
        len(families),
    )
    expected = (
        policy.expected_resolvable_proteingym_targets,
        policy.expected_resolvable_proteingym_assays,
        policy.expected_reserved_proteingym_families,
    )
    if actual != expected:
        raise Task4PreparationError(
            "ProteinGym reservation anchors changed: "
            f"expected targets/assays/families {expected}, found {actual}"
        )
    family_bytes = _family_set_bytes(families)
    return _ReservationResolution(
        families=families,
        resolvable_target_count=len(family_by_target),
        resolvable_assay_count=resolvable_assays,
        family_set_sha256=hashlib.sha256(family_bytes).hexdigest(),
    )


def _proteingym_family(
    *,
    entry_name: str,
    swiss_record: SwissProtRecord | None,
    uniref50_scan: UniRef50Scan,
) -> str | None:
    if swiss_record is not None:
        accession = swiss_record.primary_accession
        if accession in uniref50_scan.blank_group_accessions:
            return None
        group = uniref50_scan.accession_to_group.get(accession)
        if group is None:
            raise Task4PreparationError(
                f"ProteinGym Swiss-Prot target {entry_name!r} has no mapping status"
            )
        optional_match = uniref50_scan.entry_name_matches.get(entry_name)
        if optional_match is not None and (
            not optional_match.accession_in_target_population
            or optional_match.accession != accession
            or optional_match.group != group
        ):
            raise Task4PreparationError(
                f"ProteinGym target {entry_name!r} has inconsistent mappings"
            )
        return group

    match = uniref50_scan.entry_name_matches.get(entry_name)
    if match is not None:
        if match.accession_in_target_population:
            raise Task4PreparationError(
                f"ProteinGym target {entry_name!r} resolves to an inconsistent "
                "Swiss-Prot accession"
            )
        return match.group
    if entry_name in uniref50_scan.conflicting_entry_names:
        raise Task4PreparationError(
            f"ProteinGym target {entry_name!r} has conflicting UniRef50 mappings"
        )
    if (
        entry_name in uniref50_scan.blank_group_entry_names
        or entry_name in uniref50_scan.missing_entry_names
    ):
        return None
    raise Task4PreparationError(
        f"ProteinGym target {entry_name!r} has no UniRef50 mapping status"
    )


def _group_for(accession: str, scan: UniRef50Scan) -> str:
    if accession in scan.blank_group_accessions:
        return ""
    group = scan.accession_to_group.get(accession)
    if group is None:
        raise Task4PreparationError(
            f"accession {accession!r} has no usable UniRef50 mapping"
        )
    return group


def _catalog_row(record: EligibleRecord) -> str:
    values = (
        record.primary_accession,
        record.sequence,
        record.sequence_sha256,
        str(record.biological_length),
        *(_bool(getattr(record.flags, name)) for name in EXCLUSION_FLAGS),
        _bool(record.eligible),
        record.primary_exclusion_reason or "",
        record.uniref50_group,
        _bool(record.proteingym_candidate_test_reserved),
    )
    if any("\t" in value or "\n" in value or "\r" in value for value in values):
        raise Task4PreparationError("catalog value contains a tab or newline")
    return "\t".join(values)


def _write_reserved_families(
    path: Path, *, relative_path: str, families: frozenset[str]
) -> DerivedArtifact:
    content = _family_set_bytes(families)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return DerivedArtifact(
        relative_path=relative_path,
        row_count=len(families),
        byte_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _family_set_bytes(families: frozenset[str]) -> bytes:
    if not families:
        raise Task4PreparationError("reserved ProteinGym family set is empty")
    return ("\n".join(sorted(families)) + "\n").encode("ascii")


def _duplicate_audit(counts: Counter[str]) -> DuplicateAudit:
    duplicate_counts = tuple(count for count in counts.values() if count > 1)
    return DuplicateAudit(
        unique_sequence_hash_count=len(counts),
        duplicate_sequence_group_count=len(duplicate_counts),
        records_in_duplicate_groups=sum(duplicate_counts),
        redundant_record_count=sum(count - 1 for count in duplicate_counts),
        maximum_duplicate_multiplicity=max(counts.values(), default=0),
    )


def _validate_reconciliations(
    *,
    population: PopulationAudit,
    mapping: UniRef50Audit,
    source_duplicates: DuplicateAudit,
    eligible_duplicates: DuplicateAudit,
    groups: GroupAudit,
    reservation: ProteinGymReservationAudit,
    family_artifact: DerivedArtifact,
) -> None:
    if population.source.records != (
        population.eligible.records + population.excluded.records
    ):
        raise RuntimeError("source record counts do not reconcile")
    if population.source.residues != (
        population.eligible.residues + population.excluded.residues
    ):
        raise RuntimeError("source residue counts do not reconcile")
    if sum(value.records for value in population.primary_exclusions.values()) != (
        population.excluded.records
    ):
        raise RuntimeError("primary exclusion record counts do not reconcile")
    if sum(value.residues for value in population.primary_exclusions.values()) != (
        population.excluded.residues
    ):
        raise RuntimeError("primary exclusion residue counts do not reconcile")
    if mapping.target_accession_count != population.source.records:
        raise RuntimeError("mapping target count does not match source records")
    if source_duplicates.unique_sequence_hash_count > population.source.records:
        raise RuntimeError("source duplicate counts do not reconcile")
    if eligible_duplicates.unique_sequence_hash_count > population.eligible.records:
        raise RuntimeError("eligible duplicate counts do not reconcile")
    if groups.eligible_unique_group_count > groups.source_unique_group_count:
        raise RuntimeError("eligible group counts do not reconcile")
    if reservation.reserved_family_count != family_artifact.row_count:
        raise RuntimeError("reserved family artifact count does not reconcile")
    if reservation.reserved_family_set_sha256 != family_artifact.sha256:
        raise RuntimeError("reserved family artifact checksum does not reconcile")


def _validate_build_inputs(
    *,
    policy: EligibilityPolicy,
    policy_sha256: str,
    task2_report_sha256: str,
    sources: Mapping[str, SourceEvidence],
    code_revision: str,
) -> None:
    if policy.scope != TASK4_SCOPE:
        raise Task4PreparationError("eligibility policy has the wrong scope")
    if policy != APPROVED_ELIGIBILITY_POLICY:
        raise Task4PreparationError("eligibility policy is not the approved policy")
    if policy_sha256 != APPROVED_ELIGIBILITY_POLICY_SHA256:
        raise Task4PreparationError(
            "eligibility policy bytes do not match the approved checksum"
        )
    if task2_report_sha256 != policy.approved_task2_report_sha256:
        raise Task4PreparationError("Task 2 report checksum is not approved")
    if set(sources) != _SOURCE_ROLES:
        raise Task4PreparationError(
            "source evidence does not contain the three required roles"
        )
    if not code_revision:
        raise Task4PreparationError("code revision must not be empty")


def _add(count: RecordResidueCount, length: int) -> RecordResidueCount:
    return replace(
        count,
        records=count.records + 1,
        residues=count.residues + length,
    )


def _bool(value: bool) -> str:
    return "true" if value else "false"
