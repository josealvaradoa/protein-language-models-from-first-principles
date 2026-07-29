"""Build the frozen Week 1 random diagnostic split."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path

from protein_lm.data.eligibility import CATALOG_COLUMNS
from protein_lm.data.random_split_policy import (
    APPROVED_RANDOM_SPLIT_CONFIG_SHA256,
    APPROVED_RANDOM_SPLIT_POLICY,
    PARTITIONS,
    RandomSplitError,
    RandomSplitPolicy,
)
from protein_lm.data.task5_report import (
    DerivedArtifact,
    PartitionAudit,
    RandomSplitBuild,
    SplitPopulation,
)

LOCAL_ASSIGNMENT_COLUMNS = (
    "strategy",
    "stage",
    "repair_cycle",
    "stable_assignment_unit",
    "partition_or_exclusion_status",
    "accession",
)
PUBLIC_MANIFEST_COLUMNS = (
    "primary_accession",
    "partition",
    "sequence_sha256",
    "biological_length",
    "uniref50_group",
)

_HASH_SPACE = 1 << 256
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UNIREF50_PATTERN = re.compile(r"^UniRef50_[\x21-\x7e]+$")
_PERCENT_QUANTUM = Decimal("0.000001")


class DiagnosticSplitUseError(RandomSplitError):
    """Raised when code attempts to use the diagnostic for model training."""


@dataclass(frozen=True)
class SplitInputRecord:
    """The approved Task 4 fields needed for diagnostic assignment."""

    primary_accession: str
    sequence_sha256: str
    biological_length: int
    uniref50_group: str
    proteingym_candidate_test_reserved: bool


def assignment_payload(
    primary_accession: str,
    policy: RandomSplitPolicy = APPROVED_RANDOM_SPLIT_POLICY,
) -> bytes:
    """Construct the exact bytes hashed for one accession assignment."""

    _require_visible_ascii(primary_accession, "primary accession")
    _require_visible_ascii(policy.assignment_namespace, "assignment namespace")
    return (
        policy.assignment_namespace.encode("ascii")
        + b"\x00"
        + str(policy.seed).encode("ascii")
        + b"\x00"
        + primary_accession.encode("ascii")
    )


def partition_from_integer(value: int) -> str:
    """Apply the exact frozen 90/5/5 boundaries without floating point."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("assignment value must be an integer")
    if value < 0 or value >= _HASH_SPACE:
        raise ValueError("assignment value must fit in an unsigned SHA-256 digest")
    if 10 * value < 9 * _HASH_SPACE:
        return "training"
    if 20 * value < 19 * _HASH_SPACE:
        return "validation"
    return "test"


def assign_partition(
    primary_accession: str,
    policy: RandomSplitPolicy = APPROVED_RANDOM_SPLIT_POLICY,
) -> str:
    """Assign one accession independently and deterministically."""

    digest = hashlib.sha256(assignment_payload(primary_accession, policy)).digest()
    return partition_from_integer(int.from_bytes(digest, byteorder="big"))


def require_selected_training_manifest(metadata: Mapping[str, object]) -> None:
    """Reject diagnostic metadata at the future model-training boundary."""

    if metadata.get("strategy") != "group_aware":
        raise DiagnosticSplitUseError(
            "manifest strategy is not the selected group-aware candidate"
        )
    if metadata.get("stage") != "selected":
        raise DiagnosticSplitUseError("manifest stage is not the final selected stage")
    if metadata.get("diagnostic_only") is not False:
        raise DiagnosticSplitUseError(
            "manifest is not explicitly marked as non-diagnostic"
        )
    if metadata.get("selected_for_training") is not True:
        raise DiagnosticSplitUseError(
            "manifest is not explicitly selected for training"
        )
    if metadata.get("model_use") != "selected_training_allowed":
        raise DiagnosticSplitUseError(
            "manifest does not explicitly allow selected training use"
        )


def sha256_sidecar(filename: str, digest: str) -> str:
    """Return one portable SHA-256 sidecar line."""

    if Path(filename).name != filename:
        raise RandomSplitError("sidecar filename must not contain a directory")
    _require_visible_ascii(filename, "sidecar filename")
    if not _SHA256_PATTERN.fullmatch(digest):
        raise RandomSplitError("sidecar SHA-256 is malformed")
    return f"{digest}  {filename}\n"


def validate_task4_report(
    task4_report: Mapping[str, object],
    policy: RandomSplitPolicy,
) -> dict[str, dict[str, object]]:
    """Validate the Task 4 anchors needed before constructing Task 5."""

    expected = {
        "schema version": (task4_report.get("schema_version"), 1),
        "scope": (
            task4_report.get("scope"),
            "week_01_task_4_eligible_records",
        ),
        "Task 4 policy SHA-256": (
            task4_report.get("policy_sha256"),
            policy.task4_policy_sha256,
        ),
        "catalog relative path": (
            _nested(task4_report, "catalog", "relative_path"),
            "data/processed/week_01/task_04_record_catalog.tsv",
        ),
        "catalog SHA-256": (
            _nested(task4_report, "catalog", "sha256"),
            policy.task4_catalog_sha256,
        ),
        "catalog byte size": (
            _nested(task4_report, "catalog", "byte_size"),
            policy.task4_catalog_byte_size,
        ),
        "catalog row count": (
            _nested(task4_report, "catalog", "row_count"),
            policy.task4_catalog_row_count,
        ),
        "eligible records": (
            _nested(task4_report, "population", "eligible", "records"),
            policy.expected_eligible_records,
        ),
        "eligible residues": (
            _nested(task4_report, "population", "eligible", "residues"),
            policy.expected_eligible_residues,
        ),
        "eligible groups": (
            _nested(task4_report, "groups", "eligible_unique_group_count"),
            policy.expected_eligible_groups,
        ),
    }
    differences = [
        f"{name}: found {actual!r}, expected {approved!r}"
        for name, (actual, approved) in expected.items()
        if actual != approved
    ]
    if differences:
        raise RandomSplitError("Task 4 report drift: " + "; ".join(differences))

    sources = task4_report.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise RandomSplitError("Task 4 report sources are malformed")
    if any(
        not isinstance(key, str) or not isinstance(value, dict)
        for key, value in sources.items()
    ):
        raise RandomSplitError("Task 4 report sources are malformed")
    return sources


def build_random_diagnostic(
    *,
    catalog_path: Path,
    local_assignment_output_path: Path,
    public_manifest_output_path: Path,
    policy: RandomSplitPolicy,
    policy_sha256: str,
) -> RandomSplitBuild:
    """Build staged local and public Task 5 manifests."""

    _validate_build_policy(policy, policy_sha256)
    records = _read_eligible_records(catalog_path, policy)
    records.sort(key=lambda record: record.primary_accession)
    return _write_manifests(
        records,
        local_assignment_output_path=local_assignment_output_path,
        public_manifest_output_path=public_manifest_output_path,
        policy=policy,
    )


def _read_eligible_records(
    path: Path,
    policy: RandomSplitPolicy,
) -> list[SplitInputRecord]:
    records: list[SplitInputRecord] = []
    accessions: set[str] = set()
    groups: set[str] = set()
    residue_count = 0
    source_rows = 0
    byte_size = 0
    hasher = hashlib.sha256()

    try:
        source = Path(path).open("rb")
    except OSError as error:
        raise RandomSplitError(f"could not open Task 4 catalog: {error}") from error

    with source:
        raw_header = source.readline()
        byte_size += len(raw_header)
        hasher.update(raw_header)
        header = _decode_lf_line(raw_header, line_number=1)
        if tuple(header.split("\t")) != CATALOG_COLUMNS:
            raise RandomSplitError("Task 4 catalog header is not approved")

        for line_number, raw_line in enumerate(source, start=2):
            source_rows += 1
            byte_size += len(raw_line)
            hasher.update(raw_line)
            columns = _decode_lf_line(raw_line, line_number=line_number).split("\t")
            if len(columns) != len(CATALOG_COLUMNS):
                raise RandomSplitError(
                    f"catalog line {line_number}: expected "
                    f"{len(CATALOG_COLUMNS)} columns"
                )
            eligible = _parse_bool(columns[9], line_number, "eligible")
            if not eligible:
                continue
            if any(
                _parse_bool(columns[index], line_number, CATALOG_COLUMNS[index])
                for index in range(4, 9)
            ):
                raise RandomSplitError(
                    f"catalog line {line_number}: eligible row has exclusion flags"
                )
            if columns[10]:
                raise RandomSplitError(
                    f"catalog line {line_number}: eligible row has exclusion reason"
                )

            accession = columns[0]
            sequence_digest = columns[2]
            group = columns[11]
            _require_visible_ascii(accession, f"catalog line {line_number} accession")
            if not _SHA256_PATTERN.fullmatch(sequence_digest):
                raise RandomSplitError(
                    f"catalog line {line_number}: malformed sequence SHA-256"
                )
            if not _UNIREF50_PATTERN.fullmatch(group):
                raise RandomSplitError(
                    f"catalog line {line_number}: malformed UniRef50 group"
                )
            try:
                biological_length = int(columns[3])
            except ValueError as error:
                raise RandomSplitError(
                    f"catalog line {line_number}: malformed biological length"
                ) from error
            if biological_length <= 0:
                raise RandomSplitError(
                    f"catalog line {line_number}: biological length must be positive"
                )
            if accession in accessions:
                raise RandomSplitError(
                    f"catalog line {line_number}: duplicate eligible accession"
                )

            accessions.add(accession)
            groups.add(group)
            residue_count += biological_length
            records.append(
                SplitInputRecord(
                    primary_accession=accession,
                    sequence_sha256=sequence_digest,
                    biological_length=biological_length,
                    uniref50_group=group,
                    proteingym_candidate_test_reserved=_parse_bool(
                        columns[12],
                        line_number,
                        "proteingym_candidate_test_reserved",
                    ),
                )
            )

    actual = {
        "catalog SHA-256": (hasher.hexdigest(), policy.task4_catalog_sha256),
        "catalog byte size": (byte_size, policy.task4_catalog_byte_size),
        "catalog row count": (source_rows, policy.task4_catalog_row_count),
        "eligible records": (len(records), policy.expected_eligible_records),
        "eligible residues": (residue_count, policy.expected_eligible_residues),
        "eligible groups": (len(groups), policy.expected_eligible_groups),
    }
    differences = [
        f"{name}: found {found}, expected {expected}"
        for name, (found, expected) in actual.items()
        if found != expected
    ]
    if differences:
        raise RandomSplitError("Task 4 catalog drift: " + "; ".join(differences))
    return records


def _write_manifests(
    records: list[SplitInputRecord],
    *,
    local_assignment_output_path: Path,
    public_manifest_output_path: Path,
    policy: RandomSplitPolicy,
) -> RandomSplitBuild:
    local_assignment_output_path.parent.mkdir(parents=True, exist_ok=True)
    public_manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    local_hasher = hashlib.sha256()
    public_hasher = hashlib.sha256()
    local_byte_size = 0
    public_byte_size = 0
    partition_counts = {partition: [0, 0] for partition in PARTITIONS}
    partition_groups = {partition: set() for partition in PARTITIONS}

    with (
        local_assignment_output_path.open("wb") as local_output,
        public_manifest_output_path.open("wb") as public_output,
    ):
        local_header = ("\t".join(LOCAL_ASSIGNMENT_COLUMNS) + "\n").encode()
        public_header = ("\t".join(PUBLIC_MANIFEST_COLUMNS) + "\n").encode()
        local_output.write(local_header)
        public_output.write(public_header)
        local_hasher.update(local_header)
        public_hasher.update(public_header)
        local_byte_size += len(local_header)
        public_byte_size += len(public_header)

        for record in records:
            partition = assign_partition(record.primary_accession, policy)
            partition_counts[partition][0] += 1
            partition_counts[partition][1] += record.biological_length
            partition_groups[partition].add(record.uniref50_group)

            local_row = _tsv_row(
                (
                    policy.strategy,
                    policy.stage,
                    "0",
                    record.primary_accession,
                    partition,
                    record.primary_accession,
                )
            )
            public_row = _tsv_row(
                (
                    record.primary_accession,
                    partition,
                    record.sequence_sha256,
                    str(record.biological_length),
                    record.uniref50_group,
                )
            )
            local_bytes = (local_row + "\n").encode("utf-8")
            public_bytes = (public_row + "\n").encode("utf-8")
            local_output.write(local_bytes)
            public_output.write(public_bytes)
            local_hasher.update(local_bytes)
            public_hasher.update(public_bytes)
            local_byte_size += len(local_bytes)
            public_byte_size += len(public_bytes)

    if local_assignment_output_path.stat().st_size != local_byte_size:
        raise RandomSplitError("local assignment byte count changed after writing")
    if public_manifest_output_path.stat().st_size != public_byte_size:
        raise RandomSplitError("public manifest byte count changed after writing")

    population = SplitPopulation(
        records=len(records),
        residues=sum(record.biological_length for record in records),
        unique_groups=len({record.uniref50_group for record in records}),
    )
    partitions = _partition_audits(
        partition_counts=partition_counts,
        partition_groups=partition_groups,
        population=population,
        policy=policy,
    )
    _validate_partition_reconciliation(population, partitions)
    return RandomSplitBuild(
        population=population,
        partitions=partitions,
        local_assignments=DerivedArtifact(
            relative_path=policy.local_assignment_relative_path,
            row_count=population.records,
            byte_size=local_byte_size,
            sha256=local_hasher.hexdigest(),
        ),
        public_manifest=DerivedArtifact(
            relative_path=policy.public_manifest_relative_path,
            row_count=population.records,
            byte_size=public_byte_size,
            sha256=public_hasher.hexdigest(),
        ),
    )


def _partition_audits(
    *,
    partition_counts: dict[str, list[int]],
    partition_groups: dict[str, set[str]],
    population: SplitPopulation,
    policy: RandomSplitPolicy,
) -> dict[str, PartitionAudit]:
    targets = {
        "training": policy.training_target_numerator,
        "validation": policy.validation_target_numerator,
        "test": policy.test_target_numerator,
    }
    audits = {}
    for partition in PARTITIONS:
        record_count, residue_count = partition_counts[partition]
        target_numerator = targets[partition]
        audits[partition] = PartitionAudit(
            target_numerator=target_numerator,
            target_denominator=policy.target_denominator,
            target_share_percent=_format_percent(
                target_numerator,
                policy.target_denominator,
            ),
            records=record_count,
            residues=residue_count,
            unique_groups=len(partition_groups[partition]),
            record_share_percent=_format_percent(
                record_count,
                population.records,
            ),
            residue_share_percent=_format_percent(
                residue_count,
                population.residues,
            ),
            record_deviation_percentage_points=_format_deviation(
                record_count,
                population.records,
                target_numerator,
                policy.target_denominator,
            ),
            residue_deviation_percentage_points=_format_deviation(
                residue_count,
                population.residues,
                target_numerator,
                policy.target_denominator,
            ),
        )
    return audits


def _format_percent(numerator: int, denominator: int) -> str:
    with localcontext() as context:
        context.prec = 50
        percent = Decimal(numerator) * 100 / Decimal(denominator)
        return format(percent.quantize(_PERCENT_QUANTUM, ROUND_HALF_UP), "f")


def _format_deviation(
    realized_numerator: int,
    realized_denominator: int,
    target_numerator: int,
    target_denominator: int,
) -> str:
    with localcontext() as context:
        context.prec = 50
        realized = Decimal(realized_numerator) / Decimal(realized_denominator)
        target = Decimal(target_numerator) / Decimal(target_denominator)
        difference = (realized - target) * 100
        return format(difference.quantize(_PERCENT_QUANTUM, ROUND_HALF_UP), "f")


def _validate_partition_reconciliation(
    population: SplitPopulation,
    partitions: Mapping[str, PartitionAudit],
) -> None:
    if set(partitions) != set(PARTITIONS):
        raise RuntimeError("diagnostic partitions do not match the frozen names")
    if (
        sum(partition.records for partition in partitions.values())
        != population.records
    ):
        raise RuntimeError("diagnostic record counts do not reconcile")
    if (
        sum(partition.residues for partition in partitions.values())
        != population.residues
    ):
        raise RuntimeError("diagnostic residue counts do not reconcile")


def _validate_build_policy(
    policy: RandomSplitPolicy,
    policy_sha256: str,
) -> None:
    if policy != APPROVED_RANDOM_SPLIT_POLICY:
        raise RandomSplitError("random split policy is not the approved policy")
    if policy_sha256 != APPROVED_RANDOM_SPLIT_CONFIG_SHA256:
        raise RandomSplitError(
            "random split policy bytes do not match the approved checksum"
        )
    if (
        policy.training_target_numerator
        + policy.validation_target_numerator
        + policy.test_target_numerator
        != policy.target_denominator
    ):
        raise RandomSplitError("random split target fractions do not sum to one")


def _decode_lf_line(raw_line: bytes, *, line_number: int) -> str:
    if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
        raise RandomSplitError(
            f"catalog line {line_number}: expected one LF line ending"
        )
    try:
        return raw_line[:-1].decode("utf-8")
    except UnicodeDecodeError as error:
        raise RandomSplitError(f"catalog line {line_number}: invalid UTF-8") from error


def _parse_bool(value: str, line_number: int, field: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise RandomSplitError(f"catalog line {line_number}: {field} must be true or false")


def _require_visible_ascii(value: str, field: str) -> None:
    if (
        not value
        or not value.isascii()
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise RandomSplitError(f"{field} must contain only visible ASCII")


def _tsv_row(values: tuple[str, ...]) -> str:
    if any("\t" in value or "\n" in value or "\r" in value for value in values):
        raise RandomSplitError("manifest value contains a tab or newline")
    return "\t".join(values)


def _nested(parent: Mapping[str, object], *keys: str) -> object:
    value: object = parent
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value
