"""Load the frozen Week 1 eligible-record policy."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

TASK4_SCHEMA_VERSION = 1
TASK4_SCOPE = "week_01_task_4_eligible_records"
APPROVED_ELIGIBILITY_POLICY_SHA256 = (
    "69b8160e0aa972b66d28c0404af1c1b0a745bc423b8a620e47d30e5e483ce276"
)
EXCLUSION_FLAGS = (
    "noncanonical_residue",
    "fragment",
    "below_min_length",
    "above_max_length",
    "blank_uniref50_mapping",
)


class Task4PreparationError(ValueError):
    """Raised when Task 4 cannot produce a contract-compliant catalog."""


@dataclass(frozen=True)
class EligibilityPolicy:
    """The Task 3 choices that directly govern eligible-record preparation."""

    schema_version: int
    scope: str
    source_release: str
    proteingym_release: str
    approved_task2_report_sha256: str
    sequence_hash: str
    canonical_amino_acids: str
    minimum_length: int
    maximum_length: int
    primary_exclusion_precedence: tuple[str, ...]
    expected_resolvable_proteingym_targets: int
    expected_resolvable_proteingym_assays: int
    expected_reserved_proteingym_families: int
    catalog_order: str


APPROVED_ELIGIBILITY_POLICY = EligibilityPolicy(
    schema_version=TASK4_SCHEMA_VERSION,
    scope=TASK4_SCOPE,
    source_release="2026_02",
    proteingym_release="v1.3",
    approved_task2_report_sha256=(
        "ab83d9a3341694dab9b4097334f43b2036e5b4fb0417c8b3a028e54f679cdd0f"
    ),
    sequence_hash="sha256",
    canonical_amino_acids="ACDEFGHIKLMNPQRSTVWY",
    minimum_length=32,
    maximum_length=2046,
    primary_exclusion_precedence=EXCLUSION_FLAGS,
    expected_resolvable_proteingym_targets=177,
    expected_resolvable_proteingym_assays=207,
    expected_reserved_proteingym_families=175,
    catalog_order="swiss_prot_source",
)


def load_eligibility_policy(path: Path) -> EligibilityPolicy:
    """Load the committed policy and reject any drift from Task 3."""

    try:
        with Path(path).open("rb") as source:
            raw = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise Task4PreparationError(
            f"could not load eligibility policy {path}: {error}"
        ) from error

    if set(raw) != {field.name for field in fields(EligibilityPolicy)}:
        raise Task4PreparationError(
            "eligibility policy fields differ from the approved schema"
        )

    try:
        policy = EligibilityPolicy(
            schema_version=_integer(raw, "schema_version"),
            scope=_string(raw, "scope"),
            source_release=_string(raw, "source_release"),
            proteingym_release=_string(raw, "proteingym_release"),
            approved_task2_report_sha256=_string(
                raw, "approved_task2_report_sha256"
            ),
            sequence_hash=_string(raw, "sequence_hash"),
            canonical_amino_acids=_string(raw, "canonical_amino_acids"),
            minimum_length=_integer(raw, "minimum_length"),
            maximum_length=_integer(raw, "maximum_length"),
            primary_exclusion_precedence=_string_tuple(
                raw, "primary_exclusion_precedence"
            ),
            expected_resolvable_proteingym_targets=_integer(
                raw, "expected_resolvable_proteingym_targets"
            ),
            expected_resolvable_proteingym_assays=_integer(
                raw, "expected_resolvable_proteingym_assays"
            ),
            expected_reserved_proteingym_families=_integer(
                raw, "expected_reserved_proteingym_families"
            ),
            catalog_order=_string(raw, "catalog_order"),
        )
    except (TypeError, ValueError) as error:
        raise Task4PreparationError(f"invalid eligibility policy: {error}") from error

    if policy != APPROVED_ELIGIBILITY_POLICY:
        raise Task4PreparationError(
            "eligibility policy differs from the approved Week 1 decisions"
        )
    return policy


def _string(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{key} must be a nonempty string")
    return value


def _integer(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _string_tuple(raw: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise TypeError(f"{key} must be a nonempty string array")
    if any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"{key} must contain only nonempty strings")
    return tuple(value)
