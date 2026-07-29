"""Load the frozen Week 1 Task 6 group-aware split policy."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

TASK6_SCHEMA_VERSION = 1
TASK6_SCOPE = "week_01_task_06_group_aware_pre_repair"
APPROVED_GROUP_SPLIT_CONFIG_SHA256 = (
    "54708910a4d09f2c9af4be2171e0044dcc08799a5e0d64e9620a419095af10b8"
)


class GroupSplitError(ValueError):
    """Raised when Task 6 cannot produce trustworthy candidate evidence."""


@dataclass(frozen=True)
class GroupSplitPolicy:
    """Every frozen input and algorithm choice used by Task 6."""

    schema_version: int
    scope: str
    strategy: str
    stage: str
    repair_cycle: int
    seed: int
    order_namespace: str
    hash_algorithm: str
    target_denominator: int
    training_target_numerator: int
    validation_target_numerator: int
    test_target_numerator: int
    balance_tolerance_numerator: int
    balance_tolerance_denominator: int
    assignment_unit: str
    manifest_order: str
    partition_tie_order: str
    model_use: str
    selected_for_training: bool
    license_spdx: str
    task4_report_sha256: str
    task4_policy_sha256: str
    task4_catalog_sha256: str
    task4_catalog_byte_size: int
    task4_catalog_row_count: int
    expected_eligible_records: int
    expected_eligible_residues: int
    expected_eligible_groups: int
    expected_reserved_family_universe: int
    expected_eligible_reserved_groups: int
    expected_eligible_reserved_records: int
    expected_eligible_reserved_residues: int
    local_assignment_relative_path: str
    public_manifest_relative_path: str


APPROVED_GROUP_SPLIT_POLICY = GroupSplitPolicy(
    schema_version=TASK6_SCHEMA_VERSION,
    scope=TASK6_SCOPE,
    strategy="group_aware",
    stage="pre_repair",
    repair_cycle=0,
    seed=20260727,
    order_namespace="week1-group-order-v1",
    hash_algorithm="sha256",
    target_denominator=20,
    training_target_numerator=18,
    validation_target_numerator=1,
    test_target_numerator=1,
    balance_tolerance_numerator=1,
    balance_tolerance_denominator=200,
    assignment_unit="uniref50_exact_duplicate_component",
    manifest_order="primary_accession",
    partition_tie_order="training,validation,test",
    model_use="prohibited",
    selected_for_training=False,
    license_spdx="CC-BY-4.0",
    task4_report_sha256=(
        "be791d35b39c4bf1337c121ed830ab01de1d9e73adee77e9eb5d24b0bf64bc5d"
    ),
    task4_policy_sha256=(
        "69b8160e0aa972b66d28c0404af1c1b0a745bc423b8a620e47d30e5e483ce276"
    ),
    task4_catalog_sha256=(
        "7d619d7853eb6165786c0e0aca4f50ed66f5b69dfbed134a81d789d9c6dbcb70"
    ),
    task4_catalog_byte_size=286_813_587,
    task4_catalog_row_count=575_503,
    expected_eligible_records=557_718,
    expected_eligible_residues=197_375_585,
    expected_eligible_groups=185_344,
    expected_reserved_family_universe=175,
    expected_eligible_reserved_groups=157,
    expected_eligible_reserved_records=1_965,
    expected_eligible_reserved_residues=700_093,
    local_assignment_relative_path=(
        "data/processed/week_01/task_06_group_aware_pre_repair_assignments.tsv"
    ),
    public_manifest_relative_path=(
        "manifests/week_01/task_06_group_aware_pre_repair.tsv"
    ),
)


def load_group_split_policy(path: Path) -> GroupSplitPolicy:
    """Load the exact committed Task 6 policy bytes."""

    try:
        content = Path(path).read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise GroupSplitError(f"could not load group split policy: {error}") from error

    calculated_sha256 = hashlib.sha256(content).hexdigest()
    if calculated_sha256 != APPROVED_GROUP_SPLIT_CONFIG_SHA256:
        raise GroupSplitError(
            "group split policy bytes do not match the approved checksum"
        )
    if set(raw) != {field.name for field in fields(GroupSplitPolicy)}:
        raise GroupSplitError(
            "group split policy fields differ from the approved schema"
        )

    try:
        policy = GroupSplitPolicy(
            **{
                field.name: _typed_policy_value(raw, field.name, field.type)
                for field in fields(GroupSplitPolicy)
            }
        )
    except (TypeError, ValueError) as error:
        raise GroupSplitError(f"invalid group split policy: {error}") from error
    if policy != APPROVED_GROUP_SPLIT_POLICY:
        raise GroupSplitError(
            "group split policy differs from the approved Week 1 decisions"
        )
    return policy


def _typed_policy_value(
    raw: dict[str, object],
    key: str,
    annotation: object,
) -> object:
    value = raw.get(key)
    expected_type = {
        "int": int,
        "str": str,
        "bool": bool,
    }.get(str(annotation))
    if expected_type is None:
        raise TypeError(f"unsupported policy field type for {key}")
    if expected_type is int and isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    if not isinstance(value, expected_type):
        raise TypeError(f"{key} must be {expected_type.__name__}")
    if expected_type is str and not value:
        raise ValueError(f"{key} must not be empty")
    return value
