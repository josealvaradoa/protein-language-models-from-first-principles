"""Load the frozen Week 1 Task 5 random-split policy."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

TASK5_SCHEMA_VERSION = 1
TASK5_SCOPE = "week_01_task_05_random_diagnostic"
APPROVED_RANDOM_SPLIT_CONFIG_SHA256 = (
    "2d807fe74465d75937a133feff8aab01aa77faa0b82bd4c7a3b3038d954a67e7"
)
PARTITIONS = ("training", "validation", "test")


class RandomSplitError(ValueError):
    """Raised when Task 5 cannot produce approved diagnostic evidence."""


@dataclass(frozen=True)
class RandomSplitPolicy:
    """Every frozen input and algorithm choice used by Task 5."""

    schema_version: int
    scope: str
    strategy: str
    stage: str
    seed: int
    assignment_namespace: str
    hash_algorithm: str
    target_denominator: int
    training_target_numerator: int
    validation_target_numerator: int
    test_target_numerator: int
    assignment_unit: str
    manifest_order: str
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
    local_assignment_relative_path: str
    public_manifest_relative_path: str


APPROVED_RANDOM_SPLIT_POLICY = RandomSplitPolicy(
    schema_version=TASK5_SCHEMA_VERSION,
    scope=TASK5_SCOPE,
    strategy="random",
    stage="diagnostic",
    seed=20260727,
    assignment_namespace="week1-random-v1",
    hash_algorithm="sha256",
    target_denominator=20,
    training_target_numerator=18,
    validation_target_numerator=1,
    test_target_numerator=1,
    assignment_unit="primary_accession",
    manifest_order="primary_accession",
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
    local_assignment_relative_path=(
        "data/processed/week_01/task_05_random_diagnostic_assignments.tsv"
    ),
    public_manifest_relative_path=("manifests/week_01/task_05_random_diagnostic.tsv"),
)


def load_random_split_policy(path: Path) -> RandomSplitPolicy:
    """Load the exact committed Task 5 policy bytes."""

    try:
        content = Path(path).read_bytes()
        raw = tomllib.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RandomSplitError(
            f"could not load random split policy: {error}"
        ) from error

    calculated_sha256 = hashlib.sha256(content).hexdigest()
    if calculated_sha256 != APPROVED_RANDOM_SPLIT_CONFIG_SHA256:
        raise RandomSplitError(
            "random split policy bytes do not match the approved checksum"
        )
    if set(raw) != {field.name for field in fields(RandomSplitPolicy)}:
        raise RandomSplitError(
            "random split policy fields differ from the approved schema"
        )

    try:
        policy = RandomSplitPolicy(
            **{
                field.name: _typed_policy_value(raw, field.name, field.type)
                for field in fields(RandomSplitPolicy)
            }
        )
    except (TypeError, ValueError) as error:
        raise RandomSplitError(f"invalid random split policy: {error}") from error
    if policy != APPROVED_RANDOM_SPLIT_POLICY:
        raise RandomSplitError(
            "random split policy differs from the approved Week 1 decisions"
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
