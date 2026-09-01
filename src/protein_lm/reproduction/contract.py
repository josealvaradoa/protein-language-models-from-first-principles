"""Strict loading and read-only verification for the frozen Week 4 contract.

This module deliberately has no execution, Git, network, or storage behavior.
It accepts only the reviewed contract bytes and can verify the files pinned by
that contract beneath a caller-provided project root.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from protein_lm.reproduction.comparison import (
    CrossEntropyTolerances,
    MetricKey,
    MetricRecord,
)


APPROVED_FOUNDATIONS_CONTRACT_SHA256 = (
    "279b6cff2c7e9b022d850d763e575b9146149dad2046a88faf350c6405964aa6"
)
_SCHEMA_VERSION = 1
_SCHEMA_IDENTIFIER = "protein_lm.foundations_reproduction_contract"
_CONTRACT_IDENTIFIER = "2026-09-01-week-04-foundations-reproduction-v1"
_SCOPE = "closed_week_01_to_week_03_foundations_reproduction"
_STATUS = "frozen_contract_not_yet_executable"
_REEVALUATION_TOLERANCE = 0.000001
_RETRAINING_TOLERANCE = 0.0001
_MATERIAL_CROSS_ENTROPY_GAP = 0.001
_RUN_BUNDLE_ROOT_RELATIVE_PATH = "runs/week_04"
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "schema_identifier",
        "contract_identifier",
        "scope",
        "status",
        "evidence_scope",
        "tolerances",
        "identity",
        "operator_boundary",
        "run_bundle",
        "week_01_identity",
        "week_02_identity",
        "week_03_identity",
        "comparison_scope",
        "source_pins",
        "stages",
        "models",
        "metric_targets",
        "comparison_claims",
        "exclusions",
    }
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_RUN_BUNDLE_FILES = (
    "contract.toml",
    "run.json",
    "log.txt",
    "metrics.json",
    "comparison.json",
    "provenance.json",
)
_TERMINAL_STATES = ("completed", "failed", "cancelled", "runner_restarted")
_EXPECTED_WEEK2_KEYS = frozenset(
    MetricKey(model_id, collection)
    for model_id, collection in (
        ("week2_random_unigram", "random_native_validation"),
        ("week2_random_count_bigram", "random_native_validation"),
        ("week2_random_neural_bigram", "random_native_validation"),
        ("week2_family_aware_unigram", "family_aware_native_validation"),
        ("week2_family_aware_count_bigram", "family_aware_native_validation"),
        ("week2_family_aware_neural_bigram", "family_aware_native_validation"),
        ("week2_random_unigram", "shared_validation"),
        ("week2_random_count_bigram", "shared_validation"),
        ("week2_random_neural_bigram", "shared_validation"),
        ("week2_family_aware_unigram", "shared_validation"),
        ("week2_family_aware_count_bigram", "shared_validation"),
        ("week2_family_aware_neural_bigram", "shared_validation"),
    )
)
_EXPECTED_WEEK3_KEYS = frozenset(
    MetricKey(model_id, "family_aware_native_validation", seed)
    for model_id in ("c10", "c20", "e64")
    for seed in (20260821, 20260822, 20260823)
)
_CLAIM_RULES = {
    "week1_group_aware_lower_detected_strong_overlap": (
        "group_aware_rate_lt_random_rate_for_validation_and_test"
    ),
    "week2_prospective_comparison_supported": (
        "random_neural_shared_ce_minus_random_neural_native_ce_gt_"
        "family_aware_neural_shared_ce_minus_family_aware_neural_native_ce"
    ),
    "c20_beats_week2_family_aware_neural_bigram": (
        "week2_family_aware_neural_bigram_ce_minus_c20_mean_ce_gte_material_gap"
    ),
    "c20_beats_c10": "c10_mean_ce_minus_c20_mean_ce_gte_material_gap",
    "c20_beats_e64": "e64_mean_ce_minus_c20_mean_ce_gte_material_gap",
}


class ReproductionContractError(ValueError):
    """Raised when the frozen contract or its verification boundary is invalid."""


@dataclass(frozen=True)
class SourcePin:
    """One source file and the SHA-256 digest it must retain."""

    kind: str
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class ComparisonClaim:
    """One frozen comparison assertion, retained without evaluating it."""

    claim_id: str
    rule: str
    minimum_cross_entropy_gap: float
    historical_random_neural_optimism_gap: float | None = None
    historical_family_aware_neural_optimism_gap: float | None = None


@dataclass(frozen=True)
class FoundationsContract:
    """The immutable subset of the Week 4 TOML needed by later stages."""

    schema_version: int
    schema_identifier: str
    contract_identifier: str
    contract_sha256: str
    scope: str
    status: str
    tolerances: CrossEntropyTolerances
    material_cross_entropy_gap: float
    comparison_operator: str
    nonfinite_or_missing_metric_policy: str
    unexpected_or_duplicate_metric_policy: str
    source_pins: tuple[SourcePin, ...]
    metric_targets: tuple[MetricRecord, ...]
    comparison_claims: tuple[ComparisonClaim, ...]
    run_bundle_root_relative_path: str
    required_run_bundle_files: tuple[str, ...]
    terminal_states: tuple[str, ...]
    run_bundle_ignored_by_git: bool
    completed_runs_immutable: bool
    retries_require_new_run_id: bool
    atomic_json_replace: bool


@dataclass(frozen=True)
class SourcePinVerification:
    """One deterministic source-pin verification outcome."""

    kind: str
    relative_path: str
    expected_sha256: str
    observed_sha256: str | None
    passed: bool
    issue_code: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "relative_path": self.relative_path,
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
            "passed": self.passed,
            "issue_code": self.issue_code,
        }


@dataclass(frozen=True)
class SourcePinVerificationReport:
    """JSON-safe, ordered results for every pin in one contract."""

    contract_identifier: str
    contract_sha256: str
    passed: bool
    results: tuple[SourcePinVerification, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_identifier": self.contract_identifier,
            "contract_sha256": self.contract_sha256,
            "passed": self.passed,
            "results": [result.to_dict() for result in self.results],
        }

    def to_json(self) -> str:
        """Return stable JSON suitable for a later evidence record."""

        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


def load_foundations_contract(path: Path | str) -> FoundationsContract:
    """Load only the exact approved contract bytes and validate their contents."""

    try:
        content = Path(path).read_bytes()
    except OSError as error:
        raise ReproductionContractError("could not read foundations contract") from error
    digest = hashlib.sha256(content).hexdigest()
    if digest != APPROVED_FOUNDATIONS_CONTRACT_SHA256:
        raise ReproductionContractError("foundations contract bytes do not match approval")
    try:
        raw = tomllib.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ReproductionContractError("could not parse foundations contract") from error
    if not isinstance(raw, dict):
        raise ReproductionContractError("foundations contract must be a TOML table")
    _require_exact_keys(raw, _TOP_LEVEL_KEYS, "foundations contract")
    _validate_identity(raw)
    tolerances, material_gap, operator, nonfinite_policy, duplicate_policy = (
        _parse_tolerances(raw["tolerances"])
    )
    source_pins = _parse_source_pins(raw["source_pins"])
    metric_targets = _parse_metric_targets(raw["metric_targets"])
    comparison_claims = _parse_comparison_claims(raw["comparison_claims"])
    run_bundle = _parse_run_bundle(raw["run_bundle"])
    return FoundationsContract(
        schema_version=_SCHEMA_VERSION,
        schema_identifier=_SCHEMA_IDENTIFIER,
        contract_identifier=_CONTRACT_IDENTIFIER,
        contract_sha256=digest,
        scope=_SCOPE,
        status=_STATUS,
        tolerances=tolerances,
        material_cross_entropy_gap=material_gap,
        comparison_operator=operator,
        nonfinite_or_missing_metric_policy=nonfinite_policy,
        unexpected_or_duplicate_metric_policy=duplicate_policy,
        source_pins=source_pins,
        metric_targets=metric_targets,
        comparison_claims=comparison_claims,
        run_bundle_root_relative_path=run_bundle[0],
        required_run_bundle_files=run_bundle[1],
        terminal_states=run_bundle[2],
        run_bundle_ignored_by_git=run_bundle[3],
        completed_runs_immutable=run_bundle[4],
        retries_require_new_run_id=run_bundle[5],
        atomic_json_replace=run_bundle[6],
    )


def verify_source_pins(
    project_root: Path | str, contract: FoundationsContract
) -> SourcePinVerificationReport:
    """Verify every contract pin below a real project root without writing files.

    Per-pin filesystem failures are retained as results. A malformed root or
    wrong contract object cannot produce a meaningful report and raises.
    """

    root = _real_project_root(project_root)
    _validate_contract_for_verification(contract)
    results = tuple(_verify_source_pin(root, pin) for pin in contract.source_pins)
    return SourcePinVerificationReport(
        contract_identifier=contract.contract_identifier,
        contract_sha256=contract.contract_sha256,
        passed=all(result.passed for result in results),
        results=results,
    )


def _validate_identity(raw: dict[str, object]) -> None:
    expected = {
        "schema_version": _SCHEMA_VERSION,
        "schema_identifier": _SCHEMA_IDENTIFIER,
        "contract_identifier": _CONTRACT_IDENTIFIER,
        "scope": _SCOPE,
        "status": _STATUS,
    }
    for key, value in expected.items():
        if raw[key] != value or type(raw[key]) is not type(value):
            raise ReproductionContractError(f"foundations contract {key} is not approved")


def _parse_tolerances(
    value: object,
) -> tuple[CrossEntropyTolerances, float, str, str, str]:
    table = _table(value, "tolerances")
    _require_exact_keys(
        table,
        {
            "reevaluation_cross_entropy_absolute",
            "retraining_cross_entropy_absolute",
            "material_cross_entropy_gap",
            "comparison_operator",
            "nonfinite_or_missing_metric",
            "unexpected_or_duplicate_metric",
        },
        "tolerances",
    )
    reevaluation = _finite_nonnegative_float(
        table["reevaluation_cross_entropy_absolute"],
        "tolerances.reevaluation_cross_entropy_absolute",
    )
    retraining = _finite_nonnegative_float(
        table["retraining_cross_entropy_absolute"],
        "tolerances.retraining_cross_entropy_absolute",
    )
    material_gap = _finite_nonnegative_float(
        table["material_cross_entropy_gap"], "tolerances.material_cross_entropy_gap"
    )
    if (
        reevaluation != _REEVALUATION_TOLERANCE
        or retraining != _RETRAINING_TOLERANCE
        or material_gap != _MATERIAL_CROSS_ENTROPY_GAP
    ):
        raise ReproductionContractError("cross-entropy tolerances are not approved")
    operator = _nonempty_string(table["comparison_operator"], "comparison operator")
    nonfinite_policy = _nonempty_string(
        table["nonfinite_or_missing_metric"], "nonfinite metric policy"
    )
    duplicate_policy = _nonempty_string(
        table["unexpected_or_duplicate_metric"], "duplicate metric policy"
    )
    if operator != "absolute_delta_lte_tolerance":
        raise ReproductionContractError("comparison operator is not approved")
    if nonfinite_policy != "fail" or duplicate_policy != "fail":
        raise ReproductionContractError("metric failure policies are not approved")
    return (
        CrossEntropyTolerances(reevaluation=reevaluation, retraining=retraining),
        material_gap,
        operator,
        nonfinite_policy,
        duplicate_policy,
    )


def _parse_source_pins(value: object) -> tuple[SourcePin, ...]:
    if not isinstance(value, list) or len(value) != 15:
        raise ReproductionContractError("source_pins must contain exactly 15 entries")
    pins: list[SourcePin] = []
    paths: set[str] = set()
    for index, row in enumerate(value):
        table = _table(row, f"source_pins[{index}]")
        _require_exact_keys(table, {"kind", "relative_path", "sha256"}, "source pin")
        kind = _nonempty_string(table["kind"], "source pin kind")
        relative_path = _safe_relative_path(table["relative_path"], "source pin path")
        digest = _sha256(table["sha256"], "source pin sha256")
        if relative_path in paths:
            raise ReproductionContractError("source pin paths must be unique")
        paths.add(relative_path)
        pins.append(SourcePin(kind, relative_path, digest))
    return tuple(pins)


def _parse_metric_targets(value: object) -> tuple[MetricRecord, ...]:
    if not isinstance(value, list) or len(value) != 21:
        raise ReproductionContractError("metric_targets must contain exactly 21 entries")
    targets: list[MetricRecord] = []
    keys: set[MetricKey] = set()
    for index, row in enumerate(value):
        table = _table(row, f"metric_targets[{index}]")
        allowed = {
            "model_id",
            "collection",
            "cross_entropy",
            "correct_predictions",
            "token_count",
        }
        if "seed" in table:
            allowed.add("seed")
        _require_exact_keys(table, allowed, "metric target")
        model_id = _nonempty_string(table["model_id"], "metric model_id")
        collection = _nonempty_string(table["collection"], "metric collection")
        seed = table.get("seed")
        if seed is not None and type(seed) is not int:
            raise ReproductionContractError("metric seed must be an integer")
        cross_entropy = _finite_nonnegative_float(
            table["cross_entropy"], "metric cross_entropy"
        )
        correct_predictions = _nonnegative_int(
            table["correct_predictions"], "metric correct_predictions"
        )
        token_count = _positive_int(table["token_count"], "metric token_count")
        if correct_predictions > token_count:
            raise ReproductionContractError("metric correct_predictions exceeds token_count")
        key = MetricKey(model_id, collection, seed)
        if key in keys:
            raise ReproductionContractError("metric target keys must be unique")
        keys.add(key)
        targets.append(
            MetricRecord(key, cross_entropy, correct_predictions, token_count)
        )
    if keys != _EXPECTED_WEEK2_KEYS | _EXPECTED_WEEK3_KEYS:
        raise ReproductionContractError("metric target inventory is not approved")
    if sum(key.seed is None for key in keys) != 12 or sum(key.seed is not None for key in keys) != 9:
        raise ReproductionContractError("metric target seed identities are not approved")
    return tuple(targets)


def _parse_comparison_claims(value: object) -> tuple[ComparisonClaim, ...]:
    if not isinstance(value, list) or len(value) != len(_CLAIM_RULES):
        raise ReproductionContractError("comparison_claims must contain exactly five entries")
    claims: list[ComparisonClaim] = []
    claim_ids: set[str] = set()
    historical_id = "week2_prospective_comparison_supported"
    for index, row in enumerate(value):
        table = _table(row, f"comparison_claims[{index}]")
        claim_id = _nonempty_string(table.get("claim_id"), "comparison claim_id")
        required_keys = {"claim_id", "rule", "minimum_cross_entropy_gap"}
        if claim_id == historical_id:
            required_keys |= {
                "historical_random_neural_optimism_gap",
                "historical_family_aware_neural_optimism_gap",
            }
        _require_exact_keys(table, required_keys, "comparison claim")
        rule = _nonempty_string(table["rule"], "comparison claim rule")
        if _CLAIM_RULES.get(claim_id) != rule:
            raise ReproductionContractError("comparison claim rule is not approved")
        minimum_gap = _finite_nonnegative_float(
            table["minimum_cross_entropy_gap"], "comparison claim minimum gap"
        )
        historical_random = None
        historical_family = None
        if claim_id == historical_id:
            historical_random = _finite_nonnegative_float(
                table["historical_random_neural_optimism_gap"],
                "historical random neural optimism gap",
            )
            historical_family = _finite_nonnegative_float(
                table["historical_family_aware_neural_optimism_gap"],
                "historical family-aware neural optimism gap",
            )
        if claim_id in claim_ids:
            raise ReproductionContractError("comparison claim IDs must be unique")
        claim_ids.add(claim_id)
        claims.append(
            ComparisonClaim(
                claim_id,
                rule,
                minimum_gap,
                historical_random,
                historical_family,
            )
        )
    if set(_CLAIM_RULES) != claim_ids:
        raise ReproductionContractError("comparison claim inventory is not approved")
    return tuple(claims)


def _parse_run_bundle(
    value: object,
) -> tuple[str, tuple[str, ...], tuple[str, ...], bool, bool, bool, bool]:
    table = _table(value, "run_bundle")
    _require_exact_keys(
        table,
        {
            "root_relative_path",
            "ignored_by_git",
            "completed_runs_immutable",
            "retries_require_new_run_id",
            "atomic_json_replace",
            "required_files",
            "terminal_states",
        },
        "run_bundle",
    )
    root_relative_path = _safe_relative_path(
        table["root_relative_path"], "run bundle root path"
    )
    if root_relative_path != _RUN_BUNDLE_ROOT_RELATIVE_PATH:
        raise ReproductionContractError("run bundle root path is not approved")
    required_files = _string_tuple(table["required_files"], "run bundle required files")
    terminal_states = _string_tuple(table["terminal_states"], "run bundle terminal states")
    if required_files != _REQUIRED_RUN_BUNDLE_FILES:
        raise ReproductionContractError("run bundle required files are not approved")
    if terminal_states != _TERMINAL_STATES:
        raise ReproductionContractError("run bundle terminal states are not approved")
    flags = tuple(
        _true_bool(table[name], f"run bundle {name}")
        for name in (
            "ignored_by_git",
            "completed_runs_immutable",
            "retries_require_new_run_id",
            "atomic_json_replace",
        )
    )
    return (root_relative_path, required_files, terminal_states, *flags)


def _real_project_root(value: Path | str) -> Path:
    path = Path(value)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReproductionContractError("project root must be an existing real directory") from error
    if not resolved.is_dir() or path.absolute() != resolved:
        raise ReproductionContractError("project root must be a real directory, not a symlink")
    return resolved


def _validate_contract_for_verification(contract: FoundationsContract) -> None:
    if not isinstance(contract, FoundationsContract):
        raise ReproductionContractError("source-pin verification requires FoundationsContract")
    if (
        contract.schema_version != _SCHEMA_VERSION
        or contract.schema_identifier != _SCHEMA_IDENTIFIER
        or contract.contract_identifier != _CONTRACT_IDENTIFIER
        or contract.contract_sha256 != APPROVED_FOUNDATIONS_CONTRACT_SHA256
        or contract.scope != _SCOPE
        or contract.status != _STATUS
    ):
        raise ReproductionContractError("source-pin verification contract is invalid")
    if (
        not isinstance(contract.tolerances, CrossEntropyTolerances)
        or type(contract.tolerances.reevaluation) is not float
        or type(contract.tolerances.retraining) is not float
        or not math.isfinite(contract.tolerances.reevaluation)
        or not math.isfinite(contract.tolerances.retraining)
        or contract.tolerances.reevaluation != _REEVALUATION_TOLERANCE
        or contract.tolerances.retraining != _RETRAINING_TOLERANCE
        or type(contract.material_cross_entropy_gap) is not float
        or not math.isfinite(contract.material_cross_entropy_gap)
        or contract.material_cross_entropy_gap != _MATERIAL_CROSS_ENTROPY_GAP
        or contract.comparison_operator != "absolute_delta_lte_tolerance"
        or contract.nonfinite_or_missing_metric_policy != "fail"
        or contract.unexpected_or_duplicate_metric_policy != "fail"
        or not isinstance(contract.metric_targets, tuple)
        or not isinstance(contract.comparison_claims, tuple)
        or contract.run_bundle_root_relative_path != _RUN_BUNDLE_ROOT_RELATIVE_PATH
        or contract.required_run_bundle_files != _REQUIRED_RUN_BUNDLE_FILES
        or contract.terminal_states != _TERMINAL_STATES
        or contract.run_bundle_ignored_by_git is not True
        or contract.completed_runs_immutable is not True
        or contract.retries_require_new_run_id is not True
        or contract.atomic_json_replace is not True
    ):
        raise ReproductionContractError("source-pin verification contract is invalid")
    _validate_source_pins_for_verification(contract.source_pins)


def _validate_source_pins_for_verification(pins: object) -> None:
    if not isinstance(pins, tuple) or len(pins) != 15:
        raise ReproductionContractError("source-pin verification contract is invalid")
    paths: set[str] = set()
    for pin in pins:
        if not isinstance(pin, SourcePin):
            raise ReproductionContractError("source-pin verification contract is invalid")
        _nonempty_string(pin.kind, "source pin kind")
        relative_path = _safe_relative_path(pin.relative_path, "source pin path")
        _sha256(pin.sha256, "source pin sha256")
        if relative_path in paths:
            raise ReproductionContractError("source-pin verification contract is invalid")
        paths.add(relative_path)


def _verify_source_pin(root: Path, pin: object) -> SourcePinVerification:
    if not isinstance(pin, SourcePin):
        return SourcePinVerification("", "", "", None, False, "INVALID_PIN")
    if not isinstance(pin.kind, str) or not pin.kind.strip():
        return _failed_pin(pin, "INVALID_PIN")
    if not isinstance(pin.sha256, str) or not _SHA256_PATTERN.fullmatch(pin.sha256):
        return _failed_pin(pin, "INVALID_PIN")
    try:
        relative_path = _safe_relative_path(pin.relative_path, "source pin path")
    except ReproductionContractError:
        return _failed_pin(pin, "UNSAFE_PATH")
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    ancestor = root
    for part in PurePosixPath(relative_path).parts[:-1]:
        ancestor = ancestor / part
        if ancestor.is_symlink():
            return _failed_pin(pin, "SYMLINK_PATH")
    if candidate.is_symlink():
        return _failed_pin(pin, "SYMLINK_PATH")
    if not candidate.exists():
        return _failed_pin(pin, "MISSING_FILE")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return _failed_pin(pin, "UNREADABLE_FILE")
    if not _is_within(resolved, root):
        return _failed_pin(pin, "ESCAPES_PROJECT_ROOT")
    if not resolved.is_file():
        return _failed_pin(pin, "NOT_REGULAR_FILE")
    try:
        observed_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    except OSError:
        return _failed_pin(pin, "UNREADABLE_FILE")
    if observed_sha256 != pin.sha256:
        return SourcePinVerification(
            pin.kind,
            relative_path,
            pin.sha256,
            observed_sha256,
            False,
            "SHA256_MISMATCH",
        )
    return SourcePinVerification(
        pin.kind, relative_path, pin.sha256, observed_sha256, True, None
    )


def _failed_pin(pin: SourcePin, issue_code: str) -> SourcePinVerification:
    return SourcePinVerification(
        pin.kind if isinstance(pin.kind, str) else "",
        pin.relative_path if isinstance(pin.relative_path, str) else "",
        pin.sha256 if isinstance(pin.sha256, str) else "",
        None,
        False,
        issue_code,
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _table(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ReproductionContractError(f"{label} must be a TOML table")
    return value


def _require_exact_keys(
    table: dict[str, object], expected: set[str] | frozenset[str], label: str
) -> None:
    if set(table) != set(expected):
        raise ReproductionContractError(f"{label} keys differ from its frozen schema")


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReproductionContractError(f"{label} must be a nonempty string")
    return value


def _safe_relative_path(value: object, label: str) -> str:
    text = _nonempty_string(value, label)
    if "\\" in text:
        raise ReproductionContractError(f"{label} must use normalized POSIX separators")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ReproductionContractError(f"{label} must be a safe relative path")
    if path.as_posix() != text:
        raise ReproductionContractError(f"{label} must be normalized")
    return text


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ReproductionContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite_nonnegative_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0:
        raise ReproductionContractError(f"{label} must be a finite nonnegative float")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ReproductionContractError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReproductionContractError(f"{label} must be a positive integer")
    return value


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ReproductionContractError(f"{label} must be an array of strings")
    return tuple(value)


def _true_bool(value: object, label: str) -> bool:
    if value is not True:
        raise ReproductionContractError(f"{label} must be true")
    return True
