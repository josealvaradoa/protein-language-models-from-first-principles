"""Operator-gated, byte-preserving promotion of the approved Week 2 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from protein_lm.data.model_data.config import config_sha256
from protein_lm.data.model_data.contracts import ModelDataConfig, ModelDataError
from protein_lm.data.model_data.manifests import MEMBERSHIP_HEADER
from protein_lm.data.model_data.workflow import _require_clean_committed_revision


@dataclass(frozen=True)
class PromotedArtifact:
    """The frozen identity of one public manifest."""

    filename: str
    row_count: int
    byte_size: int
    sha256: str


@dataclass(frozen=True)
class PromotionContract:
    """All immutable identities required before public promotion."""

    identifier: str
    candidate_revision: str
    readiness_json_sha256: str
    artifacts: tuple[PromotedArtifact, ...]
    sealed_rows: int
    sealed_bytes: int
    sealed_sha256: str


PROMOTION_CONTRACT = PromotionContract(
    identifier="2026-08-12-week-02-model-data-promotion-v1",
    candidate_revision="e30c00b34bb07c77cd6709766b6bcf7be638ebfa",
    readiness_json_sha256=(
        "19d4ee82eae49b600e9e83e4bb19d468b7a9fc2cfd6b78ffebc995f77db9b881"
    ),
    artifacts=(
        PromotedArtifact(
            "shared_validation.tsv",
            2926,
            345572,
            "8ab7d459010b00ce87f038a263da36fc5033f274145f7ba06ee736919f378173",
        ),
        PromotedArtifact(
            "random_arm.tsv",
            463777,
            50650053,
            "98904d1d4c04151c3322e44fa8f2d773023b85e013a5a4246330f1e7ff7a1b5a",
        ),
        PromotedArtifact(
            "family_aware_arm.tsv",
            463777,
            50649387,
            "e8c307db46b3c603c4b451e7b549e619581828eae95dcf8bba9a43bb18938e50",
        ),
    ),
    sealed_rows=2609,
    sealed_bytes=311052,
    sealed_sha256="799690a3cd2ba416b648e1d3e70b8bbaa2ce8162dddc37ea45ff9533bd7fd8a4",
)

_CANDIDATE_FILES = {
    "shared_validation.tsv",
    "shared_sealed_test.tsv",
    "random_arm.tsv",
    "family_aware_arm.tsv",
    "deduplication_aliases.tsv",
    "artifact_checksums.json",
    "preparation_record.json",
}
_READINESS_GATES = {
    "artifact_and_provenance_checksums",
    "complete_per_arm_population_accounting",
    "deterministic_regeneration",
    "evaluation_bucket_bounds",
    "evaluation_token_bounds",
    "family_aware_groups_do_not_cross",
    "no_accession_or_hash_crossings_within_arm",
    "reserved_members_are_absent",
    "shared_and_sealed_groups_are_isolated",
    "source_sequence_contract",
}


def preflight_promotion(
    root: Path,
    config: ModelDataConfig,
    *,
    contract: PromotionContract = PROMOTION_CONTRACT,
) -> dict[str, object]:
    """Prove the passing candidate may be promoted without creating output.

    The sealed candidate TSV is intentionally never opened. Its aggregate identity is
    read from the public readiness evidence and candidate checksum record only.
    """

    candidate = root / config.candidate_directory_relative_path
    _verify_candidate_inventory(candidate)
    checksums = _load_candidate_checksums(candidate)
    _verify_candidate_record(candidate, config, checksums, root, contract)
    _verify_readiness(root, config, contract)
    for artifact in contract.artifacts:
        _verify_file(candidate / artifact.filename, artifact)
    _verify_sealed_commitment(checksums, contract)
    return {
        "destination": "manifests/week_02",
        "candidate_revision": contract.candidate_revision,
        "promoted_artifacts": tuple(item.filename for item in contract.artifacts),
        "sealed_membership": "not read",
        "network_requests_made": 0,
    }


def promote(
    root: Path,
    config: ModelDataConfig,
    *,
    contract: PromotionContract = PROMOTION_CONTRACT,
) -> Path:
    """Install the three approved manifests and registry as one atomic directory move."""

    _require_clean_committed_revision(root)
    plan = preflight_promotion(root, config, contract=contract)
    destination = root / str(plan["destination"])
    if destination.exists():
        raise ModelDataError("public Week 2 manifest destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidate = root / config.candidate_directory_relative_path
    with tempfile.TemporaryDirectory(
        prefix=".week_02-promotion-", dir=destination.parent
    ) as temporary:
        stage = Path(temporary) / destination.name
        stage.mkdir()
        for artifact in contract.artifacts:
            source = candidate / artifact.filename
            copied = stage / artifact.filename
            shutil.copyfile(source, copied)
            _verify_file(copied, artifact)
        (stage / "model_data_v1.json").write_text(
            json.dumps(
                registry_payload(config, contract), indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
    return destination


def registry_payload(
    config: ModelDataConfig,
    contract: PromotionContract = PROMOTION_CONTRACT,
) -> dict[str, object]:
    """Return the complete public registry without any sealed membership identifier."""

    return {
        "schema_version": 1,
        "scope": "week_02_model_data",
        "contract_identifier": contract.identifier,
        "candidate_identifier": config.candidate_identifier,
        "candidate_revision": contract.candidate_revision,
        "readiness": {
            "relative_path": config.readiness_json_relative_path,
            "sha256": contract.readiness_json_sha256,
        },
        "task4_catalog": {
            "relative_path": config.task4_catalog_relative_path,
            "sha256": config.task4_catalog_sha256,
            "byte_size": config.task4_catalog_byte_size,
            "row_count": config.task4_catalog_row_count,
        },
        "membership_header": list(MEMBERSHIP_HEADER),
        "artifacts": [asdict(item) for item in contract.artifacts],
        "sealed_test_commitment": {
            "row_count": contract.sealed_rows,
            "byte_size": contract.sealed_bytes,
            "sha256": contract.sealed_sha256,
        },
    }


def _verify_candidate_inventory(candidate: Path) -> None:
    try:
        names = {path.name for path in candidate.iterdir()}
    except OSError as error:
        raise ModelDataError("candidate directory cannot be inspected") from error
    if names != _CANDIDATE_FILES:
        raise ModelDataError("candidate artifact inventory drifted")


def _load_candidate_checksums(candidate: Path) -> dict[str, object]:
    try:
        value = json.loads(
            (candidate / "artifact_checksums.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError("candidate checksums are malformed") from error
    if not isinstance(value, dict) or set(value) != {
        "shared_validation.tsv",
        "shared_sealed_test.tsv",
        "random_arm.tsv",
        "family_aware_arm.tsv",
        "deduplication_aliases.tsv",
    }:
        raise ModelDataError("candidate checksums do not have the approved inventory")
    return value


def _verify_candidate_record(
    candidate: Path,
    config: ModelDataConfig,
    checksums: dict[str, object],
    root: Path,
    contract: PromotionContract,
) -> None:
    try:
        record = json.loads(
            (candidate / "preparation_record.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelDataError("candidate preparation record is malformed") from error
    if not isinstance(record, dict) or record.get("code_revision") != contract.candidate_revision:
        raise ModelDataError("candidate preparation revision is not approved")
    if record.get("candidate_identifier") != config.candidate_identifier:
        raise ModelDataError("candidate preparation identifier is not approved")
    if record.get("config_sha256") != config_sha256(
        root / "experiments/week_02/model_data_readiness.toml"
    ):
        raise ModelDataError("candidate preparation config identity drifted")
    if record.get("artifact_evidence") != checksums:
        raise ModelDataError("candidate preparation evidence disagrees with checksums")


def _verify_readiness(
    root: Path, config: ModelDataConfig, contract: PromotionContract
) -> None:
    json_path = root / config.readiness_json_relative_path
    sha_path = root / config.readiness_sha256_relative_path
    try:
        content = json_path.read_bytes()
        sidecar = sha_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ModelDataError("readiness evidence is malformed") from error
    digest = hashlib.sha256(content).hexdigest()
    if digest != contract.readiness_json_sha256:
        raise ModelDataError("readiness JSON identity is not approved")
    if sidecar != f"{digest}  {json_path.name}\n":
        raise ModelDataError("readiness checksum sidecar disagrees")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ModelDataError("readiness evidence is malformed") from error
    if not isinstance(payload, dict) or (
        payload.get("candidate_status") != "passed"
        or payload.get("scope") != "week_02_model_data_readiness"
        or payload.get("schema_version") != 1
        or payload.get("hard_gates") != {gate: True for gate in _READINESS_GATES}
    ):
        raise ModelDataError("readiness evidence does not prove every hard gate passed")


def _verify_sealed_commitment(
    checksums: dict[str, object], contract: PromotionContract
) -> None:
    sealed = checksums.get("shared_sealed_test.tsv")
    expected = {
        "row_count": contract.sealed_rows,
        "byte_size": contract.sealed_bytes,
        "sha256": contract.sealed_sha256,
    }
    if sealed != expected:
        raise ModelDataError("sealed aggregate commitment disagrees with approval")


def _verify_file(path: Path, expected: PromotedArtifact) -> None:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ModelDataError(f"candidate artifact is missing: {expected.filename}") from error
    actual = PromotedArtifact(
        expected.filename,
        len(content.splitlines()) - 1,
        len(content),
        hashlib.sha256(content).hexdigest(),
    )
    if actual != expected:
        raise ModelDataError(f"candidate artifact identity drifted: {expected.filename}")
