from __future__ import annotations

import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from protein_lm.data.acquisition import (
    AcquisitionValidationError,
    load_acquisition_contract,
    prove_heavy_paths_are_ignored,
    validate_acquisition_contract,
    validate_release_metadata,
    verify_local_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "experiments" / "week_01" / "acquisition.toml"
RELEASE_FIXTURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "week_01" / "reldate_2026_02.txt"
)


@pytest.fixture
def contract():
    return load_acquisition_contract(CONFIG_PATH)


def test_committed_contract_pins_approved_sources_and_ignored_paths(
    contract,
) -> None:
    assert contract.release_id == "2026_02"
    assert contract.release_date.isoformat() == "2026-06-10"
    assert contract.release_metadata_url == (
        "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
        "knowledgebase/complete/reldate.txt"
    )
    assert contract.license_spdx == "CC-BY-4.0"

    sources = {source.role: source for source in contract.sources}
    assert set(sources) == {"swiss_prot_records", "uniref50_membership"}
    assert sources["swiss_prot_records"].filename == "uniprot_sprot.dat.gz"
    assert sources["swiss_prot_records"].url == (
        "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
        "knowledgebase/complete/uniprot_sprot.dat.gz"
    )
    assert sources["swiss_prot_records"].expected_bytes == 699_031_150
    assert (
        sources["swiss_prot_records"].published_md5
        == "868b301a6ec93955f4e4355d579d8683"
    )
    assert sources["uniref50_membership"].filename == "idmapping_selected.tab.gz"
    assert sources["uniref50_membership"].url == (
        "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
        "knowledgebase/idmapping/idmapping_selected.tab.gz"
    )
    assert sources["uniref50_membership"].expected_bytes == 7_066_467_385
    assert (
        sources["uniref50_membership"].published_md5
        == "f426a0ee61882f4c86f1b0d616ae53ec"
    )
    assert sources["uniref50_membership"].uniref50_column == 10
    assert sum(source.expected_bytes for source in contract.sources) == (7_765_498_535)

    ignored_paths = prove_heavy_paths_are_ignored(contract, PROJECT_ROOT)
    assert len(ignored_paths) == 2


def test_release_metadata_fixture_matches_frozen_release(contract) -> None:
    metadata = validate_release_metadata(
        RELEASE_FIXTURE.read_text(encoding="utf-8"),
        contract,
    )

    assert metadata.release_id == "2026_02"
    assert metadata.release_date.isoformat() == "2026-06-10"


def test_release_metadata_rejects_current_release_drift(contract) -> None:
    changed_metadata = RELEASE_FIXTURE.read_text(encoding="utf-8").replace(
        "2026_02", "2026_03"
    )

    with pytest.raises(AcquisitionValidationError, match="upstream release drift"):
        validate_release_metadata(changed_metadata, contract)


def test_release_metadata_rejects_date_drift(contract) -> None:
    changed_metadata = RELEASE_FIXTURE.read_text(encoding="utf-8").replace(
        "10-Jun-2026", "11-Jun-2026"
    )

    with pytest.raises(AcquisitionValidationError, match="upstream release date drift"):
        validate_release_metadata(changed_metadata, contract)


def test_release_metadata_rejects_disagreeing_products(contract) -> None:
    changed_metadata = RELEASE_FIXTURE.read_text(encoding="utf-8").replace(
        "UniProtKB/Swiss-Prot Release 2026_02",
        "UniProtKB/Swiss-Prot Release 2026_03",
    )

    with pytest.raises(
        AcquisitionValidationError,
        match="release identifiers disagree",
    ):
        validate_release_metadata(changed_metadata, contract)


def test_contract_rejects_malformed_or_traversing_urls(contract) -> None:
    malformed = replace(contract, release_metadata_url="https://[")
    with pytest.raises(AcquisitionValidationError, match="malformed"):
        validate_acquisition_contract(malformed)

    changed_source = replace(
        contract.sources[0],
        url=(
            "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
            "../complete/uniprot_sprot.dat.gz"
        ),
    )
    traversing = replace(
        contract,
        sources=(changed_source, contract.sources[1]),
    )
    with pytest.raises(AcquisitionValidationError, match="dot path segments"):
        validate_acquisition_contract(traversing)


def test_git_boundary_rejects_force_added_raw_file(contract, tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=repository,
        check=True,
    )
    (repository / ".gitignore").write_text("/data/\n", encoding="utf-8")

    relative_path = contract.local_path_for(contract.sources[0])
    tracked_path = repository / relative_path
    tracked_path.parent.mkdir(parents=True)
    tracked_path.write_bytes(b"synthetic raw file")
    subprocess.run(
        ["git", "add", "--force", "--", relative_path.as_posix()],
        cwd=repository,
        check=True,
    )

    with pytest.raises(AcquisitionValidationError, match="heavy path is not ignored"):
        prove_heavy_paths_are_ignored(contract, repository)


def test_local_fixture_verification_returns_sha256(contract, tmp_path: Path) -> None:
    payload = b"synthetic protein-source fixture\n"
    fixture_path = tmp_path / "fixture.dat.gz"
    fixture_path.write_bytes(payload)
    source = replace(
        contract.sources[0],
        filename=fixture_path.name,
        expected_bytes=len(payload),
        published_md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    )

    result = verify_local_file(fixture_path, source, chunk_bytes=7)

    assert result.byte_size == len(payload)
    assert result.md5 == source.published_md5
    assert result.sha256 == hashlib.sha256(payload).hexdigest()


def test_local_fixture_verification_rejects_same_size_corruption(
    contract, tmp_path: Path
) -> None:
    expected_payload = b"AAAA"
    fixture_path = tmp_path / "fixture.dat.gz"
    fixture_path.write_bytes(b"AAAB")
    source = replace(
        contract.sources[0],
        filename=fixture_path.name,
        expected_bytes=len(expected_payload),
        published_md5=hashlib.md5(expected_payload, usedforsecurity=False).hexdigest(),
    )

    with pytest.raises(AcquisitionValidationError, match="MD5 mismatch"):
        verify_local_file(fixture_path, source)


def test_local_fixture_verification_rejects_wrong_size(
    contract, tmp_path: Path
) -> None:
    fixture_path = tmp_path / "fixture.dat.gz"
    fixture_path.write_bytes(b"too short")
    source = replace(
        contract.sources[0],
        filename=fixture_path.name,
        expected_bytes=10,
    )

    with pytest.raises(AcquisitionValidationError, match="byte-size mismatch"):
        verify_local_file(fixture_path, source)
