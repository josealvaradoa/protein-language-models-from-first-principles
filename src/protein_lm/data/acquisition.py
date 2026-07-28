"""Offline validation for the frozen Week 1 acquisition contract.

This module does not make network requests or download data. It validates the
committed source contract and verifies files supplied by a later acquisition
step.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import SplitResult, urlsplit

SCHEMA_VERSION = 1
REQUIRED_SOURCE_ROLES = frozenset({"swiss_prot_records", "uniref50_membership"})
OFFICIAL_HOST = "ftp.uniprot.org"
OFFICIAL_PATH_PREFIX = "/pub/databases/uniprot/current_release/"
_MD5_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_RELEASE_PATTERN = re.compile(r"^\d{4}_\d{2}$")
_KNOWLEDGEBASE_RELEASE_PATTERN = re.compile(
    r"^UniProt Knowledgebase Release (?P<release>\d{4}_\d{2})\b",
    re.MULTILINE,
)
_SWISS_PROT_RELEASE_PATTERN = re.compile(
    r"^UniProtKB/Swiss-Prot Release "
    r"(?P<release>\d{4}_\d{2}) of "
    r"(?P<release_date>\d{1,2}-[A-Za-z]{3}-\d{4})\s*$",
    re.MULTILINE,
)


class AcquisitionValidationError(ValueError):
    """Raised when acquisition evidence violates the frozen contract."""


@dataclass(frozen=True)
class SourceArtifact:
    """One upstream artifact required by the Week 1 experiment."""

    role: str
    description: str
    filename: str
    url: str
    expected_bytes: int
    published_md5: str
    uniref50_column: int | None = None


@dataclass(frozen=True)
class AcquisitionContract:
    """The release, source, storage, and verification choices frozen in TOML."""

    schema_version: int
    release_id: str
    release_date: date
    release_metadata_url: str
    license_spdx: str
    license_name: str
    license_url: str
    raw_directory: PurePosixPath
    published_checksum: str
    local_checksum: str
    sources: tuple[SourceArtifact, ...]

    def local_path_for(self, source: SourceArtifact) -> PurePosixPath:
        """Return the source path relative to the repository root."""

        return self.raw_directory / source.filename


@dataclass(frozen=True)
class ReleaseMetadata:
    """Release identity parsed from UniProt's release metadata."""

    release_id: str
    release_date: date


@dataclass(frozen=True)
class LocalFileVerification:
    """Checksums calculated from one locally acquired file."""

    path: Path
    byte_size: int
    md5: str
    sha256: str


def load_acquisition_contract(path: Path) -> AcquisitionContract:
    """Load and validate an acquisition contract from TOML."""

    try:
        with path.open("rb") as config_file:
            raw = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise AcquisitionValidationError(
            f"could not load acquisition config {path}: {error}"
        ) from error

    release = _required_table(raw, "release")
    license_data = _required_table(raw, "license")
    storage = _required_table(raw, "storage")
    verification = _required_table(raw, "verification")
    source_rows = raw.get("sources")
    if not isinstance(source_rows, list) or not source_rows:
        raise AcquisitionValidationError(
            "config field 'sources' must be a non-empty array of tables"
        )

    release_date_text = _required_string(release, "date", "release")
    try:
        release_date = date.fromisoformat(release_date_text)
    except ValueError as error:
        raise AcquisitionValidationError(
            "release.date must use ISO format YYYY-MM-DD"
        ) from error

    sources = tuple(
        _parse_source(source_row, index) for index, source_row in enumerate(source_rows)
    )
    contract = AcquisitionContract(
        schema_version=_required_integer(raw, "schema_version", "config"),
        release_id=_required_string(release, "id", "release"),
        release_date=release_date,
        release_metadata_url=_required_string(release, "metadata_url", "release"),
        license_spdx=_required_string(license_data, "spdx", "license"),
        license_name=_required_string(license_data, "name", "license"),
        license_url=_required_string(license_data, "url", "license"),
        raw_directory=PurePosixPath(
            _required_string(storage, "raw_directory", "storage")
        ),
        published_checksum=_required_string(
            verification, "published_checksum", "verification"
        ),
        local_checksum=_required_string(verification, "local_checksum", "verification"),
        sources=sources,
    )
    validate_acquisition_contract(contract)
    return contract


def validate_acquisition_contract(contract: AcquisitionContract) -> None:
    """Validate the internal consistency and safety of a parsed contract."""

    if contract.schema_version != SCHEMA_VERSION:
        raise AcquisitionValidationError(
            f"unsupported schema_version {contract.schema_version}; "
            f"expected {SCHEMA_VERSION}"
        )
    if not _RELEASE_PATTERN.fullmatch(contract.release_id):
        raise AcquisitionValidationError("release.id must use the form YYYY_NN")

    expected_raw_directory = (
        PurePosixPath("data") / "raw" / "uniprot" / contract.release_id
    )
    if contract.raw_directory != expected_raw_directory:
        raise AcquisitionValidationError(
            f"storage.raw_directory must be '{expected_raw_directory.as_posix()}'"
        )

    _validate_official_url(
        contract.release_metadata_url,
        expected_filename="reldate.txt",
        field="release.metadata_url",
    )
    _validate_https_url(contract.license_url, field="license.url")

    if contract.published_checksum != "md5":
        raise AcquisitionValidationError(
            "verification.published_checksum must be 'md5'"
        )
    if contract.local_checksum != "sha256":
        raise AcquisitionValidationError("verification.local_checksum must be 'sha256'")

    roles = [source.role for source in contract.sources]
    if len(roles) != len(set(roles)):
        raise AcquisitionValidationError("source roles must be unique")
    if set(roles) != REQUIRED_SOURCE_ROLES:
        raise AcquisitionValidationError(
            "sources must contain exactly these roles: "
            + ", ".join(sorted(REQUIRED_SOURCE_ROLES))
        )

    filenames: set[str] = set()
    urls: set[str] = set()
    for source in contract.sources:
        _validate_source(source)
        if source.filename in filenames:
            raise AcquisitionValidationError("source filenames must be unique")
        if source.url in urls:
            raise AcquisitionValidationError("source URLs must be unique")
        filenames.add(source.filename)
        urls.add(source.url)


def validate_release_metadata(
    text: str, contract: AcquisitionContract
) -> ReleaseMetadata:
    """Prove that mutable upstream metadata still names the frozen release."""

    knowledgebase_match = _KNOWLEDGEBASE_RELEASE_PATTERN.search(text)
    swiss_prot_match = _SWISS_PROT_RELEASE_PATTERN.search(text)
    if knowledgebase_match is None or swiss_prot_match is None:
        raise AcquisitionValidationError(
            "release metadata is missing the required UniProt release lines"
        )

    knowledgebase_release = knowledgebase_match.group("release")
    swiss_prot_release = swiss_prot_match.group("release")
    if knowledgebase_release != swiss_prot_release:
        raise AcquisitionValidationError(
            "Knowledgebase and Swiss-Prot release identifiers disagree"
        )
    if knowledgebase_release != contract.release_id:
        raise AcquisitionValidationError(
            "upstream release drift: "
            f"expected {contract.release_id}, found {knowledgebase_release}"
        )

    release_date_text = swiss_prot_match.group("release_date")
    try:
        parsed_date = datetime.strptime(release_date_text, "%d-%b-%Y").date()
    except ValueError as error:
        raise AcquisitionValidationError(
            f"could not parse upstream release date '{release_date_text}'"
        ) from error
    if parsed_date != contract.release_date:
        raise AcquisitionValidationError(
            "upstream release date drift: "
            f"expected {contract.release_date.isoformat()}, "
            f"found {parsed_date.isoformat()}"
        )

    return ReleaseMetadata(
        release_id=knowledgebase_release,
        release_date=parsed_date,
    )


def prove_heavy_paths_are_ignored(
    contract: AcquisitionContract, project_root: Path
) -> tuple[PurePosixPath, ...]:
    """Use Git's own matcher to prove every configured raw path is ignored."""

    ignored_paths: list[PurePosixPath] = []
    for source in contract.sources:
        relative_path = contract.local_path_for(source)
        result = subprocess.run(
            [
                "git",
                "check-ignore",
                "--quiet",
                "--",
                relative_path.as_posix(),
            ],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 1:
            raise AcquisitionValidationError(
                f"heavy path is not ignored by Git: {relative_path}"
            )
        if result.returncode != 0:
            detail = result.stderr.strip() or "git check-ignore failed"
            raise AcquisitionValidationError(detail)
        ignored_paths.append(relative_path)

    return tuple(ignored_paths)


def verify_local_file(
    path: Path, source: SourceArtifact, *, chunk_bytes: int = 1024 * 1024
) -> LocalFileVerification:
    """Verify byte size and MD5, then return a local SHA-256 provenance value."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if not path.is_file():
        raise AcquisitionValidationError(f"source file does not exist: {path}")

    byte_size = path.stat().st_size
    if byte_size != source.expected_bytes:
        raise AcquisitionValidationError(
            f"byte-size mismatch for {source.filename}: "
            f"expected {source.expected_bytes}, found {byte_size}"
        )

    published_hasher = hashlib.md5(usedforsecurity=False)
    local_hasher = hashlib.sha256()
    with path.open("rb") as source_file:
        while chunk := source_file.read(chunk_bytes):
            published_hasher.update(chunk)
            local_hasher.update(chunk)

    calculated_md5 = published_hasher.hexdigest()
    if not hmac.compare_digest(calculated_md5, source.published_md5):
        raise AcquisitionValidationError(
            f"MD5 mismatch for {source.filename}: "
            f"expected {source.published_md5}, found {calculated_md5}"
        )

    return LocalFileVerification(
        path=path,
        byte_size=byte_size,
        md5=calculated_md5,
        sha256=local_hasher.hexdigest(),
    )


def _parse_source(raw: Any, index: int) -> SourceArtifact:
    context = f"sources[{index}]"
    if not isinstance(raw, dict):
        raise AcquisitionValidationError(f"{context} must be a table")

    uniref50_column = raw.get("uniref50_column")
    if uniref50_column is not None and (
        isinstance(uniref50_column, bool) or not isinstance(uniref50_column, int)
    ):
        raise AcquisitionValidationError(
            f"{context}.uniref50_column must be an integer"
        )

    return SourceArtifact(
        role=_required_string(raw, "role", context),
        description=_required_string(raw, "description", context),
        filename=_required_string(raw, "filename", context),
        url=_required_string(raw, "url", context),
        expected_bytes=_required_integer(raw, "expected_bytes", context),
        published_md5=_required_string(raw, "published_md5", context),
        uniref50_column=uniref50_column,
    )


def _validate_source(source: SourceArtifact) -> None:
    if (
        not source.filename
        or "/" in source.filename
        or "\\" in source.filename
        or source.filename in {".", ".."}
    ):
        raise AcquisitionValidationError(
            f"source filename must be a basename: {source.filename!r}"
        )
    _validate_official_url(
        source.url,
        expected_filename=source.filename,
        field=f"sources[{source.role}].url",
    )
    if source.expected_bytes <= 0:
        raise AcquisitionValidationError(
            f"expected_bytes must be positive for {source.filename}"
        )
    if not _MD5_PATTERN.fullmatch(source.published_md5):
        raise AcquisitionValidationError(
            f"published_md5 must be 32 lowercase hex characters for {source.filename}"
        )

    if source.role == "uniref50_membership":
        if source.uniref50_column != 10:
            raise AcquisitionValidationError(
                "UniRef50 membership must be pinned to column 10"
            )
    elif source.uniref50_column is not None:
        raise AcquisitionValidationError(
            "uniref50_column is valid only for the membership source"
        )


def _validate_official_url(url: str, *, expected_filename: str, field: str) -> None:
    parsed = _split_url(url, field=field)
    if parsed.scheme != "https" or parsed.netloc != OFFICIAL_HOST:
        raise AcquisitionValidationError(f"{field} must use https://{OFFICIAL_HOST}")
    if parsed.query or parsed.fragment:
        raise AcquisitionValidationError(
            f"{field} must not contain a query or fragment"
        )
    if not parsed.path.startswith(OFFICIAL_PATH_PREFIX):
        raise AcquisitionValidationError(
            f"{field} must be below UniProt's current_release path"
        )
    if (
        "%" in parsed.path
        or "\\" in parsed.path
        or any(segment in {".", ".."} for segment in parsed.path.split("/"))
    ):
        raise AcquisitionValidationError(
            f"{field} must not contain encoded or dot path segments"
        )
    if PurePosixPath(parsed.path).name != expected_filename:
        raise AcquisitionValidationError(
            f"{field} filename does not match {expected_filename}"
        )


def _validate_https_url(url: str, *, field: str) -> None:
    parsed = _split_url(url, field=field)
    if parsed.scheme != "https" or not parsed.netloc:
        raise AcquisitionValidationError(f"{field} must be an HTTPS URL")


def _split_url(url: str, *, field: str) -> SplitResult:
    try:
        return urlsplit(url)
    except ValueError as error:
        raise AcquisitionValidationError(f"{field} is malformed") from error


def _required_table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise AcquisitionValidationError(f"config field '{key}' must be a table")
    return value


def _required_string(raw: dict[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AcquisitionValidationError(f"{context}.{key} must be a non-empty string")
    return value


def _required_integer(raw: dict[str, Any], key: str, context: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AcquisitionValidationError(f"{context}.{key} must be an integer")
    return value
