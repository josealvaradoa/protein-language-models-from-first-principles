"""Installed-package and result-provenance checks for the ESMC smoke."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any, Callable

from protein_lm.external.esmc_contract import ContractValidationError, ESMCContract, sha256_file


DistributionGetter = Callable[[str], importlib.metadata.Distribution]


def validate_installed_package_provenance(
    contract: ESMCContract,
    *,
    distribution_getter: DistributionGetter = importlib.metadata.distribution,
) -> dict[str, dict[str, str]]:
    """Fail closed unless installed ESM and Transformers come from frozen commits."""
    esm = _package_provenance("esm", distribution_getter)
    transformers = _package_provenance("transformers", distribution_getter)
    if esm["version"] != contract.code_version:
        raise ContractValidationError(
            f"esm version must be {contract.code_version}, found {esm['version']}"
        )
    if esm["commit_id"] != contract.code_revision:
        raise ContractValidationError("esm direct_url commit_id differs from the contract")
    if transformers["commit_id"] != contract.transformers_revision:
        raise ContractValidationError(
            "Transformers direct_url commit_id differs from the contract"
        )
    return {"esm": esm, "transformers": transformers}


def lockfile_sha256(project_root: Path) -> str | None:
    """Return the current lockfile hash when this source checkout has one."""
    lockfile = project_root / "uv.lock"
    return sha256_file(lockfile) if lockfile.is_file() else None


def _package_provenance(
    name: str,
    distribution_getter: DistributionGetter,
) -> dict[str, str]:
    try:
        distribution = distribution_getter(name)
        direct_url_text = distribution.read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError as exception:
        raise ContractValidationError(f"required package is not installed: {name}") from exception
    if direct_url_text is None:
        raise ContractValidationError(f"{name} does not record direct_url.json")
    try:
        direct_url: dict[str, Any] = json.loads(direct_url_text)
        commit_id = direct_url["vcs_info"]["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError) as exception:
        raise ContractValidationError(f"{name} direct_url.json lacks a commit_id") from exception
    if not isinstance(commit_id, str):
        raise ContractValidationError(f"{name} direct_url commit_id is invalid")
    return {"version": distribution.version, "commit_id": commit_id}
