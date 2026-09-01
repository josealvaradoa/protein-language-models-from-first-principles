"""Immutable on-disk storage for one reproduction run bundle.

The bundle is deliberately independent of contract loading and execution.  A
caller supplies frozen contract bytes and records the result through this small
storage boundary.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tomllib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class RunBundleError(RuntimeError):
    """Raised when a run bundle cannot be safely created or mutated."""


class RunStatus(str, Enum):
    """The only statuses represented by a run bundle."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RUNNER_RESTARTED = "runner_restarted"

    @property
    def is_terminal(self) -> bool:
        """Whether this status seals a bundle against further API mutation."""

        return self is not RunStatus.RUNNING


_RUN_ID_LENGTH = 32
_REQUIRED_FILENAMES = (
    "contract.toml",
    "run.json",
    "log.txt",
    "metrics.json",
    "comparison.json",
    "provenance.json",
)
_IDENTITY_FIELDS = frozenset({"run_id", "contract_identifier"})


@dataclass(frozen=True)
class RunBundle:
    """Immutable paths and identity for one bundle below a configured root."""

    root: Path
    run_id: str
    contract_identifier: str

    def __post_init__(self) -> None:
        """Validate direct construction as strictly as factory construction."""

        object.__setattr__(self, "root", _validated_root(self.root))
        _validate_run_id(self.run_id)
        _validate_contract_identifier(self.contract_identifier)

    @classmethod
    def create(
        cls,
        root: Path | str,
        run_id: str,
        contract_bytes: bytes,
        initial_run_record: Mapping[str, object],
    ) -> RunBundle:
        """Stage and publish a new running bundle.

        The final directory is not visible until its contract, running record,
        and empty log are all complete.
        """

        safe_root = _prepare_root(root)
        _validate_run_id(run_id)
        contract_identifier = _contract_identifier(contract_bytes)
        initial_record = _validated_initial_record(
            initial_run_record, run_id, contract_identifier
        )
        bundle = cls(safe_root, run_id, contract_identifier)
        final_directory = bundle.directory
        if final_directory.exists() or final_directory.is_symlink():
            raise RunBundleError(f"run bundle already exists: {run_id}")

        staging_directory = safe_root / f".{run_id}.staging-{uuid.uuid4().hex}"
        _assert_child(safe_root, staging_directory)
        try:
            staging_directory.mkdir(mode=0o700)
            _write_new_file(staging_directory / "contract.toml", contract_bytes)
            _write_new_file(
                staging_directory / "run.json", _json_bytes(initial_record)
            )
            _write_new_file(staging_directory / "log.txt", b"")
            if final_directory.exists() or final_directory.is_symlink():
                raise RunBundleError(f"run bundle already exists: {run_id}")
            os.replace(staging_directory, final_directory)
        except Exception:
            _remove_staging_directory(staging_directory)
            raise
        return bundle

    @property
    def directory(self) -> Path:
        """The final, caller-visible directory for this bundle."""

        return self.root / self.run_id

    @property
    def contract_path(self) -> Path:
        """The exact frozen contract bytes supplied at creation."""

        return self.directory / "contract.toml"

    @property
    def run_path(self) -> Path:
        """The status record whose terminal state seals the bundle."""

        return self.directory / "run.json"

    @property
    def log_path(self) -> Path:
        """The append-only log while a bundle remains running."""

        return self.directory / "log.txt"

    @property
    def metrics_path(self) -> Path:
        """The final metrics payload path."""

        return self.directory / "metrics.json"

    @property
    def comparison_path(self) -> Path:
        """The final comparison payload path."""

        return self.directory / "comparison.json"

    @property
    def provenance_path(self) -> Path:
        """The final provenance payload path."""

        return self.directory / "provenance.json"

    def append_log(self, text: str) -> None:
        """Append text while the run is active."""

        if not isinstance(text, str):
            raise RunBundleError("log text must be a string")
        self._running_record()
        path = self._checked_file("log.txt")
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise RunBundleError("could not append run log") from error

    def finalize(
        self,
        status: RunStatus | str,
        run_record_updates: Mapping[str, object],
        metrics: Mapping[str, object],
        comparison: Mapping[str, object],
        provenance: Mapping[str, object],
    ) -> None:
        """Write final payloads, then make the terminal record visible last."""

        terminal_status = _terminal_status(status)
        current_record = self._running_record()
        updates = _require_object(run_record_updates, "run record updates")
        _validate_identity_updates(updates, self)
        if "status" in updates:
            raise RunBundleError("run record updates cannot set status")

        final_record = dict(current_record)
        final_record.update(updates)
        final_record["status"] = terminal_status.value

        # Serialize every final payload before touching any final file.
        metrics_bytes = _json_bytes(_require_object(metrics, "metrics"))
        comparison_bytes = _json_bytes(_require_object(comparison, "comparison"))
        provenance_bytes = _json_bytes(_require_object(provenance, "provenance"))
        final_record_bytes = _json_bytes(final_record)

        contract_path = self._checked_file("contract.toml")
        self._checked_file("log.txt")
        if not contract_path.is_file() or not self.log_path.is_file():
            raise RunBundleError("running bundle is missing required initial files")
        _atomic_write_json(self._checked_file("metrics.json"), metrics_bytes)
        _atomic_write_json(self._checked_file("comparison.json"), comparison_bytes)
        _atomic_write_json(self._checked_file("provenance.json"), provenance_bytes)
        for filename in (
            "contract.toml",
            "log.txt",
            "metrics.json",
            "comparison.json",
            "provenance.json",
        ):
            path = self._checked_file(filename)
            if not path.is_file():
                raise RunBundleError("bundle is missing a required file before finalization")
        _atomic_write_json(self._checked_file("run.json"), final_record_bytes)

    def _running_record(self) -> dict[str, object]:
        record = _load_record(self._checked_file("run.json"))
        if record.get("run_id") != self.run_id:
            raise RunBundleError("run record identity does not match this bundle")
        if record.get("contract_identifier") != self.contract_identifier:
            raise RunBundleError("run record contract identifier does not match")
        try:
            status = RunStatus(record.get("status"))
        except (TypeError, ValueError) as error:
            raise RunBundleError("run record has an invalid status") from error
        if status is not RunStatus.RUNNING:
            raise RunBundleError("terminal run bundles are immutable")
        _json_bytes(record)
        return record

    def _checked_file(self, filename: str) -> Path:
        if filename not in _REQUIRED_FILENAMES:
            raise RunBundleError("unsupported bundle file")
        directory = self.directory
        if directory.is_symlink() or not directory.is_dir():
            raise RunBundleError("run bundle directory is missing or unsafe")
        _assert_child(self.root, directory)
        path = directory / filename
        _assert_child(directory, path)
        if path.is_symlink():
            raise RunBundleError("run bundle file is a symlink")
        return path


def _prepare_root(root: Path | str) -> Path:
    path = Path(root)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RunBundleError("could not prepare run bundle root") from error
    return _validated_root(path)


def _validated_root(root: Path | str) -> Path:
    path = Path(root)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RunBundleError("run bundle root must be an existing real directory") from error
    if not resolved.is_dir() or path.absolute() != resolved:
        raise RunBundleError("run bundle root must be a real directory, not a symlink")
    return resolved


def _validate_run_id(run_id: str) -> None:
    if (
        not isinstance(run_id, str)
        or len(run_id) != _RUN_ID_LENGTH
        or any(character not in "0123456789abcdef" for character in run_id)
    ):
        raise RunBundleError("run id must be exactly 32 lowercase hexadecimal characters")


def _contract_identifier(contract_bytes: bytes) -> str:
    if not isinstance(contract_bytes, bytes) or not contract_bytes:
        raise RunBundleError("contract bytes must be nonempty bytes")
    try:
        contract = tomllib.loads(contract_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RunBundleError("contract bytes must contain valid TOML") from error
    identifier = contract.get("contract_identifier")
    _validate_contract_identifier(identifier)
    return identifier


def _validate_contract_identifier(identifier: object) -> None:
    if not isinstance(identifier, str) or not identifier.strip():
        raise RunBundleError("contract identifier must be a nonempty string")


def _validated_initial_record(
    record: Mapping[str, object], run_id: str, contract_identifier: str
) -> dict[str, object]:
    value = _require_object(record, "initial run record")
    if value.get("run_id") != run_id:
        raise RunBundleError("initial run record run_id does not match")
    if value.get("contract_identifier") != contract_identifier:
        raise RunBundleError("initial run record contract_identifier does not match")
    if value.get("status") != RunStatus.RUNNING.value:
        raise RunBundleError("initial run record status must be running")
    _json_bytes(value)
    return value


def _terminal_status(value: RunStatus | str) -> RunStatus:
    try:
        status = RunStatus(value)
    except (TypeError, ValueError) as error:
        raise RunBundleError("final status is invalid") from error
    if not status.is_terminal:
        raise RunBundleError("final status must be terminal")
    return status


def _validate_identity_updates(updates: Mapping[str, object], bundle: RunBundle) -> None:
    expected = {
        "run_id": bundle.run_id,
        "contract_identifier": bundle.contract_identifier,
    }
    for field in _IDENTITY_FIELDS & updates.keys():
        if updates[field] != expected[field]:
            raise RunBundleError(f"run record update cannot change {field}")


def _require_object(value: Mapping[str, object], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RunBundleError(f"{label} must be an object mapping")
    return dict(value)


def _json_bytes(value: Mapping[str, object]) -> bytes:
    try:
        encoded = json.dumps(
            _json_safe_value(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise RunBundleError("payload must be a finite JSON-safe object") from error
    return (encoded + "\n").encode("utf-8")


def _json_safe_value(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    raise TypeError("value is not JSON-safe")


def _write_new_file(path: Path, content: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise RunBundleError(f"could not write {path.name}") from error


def _remove_staging_directory(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _atomic_write_json(path: Path, content: bytes) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    _assert_child(path.parent, temporary)
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as error:
        raise RunBundleError(f"could not atomically write {path.name}") from error
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _load_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunBundleError("run record is malformed") from error
    if not isinstance(value, dict):
        raise RunBundleError("run record must be a JSON object")
    return value


def _assert_child(parent: Path, child: Path) -> None:
    try:
        resolved_parent = parent.resolve(strict=True)
        if not resolved_parent.is_dir() or parent.absolute() != resolved_parent:
            raise RunBundleError("bundle path contains a symlink")
        resolved_child = child.resolve(strict=False)
        resolved_child.relative_to(resolved_parent)
    except ValueError as error:
        raise RunBundleError("bundle path escapes its configured root") from error
    except OSError as error:
        raise RunBundleError("could not validate bundle path") from error
