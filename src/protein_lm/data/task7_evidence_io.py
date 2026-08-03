"""Small atomic-file helper for ignored Task 7 evidence directories."""

from __future__ import annotations

import hashlib
from pathlib import Path

from protein_lm.data.similarity_audit_models import FileEvidence


class EvidenceWriter:
    """Write one evidence file through a sibling temporary file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary_path = path.with_name(f".{path.name}.incomplete")
        self.temporary_path.unlink(missing_ok=True)
        self.output = self.temporary_path.open("wb")
        self.hasher = hashlib.sha256()
        self.byte_size = 0
        self.row_count = 0

    def write(self, content: bytes) -> None:
        self.output.write(content)
        self.hasher.update(content)
        self.byte_size += len(content)
        self.row_count += 1

    def finish(self) -> FileEvidence:
        self.output.close()
        self.temporary_path.replace(self.path)
        return FileEvidence(self.row_count, self.byte_size, self.hasher.hexdigest())

    def abort(self) -> None:
        if not self.output.closed:
            self.output.close()
        self.temporary_path.unlink(missing_ok=True)
        self.path.unlink(missing_ok=True)
