"""Runtime provenance shared by the A-004 report and receipt."""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping

from protein_lm.data.similarity_audit_policy import SimilarityAuditError

HARDWARE_FIELDS = ("platform", "machine", "processor", "logical_cpu_count")


def hardware_provenance() -> dict[str, object]:
    """Capture the same host fields used by the A-003 audit report."""

    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    validate_hardware_provenance(hardware)
    return hardware


def validate_hardware_provenance(hardware: Mapping[str, object]) -> None:
    """Require one complete, JSON-safe hardware provenance object."""

    if set(hardware) != set(HARDWARE_FIELDS):
        raise SimilarityAuditError("A-004 hardware provenance fields drifted")
    if any(not isinstance(hardware[name], str) for name in HARDWARE_FIELDS[:3]):
        raise SimilarityAuditError("A-004 hardware provenance is malformed")
    cpu_count = hardware["logical_cpu_count"]
    if isinstance(cpu_count, bool) or not isinstance(cpu_count, int) or cpu_count < 1:
        raise SimilarityAuditError("A-004 logical CPU count is malformed")
