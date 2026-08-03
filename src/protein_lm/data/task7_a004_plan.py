"""The exact imported and newly executed A-004 fixed-budget pass plan."""

from __future__ import annotations

from dataclasses import dataclass

IMPORTED_TRACK = ("random", "validation", "residual")
EXECUTED_TRACKS = (
    ("random", "validation", "enforcement"),
    ("random", "test", "enforcement"),
    ("random", "test", "residual"),
    ("group_aware", "validation", "enforcement"),
    ("group_aware", "validation", "residual"),
    ("group_aware", "test", "enforcement"),
    ("group_aware", "test", "residual"),
)


@dataclass(frozen=True)
class PlannedTrack:
    """One imported or freshly executed fixed-budget pass."""

    strategy: str
    partition: str
    pass_name: str
    origin: str


def fixed_budget_stage_plan() -> tuple[PlannedTrack, ...]:
    """Return the one A-003 import plus seven fresh A-004 passes."""

    return (
        PlannedTrack(*IMPORTED_TRACK, origin="imported_a003"),
        *(PlannedTrack(*track, origin="executed_a004") for track in EXECUTED_TRACKS),
    )
