"""CLI safety checks that do not load project collections."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "train_week3_mlp", ROOT / "scripts/train_week3_mlp.py"
)
assert SPEC and SPEC.loader
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_default_is_preflight_only(capsys: pytest.CaptureFixture[str]) -> None:
    assert SCRIPT.main([]) == 0
    assert "preflight loads no collections" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args", (("--new-run",), ("--new-run", "--run-id", "abc", "--seed", "20260821"))
)
def test_execution_requires_explicit_operating_choices(args: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit):
        SCRIPT.parse_args(list(args))


def test_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        SCRIPT.parse_args(
            [
                "--new-run",
                "--resume-checkpoint",
                "checkpoint",
                "--run-id",
                "abc",
                "--seed",
                "20260821",
                "--device",
                "cpu",
            ]
        )
