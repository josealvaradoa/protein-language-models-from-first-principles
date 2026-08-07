import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_synthetic_sustained_confirmation.py"
)


def _load_script_module() -> object:
    specification = importlib.util.spec_from_file_location(
        "run_synthetic_sustained_confirmation",
        SCRIPT_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_sustained_cli_requires_explicit_execution_acknowledgement() -> None:
    module = _load_script_module()

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--source-capacity-result",
                "capacity-h.json",
                "--result-path",
                "result.json",
            ]
        )


def test_sustained_cli_accepts_source_and_new_result_paths() -> None:
    module = _load_script_module()

    args = module.parse_args(
        [
            "--execute-mps-sustained-confirmation",
            "--source-capacity-result",
            "results/capacity-h.json",
            "--result-path",
            "results/sustained-h.json",
        ]
    )

    assert args.source_capacity_result == Path("results/capacity-h.json")
    assert args.result_path == Path("results/sustained-h.json")


def test_sustained_cli_refuses_existing_result_before_starting_mps_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script_module()
    result_path = tmp_path / "existing.json"
    result_path.write_text("existing evidence\n", encoding="utf-8")

    def should_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("the sustained benchmark must not start")

    monkeypatch.setattr(module, "run_sustained_h_confirmation", should_not_run)

    assert (
        module.main(
            [
                "--execute-mps-sustained-confirmation",
                "--source-capacity-result",
                str(tmp_path / "capacity-h.json"),
                "--result-path",
                str(result_path),
            ]
        )
        == 2
    )
