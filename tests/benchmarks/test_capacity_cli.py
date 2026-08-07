import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_synthetic_capacity_benchmark.py"
)


def _load_script_module() -> object:
    specification = importlib.util.spec_from_file_location(
        "run_synthetic_capacity_benchmark",
        SCRIPT_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_capacity_cli_requires_explicit_execution_acknowledgement() -> None:
    module = _load_script_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--candidate", "E", "--result-path", "result.json"])


def test_capacity_cli_accepts_one_approved_candidate() -> None:
    module = _load_script_module()

    args = module.parse_args(
        [
            "--execute-mps-capacity-benchmark",
            "--candidate",
            "J",
            "--result-path",
            "results/candidate-j.json",
        ]
    )

    assert args.candidate == "J"
    assert args.result_path == Path("results/candidate-j.json")


def test_capacity_cli_rejects_unapproved_or_repeated_candidates() -> None:
    module = _load_script_module()

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--execute-mps-capacity-benchmark",
                "--candidate",
                "A",
                "--result-path",
                "result.json",
            ]
        )
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--execute-mps-capacity-benchmark",
                "--candidate",
                "E",
                "--candidate",
                "F",
                "--result-path",
                "result.json",
            ]
        )


def test_capacity_cli_refuses_to_overwrite_existing_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    result_path = tmp_path / "existing.json"
    result_path.write_text("existing evidence\n", encoding="utf-8")

    def should_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("the benchmark must not run for an existing output path")

    monkeypatch.setattr(module, "run_synthetic_capacity_benchmark", should_not_run)

    assert (
        module.main(
            [
                "--execute-mps-capacity-benchmark",
                "--candidate",
                "E",
                "--result-path",
                str(result_path),
            ]
        )
        == 2
    )
