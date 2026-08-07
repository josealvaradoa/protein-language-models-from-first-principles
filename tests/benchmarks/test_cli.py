import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_synthetic_device_benchmark.py"
)


def _load_script_module() -> object:
    specification = importlib.util.spec_from_file_location(
        "run_synthetic_device_benchmark",
        SCRIPT_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_cli_requires_explicit_execution_acknowledgement() -> None:
    module = _load_script_module()

    with pytest.raises(SystemExit):
        module.parse_args(["--candidate", "A", "--result-path", "result.json"])


def test_cli_accepts_exactly_one_approved_candidate_and_result_path() -> None:
    module = _load_script_module()

    args = module.parse_args(
        [
            "--execute-mps-benchmark",
            "--candidate",
            "C",
            "--result-path",
            "results/candidate-c.json",
        ]
    )

    assert args.candidate == "C"
    assert args.result_path == Path("results/candidate-c.json")


def test_cli_rejects_unknown_or_repeated_candidate_arguments() -> None:
    module = _load_script_module()

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--execute-mps-benchmark",
                "--candidate",
                "E",
                "--result-path",
                "result.json",
            ]
        )
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--execute-mps-benchmark",
                "--candidate",
                "A",
                "--candidate",
                "B",
                "--result-path",
                "result.json",
            ]
        )


def test_cli_rejects_an_existing_result_before_starting_mps_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    result_path = tmp_path / "existing.json"
    result_path.write_text("existing evidence\n", encoding="utf-8")

    def should_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("the benchmark must not run for an existing output path")

    monkeypatch.setattr(module, "run_synthetic_benchmark", should_not_run)

    assert (
        module.main(
            [
                "--execute-mps-benchmark",
                "--candidate",
                "A",
                "--result-path",
                str(result_path),
            ]
        )
        == 2
    )
