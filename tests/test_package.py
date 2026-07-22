from protein_lm import __version__


def test_package_imports_with_expected_version() -> None:
    assert __version__ == "0.1.0"
