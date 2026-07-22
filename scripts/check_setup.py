"""Report the local runtime used for protein language-model experiments."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy
import torch


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    print("protein language models setup check")
    print(f"project_root: {project_root}")
    print(f"python: {platform.python_version()}")
    print(f"python_executable: {sys.executable}")
    print(f"platform: {platform.platform()}")
    print(f"numpy: {numpy.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"mps_built: {torch.backends.mps.is_built()}")
    print(f"mps_available: {torch.backends.mps.is_available()}")


if __name__ == "__main__":
    main()
