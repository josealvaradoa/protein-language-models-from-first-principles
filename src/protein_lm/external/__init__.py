"""Optional, externally sourced model integrations with explicit contracts."""

from protein_lm.external.esmc_contract import ESMCContract, load_esmc_contract
from protein_lm.external.esmc_result import write_esmc_result
from protein_lm.external.esmc_smoke import run_esmc_smoke

__all__ = [
    "ESMCContract",
    "load_esmc_contract",
    "run_esmc_smoke",
    "write_esmc_result",
]
