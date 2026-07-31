"""Compatibility imports for the former combined Task 7 input module.

New code should import membership logic from ``similarity_manifests`` and FASTA
logic from ``similarity_fastas``. This facade keeps the original public imports
available without mixing both responsibilities in one implementation file.
"""

from protein_lm.data.similarity_fastas import (
    FastaEvidence,
    MaterializedInputs,
    iter_one_line_fasta,
    materialize_strategy_fastas,
    write_fasta_subset,
)
from protein_lm.data.similarity_manifests import (
    PARTITIONS,
    STRATEGIES,
    PartitionPopulation,
    StrategyManifest,
    StructuralMembershipAudit,
    load_strategy_manifest,
    metadata_by_partition,
)

__all__ = [
    "PARTITIONS",
    "STRATEGIES",
    "FastaEvidence",
    "MaterializedInputs",
    "PartitionPopulation",
    "StrategyManifest",
    "StructuralMembershipAudit",
    "iter_one_line_fasta",
    "load_strategy_manifest",
    "materialize_strategy_fastas",
    "metadata_by_partition",
    "write_fasta_subset",
]
