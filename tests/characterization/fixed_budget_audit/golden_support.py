"""Independent expected bytes for the synthetic fixed-budget audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FINGERPRINT = "8cbf7b374c9fb9100b92254e2aa51e80bef636e16853623e395d61473b21ba37"
SOURCE_FINGERPRINT = "synthetic-a003-fingerprint"
CODE_REVISION = "b" * 40
FORMAT_OUTPUT = (
    "query,target,fident,qcov,tcov,alnlen,qlen,tlen,qstart,qend,"
    "tstart,tend,evalue,bits"
)
TRACKS = (
    ("group_aware", "test", "enforcement", "executed_a004"),
    ("group_aware", "test", "residual", "executed_a004"),
    ("group_aware", "validation", "enforcement", "executed_a004"),
    ("group_aware", "validation", "residual", "executed_a004"),
    ("random", "test", "enforcement", "executed_a004"),
    ("random", "test", "residual", "executed_a004"),
    ("random", "validation", "enforcement", "executed_a004"),
    ("random", "validation", "residual", "imported_a003"),
)
CAPS = (1_000, 10_000, 100_000)
CATEGORIES = (
    "closest_match_prohibited",
    "identity_30_to_under_40",
    "identity_40_to_under_50",
    "identity_ge_50_below_bidirectional_coverage",
    "identity_under_30_or_no_residual_hit",
)

Track = tuple[str, str, str, str]


def json_bytes(payload: object) -> bytes:
    """Serialize the reviewed checkpoint convention without production code."""

    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def identity(content: bytes) -> dict[str, object]:
    """Calculate the byte identity directly with hashlib."""

    return {"byte_size": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def file_evidence(content: bytes, rows: int) -> dict[str, object]:
    return {"row_count": rows, **identity(content)}


def fasta_evidence(content: bytes) -> dict[str, object]:
    return {
        "record_count": 1,
        "residue_count": 4,
        **identity(content),
    }


class GoldenAudit:
    """Literal synthetic artifact graph used as byte-level expected data."""

    def __init__(self, project_root: Path, source_policy_bytes: bytes) -> None:
        self.project_root = project_root
        self.workspace = project_root / "a004"
        self.source_workspace = project_root / "a003"
        self.source_policy_bytes = source_policy_bytes
        self.artifacts: dict[str, bytes] = {}
        self.databases: dict[str, dict[str, object]] = {}
        self.stages: dict[tuple[Track, int], dict[str, object]] = {}
        self.pass_markers: dict[Track, dict[str, object] | None] = {}
        self.summaries: dict[tuple[Track, int], dict[str, object]] = {}
        self.unions: dict[tuple[str, str], dict[str, object]] = {}
        self._add_fixture_inputs()
        self._add_imported_stages()
        self._add_databases()
        self._add_executed_tracks()
        self._add_summaries()
        self._add_unions()

    def add_publication(self, markdown: bytes) -> None:
        """Add independently assembled report, receipt, and completion bytes."""

        report_payload = self._report_payload()
        report_json = json_bytes(report_payload)
        report_root = "a004/evidence/report"
        report_json_rel = f"{report_root}/a004_report.json"
        report_markdown_rel = f"{report_root}/a004_report.md"
        self.add(report_json_rel, report_json)
        self.add(report_markdown_rel, markdown)
        report_marker = json_bytes(
            {
                "schema_version": 1,
                "stage": "a004_report_artifacts",
                "fingerprint": FINGERPRINT,
                "json": identity(report_json),
                "markdown": identity(markdown),
            }
        )
        report_marker_rel = f"{report_root}/complete.json"
        self.add(report_marker_rel, report_marker)
        report = {
            "directory": str(self.path(report_root)),
            "json": identity(report_json),
            "markdown": identity(markdown),
            "marker": identity(report_marker),
        }
        receipt = json_bytes(self._receipt_payload(report))
        receipt_rel = "a004/a004_import_receipt.json"
        self.add(receipt_rel, receipt)
        completion = json_bytes(
            {
                "schema_version": 1,
                "stage": "a004_workflow_complete",
                "fingerprint": FINGERPRINT,
                "receipt": identity(receipt),
                "report": {
                    key: report[key] for key in ("json", "markdown", "marker")
                },
                "model_use": "prohibited",
                "task8_membership_use_authorized": False,
                "diagnostic_assignments_unchanged": True,
            }
        )
        self.add("a004/a004_complete.json", completion)

    def add(self, relative_path: str, content: bytes) -> None:
        if relative_path in self.artifacts:
            raise AssertionError(f"duplicate expected artifact: {relative_path}")
        self.artifacts[relative_path] = content

    def bytes(self, relative_path: str) -> bytes:
        return self.artifacts[relative_path]

    def path(self, relative_path: str) -> Path:
        return self.project_root / relative_path

    def artifact_identity(self, relative_path: str) -> dict[str, object]:
        return identity(self.bytes(relative_path))

    def query(self, strategy: str, partition: str) -> str:
        prefix = "R" if strategy == "random" else "G"
        suffix = "VALID" if partition == "validation" else "TEST"
        return f"{prefix}_{suffix}"

    def target(self, strategy: str) -> str:
        return "R_TRAIN" if strategy == "random" else "G_TRAIN"

    def fasta_rel(self, strategy: str, partition: str) -> str:
        return f"a004/fastas/{strategy}_{partition}.fasta"

    def pass_rel(self, track: Track) -> str:
        strategy, partition, pass_name, _ = track
        return f"a004/tracks/{strategy}/{partition}/{pass_name}"

    def canonical_rel(self, track: Track, cap: int) -> str:
        strategy, partition, pass_name, origin = track
        if origin == "imported_a003":
            return f"a003/tracks/{strategy}/{partition}/{pass_name}/cap_{cap}/canonical.tsv"
        return f"{self.pass_rel(track)}/cap_{cap}/canonical.tsv"

    def summary_rel(self, track: Track, cap: int) -> str:
        strategy, partition, pass_name, origin = track
        return (
            f"a004/evidence/{origin}/{strategy}/{partition}/{pass_name}/cap_{cap}"
        )

    def search_command(self, track: Track, cap: int) -> tuple[str, ...]:
        strategy, partition, pass_name, _ = track
        pass_directory = self.path(self.pass_rel(track))
        stage_directory = pass_directory / f"cap_{cap}"
        query_path = (
            pass_directory / "escalated_queries.fasta"
            if cap == 100_000
            else self.path(self.fasta_rel(strategy, partition))
        )
        minimum, coverage = (
            ("0.5", "0.8") if pass_name == "enforcement" else ("0.3", "0.0")
        )
        return (
            "/opt/homebrew/bin/mmseqs",
            "easy-search",
            str(query_path),
            str(self.workspace / "databases" / strategy / "target"),
            str(stage_directory / "raw.tsv"),
            str(stage_directory / "mmseqs_tmp"),
            "--search-type",
            "1",
            "--alignment-mode",
            "3",
            "--seq-id-mode",
            "0",
            "-s",
            "7.5",
            "-e",
            "10",
            "--mask",
            "0",
            "--comp-bias-corr",
            "0",
            "--max-seqs",
            str(cap),
            "--threads",
            "10",
            "--format-output",
            FORMAT_OUTPUT,
            "--min-seq-id",
            minimum,
            "-c",
            coverage,
            "--cov-mode",
            "0",
        )

    def _add_fixture_inputs(self) -> None:
        self.add(
            "experiments/week_01/diagnostic_similarity_audit.toml",
            self.source_policy_bytes,
        )
        for strategy, prefix in (("random", "R"), ("group_aware", "G")):
            for partition, suffix in (
                ("training", "TRAIN"),
                ("validation", "VALID"),
                ("test", "TEST"),
            ):
                self.add(
                    self.fasta_rel(strategy, partition),
                    f">{prefix}_{suffix}\nAAAA\n".encode(),
                )
        self.add("a003/fastas/random_validation.fasta", b">R_VALID\nAAAA\n")
        self.add(
            "a003/tracks/random/validation/residual/escalated_queries.fasta",
            b">R_VALID\nAAAA\n",
        )
        for name in (
            "catalog",
            "task5_local",
            "task5_public",
            "task5_report",
            "task6_local",
            "task6_public",
            "task6_report",
        ):
            suffix = "tsv" if name == "catalog" else "json"
            self.add(f"frozen/{name}.{suffix}", f"{name}\n".encode())

    def _add_imported_stages(self) -> None:
        track = TRACKS[-1]
        for cap in CAPS:
            high = cap != 1_000
            canonical = self._canonical("R_VALID", "R_TRAIN", high=high)
            canonical_rel = self.canonical_rel(track, cap)
            self.add(canonical_rel, canonical)
            marker_rel = canonical_rel.replace("canonical.tsv", "complete.json")
            marker = json_bytes(
                {
                    "schema_version": 1,
                    "stage": "search_stage",
                    "fingerprint": SOURCE_FINGERPRINT,
                    "cap": cap,
                }
            )
            self.add(marker_rel, marker)
            query_rel = (
                "a003/tracks/random/validation/residual/escalated_queries.fasta"
                if cap == 100_000
                else "a003/fastas/random_validation.fasta"
            )
            self.stages[(track, cap)] = {
                "cap": cap,
                "origin": "imported_a003",
                "marker": identity(marker),
                "query_fasta": fasta_evidence(self.bytes(query_rel)),
                "canonical": file_evidence(canonical, 1),
                "command": ["mmseqs", "easy-search", str(cap)],
                "runtime_seconds": "0.03",
            }
        self.pass_markers[track] = None

    def _add_databases(self) -> None:
        for strategy in ("random", "group_aware"):
            target_rel = f"a004/databases/{strategy}/target"
            self.add(target_rel, b"synthetic database")
            command = (
                "/opt/homebrew/bin/mmseqs",
                "createdb",
                str(self.path(self.fasta_rel(strategy, "training"))),
                str(self.workspace / "databases" / f".{strategy}.incomplete" / "target"),
                "--dbtype",
                "1",
                "--shuffle",
                "0",
                "--createdb-mode",
                "0",
                "--threads",
                "10",
            )
            training = fasta_evidence(self.bytes(self.fasta_rel(strategy, "training")))
            marker_rel = f"a004/databases/{strategy}/complete.json"
            marker = json_bytes(
                {
                    "schema_version": 1,
                    "stage": "a004_target_database",
                    "fingerprint": FINGERPRINT,
                    "strategy": strategy,
                    "training_fasta": training,
                    "database_prefix": str(
                        self.workspace / "databases" / strategy / "target"
                    ),
                    "command": list(command),
                    "runtime_seconds": "0.01",
                    "artifacts": {"target": self.artifact_identity(target_rel)},
                }
            )
            self.add(marker_rel, marker)
            self.databases[strategy] = {
                "command": list(command),
                "runtime_seconds": "0.01",
                "identity": {
                    "marker": identity(marker),
                    "training_fasta": training,
                    "artifacts": {"target": self.artifact_identity(target_rel)},
                },
            }

    def _add_executed_tracks(self) -> None:
        for track in TRACKS[:-1]:
            strategy, partition, _, _ = track
            query = self.query(strategy, partition)
            target = self.target(strategy)
            query_fasta = fasta_evidence(
                self.bytes(self.fasta_rel(strategy, partition))
            )
            stage_markers = {}
            for cap in CAPS:
                high = cap != 1_000
                canonical = self._canonical(query, target, high=high)
                raw = self._raw(query, target, high=high)
                canonical_rel = self.canonical_rel(track, cap)
                self.add(canonical_rel, canonical)
                command = self.search_command(track, cap)
                marker_rel = canonical_rel.replace("canonical.tsv", "complete.json")
                marker = json_bytes(
                    {
                        "schema_version": 1,
                        "stage": "a004_fixed_budget_search_stage",
                        "fingerprint": FINGERPRINT,
                        "strategy": strategy,
                        "partition": partition,
                        "pass_name": track[2],
                        "cap": cap,
                        "query_fasta": query_fasta,
                        "query_ids_sha256": self._query_hash(query),
                        "target_database": str(
                            self.workspace / "databases" / strategy / "target"
                        ),
                        "target_database_identity": self.databases[strategy]["identity"],
                        "command": list(command),
                        "canonical_path": str(self.path(canonical_rel)),
                        "runtime_seconds": "0.02",
                        "raw_retained": False,
                        "alignment_evidence": {
                            "raw": file_evidence(raw, 1),
                            "canonical": file_evidence(canonical, 1),
                        },
                    }
                )
                self.add(marker_rel, marker)
                record = {
                    "cap": cap,
                    "origin": "executed_a004",
                    "marker": identity(marker),
                    "query_fasta": query_fasta,
                    "canonical": file_evidence(canonical, 1),
                    "command": list(command),
                    "runtime_seconds": "0.02",
                }
                self.stages[(track, cap)] = record
                stage_markers[str(cap)] = {
                    key: record[key]
                    for key in ("marker", "canonical", "query_fasta", "command")
                }
            escalation_rel = f"{self.pass_rel(track)}/escalated_queries.fasta"
            self.add(escalation_rel, self.bytes(self.fasta_rel(strategy, partition)))
            escalation_marker_rel = f"{self.pass_rel(track)}/escalated_queries.complete.json"
            escalation_marker = json_bytes(
                {
                    "schema_version": 1,
                    "stage": "a004_escalation_fasta",
                    "fingerprint": FINGERPRINT,
                    "source_fasta": query_fasta,
                    "source_query_ids_sha256": self._query_hash(query),
                    "query_count": 1,
                    "query_ids_sha256": self._query_hash(query),
                    "fasta": query_fasta,
                }
            )
            self.add(escalation_marker_rel, escalation_marker)
            pass_rel = f"{self.pass_rel(track)}/complete.json"
            pass_marker = json_bytes(
                {
                    "schema_version": 1,
                    "stage": "a004_fixed_budget_pass",
                    "fingerprint": FINGERPRINT,
                    "strategy": strategy,
                    "partition": partition,
                    "pass_name": track[2],
                    "query_fasta": query_fasta,
                    "query_ids_sha256": self._query_hash(query),
                    "target_database": str(
                        self.workspace / "databases" / strategy / "target"
                    ),
                    "target_database_identity": self.databases[strategy]["identity"],
                    "changed_query_ids": [query],
                    "escalation": {
                        "fasta": query_fasta,
                        "marker": identity(escalation_marker),
                    },
                    "stages": stage_markers,
                }
            )
            self.add(pass_rel, pass_marker)
            self.pass_markers[track] = identity(pass_marker)

    def _add_summaries(self) -> None:
        for track in TRACKS:
            strategy, partition, _, origin = track
            query = self.query(strategy, partition)
            target = self.target(strategy)
            for cap in CAPS:
                high = cap != 1_000
                pair = f"{query}\t{target}\n".encode() if high else b""
                summary = (
                    f"{query}\t1\tclosest_match_prohibited\t1\t1\n".encode()
                    if high
                    else (
                        f"{query}\t0\tidentity_under_30_or_no_residual_hit\t1\t0\n"
                    ).encode()
                )
                directory = self.summary_rel(track, cap)
                pair_rel = f"{directory}/prohibited_pairs.tsv"
                query_rel = f"{directory}/query_summaries.tsv"
                self.add(pair_rel, pair)
                self.add(query_rel, summary)
                evidence = self._cap_evidence(cap, pair, summary, high=high)
                canonical_rel = self.canonical_rel(track, cap)
                fasta_rel = (
                    (
                        "a003/tracks/random/validation/residual/escalated_queries.fasta"
                        if origin == "imported_a003"
                        else f"{self.pass_rel(track)}/escalated_queries.fasta"
                    )
                    if cap == 100_000
                    else (
                        "a003/fastas/random_validation.fasta"
                        if origin == "imported_a003"
                        else self.fasta_rel(strategy, partition)
                    )
                )
                marker_rel = f"{directory}/complete.json"
                marker = json_bytes(
                    {
                        "schema_version": 1,
                        "stage": "a004_cap_summary",
                        "fingerprint": FINGERPRINT,
                        "source_label": origin,
                        "cap": cap,
                        "canonical_path": str(self.path(canonical_rel)),
                        "canonical": file_evidence(self.bytes(canonical_rel), 1),
                        "query_fasta": fasta_evidence(self.bytes(fasta_rel)),
                        "query_ids_sha256": self._query_hash(query),
                        "evidence": evidence,
                    }
                )
                self.add(marker_rel, marker)
                self.summaries[(track, cap)] = {
                    "source_label": origin,
                    "marker": identity(marker),
                    "evidence": evidence,
                }

    def _add_unions(self) -> None:
        for strategy in ("group_aware", "random"):
            for partition in ("test", "validation"):
                query = self.query(strategy, partition)
                target = self.target(strategy)
                pair = f"{query}\t{target}\n".encode()
                root = f"a004/pair_unions/{strategy}/{partition}"
                residual = next(
                    track
                    for track in TRACKS
                    if track[:3] == (strategy, partition, "residual")
                )
                enforcement = next(
                    track
                    for track in TRACKS
                    if track[:3] == (strategy, partition, "enforcement")
                )
                common_sources = {
                    f"enforcement_executed_a004_cap_{cap}": identity(
                        self.bytes(
                            f"{self.summary_rel(enforcement, cap)}/prohibited_pairs.tsv"
                        )
                    )
                    for cap in (1_000, 10_000)
                }
                common_sources.update(
                    {
                        f"residual_{residual[3]}_cap_{cap}": identity(
                            self.bytes(
                                f"{self.summary_rel(residual, cap)}/prohibited_pairs.tsv"
                            )
                        )
                        for cap in (1_000, 10_000)
                    }
                )
                common_labels = tuple(sorted(common_sources))
                common = self._union_record(common_labels, pair)
                common_dir = f"{root}/common_all_query_10000"
                common_pair_rel = f"{common_dir}/prohibited_pairs.tsv"
                self.add(common_pair_rel, pair)
                common_marker = json_bytes(
                    {
                        "schema_version": 1,
                        "stage": "a004_pair_union",
                        "fingerprint": FINGERPRINT,
                        "label": f"common_all_query_10000_{strategy}_{partition}",
                        "sources": common_sources,
                        "evidence": common,
                    }
                )
                common_marker_rel = f"{common_dir}/complete.json"
                self.add(common_marker_rel, common_marker)
                staged_sources = {
                    "common_all_query_10000": identity(pair),
                    "enforcement_executed_a004_cap_100000": identity(pair),
                    f"residual_{residual[3]}_cap_100000": identity(pair),
                }
                staged = self._union_record(tuple(sorted(staged_sources)), pair)
                staged_dir = f"{root}/staged_union_with_changed_query_100000"
                self.add(f"{staged_dir}/prohibited_pairs.tsv", pair)
                staged_marker = json_bytes(
                    {
                        "schema_version": 1,
                        "stage": "a004_pair_union",
                        "fingerprint": FINGERPRINT,
                        "label": (
                            "staged_union_with_changed_query_100000_"
                            f"{strategy}_{partition}"
                        ),
                        "sources": staged_sources,
                        "evidence": staged,
                    }
                )
                staged_marker_rel = f"{staged_dir}/complete.json"
                self.add(staged_marker_rel, staged_marker)
                self.unions[(strategy, partition)] = {
                    "common": {
                        "marker": identity(common_marker),
                        "evidence": common,
                    },
                    "staged": {
                        "marker": identity(staged_marker),
                        "evidence": staged,
                    },
                    "comparison": {
                        "common_pairs": 1,
                        "staged_pairs": 1,
                        "additional_pairs": 0,
                        "common_queries": 1,
                        "staged_queries": 1,
                        "newly_prohibited_queries": 0,
                    },
                }

    def _report_payload(self) -> dict[str, object]:
        partitions = []
        for strategy, partition in sorted(self.unions):
            bundle = self.unions[(strategy, partition)]
            partitions.append(
                {
                    "strategy": strategy,
                    "partition": partition,
                    "common_all_query_10000": self._report_union(
                        bundle["common"]["evidence"]
                    ),
                    "staged_union_with_changed_query_100000": self._report_union(
                        bundle["staged"]["evidence"]
                    ),
                    "staged_additions": bundle["comparison"],
                }
            )
        return {
            "schema_version": 1,
            "stage": "a004_report",
            "fingerprint": FINGERPRINT,
            "scope": "week_01_task_07_read_only_fixed_budget_audit",
            "adjustment_id": "A-004",
            "read_only": True,
            "model_use": "prohibited",
            "task8_membership_use_authorized": False,
            "diagnostic_assignments_unchanged": True,
            "hardware": self._hardware(),
            "result_semantics": {
                "common_result_name": "common_all_query_10000",
                "staged_result_name": "staged_union_with_changed_query_100000",
                "staged_cap_applies_to_all_queries": False,
                "negative_query_meaning": (
                    "no prohibited pair detected through the query's highest "
                    "executed cap"
                ),
            },
            "assignment_balance": self._assignment_balance(),
            "tracks": [self._report_track(track) for track in TRACKS],
            "partition_results": partitions,
            "limitations": [
                "Every prohibited-match numerator is a lower bound under the fixed search budget.",
                "The staged result adds 100000-cap evidence only for changed queries.",
                "Detected overlap is not an exhaustive biological relationship inventory.",
                "Length-distribution differences remain descriptive limitations.",
            ],
        }

    def _report_track(self, track: Track) -> dict[str, object]:
        query = self.query(track[0], track[1])
        caps = {}
        for cap in CAPS:
            high = cap != 1_000
            source = track[3]
            caps[str(cap)] = {
                "source_label": source,
                "query_scope": (
                    "changed_queries_1000_to_10000"
                    if cap == 100_000
                    else "all_queries"
                ),
                "query_count": 1,
                "returned_rows": 1,
                "prohibited_pairs": int(high),
                "prohibited_queries": int(high),
                "prohibited_query_rate": self._rate(int(high)),
                "closest_categories": self._categories(high=high),
                "runtime_seconds": "0.03" if source == "imported_a003" else "0.02",
            }
        return {
            "strategy": track[0],
            "partition": track[1],
            "pass_name": track[2],
            "source_label": track[3],
            "all_query_denominator": 1,
            "changed_query_count_1000_to_10000": 1,
            "caps": caps,
            "cap_sensitivity": self._comparisons(query),
        }

    def _receipt_payload(self, report: dict[str, object]) -> dict[str, object]:
        assignments = {
            name: self.artifact_identity(f"frozen/{name}.json")
            for name in (
                "task5_public",
                "task5_local",
                "task5_report",
                "task6_public",
                "task6_local",
                "task6_report",
            )
        }
        tracks = [self._receipt_track(track) for track in TRACKS]
        imported_fastas = {
            strategy: {
                partition: fasta_evidence(
                    self.bytes(self.fasta_rel(strategy, partition))
                )
                for partition in ("training", "validation", "test")
            }
            for strategy in ("random", "group_aware")
        }
        return {
            "schema_version": 1,
            "stage": "a004_import_receipt",
            "fingerprint": FINGERPRINT,
            "authority": {
                "adjustment_id": "A-004",
                "source_adjustment_id": "A-003",
                "read_only": True,
                "repair_authorized": False,
                "selected_split_authorized": False,
                "model_use": "prohibited",
                "task8_membership_use_authorized": False,
            },
            "code_revision": CODE_REVISION,
            "mmseqs_version": "18-8cc5c",
            "hardware": self._hardware(),
            "source_policy": identity(self.source_policy_bytes),
            "diagnostic_assignments": {
                "unchanged": True,
                "before": assignments,
                "after": assignments,
            },
            "imported_a003": {
                "fingerprint": SOURCE_FINGERPRINT,
                "fastas": imported_fastas,
                "database": {
                    "marker": {"byte_size": 1, "sha256": "4" * 64},
                    "artifact_count": 1,
                },
                "escalated_query_ids": ["R_VALID"],
                "stages": [self.stages[(TRACKS[-1], cap)] for cap in CAPS],
            },
            "fresh_a004_databases": self.databases,
            "imported_tracks": [tracks[-1]],
            "executed_tracks": tracks[:-1],
            "pair_unions": {
                f"{strategy}_{partition}": {
                    "common_all_query_10000": self.unions[(strategy, partition)][
                        "common"
                    ],
                    "staged_union_with_changed_query_100000": self.unions[
                        (strategy, partition)
                    ]["staged"],
                    "comparison": self.unions[(strategy, partition)]["comparison"],
                }
                for strategy, partition in sorted(self.unions)
            },
            "report": report,
        }

    def _receipt_track(self, track: Track) -> dict[str, object]:
        query = self.query(track[0], track[1])
        return {
            "strategy": track[0],
            "partition": track[1],
            "pass_name": track[2],
            "origin": track[3],
            "all_query_count": 1,
            "changed_query_ids_between_1000_and_10000": [query],
            "stages": [self.stages[(track, cap)] for cap in CAPS],
            "cap_sensitivity": self._comparisons(query),
            "cap_summaries": {
                str(cap): self.summaries[(track, cap)] for cap in CAPS
            },
            "pass_marker": self.pass_markers[track],
        }

    @staticmethod
    def _hardware() -> dict[str, object]:
        return {
            "platform": "synthetic-platform",
            "machine": "synthetic-machine",
            "processor": "synthetic-processor",
            "logical_cpu_count": 8,
        }

    @staticmethod
    def _assignment_balance() -> dict[str, object]:
        return {
            strategy: {
                partition: {"records": 1, "residues": 4, "unique_groups": 1}
                for partition in ("training", "validation", "test")
            }
            for strategy in ("random", "group_aware")
        }

    @staticmethod
    def _categories(*, high: bool) -> dict[str, int]:
        categories = {name: 0 for name in CATEGORIES}
        categories[
            "closest_match_prohibited"
            if high
            else "identity_under_30_or_no_residual_hit"
        ] = 1
        return categories

    @staticmethod
    def _rate(numerator: int) -> dict[str, object]:
        return {
            "numerator": numerator,
            "denominator": 1,
            "fraction": "1.00000000" if numerator else "0.00000000",
            "percent": "100.000000" if numerator else "0.000000",
        }

    @staticmethod
    def _comparisons(query: str) -> list[dict[str, object]]:
        return [
            {
                "baseline_cap": 1_000,
                "comparison_cap": 10_000,
                "compared_queries": 1,
                "complete_row_changes": 1,
                "complete_row_change_query_ids": [query],
                "newly_prohibited_queries": 1,
                "no_longer_prohibited_queries": 0,
                "closest_category_changes": 1,
            },
            {
                "baseline_cap": 10_000,
                "comparison_cap": 100_000,
                "compared_queries": 1,
                "complete_row_changes": 0,
                "complete_row_change_query_ids": [],
                "newly_prohibited_queries": 0,
                "no_longer_prohibited_queries": 0,
                "closest_category_changes": 0,
            },
        ]

    @staticmethod
    def _report_union(raw: object) -> dict[str, object]:
        assert isinstance(raw, dict)
        return {
            "prohibited_pairs": raw["unique_pairs"],
            "prohibited_queries": raw["unique_queries"],
            "denominator": 1,
            "rate": GoldenAudit._rate(1),
            "source_labels": raw["source_labels"],
        }

    @staticmethod
    def _query_hash(query: str) -> str:
        return hashlib.sha256(f"{query}\n".encode()).hexdigest()

    @staticmethod
    def _raw(query: str, target: str, *, high: bool) -> bytes:
        fident = "0.60" if high else "0.10"
        return (
            f"{query}\t{target}\t{fident}\t1.0\t1.0\t4\t4\t4\t1\t4\t1\t4\t1e-20\t100\n"
        ).encode()

    @staticmethod
    def _canonical(query: str, target: str, *, high: bool) -> bytes:
        fident = "6e-1" if high else "1e-1"
        return (
            f"{query}\t{target}\t{fident}\t1e0\t1e0\t4\t4\t4\t1\t4\t1\t4\t1e-20\t1e2\n"
        ).encode()

    @staticmethod
    def _cap_evidence(
        cap: int, pair: bytes, summary: bytes, *, high: bool
    ) -> dict[str, object]:
        categories = {name: 0 for name in CATEGORIES}
        categories[
            "closest_match_prohibited"
            if high
            else "identity_under_30_or_no_residual_hit"
        ] = 1
        return {
            "cap": cap,
            "query_count": 1,
            "returned_rows": 1,
            "prohibited_pairs": int(high),
            "prohibited_queries": int(high),
            "closest_categories": categories,
            "prohibited_pair_file": file_evidence(pair, int(high)),
            "query_summary_file": file_evidence(summary, 1),
        }

    @staticmethod
    def _union_record(labels: tuple[str, ...], pair: bytes) -> dict[str, object]:
        return {
            "source_labels": list(labels),
            "unique_pairs": 1,
            "unique_queries": 1,
            "prohibited_pair_file": file_evidence(pair, 1),
        }
