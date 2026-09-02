#!/usr/bin/env python3
"""Build the 4,125-plasmid tRNA-supply table from the target GBFF subset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from Bio import SeqIO


ANTICODON_SEQUENCE = re.compile(r"(?:^|[,\(])seq:([ACGTacgt]{3})(?:,|\))")
COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gbff-root",
        type=Path,
        required=True,
        help="Directory containing one <GCF_ID>/genomic.gbff for each target genome.",
    )
    parser.add_argument(
        "--target-table",
        type=Path,
        default=package_root / "inputs_small" / "01_plasmid_trna_61961.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=package_root / "inputs_small" / "trna_supply_table_rebuilt_from_all.csv",
    )
    return parser.parse_args()


def supported_codon(anticodon: str) -> str:
    return anticodon.translate(COMPLEMENT)[::-1].upper()


def main() -> None:
    args = parse_args()
    targets = pd.read_csv(args.target_table)
    targets["Total_tRNA"] = pd.to_numeric(targets["Total_tRNA"], errors="raise")
    targets = targets.loc[targets["Total_tRNA"].gt(0), ["GCF_ID", "Replicon_Acc"]].copy()
    if len(targets) != 4_125 or targets["Replicon_Acc"].duplicated().any():
        raise ValueError("Expected 4,125 unique tRNA-positive plasmids")

    wanted_by_genome = targets.groupby("GCF_ID")["Replicon_Acc"].agg(set).to_dict()
    rows: list[dict[str, object]] = []
    for genome_id, wanted_accessions in wanted_by_genome.items():
        gbff = args.gbff_root / str(genome_id) / "genomic.gbff"
        if not gbff.is_file():
            raise FileNotFoundError(gbff)
        found: set[str] = set()
        for record in SeqIO.parse(gbff, "genbank"):
            accession = str(record.id)
            if accession not in wanted_accessions:
                continue
            found.add(accession)
            codons: set[str] = set()
            for feature in record.features:
                if feature.type != "tRNA":
                    continue
                qualifiers = feature.qualifiers.get("anticodon", [])
                if len(qualifiers) != 1:
                    continue
                match = ANTICODON_SEQUENCE.search(str(qualifiers[0]).replace(" ", ""))
                if match is None:
                    raise ValueError(f"{accession}: unsupported anticodon qualifier {qualifiers[0]!r}")
                codons.add(supported_codon(match.group(1)))
            rows.append(
                {
                    "Genome_ID": genome_id,
                    "Replicon_ID": accession,
                    "tRNA_codons_set": ",".join(sorted(codons)),
                    "tRNA_type_count": len(codons),
                }
            )
        if found != wanted_accessions:
            raise ValueError(f"{genome_id}: GBFF accessions differ from the target set")

    output = pd.DataFrame(rows).sort_values(["Genome_ID", "Replicon_ID"])
    expected = set(targets["Replicon_Acc"])
    if len(output) != len(expected) or set(output["Replicon_ID"]) != expected:
        raise ValueError("Output does not exactly match the target plasmids")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
