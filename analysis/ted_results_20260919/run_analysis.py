#!/usr/bin/env python3
"""Run the complete TED results analysis and generate all deliverables."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import PIL
import reportlab

from analyze_ted_results import (
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_SEED,
    resolve_layout,
    sha256_file,
    write_csv,
    write_json,
)
from figures import generate_all_figures
from reporting import create_report
from summaries import generate_all_tables
from validation import validate_results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root / "HB_TPM_TED_results_20260919",
        help="Extracted result package root or private_data/ted_full directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    return parser.parse_args(argv)


def _deduplicate_hashes(records: list[dict]) -> list[dict]:
    by_path = {}
    for record in records:
        by_path[record["path"]] = record
    return [by_path[path] for path in sorted(by_path)]


def _verify_inputs_unchanged(records: list[dict]) -> None:
    for record in records:
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"analysis input disappeared during run: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"analysis input changed during run: {path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.bootstrap_draws < 100:
        raise ValueError("--bootstrap-draws must be at least 100")
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    package_root, data_root = resolve_layout(args.results_root)
    output_dir = args.output_dir.resolve()
    if output_dir == package_root or package_root in output_dir.parents:
        raise ValueError("output directory must not be inside the extracted read-only results package")
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    results, supplied_summary, checks, input_hashes, validation_meta = validate_results(
        package_root, data_root, repo_root, args.bootstrap_draws, args.seed
    )
    table_map = generate_all_tables(results, data_root, args.bootstrap_draws, args.seed)
    data_integrity = pd.DataFrame(checks)
    table_map["data_integrity"] = data_integrity
    table_map["supplied_aggregate_reproduction"] = supplied_summary
    for name, frame in table_map.items():
        write_csv(frame, tables_dir / f"{name}.csv")

    provenance = generate_all_figures(table_map, figures_dir, args.seed)
    provenance["output_sha256"] = provenance["output_file"].map(lambda name: sha256_file(figures_dir / name))
    provenance["input_table_sha256"] = provenance["input_tables"].map(
        lambda value: ";".join(
            f"{Path(item).name}:{sha256_file(output_dir / item)}" for item in str(value).split(";")
        )
    )
    write_csv(provenance, output_dir / "FIGURE_PROVENANCE.csv")

    report_path = output_dir / "analysis_report.md"
    create_report(report_path, table_map, data_integrity, validation_meta, args.seed, args.bootstrap_draws)

    for evidence in table_map["claim_status"]["evidence"]:
        if not (output_dir / evidence).is_file():
            raise FileNotFoundError(f"claim evidence table missing: {evidence}")
    for output in provenance["output_file"]:
        if not (figures_dir / output).is_file():
            raise FileNotFoundError(f"figure output missing: {output}")

    input_hashes = _deduplicate_hashes(input_hashes)
    _verify_inputs_unchanged(input_hashes)
    scripts = [
        script_dir / "run_analysis.py",
        script_dir / "analyze_ted_results.py",
        script_dir / "validation.py",
        script_dir / "summaries.py",
        script_dir / "figures.py",
        script_dir / "figure_factory.py",
        script_dir / "reporting.py",
        script_dir / "test_analysis.py",
    ]
    script_hashes = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in scripts if path.is_file()
    ]
    output_files = sorted(
        [p for p in tables_dir.glob("*.csv")]
        + [p for p in figures_dir.glob("*.png")]
        + [p for p in figures_dir.glob("*.pdf")]
        + [report_path, output_dir / "FIGURE_PROVENANCE.csv"]
    )
    output_hashes = [{"path": str(path.relative_to(output_dir).as_posix()), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in output_files]
    finished = datetime.now(timezone.utc)
    manifest = {
        "analysis_name": "TED Results Analysis and Diagnostic Report",
        "analysis_version": 1,
        "created_utc": finished.isoformat(),
        "started_utc": started.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "results_root": str(package_root),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "bootstrap_draws": args.bootstrap_draws,
        "statistical_unit": "patient",
        "weighting": "replicates averaged within patient-method-K-strategy; five K values and three strategies equally weighted",
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL.__version__,
            "reportlab": reportlab.Version,
            "cwd": os.getcwd(),
        },
        "validation": validation_meta,
        "analysis_scripts": script_hashes,
        "input_files": input_hashes,
        "output_files": output_hashes,
        "manifest_self_hash": "omitted because a file cannot contain its own stable cryptographic hash",
    }
    write_json(manifest, output_dir / "analysis_manifest.json")
    print(f"TED analysis complete: {output_dir}")
    print(f"Validated {len(results):,} learned-method rows and wrote {len(output_files) + 1} outputs.")


if __name__ == "__main__":
    main()
