#!/usr/bin/env python3
"""Aggregate cardiac GPU runs with patient-level paired bootstrap intervals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


KEYS = ["patient_id", "fold", "k", "strategy", "replicate"]
METRICS = (
    "mean_hidden_dice",
    "mean_hidden_hd95_px",
    "area_curve_relative_rmse",
    "hidden_brier",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_interval(
    values: np.ndarray, *, draws: int, rng: np.random.Generator
) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(int(draws), len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-method", default="oa_hb_tpm_map")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_root.rglob("patient_results.csv"))
    if not files:
        raise FileNotFoundError(
            f"no patient_results.csv found below {args.input_root}"
        )
    frames = [pd.read_csv(path, dtype={"patient_id": str}) for path in files]
    results = pd.concat(frames, ignore_index=True)
    required = set(KEYS) | {"method_id", "trust_gate_reject", *METRICS}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"result files missing columns: {missing}")
    duplicate_key = ["method_id", *KEYS]
    duplicated = results.duplicated(duplicate_key, keep=False)
    if duplicated.any():
        example = results.loc[duplicated, duplicate_key].head().to_dict("records")
        raise ValueError(f"duplicate patient observations across runs: {example}")
    results.to_csv(args.output_dir / "all_patient_results.csv", index=False)

    summary = (
        results.groupby(["method_id", "k", "strategy"], as_index=False)
        .agg(
            n_patients=("patient_id", "nunique"),
            n_patient_observations=("patient_id", "size"),
            mean_hidden_dice=("mean_hidden_dice", "mean"),
            mean_hidden_hd95_px=("mean_hidden_hd95_px", "mean"),
            mean_area_curve_relative_rmse=("area_curve_relative_rmse", "mean"),
            mean_hidden_brier=("hidden_brier", "mean"),
            trust_gate_rejection_rate=("trust_gate_reject", "mean"),
        )
        .sort_values(["k", "strategy", "method_id"])
    )
    summary.to_csv(args.output_dir / "aggregate_summary.csv", index=False)

    reference = results[results["method_id"] == args.reference_method]
    if reference.empty:
        paired = pd.DataFrame()
    else:
        rng = np.random.default_rng(args.seed)
        rows: list[dict[str, object]] = []
        for method in sorted(set(results["method_id"]) - {args.reference_method}):
            candidate = results[results["method_id"] == method]
            merged = candidate.merge(
                reference,
                on=KEYS,
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            if len(merged) != len(candidate) or len(merged) != len(reference):
                raise ValueError(
                    f"incomplete pairing for {method} vs {args.reference_method}: "
                    f"candidate={len(candidate)}, reference={len(reference)}, "
                    f"paired={len(merged)}"
                )
            for (k, strategy), group in merged.groupby(["k", "strategy"]):
                for metric in METRICS:
                    group = group.copy()
                    group["difference"] = (
                        group[f"{metric}_candidate"]
                        - group[f"{metric}_reference"]
                    )
                    patient_differences = (
                        group.groupby("patient_id")["difference"].mean().to_numpy()
                    )
                    lower, upper = bootstrap_interval(
                        patient_differences,
                        draws=args.bootstrap_draws,
                        rng=rng,
                    )
                    rows.append(
                        {
                            "candidate_method": method,
                            "reference_method": args.reference_method,
                            "k": int(k),
                            "strategy": str(strategy),
                            "metric": metric,
                            "difference_definition": "candidate_minus_reference",
                            "n_paired_patients": len(patient_differences),
                            "mean_paired_difference": float(
                                patient_differences.mean()
                            ),
                            "bootstrap_95_lower": lower,
                            "bootstrap_95_upper": upper,
                            "bootstrap_unit": "patient",
                        }
                    )
        paired = pd.DataFrame(rows)
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(args.input_root.resolve()),
        "input_files": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        "reference_method": args.reference_method,
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "statistical_unit": "patient",
        "n_rows": len(results),
        "n_unique_patients": int(results["patient_id"].nunique()),
        "methods": sorted(results["method_id"].unique().tolist()),
    }
    (args.output_dir / "aggregation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Aggregated {len(files)} run files into {args.output_dir}")


if __name__ == "__main__":
    main()
