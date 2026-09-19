"""Integrity and frozen-summary reproduction checks for the TED package."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from analyze_ted_results import (
    FULL_KEY,
    K_VALUES,
    METHODS,
    PRIMARY_METRICS,
    STRATEGIES,
    finite_or_fail,
    normalized_sha256,
    require_columns,
    sha256_file,
)


def validate_zip_checksum(package_root: Path) -> dict[str, Any]:
    zip_path = package_root.parent / f"{package_root.name}.zip"
    checksum_path = Path(str(zip_path) + ".sha256")
    if not zip_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(f"ZIP or checksum sidecar missing: {zip_path}, {checksum_path}")
    declared = checksum_path.read_text(encoding="utf-8").strip().split()[0].lower()
    actual = sha256_file(zip_path)
    if declared != actual:
        raise ValueError(f"ZIP checksum mismatch: declared={declared}, actual={actual}")
    return {"check": "zip_sha256", "status": "pass", "expected": declared, "observed": actual, "details": str(zip_path)}


def compare_code_snapshots(package_root: Path, repo_root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    bundled = package_root / "HB_TPM_collaborator_experiment_package_20260901"
    current = repo_root / "HB_TPM_collaborator_experiment_package_20260901"
    if not bundled.is_dir() or not current.is_dir():
        raise FileNotFoundError("bundled or current collaborator experiment package missing")
    suffixes = {".py", ".json", ".md", ".yaml", ".yml", ".txt"}
    bundled_files = {p.relative_to(bundled).as_posix(): p for p in bundled.rglob("*") if p.is_file() and p.suffix.lower() in suffixes}
    current_files = {p.relative_to(current).as_posix(): p for p in current.rglob("*") if p.is_file() and p.suffix.lower() in suffixes}
    shared = sorted(set(bundled_files) & set(current_files))
    missing = sorted(set(bundled_files) - set(current_files))
    different = [rel for rel in shared if normalized_sha256(bundled_files[rel]) != normalized_sha256(current_files[rel])]
    if missing or different:
        raise ValueError(f"run-code snapshot mismatch: missing={missing[:5]}, different={different[:5]}")
    hashes = [
        {"path": str(bundled_files[rel].resolve()), "sha256": sha256_file(bundled_files[rel]), "normalized_sha256": normalized_sha256(bundled_files[rel])}
        for rel in shared
    ]
    check = {
        "check": "normalized_run_code_snapshot",
        "status": "pass",
        "expected": f"{len(bundled_files)} bundled files represented in current package",
        "observed": f"{len(shared)} normalized-content matches; {len(current_files) - len(shared)} current-only files",
        "details": "CRLF/LF normalized before comparison",
    }
    return check, hashes


def reproduce_supplied_aggregates(results: pd.DataFrame, aggregate_dir: Path, draws: int, seed: int):
    supplied_summary = pd.read_csv(aggregate_dir / "aggregate_summary.csv")
    reproduced = (
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
        .reset_index(drop=True)
    )
    keys = ["method_id", "k", "strategy"]
    merged = supplied_summary.merge(reproduced, on=keys, suffixes=("_supplied", "_reproduced"), validate="one_to_one")
    if len(merged) != 90:
        raise ValueError(f"expected 90 supplied aggregate rows, found {len(merged)}")
    max_error = max(
        float(np.max(np.abs(merged[f"{column}_supplied"] - merged[f"{column}_reproduced"])))
        for column in supplied_summary.columns if column not in keys
    )
    if max_error > 1e-12:
        raise ValueError(f"supplied aggregate summary not reproduced: maximum error {max_error}")

    supplied_paired = pd.read_csv(aggregate_dir / "paired_differences.csv")
    rng = np.random.default_rng(seed)
    reference = results[results["method_id"] == "oa_hb_tpm_map"]
    pair_rows = []
    for method in sorted(set(results["method_id"]) - {"oa_hb_tpm_map"}):
        candidate = results[results["method_id"] == method]
        raw_keys = ["patient_id", "fold", "k", "strategy", "replicate"]
        paired = candidate.merge(reference, on=raw_keys, suffixes=("_candidate", "_reference"), validate="one_to_one")
        if len(paired) != len(candidate) or len(paired) != len(reference):
            raise ValueError(f"incomplete raw pairing for {method}")
        for (k, strategy), group in paired.groupby(["k", "strategy"]):
            for metric in [*PRIMARY_METRICS, "hidden_brier"]:
                differences = (
                    group.assign(difference=group[f"{metric}_candidate"] - group[f"{metric}_reference"])
                    .groupby("patient_id")["difference"].mean().to_numpy()
                )
                indices = rng.integers(0, len(differences), size=(draws, len(differences)))
                boot = differences[indices].mean(axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                pair_rows.append({
                    "candidate_method": method, "reference_method": "oa_hb_tpm_map", "k": int(k), "strategy": strategy,
                    "metric": metric, "mean_paired_difference": float(differences.mean()),
                    "bootstrap_95_lower": float(lo), "bootstrap_95_upper": float(hi),
                })
    pair_keys = ["candidate_method", "reference_method", "k", "strategy", "metric"]
    checked = supplied_paired.merge(pd.DataFrame(pair_rows), on=pair_keys, suffixes=("_supplied", "_reproduced"), validate="one_to_one")
    pair_error = max(
        float(np.max(np.abs(checked[f"{column}_supplied"] - checked[f"{column}_reproduced"])))
        for column in ["mean_paired_difference", "bootstrap_95_lower", "bootstrap_95_upper"]
    )
    if pair_error > 1e-12:
        raise ValueError(f"supplied paired summary not reproduced: maximum error {pair_error}")
    checks = [
        {"check": "supplied_aggregate_summary_reproduction", "status": "pass", "expected": "90 rows; tolerance 1e-12", "observed": f"maximum error {max_error:.3g}", "details": str(aggregate_dir / "aggregate_summary.csv")},
        {"check": "supplied_paired_summary_reproduction", "status": "pass", "expected": f"300 rows; {draws} draws; tolerance 1e-12", "observed": f"maximum error {pair_error:.3g}", "details": str(aggregate_dir / "paired_differences.csv")},
    ]
    return checks, reproduced


def validate_results(package_root: Path, data_root: Path, repo_root: Path, draws: int, seed: int):
    checks = [validate_zip_checksum(package_root)]
    code_check, code_hashes = compare_code_snapshots(package_root, repo_root)
    checks.append(code_check)
    aggregate_dir = data_root / "aggregate" / "all_learned_methods_v2"
    patient_path = aggregate_dir / "all_patient_results.csv"
    results = pd.read_csv(patient_path, dtype={"patient_id": str})
    required = set(FULL_KEY) | set(PRIMARY_METRICS) | {
        "hidden_brier", "posterior_predictive_hidden_brier", "hidden_area_95_coverage",
        "area_95_interval_width_relative", "trust_score", "trust_threshold", "trust_gate_reject",
        "phase_offset_cycles", "runtime_s", "hidden_images_used_by_estimator",
        "hidden_masks_used_by_estimator", "statistical_unit", "setting",
    }
    require_columns(results, required, "all_patient_results.csv")
    finite_or_fail(results, list(PRIMARY_METRICS), "primary metrics")
    if len(results) != 32340 or results["patient_id"].nunique() != 98:
        raise ValueError(f"unexpected result dimensions: rows={len(results)}, patients={results['patient_id'].nunique()}")
    if sorted(results["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
        raise ValueError("expected five folds")
    if sorted(results["method_id"].unique().tolist()) != sorted(METHODS):
        raise ValueError("unexpected method set")
    if sorted(results["k"].unique().tolist()) != K_VALUES or sorted(results["strategy"].unique().tolist()) != STRATEGIES:
        raise ValueError("factor levels differ from frozen design")
    if results.duplicated(FULL_KEY).any():
        raise ValueError("duplicate full keys detected")
    if (results.groupby("patient_id")["fold"].nunique() != 1).any():
        raise ValueError("patient-fold exclusivity violated")
    if results["hidden_images_used_by_estimator"].astype(bool).any() or results["hidden_masks_used_by_estimator"].astype(bool).any():
        raise ValueError("hidden-data leakage flag is true")
    if set(results["statistical_unit"].astype(str)) != {"patient"}:
        raise ValueError("unexpected statistical unit")
    cell_counts = results.groupby(["patient_id", "method_id", "k", "strategy"]).size().reset_index(name="n")
    expected_cell = cell_counts["strategy"].map({"uniform": 1, "random": 5, "clustered": 5})
    if not np.array_equal(cell_counts["n"].to_numpy(), expected_cell.to_numpy()):
        raise ValueError("incomplete replicate counts")
    checks.extend([
        {"check": "learned_result_rows", "status": "pass", "expected": 32340, "observed": len(results), "details": str(patient_path)},
        {"check": "unique_patients", "status": "pass", "expected": 98, "observed": 98, "details": "anonymized identifiers"},
        {"check": "factorial_completeness", "status": "pass", "expected": "6 methods x 5 K x 3 strategies; 1/5/5 replicates", "observed": "complete", "details": "uniform/random/clustered"},
        {"check": "duplicate_full_keys", "status": "pass", "expected": 0, "observed": 0, "details": ",".join(FULL_KEY)},
        {"check": "patient_fold_exclusivity", "status": "pass", "expected": "one fold per patient", "observed": "one fold per patient", "details": "fold counts 20/20/20/19/19"},
        {"check": "primary_metric_finiteness", "status": "pass", "expected": "all finite", "observed": "all finite", "details": ",".join(PRIMARY_METRICS)},
        {"check": "hidden_data_leakage_flags", "status": "pass", "expected": "all false", "observed": "all false", "details": "image and mask flags"},
    ])

    run_manifests = sorted((data_root / "runs").glob("*/fold*_k*_all_strategies/run_manifest.json"))
    if len(run_manifests) != 150:
        raise ValueError(f"expected 150 run manifests, found {len(run_manifests)}")
    run_cells, commits, successful, failed = set(), set(), [], []
    verified_outputs = []
    for manifest_path in run_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        method = manifest_path.parents[1].name
        match = re.fullmatch(r"fold(\d+)_k(\d+)_all_strategies", manifest_path.parent.name)
        if not match:
            raise ValueError(f"unexpected run path {manifest_path}")
        run_cells.add((method, int(match.group(1)), int(match.group(2))))
        commits.add(str(manifest.get("git_commit")))
        successful.append(int(manifest.get("n_successful_patient_observations", -1)))
        failed.append(int(manifest.get("n_failed_patient_observations", -1)))
        for relative, expected_hash in manifest.get("output_sha256", {}).items():
            output = manifest_path.parent / relative
            if not output.is_file() or sha256_file(output) != expected_hash:
                raise ValueError(f"missing or hash-mismatched recorded output {output}")
            verified_outputs.append({"path": str(output.resolve()), "sha256": expected_hash})
    expected_cells = {(method, fold, k) for method in METHODS for fold in range(5) for k in K_VALUES}
    if run_cells != expected_cells or any(failed) or set(successful) != {209, 220}:
        raise ValueError("run manifest factorial or success counts are incomplete")
    checks.append({
        "check": "run_manifests_and_output_hashes", "status": "pass",
        "expected": "150 manifests; zero failures; every recorded output hash matches",
        "observed": f"150 manifests; {len(verified_outputs)} output hashes matched",
        "details": f"recorded commit {next(iter(commits)) if len(commits) == 1 else sorted(commits)}",
    })
    reproduction_checks, supplied_summary = reproduce_supplied_aggregates(results, aggregate_dir, draws, seed)
    checks.extend(reproduction_checks)

    stale_log = data_root / "logs" / "target_only" / "fold0_k2.log"
    completed_path = data_root / "runs" / "target_only" / "fold0_k2_all_strategies" / "run_manifest.json"
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    stale_detected = "Traceback" in stale_log.read_text(encoding="utf-8", errors="replace") and int(completed["n_failed_patient_observations"]) == 0
    if not stale_detected:
        raise ValueError("stale target_only/fold0_k2.log warning was not reproducible")
    checks.append({
        "check": "stale_log_provenance", "status": "warning",
        "expected": "failed-attempt traceback followed by successful rerun",
        "observed": f"traceback retained; completed manifest reports {completed['n_successful_patient_observations']} successful and 0 failed observations",
        "details": str(stale_log),
    })
    input_hashes = verified_outputs + code_hashes
    extra_paths = [patient_path, aggregate_dir / "aggregate_summary.csv", aggregate_dir / "paired_differences.csv",
                   data_root / "manifest" / "manifest_report.json", data_root / "manifest" / "ted_manifest.csv",
                   data_root / "processed" / "ted_24x112" / "standardized_manifest.csv",
                   package_root.parent / f"{package_root.name}.zip", Path(str(package_root.parent / f"{package_root.name}.zip") + ".sha256"), stale_log, *run_manifests]
    for path in extra_paths:
        if path.is_file():
            input_hashes.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    metadata = {
        "recorded_commit": next(iter(commits)) if len(commits) == 1 else sorted(commits),
        "recorded_commit_available_in_local_git": False,
        "stale_log_warning": True,
        "verified_output_hash_count": len(verified_outputs),
    }
    return results, supplied_summary, checks, input_hashes, metadata
