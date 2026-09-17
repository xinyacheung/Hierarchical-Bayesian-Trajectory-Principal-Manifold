#!/usr/bin/env python3
"""Run the P0 observation-aware HB-TPM photon benchmark.

The runner simulates a family of smooth diffusion trajectories, fits source
and target records using ordered photon arrivals only, and joins simulation
truth only after every estimator has returned.  The smoke protocol is a code
and data-contract gate, not a final efficacy claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fcs_time_varying_simulation import (  # noqa: E402
    ExponentialDiffusion,
    TimeVaryingSimulationSettings,
    simulate_time_varying,
)
from observation_aware_hb_tpm import (  # noqa: E402
    ObservationAwareEstimate,
    fit_log_linear_event_trajectory,
    fit_observation_aware_target,
    learn_prior_from_event_records,
)


EXPERIMENT_ROOT = (
    PROJECT_ROOT / "experiments" / "07_observation_aware_hb_tpm"
)
DEFAULT_PROTOCOL = EXPERIMENT_ROOT / "settings" / "oa_hb_tpm_smoke.json"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "outputs" / "smoke"
RESULT_SCHEMA_VERSION = "oa-hb-tpm-results-0.1.0"


@dataclass
class SimulatedRecord:
    record_id: str
    role: str
    seed: int
    log_d_truth: np.ndarray
    event_time_s: np.ndarray
    diagnostics: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_ready(dict(payload)),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output directory is nonempty: {path}; pass --overwrite"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "observed_events" / "source").mkdir(parents=True)
    (path / "observed_events" / "target").mkdir(parents=True)
    (path / "simulation_truth").mkdir(parents=True)


def _truth_covariance(protocol: Mapping[str, Any]) -> np.ndarray:
    family = protocol["truth_family"]
    standard_deviations = np.asarray(family["sd_log_endpoints"], dtype=float)
    correlation = float(family["correlation"])
    return np.outer(standard_deviations, standard_deviations) * np.asarray(
        [[1.0, correlation], [correlation, 1.0]], dtype=float
    )


def _simulate_record(
    *,
    role: str,
    index: int,
    seed: int,
    log_d_truth: np.ndarray,
    protocol: Mapping[str, Any],
) -> SimulatedRecord:
    duration_s = float(protocol["duration_s"])
    brightness_key = f"{role}_molecular_brightness_cps"
    schedule = ExponentialDiffusion(
        model_id=f"OA_{role.upper()}_{index:03d}",
        label="observation-aware log-linear trajectory family",
        duration_s=duration_s,
        start_um2_s=float(np.exp(log_d_truth[0])),
        stop_um2_s=float(np.exp(log_d_truth[1])),
    )
    settings = TimeVaryingSimulationSettings(
        schedule=schedule,
        seed=int(seed),
        n_molecules=int(protocol["n_molecules"]),
        wxy_um=float(protocol["wxy_um"]),
        wz_um=float(protocol["wz_um"]),
        molecular_brightness_cps=float(protocol[brightness_key]),
        background_cps=float(protocol["background_cps"]),
        brownian_dt_s=float(protocol["brownian_dt_s"]),
    )
    result = simulate_time_varying(settings)
    return SimulatedRecord(
        record_id=f"{role}_{index:03d}",
        role=role,
        seed=int(seed),
        log_d_truth=np.asarray(log_d_truth, dtype=float),
        event_time_s=result.event_times_s,
        diagnostics=dict(result.diagnostics),
    )


def _likelihood_controls(
    protocol: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    estimator = protocol["estimator"]
    return {
        "wxy_um": float(protocol["wxy_um"]),
        "kappa": float(protocol["wz_um"]) / float(protocol["wxy_um"]),
        "molecular_brightness_cps": float(
            protocol[f"{role}_molecular_brightness_cps"]
        ),
        "background_cps": float(protocol["background_cps"]),
        "d_bounds_um2_s": tuple(estimator["d_bounds_um2_s"]),
        "occupancy_bounds": tuple(estimator["occupancy_bounds"]),
        "state_max": int(estimator["state_max"]),
        "n_refinement_intervals": int(
            estimator["n_refinement_intervals"]
        ),
        "optimizer_maxiter": int(estimator["optimizer_maxiter"]),
        "optimizer_ftol": float(estimator["optimizer_ftol"]),
        "optimizer_gtol": float(estimator["optimizer_gtol"]),
        "hessian_step": float(estimator["hessian_step"]),
        "information_eigenvalue_floor": float(
            estimator["information_eigenvalue_floor"]
        ),
    }


def _curve(log_d: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    values = np.asarray(log_d, dtype=float)
    return np.exp(
        (1.0 - fractions) * values[0] + fractions * values[1]
    )


def _evaluation_row(
    record: SimulatedRecord,
    estimate: ObservationAwareEstimate,
) -> dict[str, Any]:
    fractions = np.linspace(0.0, 1.0, 201)
    truth_curve = _curve(record.log_d_truth, fractions)
    fitted_curve = _curve(np.asarray(estimate.log_d_estimate), fractions)
    relative_curve_rmse = float(
        np.sqrt(np.mean(((fitted_curve - truth_curve) / truth_curve) ** 2))
    )
    ci = np.asarray(estimate.ci95_d_um2_s, dtype=float)
    truth_d = np.exp(record.log_d_truth)
    covered = (truth_d >= ci[:, 0]) & (truth_d <= ci[:, 1])
    return {
        "record_id": record.record_id,
        "method_id": estimate.method_id,
        "seed": record.seed,
        "n_photons": len(record.event_time_s),
        "success": estimate.success,
        "failure_reason": estimate.failure_reason,
        "runtime_s": estimate.runtime_s,
        "truth_d_start_um2_s": truth_d[0],
        "truth_d_end_um2_s": truth_d[1],
        "estimate_d_start_um2_s": estimate.d_estimate_um2_s[0],
        "estimate_d_end_um2_s": estimate.d_estimate_um2_s[1],
        "log_endpoint_squared_error": float(
            np.mean(
                (np.asarray(estimate.log_d_estimate) - record.log_d_truth) ** 2
            )
        ),
        "curve_relative_rmse": relative_curve_rmse,
        "endpoint_coverage": float(np.mean(covered)),
        "prior_used": bool(estimate.diagnostics["prior_used"]),
        "simulation_truth_used_by_estimator": bool(
            estimate.diagnostics["simulation_truth_used"]
        ),
    }


def _make_figure(
    output_path: Path,
    records: Sequence[SimulatedRecord],
    fits_by_record: Mapping[str, Mapping[str, ObservationAwareEstimate]],
    results: pd.DataFrame,
) -> None:
    colors = {
        "truth": "#111827",
        "target_only_event_qmle": "#D7BA68",
        "oa_hb_tpm_map": "#F48A83",
    }
    labels = {
        "target_only_event_qmle": "Target-only event QMLE",
        "oa_hb_tpm_map": "Observation-aware HB-TPM",
    }
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.7))
    fractions = np.linspace(0.0, 1.0, 201)
    representative = records[0]
    axes[0].plot(
        fractions,
        _curve(representative.log_d_truth, fractions),
        color=colors["truth"],
        linestyle="--",
        linewidth=2.0,
        label="Simulation truth (evaluation only)",
    )
    for method_id in ("target_only_event_qmle", "oa_hb_tpm_map"):
        estimate = fits_by_record[representative.record_id][method_id]
        axes[0].plot(
            fractions,
            _curve(np.asarray(estimate.log_d_estimate), fractions),
            color=colors[method_id],
            linewidth=2.4,
            label=labels[method_id],
        )
    axes[0].set_title("(a) Representative sparse target")
    axes[0].set_xlabel("Normalized acquisition time")
    axes[0].set_ylabel(r"$D(t)$ ($\mu$m$^2$/s)")
    axes[0].legend(frameon=False, fontsize=7)

    pivot = results.pivot(
        index="record_id", columns="method_id", values="curve_relative_rmse"
    )
    x = np.arange(len(pivot), dtype=float)
    for index in range(len(pivot)):
        axes[1].plot(
            [0, 1],
            [
                pivot.iloc[index]["target_only_event_qmle"],
                pivot.iloc[index]["oa_hb_tpm_map"],
            ],
            color="#CBD5E1",
            linewidth=1.2,
            zorder=1,
        )
    axes[1].scatter(
        np.zeros(len(pivot)),
        pivot["target_only_event_qmle"],
        color=colors["target_only_event_qmle"],
        edgecolor="#8A6A1B",
        zorder=2,
    )
    axes[1].scatter(
        np.ones(len(pivot)),
        pivot["oa_hb_tpm_map"],
        color=colors["oa_hb_tpm_map"],
        edgecolor="#B9443F",
        zorder=2,
    )
    axes[1].set_xticks([0, 1], ["Target only", "OA-HB-TPM"])
    axes[1].set_ylabel("Relative curve RMSE")
    axes[1].set_title("(b) Paired target records")

    summary = results.groupby("method_id", sort=False).agg(
        curve_rmse=("curve_relative_rmse", "mean"),
        coverage=("endpoint_coverage", "mean"),
    )
    methods = ["target_only_event_qmle", "oa_hb_tpm_map"]
    bars = axes[2].bar(
        np.arange(2),
        [summary.loc[item, "curve_rmse"] for item in methods],
        color=[colors[item] for item in methods],
        edgecolor=["#8A6A1B", "#B9443F"],
        width=0.62,
    )
    for bar, method_id in zip(bars, methods):
        coverage = summary.loc[method_id, "coverage"]
        axes[2].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"coverage {coverage:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[2].set_xticks(np.arange(2), ["Target only", "OA-HB-TPM"])
    axes[2].set_ylabel("Mean relative curve RMSE")
    axes[2].set_title("(c) Smoke summary")

    for axis in axes:
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Observation-aware HB-TPM: source prior and target posterior use photon events only",
        fontsize=12,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run(protocol_path: Path, output_dir: Path, *, overwrite: bool) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _prepare_output(output_dir, overwrite=overwrite)
    shutil.copy2(protocol_path, output_dir / "settings_resolved.json")

    rng = np.random.default_rng(int(protocol["seed"]))
    mean = np.asarray(protocol["truth_family"]["mean_log_endpoints"], dtype=float)
    covariance = _truth_covariance(protocol)
    n_source = int(protocol["source_records"])
    n_target = int(protocol["target_records"])
    all_coefficients = rng.multivariate_normal(mean, covariance, n_source + n_target)
    seeds = rng.integers(1, np.iinfo(np.int32).max, size=n_source + n_target)
    source_records: list[SimulatedRecord] = []
    target_records: list[SimulatedRecord] = []
    for index in range(n_source + n_target):
        role = "source" if index < n_source else "target"
        role_index = index if role == "source" else index - n_source
        record = _simulate_record(
            role=role,
            index=role_index,
            seed=int(seeds[index]),
            log_d_truth=all_coefficients[index],
            protocol=protocol,
        )
        record_frame = pd.DataFrame({"event_time_s": record.event_time_s})
        record_frame.to_csv(
            output_dir
            / "observed_events"
            / role
            / f"{record.record_id}.csv.gz",
            index=False,
            compression="gzip",
        )
        (source_records if role == "source" else target_records).append(record)

    source_controls = _likelihood_controls(protocol, role="source")
    prior, source_fits = learn_prior_from_event_records(
        [record.event_time_s.copy() for record in source_records],
        duration_s=float(protocol["duration_s"]),
        likelihood_controls=source_controls,
        variance_floor=float(protocol["empirical_bayes"]["variance_floor"]),
        covariance_shrinkage=float(
            protocol["empirical_bayes"]["covariance_shrinkage"]
        ),
    )

    # Only after source prior learning is complete do we join source truth for QA.
    source_rows: list[dict[str, Any]] = []
    for record, fit in zip(source_records, source_fits):
        source_rows.append(
            {
                "record_id": record.record_id,
                "seed": record.seed,
                "n_photons": len(record.event_time_s),
                "success": fit.success,
                "estimate_log_d_start": fit.log_d_estimate[0],
                "estimate_log_d_end": fit.log_d_estimate[1],
                "truth_log_d_start_evaluation_only": record.log_d_truth[0],
                "truth_log_d_end_evaluation_only": record.log_d_truth[1],
                "simulation_truth_used_by_estimator": fit.diagnostics[
                    "simulation_truth_used"
                ],
            }
        )
    pd.DataFrame(source_rows).to_csv(
        output_dir / "source_fit_results.csv", index=False
    )
    _write_json(output_dir / "learned_source_prior.json", prior.to_dict())

    target_controls = _likelihood_controls(protocol, role="target")
    evaluation_rows: list[dict[str, Any]] = []
    fits_by_record: dict[str, dict[str, ObservationAwareEstimate]] = {}
    for record in target_records:
        target_only = fit_log_linear_event_trajectory(
            record.event_time_s.copy(),
            duration_s=float(protocol["duration_s"]),
            prior=None,
            **target_controls,
        )
        hierarchical = fit_observation_aware_target(
            record.event_time_s.copy(),
            prior=prior,
            duration_s=float(protocol["duration_s"]),
            likelihood_controls=target_controls,
        )
        fits_by_record[record.record_id] = {
            target_only.method_id: target_only,
            hierarchical.method_id: hierarchical,
        }
        # Truth is joined here, after both fits are immutable results.
        evaluation_rows.extend(
            [
                _evaluation_row(record, target_only),
                _evaluation_row(record, hierarchical),
            ]
        )

    results = pd.DataFrame(evaluation_rows)
    results.to_csv(output_dir / "target_estimator_results.csv", index=False)
    summary = (
        results.groupby("method_id", sort=False)
        .agg(
            n_records=("record_id", "count"),
            success_rate=("success", "mean"),
            log_endpoint_rmse=("log_endpoint_squared_error", lambda x: np.sqrt(np.mean(x))),
            mean_curve_relative_rmse=("curve_relative_rmse", "mean"),
            median_curve_relative_rmse=("curve_relative_rmse", "median"),
            endpoint_coverage=("endpoint_coverage", "mean"),
            mean_runtime_s=("runtime_s", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "method_summary.csv", index=False)

    truth_rows = [
        {
            "record_id": record.record_id,
            "role": record.role,
            "seed": record.seed,
            "truth_log_d_start": record.log_d_truth[0],
            "truth_log_d_end": record.log_d_truth[1],
            "truth_d_start_um2_s": float(np.exp(record.log_d_truth[0])),
            "truth_d_end_um2_s": float(np.exp(record.log_d_truth[1])),
        }
        for record in [*source_records, *target_records]
    ]
    pd.DataFrame(truth_rows).to_csv(
        output_dir / "simulation_truth" / "trajectory_coefficients.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "field": "event_time_s",
                "file_group": "observed_events",
                "estimator_visible": True,
                "unit": "s",
                "description": "Ordered detected photon arrival time",
            },
            {
                "field": "truth_log_d_start/truth_log_d_end",
                "file_group": "simulation_truth",
                "estimator_visible": False,
                "unit": "log(um^2/s)",
                "description": "Simulation-only trajectory coefficients",
            },
        ]
    ).to_csv(output_dir / "data_dictionary.csv", index=False)

    _make_figure(
        output_dir / "figure_observation_aware_hb_tpm_smoke.png",
        target_records,
        fits_by_record,
        results,
    )
    all_invariants = [
        record.diagnostics["all_invariants_ok"]
        for record in [*source_records, *target_records]
    ]
    _write_json(
        output_dir / "benchmark_manifest.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stage": protocol["stage"],
            "protocol": str(protocol_path.resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "truth_separation": protocol["truth_separation"],
            "source_prior_input": "source event_time_s arrays only",
            "target_estimator_input": "target event_time_s array only",
            "simulation_truth_joined_after_all_fits": True,
            "all_simulation_invariants_ok": bool(all(all_invariants)),
            "n_source_records": n_source,
            "n_target_records": n_target,
            "output_files": sorted(
                str(path.relative_to(output_dir))
                for path in output_dir.rglob("*")
                if path.is_file()
            ),
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args.protocol, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
