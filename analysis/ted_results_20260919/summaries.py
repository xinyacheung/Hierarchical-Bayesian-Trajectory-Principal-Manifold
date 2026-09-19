"""Patient-level summaries, diagnostics, and claim mapping for TED results."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_ted_results import (
    ABLATIONS,
    ALL_SUMMARY_METRICS,
    CONTRASTS,
    K_VALUES,
    METHODS,
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    STRATEGIES,
    analyze_contrasts,
    build_canonical_patient_table,
    fold_stratified_bootstrap_mean,
    load_checkpoint_state_without_torch,
    pair_methods,
)


def method_summary(canonical: pd.DataFrame) -> pd.DataFrame:
    metrics = [*PRIMARY_METRICS, "hidden_brier", "posterior_predictive_hidden_brier", "hidden_area_95_coverage", "area_95_interval_width_relative", "trust_gate_reject"]
    result = canonical.groupby("method_id", as_index=False)[metrics].mean()
    result = result.rename(columns={"trust_gate_reject": "trust_gate_rejection_rate"})
    result["weighting"] = "equal patient x K x strategy cells"
    return result.sort_values("mean_hidden_dice", ascending=False).reset_index(drop=True)


def full_cell_summary(canonical: pd.DataFrame) -> pd.DataFrame:
    metrics = [*PRIMARY_METRICS, "hidden_brier", "posterior_predictive_hidden_brier", "hidden_area_95_coverage", "area_95_interval_width_relative", "trust_gate_reject", "trust_score", "runtime_s"]
    grouped = canonical.groupby(["method_id", "k", "strategy"], as_index=False)
    out = grouped[metrics].mean().rename(columns={"trust_gate_reject": "trust_gate_rejection_rate"})
    out.insert(3, "n_patients", grouped["patient_id"].nunique()["patient_id"])
    return out


def bootstrap_method_profiles(canonical: pd.DataFrame, draws: int, seed: int):
    rng = np.random.default_rng(seed + 101)
    rows = []
    for method in METHODS:
        for k in K_VALUES:
            group = canonical[(canonical["method_id"] == method) & (canonical["k"] == k)]
            for metric in PRIMARY_METRICS:
                patient = group.groupby(["patient_id", "fold"], as_index=False)[metric].mean()
                lo, hi = fold_stratified_bootstrap_mean(patient, metric, draws, rng)
                rows.append({"method_id": method, "k": k, "metric": metric, "mean": float(patient[metric].mean()), "bootstrap_95_lower": lo, "bootstrap_95_upper": hi})
    k_profile = pd.DataFrame(rows)
    strategy = canonical.groupby(["method_id", "strategy"], as_index=False)[[*PRIMARY_METRICS, "hidden_brier", "trust_gate_reject"]].mean()
    return k_profile, strategy.rename(columns={"trust_gate_reject": "trust_gate_rejection_rate"})


def cell_rankings(canonical: pd.DataFrame) -> pd.DataFrame:
    cell = canonical.groupby(["method_id", "k", "strategy"], as_index=False)[list(PRIMARY_METRICS)].mean()
    rows = []
    for (k, strategy), group in cell.groupby(["k", "strategy"], sort=True):
        for metric, spec in PRIMARY_METRICS.items():
            ranked = group[["method_id", metric]].copy()
            ranked["rank"] = ranked[metric].rank(method="min", ascending=not spec["higher_is_better"]).astype(int)
            for record in ranked.sort_values("rank").to_dict("records"):
                rows.append({"k": int(k), "strategy": strategy, "metric": metric, "method_id": record["method_id"], "mean": record[metric], "rank": record["rank"]})
    return pd.DataFrame(rows)


def fold_diagnostics(canonical: pd.DataFrame, primary_contrasts: pd.DataFrame):
    fold_level = canonical.groupby(["method_id", "fold"], as_index=False)[list(PRIMARY_METRICS)].mean()
    heterogeneity_rows = []
    for method, group in fold_level.groupby("method_id"):
        for metric in PRIMARY_METRICS:
            values = group[metric]
            heterogeneity_rows.append({"method_id": method, "metric": metric, "fold_min": values.min(), "fold_max": values.max(), "fold_range": values.max() - values.min(), "fold_sd": values.std(ddof=1)})
    loo_rows = []
    for contrast, (candidate, reference) in CONTRASTS.items():
        paired = pair_methods(canonical, candidate, reference, list(PRIMARY_METRICS))
        for left_out in range(5):
            subset = paired[paired["fold"] != left_out]
            for metric in PRIMARY_METRICS:
                patient = subset.groupby(["patient_id", "fold"], as_index=False)[f"{metric}_difference"].mean()
                loo_rows.append({"contrast": contrast, "candidate_method": candidate, "reference_method": reference, "left_out_fold": left_out, "metric": metric, "mean_paired_difference": patient[f"{metric}_difference"].mean(), "n_patients": patient["patient_id"].nunique()})
    return fold_level, pd.DataFrame(heterogeneity_rows), pd.DataFrame(loo_rows)


def uncertainty_summary(canonical: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    metrics = ["hidden_brier", "posterior_predictive_hidden_brier", "hidden_area_95_coverage", "area_95_interval_width_relative"]
    rng = np.random.default_rng(seed + 211)
    rows = []
    for method in METHODS:
        group = canonical[canonical["method_id"] == method]
        for metric in metrics:
            valid = group.dropna(subset=[metric])
            if valid.empty:
                rows.append({"method_id": method, "metric": metric, "n_patients": 0, "mean": np.nan, "bootstrap_95_lower": np.nan, "bootstrap_95_upper": np.nan, "nominal_target": np.nan})
                continue
            patient = valid.groupby(["patient_id", "fold"], as_index=False)[metric].mean()
            lo, hi = fold_stratified_bootstrap_mean(patient, metric, draws, rng)
            rows.append({"method_id": method, "metric": metric, "n_patients": patient["patient_id"].nunique(), "mean": patient[metric].mean(), "bootstrap_95_lower": lo, "bootstrap_95_upper": hi, "nominal_target": 0.95 if metric == "hidden_area_95_coverage" else np.nan})
    return pd.DataFrame(rows)


def _vectorized_bootstrap_correlation(x: np.ndarray, y: np.ndarray, draws: int, rng: np.random.Generator):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    observed = float(np.corrcoef(rx, ry)[0, 1])
    indices = rng.integers(0, len(rx), size=(draws, len(rx)))
    bx, by = rx[indices], ry[indices]
    bx = bx - bx.mean(axis=1, keepdims=True)
    by = by - by.mean(axis=1, keepdims=True)
    denominator = np.sqrt((bx * bx).sum(axis=1) * (by * by).sum(axis=1))
    correlations = np.divide((bx * by).sum(axis=1), denominator, out=np.full(draws, np.nan), where=denominator > 0)
    lo, hi = np.nanquantile(correlations, [0.025, 0.975])
    return observed, float(lo), float(hi)


def trust_gate_diagnostics(results: pd.DataFrame, canonical: pd.DataFrame, draws: int, seed: int):
    rates = (
        results.groupby(["method_id", "k", "strategy"], as_index=False)
        .agg(n_patient_observations=("patient_id", "size"), n_patients=("patient_id", "nunique"), rejection_rate=("trust_gate_reject", "mean"), trust_score_mean=("trust_score", "mean"), trust_score_median=("trust_score", "median"), threshold_mean=("trust_threshold", "mean"))
    )
    quantiles = results.groupby(["method_id", "k", "strategy"])["trust_score"].quantile([0.05, 0.25, 0.75, 0.95]).unstack().reset_index()
    quantiles = quantiles.rename(columns={0.05: "trust_score_q05", 0.25: "trust_score_q25", 0.75: "trust_score_q75", 0.95: "trust_score_q95"})
    rates = rates.merge(quantiles, on=["method_id", "k", "strategy"], validate="one_to_one")

    accepted_rows = []
    for (method, strategy), group in results.groupby(["method_id", "strategy"], sort=True):
        for metric in PRIMARY_METRICS:
            accepted = group.loc[~group["trust_gate_reject"].astype(bool), metric]
            rejected = group.loc[group["trust_gate_reject"].astype(bool), metric]
            accepted_rows.append({
                "method_id": method, "strategy": strategy, "metric": metric,
                "n_accepted": len(accepted), "n_rejected": len(rejected),
                "accepted_mean": accepted.mean() if len(accepted) else np.nan,
                "rejected_mean": rejected.mean() if len(rejected) else np.nan,
                "rejected_minus_accepted": rejected.mean() - accepted.mean() if len(accepted) and len(rejected) else np.nan,
            })
    accepted = pd.DataFrame(accepted_rows)

    rng = np.random.default_rng(seed + 307)
    correlation_rows = []
    for (method, strategy), group in canonical.groupby(["method_id", "strategy"], sort=True):
        for metric in PRIMARY_METRICS:
            patient = group.groupby("patient_id", as_index=False)[["trust_score", metric]].mean().dropna()
            error = 1.0 - patient[metric].to_numpy() if PRIMARY_METRICS[metric]["higher_is_better"] else patient[metric].to_numpy()
            rho, lo, hi = _vectorized_bootstrap_correlation(patient["trust_score"].to_numpy(), error, draws, rng)
            correlation_rows.append({"method_id": method, "strategy": strategy, "metric": metric, "n_patients": len(patient), "spearman_rho_trust_vs_error": rho, "bootstrap_95_lower": lo, "bootstrap_95_upper": hi, "bootstrap_unit": "patient"})
    return rates, accepted, pd.DataFrame(correlation_rows)


def training_diagnostics(data_root: Path) -> pd.DataFrame:
    rows = []
    for manifest_path in sorted((data_root / "runs").glob("*/fold*_k*_all_strategies/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = json.loads((manifest_path.parent / "config_resolved.json").read_text(encoding="utf-8"))
        history_path = manifest_path.parent / "history.csv"
        history = pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame()
        row = {
            "method_id": manifest_path.parents[1].name,
            "fold": int(config.get("fold", re.search(r"fold(\d+)", manifest_path.parent.name).group(1))),
            "k": int(config.get("k", re.search(r"_k(\d+)_", manifest_path.parent.name).group(1))),
            "best_epoch": manifest.get("best_epoch"),
            "best_validation_patient_mean_dice": manifest.get("best_validation_patient_mean_dice"),
            "n_history_epochs": len(history),
            "final_validation_mean_full_cycle_dice": history["validation_mean_full_cycle_dice"].iloc[-1] if "validation_mean_full_cycle_dice" in history and len(history) else np.nan,
            "minimum_validation_total": history["validation_total"].min() if "validation_total" in history and len(history) else np.nan,
            "final_validation_total": history["validation_total"].iloc[-1] if "validation_total" in history and len(history) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def checkpoint_diagnostics(data_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((data_root / "runs").glob("*/fold*_k*_all_strategies/checkpoints/best_model.pt")):
        method = path.parents[2].name
        match = re.fullmatch(r"fold(\d+)_k(\d+)_all_strategies", path.parents[1].name)
        state = load_checkpoint_state_without_torch(path)
        prior_mean_keys = [key for key in state if key.endswith("prior_mean")]
        chol_keys = [key for key in state if "prior_cholesky_unconstrained" in key]
        iso_keys = [key for key in state if "prior_log_variance" in key]
        harmonic_fraction = np.nan
        prior_mean_l2 = np.nan
        if prior_mean_keys:
            mean = np.asarray(state[prior_mean_keys[0]], dtype=float)
            prior_mean_l2 = float(np.linalg.norm(mean))
            if mean.ndim >= 2 and mean.shape[0] > 1:
                total = float(np.sum(mean * mean))
                harmonic_fraction = float(np.sum(mean[1:] * mean[1:]) / total) if total > 0 else 0.0
        covariance_trace = covariance_condition = mean_abs_correlation = np.nan
        if chol_keys:
            raw = np.asarray(state[chol_keys[0]], dtype=float)
            if raw.ndim == 2:
                raw = raw[None, ...]
            traces, conditions, correlations = [], [], []
            for matrix in raw:
                lower = np.tril(matrix.copy())
                diag = np.diag_indices_from(lower)
                x = lower[diag]
                lower[diag] = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0) + 1e-4
                covariance = lower @ lower.T
                eig = np.linalg.eigvalsh(covariance)
                traces.append(np.trace(covariance))
                conditions.append(float(eig.max() / max(eig.min(), 1e-12)))
                sd = np.sqrt(np.diag(covariance))
                corr = covariance / np.outer(sd, sd)
                correlations.append(np.mean(np.abs(corr[np.triu_indices_from(corr, 1)])))
            covariance_trace = float(np.mean(traces))
            covariance_condition = float(np.median(conditions))
            mean_abs_correlation = float(np.mean(correlations))
        elif iso_keys:
            variance = np.exp(np.asarray(state[iso_keys[0]], dtype=float))
            covariance_trace = float(np.mean(variance) * 7.0)
            covariance_condition = 1.0
            mean_abs_correlation = 0.0
        rows.append({
            "method_id": method, "fold": int(match.group(1)), "k": int(match.group(2)),
            "prior_mean_l2": prior_mean_l2, "prior_harmonic_energy_fraction": harmonic_fraction,
            "mean_prior_covariance_trace": covariance_trace, "median_prior_covariance_condition_number": covariance_condition,
            "mean_absolute_prior_correlation": mean_abs_correlation, "checkpoint_tensor_count": len(state),
        })
    return pd.DataFrame(rows)


def model_behavior(canonical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in canonical.groupby("method_id", sort=True):
        k_profile = group.groupby("k")[list(PRIMARY_METRICS)].mean()
        for metric, spec in PRIMARY_METRICS.items():
            values = k_profile[metric]
            delta = float(values.loc[10] - values.loc[2])
            improvement = delta if spec["higher_is_better"] else -delta
            rows.append({"method_id": method, "diagnostic": "K10_minus_K2", "metric": metric, "value": delta, "improvement_oriented_value": improvement})
        rows.extend([
            {"method_id": method, "diagnostic": "mean_absolute_phase_offset_cycles", "metric": "phase_offset_cycles", "value": float(group["phase_offset_cycles"].abs().mean()), "improvement_oriented_value": np.nan},
            {"method_id": method, "diagnostic": "across_cell_area_rmse_range", "metric": "area_curve_relative_rmse", "value": float(group.groupby(["k", "strategy"])["area_curve_relative_rmse"].mean().max() - group.groupby(["k", "strategy"])["area_curve_relative_rmse"].mean().min()), "improvement_oriented_value": np.nan},
        ])
    return pd.DataFrame(rows)


def oracle_analysis(data_root: Path, learned_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    oracle_dir = data_root.parent.parent / "oracle" / "aggregate"
    if not oracle_dir.is_dir():
        oracle_dir = data_root / "oracle" / "aggregate"
    oracle_path = oracle_dir / "all_patient_results.csv"
    if not oracle_path.is_file():
        raise FileNotFoundError(f"oracle patient table missing: {oracle_path}")
    oracle = pd.read_csv(oracle_path, dtype={"patient_id": str})
    oracle_summary = oracle.groupby(["k", "strategy"], as_index=False)[list(PRIMARY_METRICS)].mean()
    oracle_global = oracle_summary[list(PRIMARY_METRICS)].mean()
    best_rows = []
    for metric, spec in PRIMARY_METRICS.items():
        sorted_learned = learned_summary.sort_values(metric, ascending=not spec["higher_is_better"])
        best = sorted_learned.iloc[0]
        oracle_value = float(oracle_global[metric])
        learned_value = float(best[metric])
        gap = oracle_value - learned_value if spec["higher_is_better"] else learned_value - oracle_value
        best_rows.append({"metric": metric, "oracle_mean": oracle_value, "best_learned_method": best["method_id"], "best_learned_mean": learned_value, "oracle_advantage_improvement_oriented": gap, "oracle_is_end_to_end": False})
    return oracle_summary, pd.DataFrame(best_rows)


def claim_status(primary: pd.DataFrame, uncertainty: pd.DataFrame, trust_rates: pd.DataFrame, oracle_gap: pd.DataFrame) -> pd.DataFrame:
    global_primary = primary[(primary["scope"] == "global") & (primary["metric"].isin(PRIMARY_METRICS))]
    rows = []
    source = global_primary[global_primary["contrast"] == "source_prior_contribution"]
    source_status = "mixed/inconclusive" if (source["interval_interpretation"] == "inconclusive").any() else "supported"
    rows.append({"claim": "Source-prior benefit (OA-HB-TPM MAP vs target-only)", "status": source_status, "evidence": "tables/primary_paired_contrasts.csv", "boundary": "Patient-paired global primary metrics; no clinical threshold prespecified"})
    likelihood = global_primary[global_primary["contrast"] == "observation_likelihood_vs_latent_mse"]
    likelihood_status = "not supported; latent-MSE is better" if (likelihood["interval_interpretation"] == "harmful").any() else "inconclusive"
    rows.append({"claim": "Observation-aware likelihood benefit over latent-MSE", "status": likelihood_status, "evidence": "tables/primary_paired_contrasts.csv", "boundary": "Training objective comparison within delivered learned methods"})
    update = global_primary[global_primary["contrast"] == "target_specific_updating"]
    update_status = "not supported" if (update["interval_interpretation"] == "harmful").any() else "inconclusive"
    rows.append({"claim": "Target-specific posterior updating improves on source mean", "status": update_status, "evidence": "tables/primary_paired_contrasts.csv", "boundary": "Source mean has no posterior interval outputs"})
    coverage = uncertainty[(uncertainty["metric"] == "hidden_area_95_coverage") & (uncertainty["method_id"].isin(["oa_hb_tpm_map", "hb_tpm_latent_mse"]))]
    rows.append({"claim": "Nominal 95% posterior area intervals are calibrated", "status": "not supported", "evidence": "tables/uncertainty_summary.csv", "boundary": f"Observed coverage ranges {coverage['mean'].min():.3f}-{coverage['mean'].max():.3f}"})
    map_rates = trust_rates[trust_rates["method_id"] == "oa_hb_tpm_map"].groupby("strategy")["rejection_rate"].mean()
    rows.append({"claim": "Trust gate is sensitive to sampling pattern", "status": "supported descriptively", "evidence": "tables/trust_gate_rates.csv", "boundary": f"Uniform/random/clustered rejection {map_rates.get('uniform', np.nan):.3f}/{map_rates.get('random', np.nan):.3f}/{map_rates.get('clustered', np.nan):.3f}; family shift not tested"})
    rows.append({"claim": "Oracle gap identifies image-to-latent estimation as the dominant bottleneck", "status": "supported as a diagnostic inference", "evidence": "tables/oracle_upper_bound.csv", "boundary": "Oracle uses manual contours and is not an end-to-end comparator"})
    rows.append({"claim": "Generalization under acquisition/vendor/family shift", "status": "not assessable", "evidence": "tables/missing_deliverables.csv", "boundary": "No family-shift split or external cohort was delivered"})
    return pd.DataFrame(rows)


def missing_deliverables() -> pd.DataFrame:
    rows = [
        ("Classical trajectory baselines", "absent", "Linear/spline/GP or other frozen-design classical comparators were not delivered", "No superiority claim versus classical methods"),
        ("Family-shift validation", "absent", "No held-out family/vendor/acquisition-shift test is present", "Trust-gate evidence is limited to sampling-pattern sensitivity"),
        ("ED/ES timing error", "absent", "No ED/ES timing metric appears in patient results", "Do not claim improved cardiac event timing"),
        ("Temporal jerk/smoothness secondary outcome", "absent", "No jerk metric appears in delivered results", "Do not claim improved physiological smoothness from outcomes"),
        ("Raw image arrays", "absent", "Package contains derived results, checkpoints, and manifests but not target image arrays", "Inference cannot be independently rerun from this package"),
        ("Clinical significance threshold", "not prespecified", "No minimally important difference is defined", "Report statistical uncertainty without clinical-significance language"),
    ]
    return pd.DataFrame(rows, columns=["item", "delivery_status", "evidence", "manuscript_boundary"])


def generate_all_tables(results: pd.DataFrame, data_root: Path, draws: int, seed: int):
    canonical = build_canonical_patient_table(results)
    summary = method_summary(canonical)
    cell = full_cell_summary(canonical)
    primary = analyze_contrasts(canonical, CONTRASTS, draws, seed)
    ablations = analyze_contrasts(canonical, ABLATIONS, draws, seed + 17)
    performance_k, strategy = bootstrap_method_profiles(canonical, draws, seed)
    rankings = cell_rankings(canonical)
    fold_level, fold_heterogeneity, fold_loo = fold_diagnostics(canonical, primary)
    uncertainty = uncertainty_summary(canonical, draws, seed)
    trust_rates, trust_errors, trust_correlations = trust_gate_diagnostics(results, canonical, draws, seed)
    training = training_diagnostics(data_root)
    checkpoints = checkpoint_diagnostics(data_root)
    behavior = model_behavior(canonical)
    oracle_cells, oracle_gap = oracle_analysis(data_root, summary)
    claims = claim_status(primary, uncertainty, trust_rates, oracle_gap)
    missing = missing_deliverables()
    tables = OrderedDict([
        ("canonical_patient_cells", canonical),
        ("method_summary_equal_cell", summary),
        ("full_cell_summary", cell),
        ("primary_paired_contrasts", primary),
        ("secondary_ablations", ablations),
        ("performance_by_k", performance_k),
        ("strategy_interactions", strategy),
        ("per_cell_rankings", rankings),
        ("fold_level_performance", fold_level),
        ("fold_heterogeneity", fold_heterogeneity),
        ("leave_one_fold_out", fold_loo),
        ("uncertainty_summary", uncertainty),
        ("trust_gate_rates", trust_rates),
        ("trust_gate_accepted_vs_rejected", trust_errors),
        ("trust_gate_correlations", trust_correlations),
        ("training_diagnostics", training),
        ("checkpoint_prior_diagnostics", checkpoints),
        ("model_behavior_diagnostics", behavior),
        ("oracle_cell_summary", oracle_cells),
        ("oracle_upper_bound", oracle_gap),
        ("claim_status", claims),
        ("missing_deliverables", missing),
    ])
    return tables
