"""Publication-oriented PNG/PDF figures for the TED analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_ted_results import CONTRASTS, K_VALUES, PRIMARY_METRICS, STRATEGIES
from figure_factory import DISPLAY_NAMES, PALETTE, DualCanvas, legend, map_value, nice_limits, panel_axes


SELECTED_METHODS = ["oa_hb_tpm_map", "hb_tpm_latent_mse", "source_mean", "target_only"]
ALL_METHODS = [*SELECTED_METHODS, "oa_hb_tpm_isotropic_prior", "oa_hb_tpm_no_phase_alignment"]


def _ticks(lo: float, hi: float, count: int = 5, decimals: int = 3):
    values = np.linspace(lo, hi, count)
    return [(float(v), f"{v:.{decimals}f}") for v in values]


def performance_vs_k(frame: pd.DataFrame, figures_dir: Path):
    png, pdf = figures_dir / "performance_vs_k.png", figures_dir / "performance_vs_k.pdf"
    c = DualCanvas(png, pdf, 1800, 1120)
    c.text(70, 38, "End-to-end performance across observation budgets", 40, bold=True)
    c.text(70, 92, "Means give each patient and sampling strategy equal weight; intervals are fold-stratified patient bootstrap 95% CIs.", 23, PALETTE["muted"])
    panels = [(100, 220, 590, 850), (680, 220, 1170, 850), (1260, 220, 1750, 850)]
    for bounds, (metric, spec) in zip(panels, PRIMARY_METRICS.items()):
        data = frame[frame["metric"] == metric]
        lo, hi = nice_limits([*data["bootstrap_95_lower"], *data["bootstrap_95_upper"]], pad=0.08)
        decimals = 3 if metric != "mean_hidden_hd95_px" else 1
        panel_axes(c, bounds, [(k, str(k)) for k in K_VALUES], _ticks(lo, hi, 5, decimals), (2, 10), (lo, hi), spec["label"], "Observed frames (K)")
        left, top, right, bottom = bounds
        for method in ALL_METHODS:
            group = data[data["method_id"] == method].sort_values("k")
            points = []
            for row in group.itertuples():
                x = map_value(row.k, 2, 10, left, right)
                y = map_value(row.mean, lo, hi, bottom, top)
                ylo = map_value(row.bootstrap_95_lower, lo, hi, bottom, top)
                yhi = map_value(row.bootstrap_95_upper, lo, hi, bottom, top)
                color = PALETTE[method]
                c.line((x, ylo, x, yhi), color, 2)
                c.line((x - 5, ylo, x + 5, ylo), color, 2)
                c.line((x - 5, yhi, x + 5, yhi), color, 2)
                points.append((x, y))
            c.polyline(points, PALETTE[method], 5 if method in SELECTED_METHODS else 3)
            for x, y in points:
                c.circle(x, y, 6 if method in SELECTED_METHODS else 4, PALETTE[method])
    legend(c, ALL_METHODS, 320, 930, columns=3)
    c.finish()
    return png, pdf


def paired_forest(frame: pd.DataFrame, figures_dir: Path):
    png, pdf = figures_dir / "paired_effect_forest.png", figures_dir / "paired_effect_forest.pdf"
    c = DualCanvas(png, pdf, 1800, 1080)
    c.text(70, 38, "Prespecified patient-paired primary contrasts", 40, bold=True)
    c.text(70, 92, "Points are candidate minus reference. Intervals crossing zero are inconclusive; lower is better for HD95 and RMSE.", 23, PALETTE["muted"])
    labels = {
        "source_prior_contribution": "MAP - target only",
        "observation_likelihood_vs_latent_mse": "MAP - latent-MSE",
        "target_specific_updating": "MAP - source mean",
    }
    global_rows = frame[(frame["scope"] == "global") & (frame["metric"].isin(PRIMARY_METRICS))]
    panels = [(140, 220, 590, 850), (700, 220, 1150, 850), (1260, 220, 1710, 850)]
    for bounds, (metric, spec) in zip(panels, PRIMARY_METRICS.items()):
        data = global_rows[global_rows["metric"] == metric].copy()
        lo, hi = nice_limits([*data["bootstrap_95_lower"], *data["bootstrap_95_upper"], 0], pad=0.18, include_zero=True)
        left, top, right, bottom = bounds
        panel_axes(c, bounds, _ticks(lo, hi, 5, 3 if metric != "mean_hidden_hd95_px" else 2), [], (lo, hi), (0, 4), spec["label"], "Candidate - reference")
        x0 = map_value(0, lo, hi, left, right)
        c.line((x0, top, x0, bottom), PALETTE["muted"], 3)
        for i, contrast in enumerate(CONTRASTS, start=1):
            row = data[data["contrast"] == contrast].iloc[0]
            y = top + i * (bottom - top) / 4
            color = PALETTE[row.candidate_method] if contrast == "source_prior_contribution" else (PALETTE["hb_tpm_latent_mse"] if contrast == "observation_likelihood_vs_latent_mse" else PALETTE["source_mean"])
            xl = map_value(row.bootstrap_95_lower, lo, hi, left, right)
            xu = map_value(row.bootstrap_95_upper, lo, hi, left, right)
            xm = map_value(row.mean_paired_difference, lo, hi, left, right)
            c.line((xl, y, xu, y), color, 7)
            c.circle(xm, y, 10, color, PALETTE["paper"], 2)
            c.text(left, y - 52, labels[contrast], 22, PALETTE["ink"])
            c.text(right, y - 52, row.interval_interpretation, 20, PALETTE["muted"], anchor="right")
    c.finish()
    return png, pdf


def fold_stability(frame: pd.DataFrame, figures_dir: Path):
    png, pdf = figures_dir / "fold_stability.png", figures_dir / "fold_stability.pdf"
    c = DualCanvas(png, pdf, 1800, 1100)
    c.text(70, 38, "Fold-level stability", 40, bold=True)
    c.text(70, 92, "Each point is the equal-cell patient mean within a held-out fold; wide ranges indicate cohort heterogeneity.", 23, PALETTE["muted"])
    panels = [(100, 220, 590, 830), (680, 220, 1170, 830), (1260, 220, 1750, 830)]
    for bounds, (metric, spec) in zip(panels, PRIMARY_METRICS.items()):
        data = frame[frame["method_id"].isin(SELECTED_METHODS)]
        lo, hi = nice_limits(data[metric], pad=0.08)
        panel_axes(c, bounds, [(f, str(f)) for f in range(5)], _ticks(lo, hi, 5, 3 if metric != "mean_hidden_hd95_px" else 1), (0, 4), (lo, hi), spec["label"], "Held-out fold")
        left, top, right, bottom = bounds
        for method in SELECTED_METHODS:
            group = data[data["method_id"] == method].sort_values("fold")
            points = [(map_value(row.fold, 0, 4, left, right), map_value(getattr(row, metric), lo, hi, bottom, top)) for row in group.itertuples()]
            c.polyline(points, PALETTE[method], 4)
            for x, y in points:
                c.circle(x, y, 7, PALETTE[method])
    legend(c, SELECTED_METHODS, 330, 920, columns=2)
    c.finish()
    return png, pdf


def calibration_coverage(frame: pd.DataFrame, figures_dir: Path):
    png, pdf = figures_dir / "calibration_coverage.png", figures_dir / "calibration_coverage.pdf"
    c = DualCanvas(png, pdf, 1800, 1060)
    c.text(70, 38, "Posterior uncertainty is materially under-calibrated", 40, bold=True)
    c.text(70, 92, "Coverage is compared with the nominal 0.95 target; width is relative to the true area range.", 23, PALETTE["muted"])
    methods = [m for m in SELECTED_METHODS if m != "source_mean"]
    coverage = frame[(frame["metric"] == "hidden_area_95_coverage") & (frame["method_id"].isin(methods))]
    width = frame[(frame["metric"] == "area_95_interval_width_relative") & (frame["method_id"].isin(methods))]
    panels = [(160, 220, 820, 830), (1010, 220, 1670, 830)]
    for bounds, data, title, ymax in [(panels[0], coverage, "Empirical 95% area coverage", 1.0), (panels[1], width, "Relative interval width", max(3.5, width["bootstrap_95_upper"].max() * 1.08))]:
        left, top, right, bottom = bounds
        panel_axes(c, bounds, [(i, DISPLAY_NAMES[m]) for i, m in enumerate(methods)], _ticks(0, ymax, 6, 2), (-0.5, len(methods) - 0.5), (0, ymax), title)
        if title.startswith("Empirical"):
            ynominal = map_value(0.95, 0, ymax, bottom, top)
            c.line((left, ynominal, right, ynominal), PALETTE["muted"], 4)
            c.text(right, ynominal - 32, "Nominal 0.95", 21, PALETTE["muted"], anchor="right")
        for i, method in enumerate(methods):
            row = data[data["method_id"] == method].iloc[0]
            x = map_value(i, -0.5, len(methods) - 0.5, left, right)
            y = map_value(row["mean"], 0, ymax, bottom, top)
            ylo = map_value(row.bootstrap_95_lower, 0, ymax, bottom, top)
            yhi = map_value(row.bootstrap_95_upper, 0, ymax, bottom, top)
            c.rectangle((x - 55, y, x + 55, bottom), PALETTE[method])
            c.line((x, ylo, x, yhi), PALETTE["ink"], 3)
            c.text(x, y - 35, f"{row['mean']:.2f}", 23, bold=True, anchor="center")
    c.finish()
    return png, pdf


def trust_gate_behavior(rates: pd.DataFrame, correlations: pd.DataFrame, figures_dir: Path):
    png, pdf = figures_dir / "trust_gate_behavior.png", figures_dir / "trust_gate_behavior.pdf"
    c = DualCanvas(png, pdf, 1800, 1060)
    c.text(70, 38, "Trust-gate behavior depends strongly on sampling pattern", 40, bold=True)
    c.text(70, 92, "Rejection rates are descriptive. Correlations relate patient-mean trust score to reconstruction error.", 23, PALETTE["muted"])
    left, top, right, bottom = (150, 220, 980, 820)
    map_rates = rates[rates["method_id"] == "oa_hb_tpm_map"].groupby("strategy", as_index=False)["rejection_rate"].mean()
    panel_axes(c, (left, top, right, bottom), [(i, s.title()) for i, s in enumerate(STRATEGIES)], _ticks(0, 1, 6, 1), (-0.5, 2.5), (0, 1), "OA-HB-TPM MAP rejection rate")
    for i, strategy in enumerate(STRATEGIES):
        value = float(map_rates.loc[map_rates["strategy"] == strategy, "rejection_rate"].iloc[0])
        x = map_value(i, -0.5, 2.5, left, right)
        y = map_value(value, 0, 1, bottom, top)
        color = {"uniform": "#2A9D8F", "random": "#E9C46A", "clustered": "#D1495B"}[strategy]
        c.rectangle((x - 75, y, x + 75, bottom), color)
        c.text(x, y - 38, f"{value:.1%}", 27, bold=True, anchor="center")

    left2, top2, right2, bottom2 = (1160, 220, 1680, 820)
    corr = correlations[(correlations["method_id"] == "oa_hb_tpm_map") & (correlations["metric"] == "area_curve_relative_rmse")]
    lo, hi = -1.0, 1.0
    panel_axes(c, (left2, top2, right2, bottom2), _ticks(lo, hi, 5, 1), [], (lo, hi), (0, 4), "Trust score vs area-RMSE")
    xzero = map_value(0, lo, hi, left2, right2)
    c.line((xzero, top2, xzero, bottom2), PALETTE["muted"], 3)
    for i, strategy in enumerate(STRATEGIES, start=1):
        row = corr[corr["strategy"] == strategy].iloc[0]
        y = top2 + i * (bottom2 - top2) / 4
        xl = map_value(row.bootstrap_95_lower, lo, hi, left2, right2)
        xu = map_value(row.bootstrap_95_upper, lo, hi, left2, right2)
        xm = map_value(row.spearman_rho_trust_vs_error, lo, hi, left2, right2)
        c.line((xl, y, xu, y), PALETTE["oa_hb_tpm_map"], 7)
        c.circle(xm, y, 9, PALETTE["oa_hb_tpm_map"])
        c.text(left2, y - 48, strategy.title(), 23)
    c.finish()
    return png, pdf


def oracle_gap(frame: pd.DataFrame, figures_dir: Path):
    png, pdf = figures_dir / "oracle_gap.png", figures_dir / "oracle_gap.pdf"
    c = DualCanvas(png, pdf, 1800, 1030)
    c.text(70, 38, "Oracle-contour upper bound remains far above end-to-end models", 40, bold=True)
    c.text(70, 92, "The oracle uses manual contours and is shown separately; it is not an end-to-end imaging comparator.", 23, PALETTE["muted"])
    panels = [(120, 220, 580, 800), (690, 220, 1150, 800), (1260, 220, 1720, 800)]
    for bounds, (metric, spec) in zip(panels, PRIMARY_METRICS.items()):
        row = frame[frame["metric"] == metric].iloc[0]
        values = [row.best_learned_mean, row.oracle_mean]
        ymax = max(values) * 1.18
        ymin = 0
        left, top, right, bottom = bounds
        panel_axes(c, bounds, [(0, "Best learned"), (1, "Oracle")], _ticks(ymin, ymax, 5, 2 if metric != "mean_hidden_hd95_px" else 1), (-0.5, 1.5), (ymin, ymax), spec["label"])
        for i, (value, color) in enumerate(zip(values, [PALETTE[row.best_learned_method], PALETTE["oracle"]])):
            x = map_value(i, -0.5, 1.5, left, right)
            y = map_value(value, ymin, ymax, bottom, top)
            c.rectangle((x - 65, y, x + 65, bottom), color)
            c.text(x, y - 38, f"{value:.3f}", 25, bold=True, anchor="center")
        c.text((left + right) / 2, bottom + 80, f"Oracle advantage: {row.oracle_advantage_improvement_oriented:.3f}", 22, PALETTE["muted"], anchor="center")
    c.finish()
    return png, pdf


def generate_all_figures(tables: dict[str, pd.DataFrame], figures_dir: Path, seed: int):
    figures_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("Figure 1", performance_vs_k, (tables["performance_by_k"],), ["tables/performance_by_k.csv"], "K-response curves with fold-stratified patient-bootstrap intervals"),
        ("Figure 2", paired_forest, (tables["primary_paired_contrasts"],), ["tables/primary_paired_contrasts.csv"], "Prespecified global paired effects; candidate minus reference"),
        ("Figure 3", fold_stability, (tables["fold_level_performance"],), ["tables/fold_level_performance.csv"], "Held-out fold means"),
        ("Figure 4", calibration_coverage, (tables["uncertainty_summary"],), ["tables/uncertainty_summary.csv"], "Coverage and interval width; nominal target explicitly shown"),
        ("Figure 5", trust_gate_behavior, (tables["trust_gate_rates"], tables["trust_gate_correlations"]), ["tables/trust_gate_rates.csv", "tables/trust_gate_correlations.csv"], "Sampling-pattern rejection and patient-bootstrap rank correlations"),
        ("Figure 6", oracle_gap, (tables["oracle_upper_bound"],), ["tables/oracle_upper_bound.csv"], "Manual-contour oracle displayed separately from end-to-end learned models"),
    ]
    provenance = []
    for label, function, args, inputs, notes in specs:
        png, pdf = function(*args, figures_dir)
        for output in [png, pdf]:
            provenance.append({
                "figure_label": label,
                "script": "figures.py",
                "input_tables": ";".join(inputs),
                "seed": seed,
                "output_file": str(output.name),
                "manual_processing": "none",
                "notes": notes,
            })
    return pd.DataFrame(provenance)
