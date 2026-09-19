"""Generate the English manuscript-oriented TED analysis report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_ted_results import PRIMARY_METRICS
from figure_factory import DISPLAY_NAMES


def _fmt(value: float, metric: str) -> str:
    if pd.isna(value):
        return "NA"
    return format(float(value), PRIMARY_METRICS.get(metric, {}).get("fmt", ".4f"))


def _global_contrast_line(row: pd.Series) -> str:
    metric = row["metric"]
    label = PRIMARY_METRICS.get(metric, {"label": metric})["label"]
    return (
        f"{label}: {_fmt(row['mean_paired_difference'], metric)} "
        f"(95% CI {_fmt(row['bootstrap_95_lower'], metric)} to {_fmt(row['bootstrap_95_upper'], metric)}; "
        f"{row['relative_change_percent']:.2f}%; {row['interval_interpretation']})"
    )


def create_report(
    output_path: Path,
    tables: dict[str, pd.DataFrame],
    checks: pd.DataFrame,
    validation_meta: dict[str, Any],
    seed: int,
    draws: int,
) -> None:
    summary = tables["method_summary_equal_cell"]
    rankings = tables["per_cell_rankings"]
    primary = tables["primary_paired_contrasts"]
    ablations = tables["secondary_ablations"]
    uncertainty = tables["uncertainty_summary"]
    trust = tables["trust_gate_rates"]
    oracle = tables["oracle_upper_bound"]
    fold_heterogeneity = tables["fold_heterogeneity"]
    behavior = tables["model_behavior_diagnostics"]
    checkpoints = tables["checkpoint_prior_diagnostics"]
    training = tables["training_diagnostics"]

    latent = summary[summary["method_id"] == "hb_tpm_latent_mse"].iloc[0]
    oa = summary[summary["method_id"] == "oa_hb_tpm_map"].iloc[0]
    source = summary[summary["method_id"] == "source_mean"].iloc[0]
    wins = rankings[rankings["rank"] == 1].groupby(["metric", "method_id"]).size().reset_index(name="n_cells")
    latent_wins = {metric: int(wins.loc[(wins["metric"] == metric) & (wins["method_id"] == "hb_tpm_latent_mse"), "n_cells"].sum()) for metric in PRIMARY_METRICS}

    global_primary = primary[(primary["scope"] == "global") & (primary["metric"].isin(PRIMARY_METRICS))]
    source_prior = global_primary[global_primary["contrast"] == "source_prior_contribution"]
    likelihood = global_primary[global_primary["contrast"] == "observation_likelihood_vs_latent_mse"]
    target_update = global_primary[global_primary["contrast"] == "target_specific_updating"]
    global_ablation = ablations[(ablations["scope"] == "global") & (ablations["metric"].isin(PRIMARY_METRICS))]

    coverage = uncertainty[uncertainty["metric"] == "hidden_area_95_coverage"].set_index("method_id")
    width = uncertainty[uncertainty["metric"] == "area_95_interval_width_relative"].set_index("method_id")
    predictive_brier = uncertainty[uncertainty["metric"] == "posterior_predictive_hidden_brier"].set_index("method_id")
    map_reject = trust[trust["method_id"] == "oa_hb_tpm_map"].groupby("strategy")["rejection_rate"].mean()
    source_range = behavior[(behavior["method_id"] == "source_mean") & (behavior["diagnostic"] == "across_cell_area_rmse_range")]["value"].iloc[0]
    best_epochs = training.groupby("method_id")["best_epoch"].agg(["median", "min", "max"])
    map_checkpoint = checkpoints[checkpoints["method_id"] == "oa_hb_tpm_map"]
    stale = checks[checks["check"] == "stale_log_provenance"].iloc[0]

    lines = [
        "# TED Results Analysis and Diagnostic Report",
        "",
        "## Executive conclusion",
        "",
        (
            "The delivered 98-patient experiment does not support the strongest frozen claims for the observation-aware MAP model. "
            "The latent-MSE HB-TPM is the strongest end-to-end method on the three primary outcomes: it ranks first in "
            f"{latent_wins['mean_hidden_dice']}/15 Dice cells, {latent_wins['mean_hidden_hd95_px']}/15 HD95 cells, and "
            f"{latent_wins['area_curve_relative_rmse']}/15 area-RMSE cells. Its equal-cell global means are "
            f"Dice {latent['mean_hidden_dice']:.4f}, HD95 {latent['mean_hidden_hd95_px']:.3f} px, and area RMSE {latent['area_curve_relative_rmse']:.4f}."
        ),
        "",
        (
            f"OA-HB-TPM MAP is nearly tied with its isotropic-prior and no-phase ablations, does not improve overall on source mean "
            f"(MAP Dice {oa['mean_hidden_dice']:.4f} versus source mean {source['mean_hidden_dice']:.4f}), and shows poor nominal-95% "
            f"area coverage ({coverage.loc['oa_hb_tpm_map', 'mean']:.1%}; latent-MSE {coverage.loc['hb_tpm_latent_mse', 'mean']:.1%}). "
            "The intervals below are descriptive uncertainty intervals for patient-paired mean differences, not post-hoc significance tests."
        ),
        "",
        (
            f"The trust gate is highly sensitive to observation geometry: OA-HB-TPM MAP rejects approximately "
            f"{map_reject['uniform']:.1%} of uniform, {map_reject['random']:.1%} of random, and {map_reject['clustered']:.1%} of clustered observations. "
            "This supports a sampling-pattern sensitivity claim only; family-shift detection was not tested."
        ),
        "",
        "![Performance versus K](figures/performance_vs_k.png)",
        "",
        "## Analysis population and methods",
        "",
        (
            "The statistical unit is the patient. The analysis first averaged the five random or clustered sampling replicates within each "
            "patient-method-K-strategy cell; uniform has one replicate. Global estimates then gave each of the five K values and three sampling "
            "strategies equal weight, preventing the five-replicate strategies from dominating uniform sampling."
        ),
        "",
        (
            f"All confidence intervals use {draws:,} deterministic bootstrap draws (seed {seed}) with patients resampled independently within each "
            "held-out fold. Patient pairing is complete for every prespecified contrast. A confidence interval containing zero is labeled inconclusive. "
            "No clinical-importance threshold was prespecified, so this report makes no clinical-significance claim."
        ),
        "",
        "The three primary contrasts are OA-HB-TPM MAP minus target-only (source-prior contribution), OA-HB-TPM MAP minus latent-MSE HB-TPM (likelihood versus latent-MSE training), and OA-HB-TPM MAP minus source mean (value of patient-specific updating). Higher Dice is better; lower HD95 and area RMSE are better.",
        "",
        "## Primary results",
        "",
        "### Source-prior contribution: OA-HB-TPM MAP versus target-only",
        "",
        "; ".join(_global_contrast_line(row) for _, row in source_prior.iterrows()) + ".",
        "",
        "The estimated effects are small relative to fold-to-fold heterogeneity. Interpret any interval excluding zero as evidence about this frozen experiment, not as a clinical effect.",
        "",
        "### Observation-aware likelihood versus latent-MSE training",
        "",
        "; ".join(_global_contrast_line(row) for _, row in likelihood.iterrows()) + ".",
        "",
        "The sign of the delivered results favors latent-MSE on the primary outcomes. This directly contradicts a manuscript claim that the observation-aware likelihood outperforms latent-point training in this experiment.",
        "",
        "### Patient-specific updating versus source mean",
        "",
        "; ".join(_global_contrast_line(row) for _, row in target_update.iterrows()) + ".",
        "",
        "Patient-specific MAP updating does not outperform the fixed source mean overall. The source mean is essentially invariant to K and strategy, as expected from its implementation; its across-cell area-RMSE range is " + f"{source_range:.6f}.",
        "",
        "![Paired effects](figures/paired_effect_forest.png)",
        "",
        "## Secondary ablations and response to observation budget",
        "",
        "The full OA-HB-TPM MAP model and its isotropic-prior and no-phase-alignment ablations are practically indistinguishable at the delivered resolution:",
        "",
    ]
    for contrast in global_ablation["contrast"].unique():
        block = global_ablation[global_ablation["contrast"] == contrast]
        lines.append(f"- {contrast.replace('_', ' ')}: " + "; ".join(_global_contrast_line(row) for _, row in block.iterrows()) + ".")
        lines.append("")
    lines.extend([
        "Latent-MSE improves with larger K, whereas OA-HB-TPM MAP is nearly flat and non-monotonic. The detailed response curves and strategy-specific means are in `tables/performance_by_k.csv` and `tables/strategy_interactions.csv`.",
        "",
        "## Fold stability",
        "",
        (
            "Performance varies materially across folds. For latent-MSE, the held-out-fold range is "
            f"{fold_heterogeneity.loc[(fold_heterogeneity['method_id'] == 'hb_tpm_latent_mse') & (fold_heterogeneity['metric'] == 'mean_hidden_dice'), 'fold_min'].iloc[0]:.4f} to "
            f"{fold_heterogeneity.loc[(fold_heterogeneity['method_id'] == 'hb_tpm_latent_mse') & (fold_heterogeneity['metric'] == 'mean_hidden_dice'), 'fold_max'].iloc[0]:.4f} for Dice and "
            f"{fold_heterogeneity.loc[(fold_heterogeneity['method_id'] == 'hb_tpm_latent_mse') & (fold_heterogeneity['metric'] == 'area_curve_relative_rmse'), 'fold_min'].iloc[0]:.4f} to "
            f"{fold_heterogeneity.loc[(fold_heterogeneity['method_id'] == 'hb_tpm_latent_mse') & (fold_heterogeneity['metric'] == 'area_curve_relative_rmse'), 'fold_max'].iloc[0]:.4f} for area RMSE. "
            "Leave-one-fold-out estimates retain direction for the largest effects but show why single pooled means are insufficient."
        ),
        "",
        "![Fold stability](figures/fold_stability.png)",
        "",
        "## Calibration and uncertainty",
        "",
        (
            f"Nominal 95% area coverage is {coverage.loc['oa_hb_tpm_map', 'mean']:.1%} for OA-HB-TPM MAP and "
            f"{coverage.loc['hb_tpm_latent_mse', 'mean']:.1%} for latent-MSE, both far below 95%. Their mean relative interval widths are "
            f"{width.loc['oa_hb_tpm_map', 'mean']:.3f} and {width.loc['hb_tpm_latent_mse', 'mean']:.3f}, respectively. "
            f"Posterior-predictive Brier scores are {predictive_brier.loc['oa_hb_tpm_map', 'mean']:.5f} and "
            f"{predictive_brier.loc['hb_tpm_latent_mse', 'mean']:.5f}. Better Brier score does not rescue the severe coverage deficit."
        ),
        "",
        "Source mean has no posterior interval outputs, so its coverage and width are correctly reported as unavailable rather than zero.",
        "",
        "![Calibration and coverage](figures/calibration_coverage.png)",
        "",
        "## Trust-gate diagnostics",
        "",
        (
            "The gate largely acts as a detector of nonuniform observation geometry under this package's calibration scheme. Accepted-versus-rejected "
            "errors and patient-bootstrap rank correlations are reported in `tables/trust_gate_accepted_vs_rejected.csv` and "
            "`tables/trust_gate_correlations.csv`. These analyses do not establish detection of anatomical family shift, scanner shift, or clinical failure."
        ),
        "",
        "![Trust-gate behavior](figures/trust_gate_behavior.png)",
        "",
        "## Model and training diagnostics",
        "",
        (
            "Training histories, selected epochs, phase offsets, K responsiveness, and checkpoint priors were read directly from the frozen run outputs. "
            f"Across methods, selected best epochs range from {int(best_epochs['min'].min())} to {int(best_epochs['max'].max())}. "
            "The checkpoint analysis reconstructs tensor-only state dictionaries without executing model code and summarizes prior mean harmonic energy, covariance trace, condition number, and off-diagonal correlation."
        ),
        "",
        (
            f"For OA-HB-TPM MAP checkpoints, mean prior harmonic-energy fraction is {map_checkpoint['prior_harmonic_energy_fraction'].mean():.3f}; "
            f"the median recorded covariance condition-number summary is {map_checkpoint['median_prior_covariance_condition_number'].median():.2f}. "
            "These are descriptive diagnostics, not evidence that the learned covariance is beneficial: the isotropic-prior ablation is nearly identical in outcome space."
        ),
        "",
        "## Oracle-contour upper bound",
        "",
    ])
    for row in oracle.itertuples():
        lines.append(
            f"- {PRIMARY_METRICS[row.metric]['label']}: oracle {_fmt(row.oracle_mean, row.metric)} versus best learned "
            f"{_fmt(row.best_learned_mean, row.metric)} ({DISPLAY_NAMES.get(row.best_learned_method, row.best_learned_method)}); "
            f"improvement-oriented oracle advantage {_fmt(row.oracle_advantage_improvement_oriented, row.metric)}."
        )
    lines.extend([
        "",
        "The large gap is consistent with image-to-latent estimation being the main bottleneck. Because the oracle consumes manual contours, it is an upper-bound diagnostic and must not be placed in an undifferentiated end-to-end model ranking.",
        "",
        "![Oracle gap](figures/oracle_gap.png)",
        "",
        "## Data integrity and provenance",
        "",
        (
            "The analysis verified the ZIP SHA-256 checksum, 150 run manifests, every recorded run-output hash, 32,340 learned-method rows, "
            "98 unique patients, five mutually exclusive folds, six methods, five K values, three strategies, complete replicate counts, zero duplicate "
            "full keys, finite primary metrics, and false hidden-image/hidden-mask leakage flags. Every supplied aggregate value and frozen bootstrap interval "
            "was independently reproduced to numerical tolerance."
        ),
        "",
        (
            f"The bundled experiment-code snapshot matches the current tracked package after line-ending normalization. The recorded run commit "
            f"`{validation_meta['recorded_commit']}` is not present in local Git history. {stale['observed']}; therefore "
            "`target_only/fold0_k2.log` is retained as a warning about an earlier failed attempt, not evidence that the final run failed."
        ),
        "",
        "Exact input and output hashes, environment versions, seed, and bootstrap settings are recorded in `analysis_manifest.json`. Figure-to-table provenance is in `FIGURE_PROVENANCE.csv`.",
        "",
        "## Limitations and manuscript implications",
        "",
        "- Classical baselines specified by the frozen design are absent. The package cannot support superiority claims against linear interpolation, splines, Gaussian processes, or other conventional comparators.",
        "",
        "- Family-shift validation is absent. Trust-gate conclusions are limited to sampling-pattern sensitivity.",
        "",
        "- ED/ES timing error, temporal jerk, and several other planned secondary outcomes are absent.",
        "",
        "- Raw target image arrays are not included, so inference cannot be rerun independently from this package. This report analyzes the frozen outputs without recomputing predictions.",
        "",
        "- The sample contains 98 patients across five folds and exhibits substantial fold heterogeneity. External validity and clinical relevance remain unestablished.",
        "",
        "A defensible manuscript can state that latent-MSE HB-TPM was the strongest delivered end-to-end method and that a large oracle gap remains. It should not state that OA-HB-TPM MAP outperforms latent-MSE or source mean, that its nominal intervals are calibrated, that the trust gate detects family shift, or that clinical significance has been shown.",
        "",
        "## Recommended next experiments",
        "",
        "1. Add the frozen classical baselines and report identical patient-level paired contrasts.",
        "",
        "2. Calibrate posterior scale on held-out source/validation patients, then evaluate coverage and Brier scores without reusing the test patients.",
        "",
        "3. Isolate the image-to-latent bottleneck with controlled encoder/initializer improvements while preserving the same trajectory decoder and folds.",
        "",
        "4. Run a prespecified external or family/vendor-shift evaluation for the trust gate, including error-detection AUROC/AUPRC and coverage conditional on acceptance.",
        "",
        "5. Add ED/ES timing, temporal smoothness/jerk, runtime, and clinically motivated effect thresholds before drafting clinical conclusions.",
        "",
        "## Machine-readable outputs",
        "",
        "All patient identifiers are confined to `tables/canonical_patient_cells.csv`; the narrative and figures contain aggregates only. The claim-to-evidence map is `tables/claim_status.csv`, and missing planned outcomes are enumerated in `tables/missing_deliverables.csv`.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")
