"""Photon-time estimators for prespecified time-varying diffusion models.

This module extends :mod:`fcs_d_estimators` without changing the stationary
benchmark implementation.  Every public estimator consumes only an ordered
one-dimensional ``event_time_s`` array.  It never accepts latent Brownian
positions, molecule identities, conditional rates, or a true diffusion
coefficient.

Two deliberately simple estimator families are provided.

``estimate_known_block_acf``
    Applies the existing Equal-LSE and feasible diagonal W-LSE separately on
    prespecified half-open time blocks.  These are window estimators: they do
    not use events outside the block being fitted.

``estimate_known_block_event_qmle``
    Jointly optimizes all block-specific diffusion coefficients using one
    ordered-event immigration--death MMPP surrogate likelihood.  The finite
    state filter is initialized once at time zero and is carried across every
    fixed cut; it is never reset to stationarity.  Window W-LSE estimates
    provide deterministic warm starts, with Equal-LSE and a geometric bound
    midpoint as prespecified fallbacks.

For the smooth model

``D(t) = D_start * (D_end / D_start) ** (t / T)``,

``estimate_local_window_wlse`` supplies a descriptive local estimate and
``estimate_exponential_d_event_qmle`` optimizes a time-inhomogeneous event
quasi-likelihood.  The latter approximates the time-ordered MMPP transition by
piecewise-constant midpoint generators on a declared refinement grid.  It is
therefore approximate even for the surrogate, and remains a quasi-likelihood
for Brownian motion observed through a Gaussian PSF.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Sequence

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import gammaln

from fcs_d_estimators import DEstimate, estimate_acf, estimate_acf_weighted


@dataclass(frozen=True)
class KnownBlockACFResult:
    """Equal-LSE and W-LSE results on prespecified half-open blocks."""

    method_id: str
    segment_edges_s: tuple[float, ...]
    equal_lse: tuple[DEstimate, ...]
    weighted_lse: tuple[DEstimate, ...]
    success: bool
    runtime_s: float
    n_input_photons: int
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalWindowEstimate:
    """One local W-LSE result and its half-open observation window."""

    window_index: int
    start_s: float
    end_s: float
    center_s: float
    n_photons: int
    estimate: DEstimate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalWLSEResult:
    """Collection of local-window W-LSE fits for a smooth ``D(t)``."""

    method_id: str
    windows: tuple[LocalWindowEstimate, ...]
    success: bool
    runtime_s: float
    n_input_photons: int
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalEqualLSEResult:
    """Collection of local-window Equal-LSE fits for a smooth ``D(t)``."""

    method_id: str
    windows: tuple[LocalWindowEstimate, ...]
    success: bool
    runtime_s: float
    n_input_photons: int
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VectorDEstimate:
    """A vector-valued diffusion estimate with JSON-friendly diagnostics."""

    method_id: str
    parameter_names: tuple[str, ...]
    d_estimates_um2_s: tuple[float, ...]
    objective: float
    success: bool
    at_search_boundary: bool
    failure_reason: str
    runtime_s: float
    n_input_photons: int
    n_analysis_points: int
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_event_times(event_time_s: np.ndarray, duration_s: float) -> np.ndarray:
    """Return sorted finite photon times in ``[0, duration_s)``."""

    times = np.asarray(event_time_s, dtype=float)
    if times.ndim != 1:
        raise ValueError("event_time_s must be a one-dimensional array")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not np.all(np.isfinite(times)):
        raise ValueError("event_time_s contains non-finite values")
    if np.any(times < 0) or np.any(times >= duration_s):
        raise ValueError("event_time_s must lie inside [0, duration_s)")
    if len(times) > 1 and np.any(np.diff(times) < 0):
        times = np.sort(times)
    return times


def _segment_edges(
    duration_s: float,
    cut_times_s: Sequence[float],
) -> np.ndarray:
    """Validate fixed cuts and return ``[0, cuts..., duration]``."""

    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    cuts = np.asarray(tuple(cut_times_s), dtype=float)
    if cuts.ndim != 1 or not np.all(np.isfinite(cuts)):
        raise ValueError("cut_times_s must be a finite one-dimensional sequence")
    if len(cuts) and (
        np.any(cuts <= 0)
        or np.any(cuts >= float(duration_s))
        or np.any(np.diff(cuts) <= 0)
    ):
        raise ValueError(
            "cut_times_s must be strictly increasing and lie in (0, duration_s)"
        )
    return np.concatenate(([0.0], cuts, [float(duration_s)]))


def split_events_half_open(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    cut_times_s: Sequence[float],
) -> tuple[np.ndarray, ...]:
    """Split events into ``[edge_j, edge_{j+1})`` blocks.

    An event exactly equal to a cut belongs to the block beginning at that
    cut.  Returned times retain the acquisition clock; callers that fit a
    window model should subtract the corresponding left edge.
    """

    times = _validate_event_times(event_time_s, duration_s)
    edges = _segment_edges(duration_s, cut_times_s)
    split_indices = np.searchsorted(times, edges[1:-1], side="left")
    return tuple(np.split(times, split_indices))


def _validate_acf_controls(
    *,
    bin_s: float,
    curve_max_lag_s: float,
    fit_max_lag_s: float,
    n_lags: int,
    wxy_um: float,
    kappa: float,
) -> None:
    if not np.isfinite(bin_s) or bin_s <= 0:
        raise ValueError("bin_s must be finite and positive")
    if not np.isfinite(curve_max_lag_s) or curve_max_lag_s <= 0:
        raise ValueError("curve_max_lag_s must be finite and positive")
    if not np.isfinite(fit_max_lag_s) or fit_max_lag_s <= 0:
        raise ValueError("fit_max_lag_s must be finite and positive")
    if int(n_lags) < 8:
        raise ValueError("n_lags must be at least 8")
    if not np.isfinite(wxy_um) or wxy_um <= 0:
        raise ValueError("wxy_um must be finite and positive")
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")


def estimate_known_block_acf(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    cut_times_s: Sequence[float],
    bin_s: float,
    curve_max_lag_s: float,
    fit_max_lag_s: float,
    n_lags: int,
    wxy_um: float,
    kappa: float,
    free_baseline: bool = False,
    tau_grid_size: int = 180,
    n_bootstrap: int = 100,
    block_length_s: float = 0.004,
    weight_shrinkage: float = 0.10,
    max_weight_ratio: float = 100.0,
    bootstrap_seed: int = 20260728,
) -> KnownBlockACFResult:
    """Fit Equal-LSE and W-LSE independently within each known block."""

    started = time.perf_counter()
    times = _validate_event_times(event_time_s, duration_s)
    edges = _segment_edges(duration_s, cut_times_s)
    _validate_acf_controls(
        bin_s=bin_s,
        curve_max_lag_s=curve_max_lag_s,
        fit_max_lag_s=fit_max_lag_s,
        n_lags=n_lags,
        wxy_um=wxy_um,
        kappa=kappa,
    )
    blocks = split_events_half_open(
        times,
        duration_s=duration_s,
        cut_times_s=cut_times_s,
    )
    equal_results: list[DEstimate] = []
    weighted_results: list[DEstimate] = []
    for index, (left, right, block) in enumerate(
        zip(edges[:-1], edges[1:], blocks)
    ):
        block_duration = float(right - left)
        local_times = block - float(left)
        equal_results.append(
            estimate_acf(
                local_times,
                duration_s=block_duration,
                bin_s=bin_s,
                curve_max_lag_s=min(curve_max_lag_s, block_duration / 4.0),
                fit_max_lag_s=min(fit_max_lag_s, block_duration / 4.0),
                n_lags=n_lags,
                wxy_um=wxy_um,
                kappa=kappa,
                free_baseline=free_baseline,
                tau_grid_size=tau_grid_size,
            )
        )
        weighted_results.append(
            estimate_acf_weighted(
                local_times,
                duration_s=block_duration,
                bin_s=bin_s,
                curve_max_lag_s=min(curve_max_lag_s, block_duration / 4.0),
                fit_max_lag_s=min(fit_max_lag_s, block_duration / 4.0),
                n_lags=n_lags,
                wxy_um=wxy_um,
                kappa=kappa,
                free_baseline=free_baseline,
                tau_grid_size=tau_grid_size,
                n_bootstrap=n_bootstrap,
                block_length_s=min(block_length_s, block_duration / 4.0),
                weight_shrinkage=weight_shrinkage,
                max_weight_ratio=max_weight_ratio,
                bootstrap_seed=int(bootstrap_seed) + index,
            )
        )

    all_success = bool(
        all(result.success for result in equal_results)
        and all(result.success for result in weighted_results)
    )
    return KnownBlockACFResult(
        method_id="known_block_window_acf",
        segment_edges_s=tuple(float(value) for value in edges),
        equal_lse=tuple(equal_results),
        weighted_lse=tuple(weighted_results),
        success=all_success,
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        diagnostics={
            "estimator_visible_input": "event_time_s only",
            "simulation_truth_used": False,
            "event_assignment": "half-open [left, right)",
            "segments_are_fitted_independently": True,
            "segment_event_counts": [int(len(block)) for block in blocks],
            "bootstrap_seed_by_segment": [
                int(bootstrap_seed) + index for index in range(len(blocks))
            ],
        },
    )


def _truncated_poisson_stationary(
    mean_occupancy: float,
    state_max: int,
) -> np.ndarray:
    if not np.isfinite(mean_occupancy) or mean_occupancy <= 0:
        raise ValueError("mean_occupancy must be finite and positive")
    if int(state_max) < 4:
        raise ValueError("state_max must be at least 4")
    states = np.arange(int(state_max) + 1, dtype=float)
    log_weights = states * math.log(float(mean_occupancy)) - gammaln(states + 1.0)
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    return weights / float(np.sum(weights))


def _surrogate_kernel(
    *,
    d_um2_s: float,
    wxy_um: float,
    kappa: float,
    effective_brightness_cps: float,
    background_cps: float,
    mean_occupancy: float,
    state_max: int,
    stationary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Build one reversible killed immigration--death generator."""

    states = np.arange(int(state_max) + 1, dtype=float)
    mu_s_inv = (
        4.0
        * float(d_um2_s)
        / float(wxy_um) ** 2
        * (1.0 + 1.0 / (2.0 * float(kappa) ** 2))
    )
    immigration_rate_s_inv = float(mean_occupancy) * mu_s_inv
    generator = np.zeros((int(state_max) + 1, int(state_max) + 1), dtype=float)
    indices = np.arange(int(state_max))
    generator[indices, indices + 1] = immigration_rate_s_inv
    generator[indices + 1, indices] = (indices + 1.0) * mu_s_inv
    generator[np.arange(int(state_max) + 1), np.arange(int(state_max) + 1)] = (
        -generator.sum(axis=1)
    )
    photon_rates = (
        float(background_cps) + float(effective_brightness_cps) * states
    )
    killed_generator = generator - np.diag(photon_rates)
    stationary_sqrt = np.sqrt(stationary)
    symmetric_generator = (
        stationary_sqrt[:, None]
        * killed_generator
        / stationary_sqrt[None, :]
    )
    eigenvalues, eigenvectors = eigh(
        0.5 * (symmetric_generator + symmetric_generator.T)
    )
    return (
        photon_rates,
        stationary_sqrt,
        eigenvalues,
        eigenvectors,
        float(mu_s_inv),
        float(immigration_rate_s_inv),
    )


def _propagate_filter(
    probabilities: np.ndarray,
    gap_s: float,
    *,
    stationary_sqrt: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    if gap_s < -1e-15:
        raise ValueError("event times must be chronologically ordered")
    transformed = probabilities / stationary_sqrt
    coefficients = transformed @ eigenvectors
    propagated = (
        (coefficients * np.exp(eigenvalues * max(0.0, float(gap_s))))
        @ eigenvectors.T
        * stationary_sqrt
    )
    return np.maximum(propagated, 0.0)


def _validate_likelihood_controls(
    *,
    wxy_um: float,
    kappa: float,
    molecular_brightness_cps: float,
    background_cps: float,
    mean_occupancy: float,
    state_max: int,
) -> None:
    if not np.isfinite(wxy_um) or wxy_um <= 0:
        raise ValueError("wxy_um must be finite and positive")
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    if not np.isfinite(molecular_brightness_cps) or molecular_brightness_cps <= 0:
        raise ValueError("molecular_brightness_cps must be finite and positive")
    if not np.isfinite(background_cps) or background_cps < 0:
        raise ValueError("background_cps must be finite and nonnegative")
    if not np.isfinite(mean_occupancy) or mean_occupancy <= 0:
        raise ValueError("mean_occupancy must be finite and positive")
    if int(state_max) < 4:
        raise ValueError("state_max must be at least 4")


def evaluate_fixed_cut_event_qmle_loglikelihood(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    cut_times_s: Sequence[float],
    d_by_segment_um2_s: Sequence[float],
    wxy_um: float,
    kappa: float,
    molecular_brightness_cps: float,
    background_cps: float,
    mean_occupancy: float,
    state_max: int = 18,
) -> tuple[float, dict[str, Any]]:
    """Evaluate the joint fixed-cut ordered-event surrogate log likelihood.

    This likelihood is exact for the declared finite-state, piecewise
    immigration--death MMPP.  It is a quasi-likelihood for Brownian molecules
    observed through the Gaussian PSF.  The filter is initialized exactly once
    and its posterior distribution is carried unchanged across fixed cuts.
    """

    times = _validate_event_times(event_time_s, duration_s)
    edges = _segment_edges(duration_s, cut_times_s)
    d_values = np.asarray(tuple(d_by_segment_um2_s), dtype=float)
    if d_values.ndim != 1 or len(d_values) != len(edges) - 1:
        raise ValueError("one diffusion coefficient is required per segment")
    if not np.all(np.isfinite(d_values)) or np.any(d_values <= 0):
        raise ValueError("all segment diffusion coefficients must be positive")
    _validate_likelihood_controls(
        wxy_um=wxy_um,
        kappa=kappa,
        molecular_brightness_cps=molecular_brightness_cps,
        background_cps=background_cps,
        mean_occupancy=mean_occupancy,
        state_max=state_max,
    )

    effective_brightness_cps = float(molecular_brightness_cps) / (2.0**1.5)
    stationary = _truncated_poisson_stationary(mean_occupancy, state_max)
    kernels = [
        _surrogate_kernel(
            d_um2_s=float(d_value),
            wxy_um=wxy_um,
            kappa=kappa,
            effective_brightness_cps=effective_brightness_cps,
            background_cps=background_cps,
            mean_occupancy=mean_occupancy,
            state_max=state_max,
            stationary=stationary,
        )
        for d_value in d_values
    ]
    blocks = split_events_half_open(
        times,
        duration_s=duration_s,
        cut_times_s=cut_times_s,
    )

    forward = stationary.copy()
    log_likelihood = 0.0
    maximum_top_probability = float(forward[-1])
    boundary_filters: list[dict[str, Any]] = []
    mu_values: list[float] = []
    immigration_values: list[float] = []

    for index, (left, right, block, kernel) in enumerate(
        zip(edges[:-1], edges[1:], blocks, kernels)
    ):
        (
            photon_rates,
            stationary_sqrt,
            eigenvalues,
            eigenvectors,
            mu_s_inv,
            immigration_rate_s_inv,
        ) = kernel
        mu_values.append(mu_s_inv)
        immigration_values.append(immigration_rate_s_inv)
        previous_time = float(left)
        for event_time in block:
            forward = _propagate_filter(
                forward,
                float(event_time) - previous_time,
                stationary_sqrt=stationary_sqrt,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
            )
            forward *= photon_rates
            scale = float(np.sum(forward))
            if not np.isfinite(scale) or scale <= 0:
                return float("-inf"), {
                    "failure_reason": "nonpositive_event_filter_scale",
                    "failure_segment_index": int(index),
                }
            log_likelihood += math.log(scale)
            forward /= scale
            maximum_top_probability = max(
                maximum_top_probability,
                float(forward[-1]),
            )
            previous_time = float(event_time)

        forward = _propagate_filter(
            forward,
            float(right) - previous_time,
            stationary_sqrt=stationary_sqrt,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        survival_scale = float(np.sum(forward))
        if not np.isfinite(survival_scale) or survival_scale <= 0:
            return float("-inf"), {
                "failure_reason": "nonpositive_survival_filter_scale",
                "failure_segment_index": int(index),
            }
        log_likelihood += math.log(survival_scale)
        forward /= survival_scale
        maximum_top_probability = max(
            maximum_top_probability,
            float(forward[-1]),
        )
        if index < len(blocks) - 1:
            carried = forward.copy()
            boundary_filters.append(
                {
                    "cut_time_s": float(right),
                    "left_posterior": carried.astype(float).tolist(),
                    "right_initial": carried.astype(float).tolist(),
                    "carried_l1_error": 0.0,
                    "reset_to_stationary": False,
                }
            )

    return float(log_likelihood), {
        "likelihood_type": (
            "joint ordered-event likelihood for a fixed-cut finite-state "
            "immigration-death MMPP surrogate"
        ),
        "exactness_scope": (
            "exact for the declared truncated piecewise MMPP; "
            "quasi-likelihood for Brownian Gaussian-PSF photon data"
        ),
        "brownian_gaussian_psf_status": "approximate quasi-likelihood",
        "estimator_visible_input": "event_time_s only",
        "simulation_truth_used": False,
        "arrival_time_bins_used": False,
        "event_assignment": "half-open [left, right)",
        "filter_initializations": 1,
        "stationary_resets_at_cuts": 0,
        "filter_carried_across_cuts": True,
        "boundary_filters": boundary_filters,
        "segment_event_counts": [int(len(block)) for block in blocks],
        "stationary_top_probability": float(stationary[-1]),
        "max_filtered_top_probability": float(maximum_top_probability),
        "effective_brightness_cps": float(effective_brightness_cps),
        "mu_by_segment_s_inv": mu_values,
        "immigration_rate_by_segment_s_inv": immigration_values,
    }


def _plugin_occupancy(
    *,
    n_events: int,
    duration_s: float,
    molecular_brightness_cps: float,
    background_cps: float,
    occupancy_min: float,
    occupancy_max: float,
) -> tuple[float, float, bool]:
    if not (0 < occupancy_min < occupancy_max):
        raise ValueError("occupancy bounds must satisfy 0 < min < max")
    effective_brightness = float(molecular_brightness_cps) / (2.0**1.5)
    signal_rate = int(n_events) / float(duration_s) - float(background_cps)
    if signal_rate <= 0:
        return float("nan"), float("nan"), False
    raw = signal_rate / effective_brightness
    clipped = float(np.clip(raw, occupancy_min, occupancy_max))
    return clipped, float(raw), bool(
        not np.isclose(clipped, raw, rtol=0.0, atol=1e-15)
    )


def _warm_starts_from_block_acf(
    result: KnownBlockACFResult,
    *,
    d_min_um2_s: float,
    d_max_um2_s: float,
) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    sources: list[str] = []
    fallback = math.sqrt(float(d_min_um2_s) * float(d_max_um2_s))
    for weighted, equal in zip(result.weighted_lse, result.equal_lse):
        if weighted.success and np.isfinite(weighted.d_estimate_um2_s):
            value = float(weighted.d_estimate_um2_s)
            source = weighted.method_id
        elif equal.success and np.isfinite(equal.d_estimate_um2_s):
            value = float(equal.d_estimate_um2_s)
            source = equal.method_id
        else:
            value = fallback
            source = "geometric_bounds_midpoint_fallback"
        values.append(float(np.clip(value, d_min_um2_s, d_max_um2_s)))
        sources.append(source)
    return np.asarray(values, dtype=float), sources


def estimate_known_block_event_qmle(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    cut_times_s: Sequence[float],
    wxy_um: float,
    kappa: float,
    molecular_brightness_cps: float,
    background_cps: float,
    d_min_um2_s: float,
    d_max_um2_s: float,
    bin_s: float,
    curve_max_lag_s: float,
    fit_max_lag_s: float,
    n_lags: int,
    state_max: int = 18,
    occupancy_min: float = 0.02,
    occupancy_max: float = 6.0,
    free_baseline: bool = False,
    tau_grid_size: int = 180,
    n_bootstrap: int = 100,
    block_length_s: float = 0.004,
    weight_shrinkage: float = 0.10,
    max_weight_ratio: float = 100.0,
    bootstrap_seed: int = 20260728,
    optimizer_maxiter: int = 80,
    optimizer_ftol: float = 1e-10,
    optimizer_gtol: float = 1e-6,
    optimizer_finite_difference_step: float = 1e-5,
    optimizer_boundary_log_tolerance: float = 1e-4,
) -> VectorDEstimate:
    """Joint fixed-cut event-time QMLE with window-estimator warm starts."""

    started = time.perf_counter()
    method_id = "known_block_event_qmle_warm"
    times = _validate_event_times(event_time_s, duration_s)
    edges = _segment_edges(duration_s, cut_times_s)
    if len(times) < 20:
        return VectorDEstimate(
            method_id=method_id,
            parameter_names=tuple(
                f"D_segment_{index + 1}" for index in range(len(edges) - 1)
            ),
            d_estimates_um2_s=tuple(float("nan") for _ in range(len(edges) - 1)),
            objective=float("nan"),
            success=False,
            at_search_boundary=False,
            failure_reason="too_few_photons",
            runtime_s=time.perf_counter() - started,
            n_input_photons=len(times),
            n_analysis_points=len(times),
            diagnostics={
                "estimator_visible_input": "event_time_s only",
                "simulation_truth_used": False,
            },
        )
    if not (0 < d_min_um2_s < d_max_um2_s):
        raise ValueError("D bounds must satisfy 0 < d_min < d_max")
    if optimizer_maxiter < 1:
        raise ValueError("optimizer_maxiter must be positive")
    if (
        optimizer_ftol <= 0
        or optimizer_gtol <= 0
        or optimizer_finite_difference_step <= 0
        or optimizer_boundary_log_tolerance <= 0
    ):
        raise ValueError("optimizer tolerances must be positive")
    _validate_likelihood_controls(
        wxy_um=wxy_um,
        kappa=kappa,
        molecular_brightness_cps=molecular_brightness_cps,
        background_cps=background_cps,
        mean_occupancy=1.0,
        state_max=state_max,
    )

    acf_result = estimate_known_block_acf(
        times,
        duration_s=duration_s,
        cut_times_s=cut_times_s,
        bin_s=bin_s,
        curve_max_lag_s=curve_max_lag_s,
        fit_max_lag_s=fit_max_lag_s,
        n_lags=n_lags,
        wxy_um=wxy_um,
        kappa=kappa,
        free_baseline=free_baseline,
        tau_grid_size=tau_grid_size,
        n_bootstrap=n_bootstrap,
        block_length_s=block_length_s,
        weight_shrinkage=weight_shrinkage,
        max_weight_ratio=max_weight_ratio,
        bootstrap_seed=bootstrap_seed,
    )
    warm_d, warm_sources = _warm_starts_from_block_acf(
        acf_result,
        d_min_um2_s=d_min_um2_s,
        d_max_um2_s=d_max_um2_s,
    )
    occupancy, raw_occupancy, occupancy_clipped = _plugin_occupancy(
        n_events=len(times),
        duration_s=duration_s,
        molecular_brightness_cps=molecular_brightness_cps,
        background_cps=background_cps,
        occupancy_min=occupancy_min,
        occupancy_max=occupancy_max,
    )
    if not np.isfinite(occupancy):
        return VectorDEstimate(
            method_id=method_id,
            parameter_names=tuple(
                f"D_segment_{index + 1}" for index in range(len(edges) - 1)
            ),
            d_estimates_um2_s=tuple(float("nan") for _ in range(len(edges) - 1)),
            objective=float("nan"),
            success=False,
            at_search_boundary=False,
            failure_reason="observed_rate_not_above_background",
            runtime_s=time.perf_counter() - started,
            n_input_photons=len(times),
            n_analysis_points=len(times),
            diagnostics={
                "estimator_visible_input": "event_time_s only",
                "simulation_truth_used": False,
                "warm_start_d_um2_s": warm_d.tolist(),
                "warm_start_source_by_segment": warm_sources,
            },
        )

    log_bounds = (
        math.log(float(d_min_um2_s)),
        math.log(float(d_max_um2_s)),
    )

    def objective(log_d: np.ndarray) -> float:
        log_likelihood, _ = evaluate_fixed_cut_event_qmle_loglikelihood(
            times,
            duration_s=duration_s,
            cut_times_s=cut_times_s,
            d_by_segment_um2_s=np.exp(np.asarray(log_d, dtype=float)),
            wxy_um=wxy_um,
            kappa=kappa,
            molecular_brightness_cps=molecular_brightness_cps,
            background_cps=background_cps,
            mean_occupancy=occupancy,
            state_max=state_max,
        )
        return -float(log_likelihood)

    optimization = minimize(
        objective,
        x0=np.log(warm_d),
        bounds=[log_bounds] * len(warm_d),
        method="L-BFGS-B",
        options={
            "ftol": float(optimizer_ftol),
            "gtol": float(optimizer_gtol),
            "eps": float(optimizer_finite_difference_step),
            "maxiter": int(optimizer_maxiter),
            "maxls": 30,
        },
    )
    log_d_hat = np.asarray(optimization.x, dtype=float)
    d_hat = np.exp(log_d_hat)
    log_likelihood, likelihood_diagnostics = (
        evaluate_fixed_cut_event_qmle_loglikelihood(
            times,
            duration_s=duration_s,
            cut_times_s=cut_times_s,
            d_by_segment_um2_s=d_hat,
            wxy_um=wxy_um,
            kappa=kappa,
            molecular_brightness_cps=molecular_brightness_cps,
            background_cps=background_cps,
            mean_occupancy=occupancy,
            state_max=state_max,
        )
    )
    at_boundary_by_parameter = [
        bool(
            value - log_bounds[0] <= optimizer_boundary_log_tolerance
            or log_bounds[1] - value <= optimizer_boundary_log_tolerance
        )
        for value in log_d_hat
    ]
    truncation_ok = bool(
        likelihood_diagnostics.get("stationary_top_probability", 1.0) < 1e-8
        and likelihood_diagnostics.get("max_filtered_top_probability", 1.0)
        < 1e-6
    )
    success = bool(
        optimization.success
        and np.all(np.isfinite(d_hat))
        and np.isfinite(log_likelihood)
        and not any(at_boundary_by_parameter)
        and not occupancy_clipped
        and truncation_ok
    )
    reasons: list[str] = []
    if not optimization.success:
        reasons.append("optimizer_failure")
    if not np.all(np.isfinite(d_hat)):
        reasons.append("nonfinite_d")
    if not np.isfinite(log_likelihood):
        reasons.append("nonfinite_loglikelihood")
    if any(at_boundary_by_parameter):
        reasons.append("d_search_boundary")
    if occupancy_clipped:
        reasons.append("occupancy_plugin_boundary")
    if not truncation_ok:
        reasons.append("state_truncation_mass_too_large")

    return VectorDEstimate(
        method_id=method_id,
        parameter_names=tuple(
            f"D_segment_{index + 1}" for index in range(len(d_hat))
        ),
        d_estimates_um2_s=tuple(float(value) for value in d_hat),
        objective=float(-log_likelihood),
        success=success,
        at_search_boundary=bool(any(at_boundary_by_parameter)),
        failure_reason=";".join(reasons),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        n_analysis_points=len(times),
        diagnostics={
            **likelihood_diagnostics,
            "segment_edges_s": edges.astype(float).tolist(),
            "d_bounds_um2_s": [
                float(d_min_um2_s),
                float(d_max_um2_s),
            ],
            "d_at_search_boundary_by_parameter": at_boundary_by_parameter,
            "plugin_mean_occupancy": float(occupancy),
            "raw_plugin_mean_occupancy": float(raw_occupancy),
            "occupancy_plugin_clipped": bool(occupancy_clipped),
            "warm_start_method": (
                "segment W-LSE; segment Equal-LSE then geometric bound "
                "midpoint are deterministic fallbacks"
            ),
            "warm_start_d_um2_s": warm_d.astype(float).tolist(),
            "warm_start_source_by_segment": warm_sources,
            "warm_start_runtime_s": float(acf_result.runtime_s),
            "window_acf_success": bool(acf_result.success),
            "optimizer": "bounded L-BFGS-B on the vector log(D_1),...,log(D_J)",
            "optimizer_success": bool(optimization.success),
            "optimizer_status": int(getattr(optimization, "status", 0)),
            "optimizer_message": str(optimization.message),
            "optimizer_iterations": int(getattr(optimization, "nit", 0)),
            "optimizer_function_evaluations": int(
                getattr(optimization, "nfev", 0)
            ),
        },
    )


def _equal_lse_by_segments(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    cut_times_s: Sequence[float],
    bin_s: float,
    curve_max_lag_s: float,
    fit_max_lag_s: float,
    n_lags: int,
    wxy_um: float,
    kappa: float,
    free_baseline: bool,
    tau_grid_size: int,
) -> tuple[tuple[DEstimate, ...], float]:
    """Compute only the Equal-LSE warm starts on fixed half-open blocks."""

    started = time.perf_counter()
    times = _validate_event_times(event_time_s, duration_s)
    edges = _segment_edges(duration_s, cut_times_s)
    _validate_acf_controls(
        bin_s=bin_s,
        curve_max_lag_s=curve_max_lag_s,
        fit_max_lag_s=fit_max_lag_s,
        n_lags=n_lags,
        wxy_um=wxy_um,
        kappa=kappa,
    )
    blocks = split_events_half_open(
        times,
        duration_s=duration_s,
        cut_times_s=cut_times_s,
    )
    results: list[DEstimate] = []
    for left, right, block in zip(edges[:-1], edges[1:], blocks):
        block_duration = float(right - left)
        results.append(
            estimate_acf(
                block - float(left),
                duration_s=block_duration,
                bin_s=bin_s,
                curve_max_lag_s=min(
                    curve_max_lag_s,
                    block_duration / 4.0,
                ),
                fit_max_lag_s=min(fit_max_lag_s, block_duration / 4.0),
                n_lags=n_lags,
                wxy_um=wxy_um,
                kappa=kappa,
                free_baseline=free_baseline,
                tau_grid_size=tau_grid_size,
            )
        )
    return tuple(results), time.perf_counter() - started


def _warm_starts_from_equal_lse(
    equal_results: Sequence[DEstimate],
    *,
    d_min_um2_s: float,
    d_max_um2_s: float,
) -> tuple[np.ndarray, list[str]]:
    midpoint = math.sqrt(float(d_min_um2_s) * float(d_max_um2_s))
    values: list[float] = []
    sources: list[str] = []
    for result in equal_results:
        if result.success and np.isfinite(result.d_estimate_um2_s):
            value = float(result.d_estimate_um2_s)
            source = result.method_id
        else:
            value = midpoint
            source = "geometric_bounds_midpoint_fallback"
        values.append(float(np.clip(value, d_min_um2_s, d_max_um2_s)))
        sources.append(source)
    return np.asarray(values, dtype=float), sources


def estimate_known_block_joint_event_qmle(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    cut_times_s: Sequence[float],
    wxy_um: float,
    kappa: float,
    molecular_brightness_cps: float,
    background_cps: float,
    d_min_um2_s: float,
    d_max_um2_s: float,
    bin_s: float,
    curve_max_lag_s: float,
    fit_max_lag_s: float,
    n_lags: int,
    state_max: int = 26,
    occupancy_min: float = 0.02,
    occupancy_max: float = 6.0,
    free_baseline: bool = False,
    tau_grid_size: int = 180,
    optimizer_maxiter: int = 80,
    optimizer_ftol: float = 1e-9,
    optimizer_gtol: float = 1e-5,
    optimizer_finite_difference_step: float = 1e-6,
    optimizer_boundary_log_tolerance: float = 1e-4,
) -> VectorDEstimate:
    """Jointly optimize fixed-block ``D`` values and one shared occupancy.

    The ordered-event objective is an exact likelihood for the calibrated
    truncated fixed-cut MMPP surrogate and a quasi-likelihood for Brownian
    Gaussian-PSF photon data.  Equal-LSE supplies each ``D_j`` warm start; the
    whole-record photon-rate moment supplies the shared occupancy warm start.
    No count bins enter the likelihood itself.
    """

    started = time.perf_counter()
    method_id = "known_block_joint_occupancy_event_qmle_warm"
    times = _validate_event_times(event_time_s, duration_s)
    edges = _segment_edges(duration_s, cut_times_s)
    parameter_names = tuple(
        f"D_segment_{index + 1}" for index in range(len(edges) - 1)
    )
    if len(times) < 20:
        return VectorDEstimate(
            method_id=method_id,
            parameter_names=parameter_names,
            d_estimates_um2_s=tuple(
                float("nan") for _ in range(len(edges) - 1)
            ),
            objective=float("nan"),
            success=False,
            at_search_boundary=False,
            failure_reason="too_few_photons",
            runtime_s=time.perf_counter() - started,
            n_input_photons=len(times),
            n_analysis_points=len(times),
            diagnostics={
                "estimator_visible_input": "event_time_s only",
                "simulation_truth_used": False,
            },
        )
    if not (0 < d_min_um2_s < d_max_um2_s):
        raise ValueError("D bounds must satisfy 0 < d_min < d_max")
    if not (0 < occupancy_min < occupancy_max):
        raise ValueError("occupancy bounds must satisfy 0 < min < max")
    if optimizer_maxiter < 1:
        raise ValueError("optimizer_maxiter must be positive")
    if (
        optimizer_ftol <= 0
        or optimizer_gtol <= 0
        or optimizer_finite_difference_step <= 0
        or optimizer_boundary_log_tolerance <= 0
    ):
        raise ValueError("optimizer tolerances must be positive")
    _validate_likelihood_controls(
        wxy_um=wxy_um,
        kappa=kappa,
        molecular_brightness_cps=molecular_brightness_cps,
        background_cps=background_cps,
        mean_occupancy=1.0,
        state_max=state_max,
    )

    equal_results, warm_start_runtime_s = _equal_lse_by_segments(
        times,
        duration_s=duration_s,
        cut_times_s=cut_times_s,
        bin_s=bin_s,
        curve_max_lag_s=curve_max_lag_s,
        fit_max_lag_s=fit_max_lag_s,
        n_lags=n_lags,
        wxy_um=wxy_um,
        kappa=kappa,
        free_baseline=free_baseline,
        tau_grid_size=tau_grid_size,
    )
    warm_d, warm_sources = _warm_starts_from_equal_lse(
        equal_results,
        d_min_um2_s=d_min_um2_s,
        d_max_um2_s=d_max_um2_s,
    )
    warm_occupancy, raw_occupancy, occupancy_start_clipped = _plugin_occupancy(
        n_events=len(times),
        duration_s=duration_s,
        molecular_brightness_cps=molecular_brightness_cps,
        background_cps=background_cps,
        occupancy_min=occupancy_min,
        occupancy_max=occupancy_max,
    )
    if not np.isfinite(warm_occupancy):
        return VectorDEstimate(
            method_id=method_id,
            parameter_names=parameter_names,
            d_estimates_um2_s=tuple(
                float("nan") for _ in range(len(edges) - 1)
            ),
            objective=float("nan"),
            success=False,
            at_search_boundary=False,
            failure_reason="observed_rate_not_above_background",
            runtime_s=time.perf_counter() - started,
            n_input_photons=len(times),
            n_analysis_points=len(times),
            diagnostics={
                "estimator_visible_input": "event_time_s only",
                "simulation_truth_used": False,
                "warm_start_d_um2_s": warm_d.tolist(),
                "warm_start_source_by_segment": warm_sources,
            },
        )

    log_d_bounds = (
        math.log(float(d_min_um2_s)),
        math.log(float(d_max_um2_s)),
    )
    log_occupancy_bounds = (
        math.log(float(occupancy_min)),
        math.log(float(occupancy_max)),
    )

    def objective(log_parameters: np.ndarray) -> float:
        values = np.exp(np.asarray(log_parameters, dtype=float))
        log_likelihood, _ = evaluate_fixed_cut_event_qmle_loglikelihood(
            times,
            duration_s=duration_s,
            cut_times_s=cut_times_s,
            d_by_segment_um2_s=values[:-1],
            wxy_um=wxy_um,
            kappa=kappa,
            molecular_brightness_cps=molecular_brightness_cps,
            background_cps=background_cps,
            mean_occupancy=float(values[-1]),
            state_max=state_max,
        )
        return -float(log_likelihood)

    optimization = minimize(
        objective,
        x0=np.log(np.append(warm_d, warm_occupancy)),
        bounds=[log_d_bounds] * len(warm_d) + [log_occupancy_bounds],
        method="L-BFGS-B",
        options={
            "ftol": float(optimizer_ftol),
            "gtol": float(optimizer_gtol),
            "eps": float(optimizer_finite_difference_step),
            "maxiter": int(optimizer_maxiter),
            "maxls": 30,
        },
    )
    log_hat = np.asarray(optimization.x, dtype=float)
    d_hat = np.exp(log_hat[:-1])
    occupancy_hat = math.exp(float(log_hat[-1]))
    log_likelihood, likelihood_diagnostics = (
        evaluate_fixed_cut_event_qmle_loglikelihood(
            times,
            duration_s=duration_s,
            cut_times_s=cut_times_s,
            d_by_segment_um2_s=d_hat,
            wxy_um=wxy_um,
            kappa=kappa,
            molecular_brightness_cps=molecular_brightness_cps,
            background_cps=background_cps,
            mean_occupancy=occupancy_hat,
            state_max=state_max,
        )
    )
    d_at_boundary = [
        bool(
            value - log_d_bounds[0] <= optimizer_boundary_log_tolerance
            or log_d_bounds[1] - value <= optimizer_boundary_log_tolerance
        )
        for value in log_hat[:-1]
    ]
    occupancy_at_boundary = bool(
        log_hat[-1] - log_occupancy_bounds[0]
        <= optimizer_boundary_log_tolerance
        or log_occupancy_bounds[1] - log_hat[-1]
        <= optimizer_boundary_log_tolerance
    )
    truncation_ok = bool(
        likelihood_diagnostics.get("stationary_top_probability", 1.0) < 1e-8
        and likelihood_diagnostics.get("max_filtered_top_probability", 1.0)
        < 1e-6
    )
    success = bool(
        optimization.success
        and np.all(np.isfinite(d_hat))
        and np.isfinite(occupancy_hat)
        and np.isfinite(log_likelihood)
        and not any(d_at_boundary)
        and not occupancy_at_boundary
        and truncation_ok
    )
    reasons: list[str] = []
    if not optimization.success:
        reasons.append("optimizer_failure")
    if not np.all(np.isfinite(d_hat)):
        reasons.append("nonfinite_d")
    if not np.isfinite(occupancy_hat):
        reasons.append("nonfinite_occupancy")
    if not np.isfinite(log_likelihood):
        reasons.append("nonfinite_loglikelihood")
    if any(d_at_boundary):
        reasons.append("d_search_boundary")
    if occupancy_at_boundary:
        reasons.append("occupancy_search_boundary")
    if not truncation_ok:
        reasons.append("state_truncation_mass_too_large")

    return VectorDEstimate(
        method_id=method_id,
        parameter_names=parameter_names,
        d_estimates_um2_s=tuple(float(value) for value in d_hat),
        objective=float(-log_likelihood),
        success=success,
        at_search_boundary=bool(
            any(d_at_boundary) or occupancy_at_boundary
        ),
        failure_reason=";".join(reasons),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        n_analysis_points=len(times),
        diagnostics={
            **likelihood_diagnostics,
            "segment_edges_s": edges.astype(float).tolist(),
            "d_bounds_um2_s": [
                float(d_min_um2_s),
                float(d_max_um2_s),
            ],
            "d_at_search_boundary_by_parameter": d_at_boundary,
            "occupancy_bounds": [
                float(occupancy_min),
                float(occupancy_max),
            ],
            "occupancy_at_search_boundary": occupancy_at_boundary,
            "occupancy_is_optimized": True,
            "initial_mean_occupancy": float(warm_occupancy),
            "raw_moment_mean_occupancy": float(raw_occupancy),
            "occupancy_start_clipped": bool(occupancy_start_clipped),
            "estimated_mean_occupancy": float(occupancy_hat),
            "finite_state_surrogate_mle": True,
            "full_latent_brownian_likelihood": False,
            "initialization_uses_binned_estimator": True,
            "likelihood_objective_uses_arrival_time_bins": False,
            "warm_start_method": (
                "segment Equal-LSE for D; whole-record photon-rate moment "
                "for shared mean occupancy"
            ),
            "warm_start_d_um2_s": warm_d.astype(float).tolist(),
            "warm_start_source_by_segment": warm_sources,
            "warm_start_runtime_s": float(warm_start_runtime_s),
            "optimizer": (
                "bounded L-BFGS-B on "
                "(log D_1,...,log D_J,log mean occupancy)"
            ),
            "optimizer_success": bool(optimization.success),
            "optimizer_status": int(getattr(optimization, "status", 0)),
            "optimizer_message": str(optimization.message),
            "optimizer_iterations": int(getattr(optimization, "nit", 0)),
            "optimizer_function_evaluations": int(
                getattr(optimization, "nfev", 0)
            ),
        },
    )


def _window_starts(
    duration_s: float,
    window_s: float,
    step_s: float,
) -> np.ndarray:
    if (
        not np.isfinite(window_s)
        or not np.isfinite(step_s)
        or window_s <= 0
        or step_s <= 0
        or window_s > duration_s
    ):
        raise ValueError(
            "window_s and step_s must be positive, with window_s <= duration_s"
        )
    last = float(duration_s) - float(window_s)
    starts = np.arange(
        0.0,
        last + max(1e-15, float(step_s) * 1e-10),
        float(step_s),
    )
    if len(starts) == 0 or not np.isclose(starts[-1], last, atol=1e-12, rtol=0):
        starts = np.append(starts, last)
    return np.unique(np.round(starts, 15))


def estimate_local_window_wlse(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    window_s: float,
    step_s: float,
    bin_s: float,
    curve_max_lag_s: float,
    fit_max_lag_s: float,
    n_lags: int,
    wxy_um: float,
    kappa: float,
    free_baseline: bool = False,
    tau_grid_size: int = 180,
    n_bootstrap: int = 100,
    block_length_s: float = 0.004,
    weight_shrinkage: float = 0.10,
    max_weight_ratio: float = 100.0,
    bootstrap_seed: int = 20260728,
) -> LocalWLSEResult:
    """Estimate a smooth diffusion profile by overlapping local W-LSE fits."""

    started = time.perf_counter()
    times = _validate_event_times(event_time_s, duration_s)
    _validate_acf_controls(
        bin_s=bin_s,
        curve_max_lag_s=curve_max_lag_s,
        fit_max_lag_s=fit_max_lag_s,
        n_lags=n_lags,
        wxy_um=wxy_um,
        kappa=kappa,
    )
    starts = _window_starts(duration_s, window_s, step_s)
    windows: list[LocalWindowEstimate] = []
    for index, left in enumerate(starts):
        right = float(left + window_s)
        first = int(np.searchsorted(times, left, side="left"))
        last = int(np.searchsorted(times, right, side="left"))
        local_times = times[first:last] - float(left)
        estimate = estimate_acf_weighted(
            local_times,
            duration_s=float(window_s),
            bin_s=bin_s,
            curve_max_lag_s=min(curve_max_lag_s, window_s / 4.0),
            fit_max_lag_s=min(fit_max_lag_s, window_s / 4.0),
            n_lags=n_lags,
            wxy_um=wxy_um,
            kappa=kappa,
            free_baseline=free_baseline,
            tau_grid_size=tau_grid_size,
            n_bootstrap=n_bootstrap,
            block_length_s=min(block_length_s, window_s / 4.0),
            weight_shrinkage=weight_shrinkage,
            max_weight_ratio=max_weight_ratio,
            bootstrap_seed=int(bootstrap_seed) + index,
        )
        windows.append(
            LocalWindowEstimate(
                window_index=int(index),
                start_s=float(left),
                end_s=right,
                center_s=float(left + window_s / 2.0),
                n_photons=len(local_times),
                estimate=estimate,
            )
        )
    return LocalWLSEResult(
        method_id="local_window_acf_weighted",
        windows=tuple(windows),
        success=bool(any(window.estimate.success for window in windows)),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        diagnostics={
            "estimator_visible_input": "event_time_s only",
            "simulation_truth_used": False,
            "event_assignment": "half-open [window_start, window_end)",
            "overlapping_windows_are_not_independent_replicates": True,
            "window_s": float(window_s),
            "step_s": float(step_s),
            "bootstrap_seed_by_window": [
                int(bootstrap_seed) + index for index in range(len(windows))
            ],
        },
    )


def estimate_local_window_equal_lse(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    window_s: float,
    step_s: float,
    bin_s: float,
    curve_max_lag_s: float,
    fit_max_lag_s: float,
    n_lags: int,
    wxy_um: float,
    kappa: float,
    free_baseline: bool = False,
    tau_grid_size: int = 180,
) -> LocalEqualLSEResult:
    """Estimate a smooth diffusion profile by overlapping local Equal-LSE."""

    started = time.perf_counter()
    times = _validate_event_times(event_time_s, duration_s)
    _validate_acf_controls(
        bin_s=bin_s,
        curve_max_lag_s=curve_max_lag_s,
        fit_max_lag_s=fit_max_lag_s,
        n_lags=n_lags,
        wxy_um=wxy_um,
        kappa=kappa,
    )
    starts = _window_starts(duration_s, window_s, step_s)
    windows: list[LocalWindowEstimate] = []
    for index, left in enumerate(starts):
        right = float(left + window_s)
        first = int(np.searchsorted(times, left, side="left"))
        last = int(np.searchsorted(times, right, side="left"))
        local_times = times[first:last] - float(left)
        estimate = estimate_acf(
            local_times,
            duration_s=float(window_s),
            bin_s=bin_s,
            curve_max_lag_s=min(curve_max_lag_s, window_s / 4.0),
            fit_max_lag_s=min(fit_max_lag_s, window_s / 4.0),
            n_lags=n_lags,
            wxy_um=wxy_um,
            kappa=kappa,
            free_baseline=free_baseline,
            tau_grid_size=tau_grid_size,
        )
        windows.append(
            LocalWindowEstimate(
                window_index=int(index),
                start_s=float(left),
                end_s=right,
                center_s=float(left + window_s / 2.0),
                n_photons=len(local_times),
                estimate=estimate,
            )
        )
    return LocalEqualLSEResult(
        method_id="local_window_acf_equal_weight",
        windows=tuple(windows),
        success=bool(any(window.estimate.success for window in windows)),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        diagnostics={
            "estimator_visible_input": "event_time_s only",
            "simulation_truth_used": False,
            "event_assignment": "half-open [window_start, window_end)",
            "overlapping_windows_are_not_independent_replicates": True,
            "window_s": float(window_s),
            "step_s": float(step_s),
        },
    )


def _exponential_profile(
    time_s: np.ndarray,
    *,
    duration_s: float,
    d_start_um2_s: float,
    d_end_um2_s: float,
) -> np.ndarray:
    fraction = np.asarray(time_s, dtype=float) / float(duration_s)
    return float(d_start_um2_s) * (
        float(d_end_um2_s) / float(d_start_um2_s)
    ) ** fraction


def evaluate_exponential_d_event_qmle_loglikelihood(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    d_start_um2_s: float,
    d_end_um2_s: float,
    wxy_um: float,
    kappa: float,
    molecular_brightness_cps: float,
    background_cps: float,
    mean_occupancy: float,
    state_max: int = 18,
    n_refinement_intervals: int = 64,
) -> tuple[float, dict[str, Any]]:
    """Approximate the smooth-``D(t)`` event QML by midpoint propagation."""

    if (
        not np.isfinite(d_start_um2_s)
        or not np.isfinite(d_end_um2_s)
        or d_start_um2_s <= 0
        or d_end_um2_s <= 0
    ):
        raise ValueError("D_start and D_end must be finite and positive")
    if int(n_refinement_intervals) < 1:
        raise ValueError("n_refinement_intervals must be positive")
    grid_edges = np.linspace(
        0.0,
        float(duration_s),
        int(n_refinement_intervals) + 1,
    )
    midpoints = 0.5 * (grid_edges[:-1] + grid_edges[1:])
    d_midpoint = _exponential_profile(
        midpoints,
        duration_s=duration_s,
        d_start_um2_s=d_start_um2_s,
        d_end_um2_s=d_end_um2_s,
    )
    log_likelihood, diagnostics = (
        evaluate_fixed_cut_event_qmle_loglikelihood(
            event_time_s,
            duration_s=duration_s,
            cut_times_s=grid_edges[1:-1],
            d_by_segment_um2_s=d_midpoint,
            wxy_um=wxy_um,
            kappa=kappa,
            molecular_brightness_cps=molecular_brightness_cps,
            background_cps=background_cps,
            mean_occupancy=mean_occupancy,
            state_max=state_max,
        )
    )
    diagnostics = {
        **diagnostics,
        "likelihood_type": (
            "time-inhomogeneous immigration-death event quasi-likelihood "
            "with midpoint piecewise-constant propagation"
        ),
        "exactness_scope": (
            "exact only for the midpoint-frozen truncated piecewise MMPP; "
            "approximate for the smooth time-inhomogeneous surrogate and a "
            "quasi-likelihood for Brownian Gaussian-PSF photon data"
        ),
        "time_inhomogeneous_transition_status": (
            "approximate midpoint refinement of the time-ordered transition"
        ),
        "smooth_d_model": (
            "D(t)=D_start*(D_end/D_start)^(t/T)"
        ),
        "n_refinement_intervals": int(n_refinement_intervals),
        "refinement_grid_edges_s": grid_edges.astype(float).tolist(),
        "d_at_refinement_midpoints_um2_s": d_midpoint.astype(float).tolist(),
    }
    return log_likelihood, diagnostics


def _warm_start_exponential_profile(
    local_result: LocalWLSEResult | LocalEqualLSEResult,
    *,
    duration_s: float,
    d_min_um2_s: float,
    d_max_um2_s: float,
    estimator_label: str = "local W-LSE",
) -> tuple[np.ndarray, str]:
    usable = [
        window
        for window in local_result.windows
        if window.estimate.success
        and np.isfinite(window.estimate.d_estimate_um2_s)
        and window.estimate.d_estimate_um2_s > 0
    ]
    midpoint = math.sqrt(float(d_min_um2_s) * float(d_max_um2_s))
    if len(usable) >= 2:
        fraction = np.asarray(
            [window.center_s / float(duration_s) for window in usable],
            dtype=float,
        )
        log_d = np.log(
            np.asarray(
                [window.estimate.d_estimate_um2_s for window in usable],
                dtype=float,
            )
        )
        design = np.column_stack((np.ones_like(fraction), fraction))
        intercept, slope = np.linalg.lstsq(design, log_d, rcond=None)[0]
        starts = np.exp([intercept, intercept + slope])
        source = (
            "log-linear regression through successful "
            f"{estimator_label} fits"
        )
    elif len(usable) == 1:
        starts = np.repeat(usable[0].estimate.d_estimate_um2_s, 2)
        source = (
            f"single successful {estimator_label} used at both endpoints"
        )
    else:
        starts = np.repeat(midpoint, 2)
        source = "geometric bounds midpoint fallback"
    return (
        np.clip(
            np.asarray(starts, dtype=float),
            float(d_min_um2_s),
            float(d_max_um2_s),
        ),
        source,
    )


def estimate_exponential_d_event_qmle(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    wxy_um: float,
    kappa: float,
    molecular_brightness_cps: float,
    background_cps: float,
    d_min_um2_s: float,
    d_max_um2_s: float,
    window_s: float,
    step_s: float,
    bin_s: float,
    curve_max_lag_s: float,
    fit_max_lag_s: float,
    n_lags: int,
    state_max: int = 18,
    occupancy_min: float = 0.02,
    occupancy_max: float = 6.0,
    n_refinement_intervals: int = 64,
    free_baseline: bool = False,
    tau_grid_size: int = 180,
    n_bootstrap: int = 100,
    block_length_s: float = 0.004,
    weight_shrinkage: float = 0.10,
    max_weight_ratio: float = 100.0,
    bootstrap_seed: int = 20260728,
    optimizer_maxiter: int = 80,
    optimizer_ftol: float = 1e-10,
    optimizer_gtol: float = 1e-6,
    optimizer_finite_difference_step: float = 1e-5,
    optimizer_boundary_log_tolerance: float = 1e-4,
) -> VectorDEstimate:
    """Fit ``D_start,D_end`` by an approximate time-inhomogeneous event QMLE."""

    started = time.perf_counter()
    method_id = "exponential_d_event_qmle_warm"
    times = _validate_event_times(event_time_s, duration_s)
    if len(times) < 20:
        return VectorDEstimate(
            method_id=method_id,
            parameter_names=("D_start", "D_end"),
            d_estimates_um2_s=(float("nan"), float("nan")),
            objective=float("nan"),
            success=False,
            at_search_boundary=False,
            failure_reason="too_few_photons",
            runtime_s=time.perf_counter() - started,
            n_input_photons=len(times),
            n_analysis_points=len(times),
            diagnostics={
                "estimator_visible_input": "event_time_s only",
                "simulation_truth_used": False,
            },
        )
    if not (0 < d_min_um2_s < d_max_um2_s):
        raise ValueError("D bounds must satisfy 0 < d_min < d_max")
    if optimizer_maxiter < 1 or int(n_refinement_intervals) < 1:
        raise ValueError("optimizer and refinement counts must be positive")
    if (
        optimizer_ftol <= 0
        or optimizer_gtol <= 0
        or optimizer_finite_difference_step <= 0
        or optimizer_boundary_log_tolerance <= 0
    ):
        raise ValueError("optimizer tolerances must be positive")
    _validate_likelihood_controls(
        wxy_um=wxy_um,
        kappa=kappa,
        molecular_brightness_cps=molecular_brightness_cps,
        background_cps=background_cps,
        mean_occupancy=1.0,
        state_max=state_max,
    )

    local_result = estimate_local_window_wlse(
        times,
        duration_s=duration_s,
        window_s=window_s,
        step_s=step_s,
        bin_s=bin_s,
        curve_max_lag_s=curve_max_lag_s,
        fit_max_lag_s=fit_max_lag_s,
        n_lags=n_lags,
        wxy_um=wxy_um,
        kappa=kappa,
        free_baseline=free_baseline,
        tau_grid_size=tau_grid_size,
        n_bootstrap=n_bootstrap,
        block_length_s=block_length_s,
        weight_shrinkage=weight_shrinkage,
        max_weight_ratio=max_weight_ratio,
        bootstrap_seed=bootstrap_seed,
    )
    warm_d, warm_source = _warm_start_exponential_profile(
        local_result,
        duration_s=duration_s,
        d_min_um2_s=d_min_um2_s,
        d_max_um2_s=d_max_um2_s,
    )
    occupancy, raw_occupancy, occupancy_clipped = _plugin_occupancy(
        n_events=len(times),
        duration_s=duration_s,
        molecular_brightness_cps=molecular_brightness_cps,
        background_cps=background_cps,
        occupancy_min=occupancy_min,
        occupancy_max=occupancy_max,
    )
    if not np.isfinite(occupancy):
        return VectorDEstimate(
            method_id=method_id,
            parameter_names=("D_start", "D_end"),
            d_estimates_um2_s=(float("nan"), float("nan")),
            objective=float("nan"),
            success=False,
            at_search_boundary=False,
            failure_reason="observed_rate_not_above_background",
            runtime_s=time.perf_counter() - started,
            n_input_photons=len(times),
            n_analysis_points=len(times),
            diagnostics={
                "estimator_visible_input": "event_time_s only",
                "simulation_truth_used": False,
                "warm_start_d_um2_s": warm_d.tolist(),
                "warm_start_source": warm_source,
            },
        )

    log_bounds = (
        math.log(float(d_min_um2_s)),
        math.log(float(d_max_um2_s)),
    )

    def objective(log_parameters: np.ndarray) -> float:
        d_start, d_end = np.exp(np.asarray(log_parameters, dtype=float))
        log_likelihood, _ = (
            evaluate_exponential_d_event_qmle_loglikelihood(
                times,
                duration_s=duration_s,
                d_start_um2_s=float(d_start),
                d_end_um2_s=float(d_end),
                wxy_um=wxy_um,
                kappa=kappa,
                molecular_brightness_cps=molecular_brightness_cps,
                background_cps=background_cps,
                mean_occupancy=occupancy,
                state_max=state_max,
                n_refinement_intervals=n_refinement_intervals,
            )
        )
        return -float(log_likelihood)

    optimization = minimize(
        objective,
        x0=np.log(warm_d),
        bounds=[log_bounds, log_bounds],
        method="L-BFGS-B",
        options={
            "ftol": float(optimizer_ftol),
            "gtol": float(optimizer_gtol),
            "eps": float(optimizer_finite_difference_step),
            "maxiter": int(optimizer_maxiter),
            "maxls": 30,
        },
    )
    log_d_hat = np.asarray(optimization.x, dtype=float)
    d_hat = np.exp(log_d_hat)
    log_likelihood, likelihood_diagnostics = (
        evaluate_exponential_d_event_qmle_loglikelihood(
            times,
            duration_s=duration_s,
            d_start_um2_s=float(d_hat[0]),
            d_end_um2_s=float(d_hat[1]),
            wxy_um=wxy_um,
            kappa=kappa,
            molecular_brightness_cps=molecular_brightness_cps,
            background_cps=background_cps,
            mean_occupancy=occupancy,
            state_max=state_max,
            n_refinement_intervals=n_refinement_intervals,
        )
    )
    at_boundary_by_parameter = [
        bool(
            value - log_bounds[0] <= optimizer_boundary_log_tolerance
            or log_bounds[1] - value <= optimizer_boundary_log_tolerance
        )
        for value in log_d_hat
    ]
    truncation_ok = bool(
        likelihood_diagnostics.get("stationary_top_probability", 1.0) < 1e-8
        and likelihood_diagnostics.get("max_filtered_top_probability", 1.0)
        < 1e-6
    )
    success = bool(
        optimization.success
        and np.all(np.isfinite(d_hat))
        and np.isfinite(log_likelihood)
        and not any(at_boundary_by_parameter)
        and not occupancy_clipped
        and truncation_ok
    )
    reasons: list[str] = []
    if not optimization.success:
        reasons.append("optimizer_failure")
    if not np.all(np.isfinite(d_hat)):
        reasons.append("nonfinite_d")
    if not np.isfinite(log_likelihood):
        reasons.append("nonfinite_loglikelihood")
    if any(at_boundary_by_parameter):
        reasons.append("d_search_boundary")
    if occupancy_clipped:
        reasons.append("occupancy_plugin_boundary")
    if not truncation_ok:
        reasons.append("state_truncation_mass_too_large")

    return VectorDEstimate(
        method_id=method_id,
        parameter_names=("D_start", "D_end"),
        d_estimates_um2_s=(float(d_hat[0]), float(d_hat[1])),
        objective=float(-log_likelihood),
        success=success,
        at_search_boundary=bool(any(at_boundary_by_parameter)),
        failure_reason=";".join(reasons),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        n_analysis_points=len(times),
        diagnostics={
            **likelihood_diagnostics,
            "d_bounds_um2_s": [
                float(d_min_um2_s),
                float(d_max_um2_s),
            ],
            "d_at_search_boundary_by_parameter": at_boundary_by_parameter,
            "plugin_mean_occupancy": float(occupancy),
            "raw_plugin_mean_occupancy": float(raw_occupancy),
            "occupancy_plugin_clipped": bool(occupancy_clipped),
            "warm_start_method": "local W-LSE log-linear endpoint extrapolation",
            "warm_start_d_um2_s": warm_d.astype(float).tolist(),
            "warm_start_source": warm_source,
            "warm_start_runtime_s": float(local_result.runtime_s),
            "n_successful_local_wlse": int(
                sum(window.estimate.success for window in local_result.windows)
            ),
            "optimizer": "bounded L-BFGS-B on (log D_start, log D_end)",
            "optimizer_success": bool(optimization.success),
            "optimizer_status": int(getattr(optimization, "status", 0)),
            "optimizer_message": str(optimization.message),
            "optimizer_iterations": int(getattr(optimization, "nit", 0)),
            "optimizer_function_evaluations": int(
                getattr(optimization, "nfev", 0)
            ),
        },
    )


def estimate_exponential_d_joint_event_qmle(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    wxy_um: float,
    kappa: float,
    molecular_brightness_cps: float,
    background_cps: float,
    d_min_um2_s: float,
    d_max_um2_s: float,
    window_s: float,
    step_s: float,
    bin_s: float,
    curve_max_lag_s: float,
    fit_max_lag_s: float,
    n_lags: int,
    state_max: int = 26,
    occupancy_min: float = 0.02,
    occupancy_max: float = 6.0,
    n_refinement_intervals: int = 64,
    free_baseline: bool = False,
    tau_grid_size: int = 180,
    optimizer_maxiter: int = 80,
    optimizer_ftol: float = 1e-9,
    optimizer_gtol: float = 1e-5,
    optimizer_finite_difference_step: float = 1e-6,
    optimizer_boundary_log_tolerance: float = 1e-4,
) -> VectorDEstimate:
    """Jointly fit exponential-profile endpoints and shared occupancy.

    Local Equal-LSE estimates initialize ``D_start`` and ``D_end`` through a
    log-linear endpoint extrapolation.  The whole-record photon-rate moment
    initializes the shared mean occupancy.  The likelihood uses exact photon
    times and the same midpoint-refined, time-inhomogeneous MMPP surrogate as
    :func:`estimate_exponential_d_event_qmle`; no bins enter its objective.
    """

    started = time.perf_counter()
    method_id = "exponential_d_joint_occupancy_event_qmle_warm"
    times = _validate_event_times(event_time_s, duration_s)
    if len(times) < 20:
        return VectorDEstimate(
            method_id=method_id,
            parameter_names=("D_start", "D_end"),
            d_estimates_um2_s=(float("nan"), float("nan")),
            objective=float("nan"),
            success=False,
            at_search_boundary=False,
            failure_reason="too_few_photons",
            runtime_s=time.perf_counter() - started,
            n_input_photons=len(times),
            n_analysis_points=len(times),
            diagnostics={
                "estimator_visible_input": "event_time_s only",
                "simulation_truth_used": False,
            },
        )
    if not (0 < d_min_um2_s < d_max_um2_s):
        raise ValueError("D bounds must satisfy 0 < d_min < d_max")
    if not (0 < occupancy_min < occupancy_max):
        raise ValueError("occupancy bounds must satisfy 0 < min < max")
    if optimizer_maxiter < 1 or int(n_refinement_intervals) < 1:
        raise ValueError("optimizer and refinement counts must be positive")
    if (
        optimizer_ftol <= 0
        or optimizer_gtol <= 0
        or optimizer_finite_difference_step <= 0
        or optimizer_boundary_log_tolerance <= 0
    ):
        raise ValueError("optimizer tolerances must be positive")
    _validate_likelihood_controls(
        wxy_um=wxy_um,
        kappa=kappa,
        molecular_brightness_cps=molecular_brightness_cps,
        background_cps=background_cps,
        mean_occupancy=1.0,
        state_max=state_max,
    )

    local_result = estimate_local_window_equal_lse(
        times,
        duration_s=duration_s,
        window_s=window_s,
        step_s=step_s,
        bin_s=bin_s,
        curve_max_lag_s=curve_max_lag_s,
        fit_max_lag_s=fit_max_lag_s,
        n_lags=n_lags,
        wxy_um=wxy_um,
        kappa=kappa,
        free_baseline=free_baseline,
        tau_grid_size=tau_grid_size,
    )
    warm_d, warm_source = _warm_start_exponential_profile(
        local_result,
        duration_s=duration_s,
        d_min_um2_s=d_min_um2_s,
        d_max_um2_s=d_max_um2_s,
        estimator_label="local Equal-LSE",
    )
    warm_occupancy, raw_occupancy, occupancy_start_clipped = _plugin_occupancy(
        n_events=len(times),
        duration_s=duration_s,
        molecular_brightness_cps=molecular_brightness_cps,
        background_cps=background_cps,
        occupancy_min=occupancy_min,
        occupancy_max=occupancy_max,
    )
    if not np.isfinite(warm_occupancy):
        return VectorDEstimate(
            method_id=method_id,
            parameter_names=("D_start", "D_end"),
            d_estimates_um2_s=(float("nan"), float("nan")),
            objective=float("nan"),
            success=False,
            at_search_boundary=False,
            failure_reason="observed_rate_not_above_background",
            runtime_s=time.perf_counter() - started,
            n_input_photons=len(times),
            n_analysis_points=len(times),
            diagnostics={
                "estimator_visible_input": "event_time_s only",
                "simulation_truth_used": False,
                "warm_start_d_um2_s": warm_d.tolist(),
                "warm_start_source": warm_source,
            },
        )

    log_d_bounds = (
        math.log(float(d_min_um2_s)),
        math.log(float(d_max_um2_s)),
    )
    log_occupancy_bounds = (
        math.log(float(occupancy_min)),
        math.log(float(occupancy_max)),
    )

    def objective(log_parameters: np.ndarray) -> float:
        d_start, d_end, occupancy = np.exp(
            np.asarray(log_parameters, dtype=float)
        )
        log_likelihood, _ = (
            evaluate_exponential_d_event_qmle_loglikelihood(
                times,
                duration_s=duration_s,
                d_start_um2_s=float(d_start),
                d_end_um2_s=float(d_end),
                wxy_um=wxy_um,
                kappa=kappa,
                molecular_brightness_cps=molecular_brightness_cps,
                background_cps=background_cps,
                mean_occupancy=float(occupancy),
                state_max=state_max,
                n_refinement_intervals=n_refinement_intervals,
            )
        )
        return -float(log_likelihood)

    optimization = minimize(
        objective,
        x0=np.log(np.append(warm_d, warm_occupancy)),
        bounds=[log_d_bounds, log_d_bounds, log_occupancy_bounds],
        method="L-BFGS-B",
        options={
            "ftol": float(optimizer_ftol),
            "gtol": float(optimizer_gtol),
            "eps": float(optimizer_finite_difference_step),
            "maxiter": int(optimizer_maxiter),
            "maxls": 30,
        },
    )
    log_hat = np.asarray(optimization.x, dtype=float)
    d_hat = np.exp(log_hat[:2])
    occupancy_hat = math.exp(float(log_hat[2]))
    log_likelihood, likelihood_diagnostics = (
        evaluate_exponential_d_event_qmle_loglikelihood(
            times,
            duration_s=duration_s,
            d_start_um2_s=float(d_hat[0]),
            d_end_um2_s=float(d_hat[1]),
            wxy_um=wxy_um,
            kappa=kappa,
            molecular_brightness_cps=molecular_brightness_cps,
            background_cps=background_cps,
            mean_occupancy=occupancy_hat,
            state_max=state_max,
            n_refinement_intervals=n_refinement_intervals,
        )
    )
    d_at_boundary = [
        bool(
            value - log_d_bounds[0] <= optimizer_boundary_log_tolerance
            or log_d_bounds[1] - value <= optimizer_boundary_log_tolerance
        )
        for value in log_hat[:2]
    ]
    occupancy_at_boundary = bool(
        log_hat[2] - log_occupancy_bounds[0]
        <= optimizer_boundary_log_tolerance
        or log_occupancy_bounds[1] - log_hat[2]
        <= optimizer_boundary_log_tolerance
    )
    truncation_ok = bool(
        likelihood_diagnostics.get("stationary_top_probability", 1.0) < 1e-8
        and likelihood_diagnostics.get("max_filtered_top_probability", 1.0)
        < 1e-6
    )
    success = bool(
        optimization.success
        and np.all(np.isfinite(d_hat))
        and np.isfinite(occupancy_hat)
        and np.isfinite(log_likelihood)
        and not any(d_at_boundary)
        and not occupancy_at_boundary
        and truncation_ok
    )
    reasons: list[str] = []
    if not optimization.success:
        reasons.append("optimizer_failure")
    if not np.all(np.isfinite(d_hat)):
        reasons.append("nonfinite_d")
    if not np.isfinite(occupancy_hat):
        reasons.append("nonfinite_occupancy")
    if not np.isfinite(log_likelihood):
        reasons.append("nonfinite_loglikelihood")
    if any(d_at_boundary):
        reasons.append("d_search_boundary")
    if occupancy_at_boundary:
        reasons.append("occupancy_search_boundary")
    if not truncation_ok:
        reasons.append("state_truncation_mass_too_large")

    return VectorDEstimate(
        method_id=method_id,
        parameter_names=("D_start", "D_end"),
        d_estimates_um2_s=(float(d_hat[0]), float(d_hat[1])),
        objective=float(-log_likelihood),
        success=success,
        at_search_boundary=bool(
            any(d_at_boundary) or occupancy_at_boundary
        ),
        failure_reason=";".join(reasons),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        n_analysis_points=len(times),
        diagnostics={
            **likelihood_diagnostics,
            "d_bounds_um2_s": [
                float(d_min_um2_s),
                float(d_max_um2_s),
            ],
            "d_at_search_boundary_by_parameter": d_at_boundary,
            "occupancy_bounds": [
                float(occupancy_min),
                float(occupancy_max),
            ],
            "occupancy_at_search_boundary": occupancy_at_boundary,
            "occupancy_is_optimized": True,
            "initial_mean_occupancy": float(warm_occupancy),
            "raw_moment_mean_occupancy": float(raw_occupancy),
            "occupancy_start_clipped": bool(occupancy_start_clipped),
            "estimated_mean_occupancy": float(occupancy_hat),
            "finite_state_surrogate_mle": True,
            "full_latent_brownian_likelihood": False,
            "initialization_uses_binned_estimator": True,
            "likelihood_objective_uses_arrival_time_bins": False,
            "warm_start_method": (
                "local Equal-LSE log-linear endpoint extrapolation for D; "
                "whole-record photon-rate moment for shared mean occupancy"
            ),
            "warm_start_d_um2_s": warm_d.astype(float).tolist(),
            "warm_start_source": warm_source,
            "warm_start_runtime_s": float(local_result.runtime_s),
            "n_successful_local_equal_lse": int(
                sum(window.estimate.success for window in local_result.windows)
            ),
            "optimizer": (
                "bounded L-BFGS-B on "
                "(log D_start,log D_end,log mean occupancy)"
            ),
            "optimizer_success": bool(optimization.success),
            "optimizer_status": int(getattr(optimization, "status", 0)),
            "optimizer_message": str(optimization.message),
            "optimizer_iterations": int(getattr(optimization, "nit", 0)),
            "optimizer_function_evaluations": int(
                getattr(optimization, "nfev", 0)
            ),
        },
    )


__all__ = [
    "KnownBlockACFResult",
    "LocalEqualLSEResult",
    "LocalWindowEstimate",
    "LocalWLSEResult",
    "VectorDEstimate",
    "estimate_exponential_d_event_qmle",
    "estimate_exponential_d_joint_event_qmle",
    "estimate_known_block_acf",
    "estimate_known_block_event_qmle",
    "estimate_known_block_joint_event_qmle",
    "estimate_local_window_equal_lse",
    "estimate_local_window_wlse",
    "evaluate_exponential_d_event_qmle_loglikelihood",
    "evaluate_fixed_cut_event_qmle_loglikelihood",
    "split_events_half_open",
]
