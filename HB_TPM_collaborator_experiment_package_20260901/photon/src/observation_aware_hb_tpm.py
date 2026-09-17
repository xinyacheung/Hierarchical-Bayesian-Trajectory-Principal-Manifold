"""Observation-aware hierarchical trajectory transfer for photon arrivals.

This module implements the first auditable HB-TPM bridge to the PEX5/FCS
project.  A trajectory is represented by the two coefficients of a log-linear
diffusion curve,

    log D(t) = (1 - t/T) * w_start + (t/T) * w_end.

Both source fitting and target adaptation use only ordered photon arrival
times.  Latent Brownian paths, molecule identities, conditional intensities,
and the true diffusion schedule are intentionally absent from every public
estimation API.

The photon likelihood is the existing time-inhomogeneous immigration--death
MMPP quasi-likelihood.  It is exact for its midpoint-frozen finite-state
surrogate, approximate for the time-inhomogeneous surrogate, and a
quasi-likelihood for Brownian motion observed through a Gaussian PSF.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from fcs_time_varying_estimators import (
    evaluate_exponential_d_event_qmle_loglikelihood,
)


@dataclass(frozen=True)
class GaussianTrajectoryPrior:
    """Empirical-Bayes prior for ``(log D_start, log D_end)``."""

    mean_log_d: tuple[float, float]
    covariance_log_d: tuple[tuple[float, float], tuple[float, float]]
    n_source_records: int
    n_successful_source_fits: int
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationAwareEstimate:
    """One target-only or hierarchical fit with Laplace uncertainty."""

    method_id: str
    log_d_estimate: tuple[float, float]
    d_estimate_um2_s: tuple[float, float]
    covariance_log_d: tuple[tuple[float, float], tuple[float, float]]
    ci95_d_um2_s: tuple[tuple[float, float], tuple[float, float]]
    objective: float
    success: bool
    at_search_boundary: bool
    failure_reason: str
    runtime_s: float
    n_input_photons: int
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_events(event_time_s: np.ndarray, duration_s: float) -> np.ndarray:
    times = np.asarray(event_time_s, dtype=float)
    if times.ndim != 1:
        raise ValueError("event_time_s must be one-dimensional")
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive and finite")
    if np.any(~np.isfinite(times)):
        raise ValueError("event_time_s contains non-finite values")
    if np.any(times < 0) or np.any(times >= duration_s):
        raise ValueError("event_time_s must lie in [0, duration_s)")
    if len(times) > 1 and np.any(np.diff(times) < 0):
        times = np.sort(times)
    return times


def _plugin_mean_occupancy(
    *,
    n_events: int,
    duration_s: float,
    molecular_brightness_cps: float,
    background_cps: float,
    occupancy_bounds: tuple[float, float],
) -> tuple[float, float, bool]:
    lower, upper = (float(value) for value in occupancy_bounds)
    if not (0 < lower < upper):
        raise ValueError("occupancy bounds must satisfy 0 < lower < upper")
    effective_brightness = float(molecular_brightness_cps) / (2.0**1.5)
    raw = (
        float(n_events) / float(duration_s) - float(background_cps)
    ) / effective_brightness
    if not np.isfinite(raw) or raw <= 0:
        return float("nan"), float(raw), False
    clipped = float(np.clip(raw, lower, upper))
    return clipped, float(raw), bool(
        not np.isclose(clipped, raw, rtol=0.0, atol=1e-14)
    )


def finite_difference_hessian(
    objective,
    point: np.ndarray,
    *,
    step: float = 2e-3,
) -> np.ndarray:
    """Return a symmetric central-difference Hessian at ``point``."""

    x = np.asarray(point, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("point must be a finite one-dimensional vector")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be positive and finite")
    dimension = len(x)
    hessian = np.zeros((dimension, dimension), dtype=float)
    center = float(objective(x))
    for row in range(dimension):
        direction = np.zeros(dimension, dtype=float)
        direction[row] = step
        hessian[row, row] = (
            float(objective(x + direction))
            - 2.0 * center
            + float(objective(x - direction))
        ) / step**2
        for column in range(row + 1, dimension):
            second = np.zeros(dimension, dtype=float)
            second[column] = step
            value = (
                float(objective(x + direction + second))
                - float(objective(x + direction - second))
                - float(objective(x - direction + second))
                + float(objective(x - direction - second))
            ) / (4.0 * step**2)
            hessian[row, column] = value
            hessian[column, row] = value
    return 0.5 * (hessian + hessian.T)


def _regularized_inverse_information(
    information: np.ndarray,
    *,
    eigenvalue_floor: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    matrix = np.asarray(information, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    regularized = np.maximum(values, float(eigenvalue_floor))
    covariance = (vectors * (1.0 / regularized)) @ vectors.T
    return covariance, values, bool(np.any(values < eigenvalue_floor))


def estimate_gaussian_coefficient_prior(
    source_log_d_estimates: np.ndarray,
    source_covariances_log_d: np.ndarray,
    *,
    variance_floor: float = 0.02**2,
    covariance_shrinkage: float = 0.15,
) -> GaussianTrajectoryPrior:
    """Estimate a deconvolved Gaussian prior from photon-derived fits.

    The between-record covariance is the sample covariance of source QMLEs
    minus their average Laplace covariance.  A small eigenvalue floor and
    diagonal shrinkage keep the empirical prior proper in a small source set.
    """

    estimates = np.asarray(source_log_d_estimates, dtype=float)
    covariances = np.asarray(source_covariances_log_d, dtype=float)
    if estimates.ndim != 2 or estimates.shape[1] != 2:
        raise ValueError("source_log_d_estimates must have shape (n, 2)")
    if covariances.shape != (len(estimates), 2, 2):
        raise ValueError("source_covariances_log_d must have shape (n, 2, 2)")
    if len(estimates) < 3:
        raise ValueError("at least three successful source fits are required")
    if not np.all(np.isfinite(estimates)) or not np.all(np.isfinite(covariances)):
        raise ValueError("source estimates and covariances must be finite")
    if not (0.0 <= covariance_shrinkage <= 1.0):
        raise ValueError("covariance_shrinkage must lie in [0, 1]")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")

    mean = np.mean(estimates, axis=0)
    observed_covariance = np.cov(estimates, rowvar=False, ddof=1)
    mean_measurement_covariance = np.mean(covariances, axis=0)
    raw_between = 0.5 * (
        observed_covariance
        - mean_measurement_covariance
        + (observed_covariance - mean_measurement_covariance).T
    )
    values, vectors = np.linalg.eigh(raw_between)
    clipped_values = np.maximum(values, float(variance_floor))
    positive_between = (vectors * clipped_values) @ vectors.T
    diagonal = np.diag(np.diag(positive_between))
    covariance = (
        (1.0 - float(covariance_shrinkage)) * positive_between
        + float(covariance_shrinkage) * diagonal
    )
    covariance = 0.5 * (covariance + covariance.T)
    return GaussianTrajectoryPrior(
        mean_log_d=(float(mean[0]), float(mean[1])),
        covariance_log_d=(
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        ),
        n_source_records=int(len(estimates)),
        n_successful_source_fits=int(len(estimates)),
        diagnostics={
            "source_information": "photon event quasi-likelihood fits only",
            "simulation_truth_used": False,
            "observed_covariance_log_d": observed_covariance.tolist(),
            "mean_source_measurement_covariance_log_d": (
                mean_measurement_covariance.tolist()
            ),
            "raw_deconvolved_eigenvalues": values.tolist(),
            "variance_floor": float(variance_floor),
            "covariance_shrinkage": float(covariance_shrinkage),
        },
    )


def fit_log_linear_event_trajectory(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    wxy_um: float,
    kappa: float,
    molecular_brightness_cps: float,
    background_cps: float,
    d_bounds_um2_s: tuple[float, float] = (5.0, 300.0),
    occupancy_bounds: tuple[float, float] = (0.02, 6.0),
    state_max: int = 12,
    n_refinement_intervals: int = 16,
    prior: GaussianTrajectoryPrior | None = None,
    warm_start_log_d: Sequence[float] | None = None,
    optimizer_maxiter: int = 80,
    optimizer_ftol: float = 1e-9,
    optimizer_gtol: float = 1e-5,
    hessian_step: float = 2e-3,
    information_eigenvalue_floor: float = 1e-4,
) -> ObservationAwareEstimate:
    """Fit a target-only QMLE or an observation-aware hierarchical MAP."""

    started = time.perf_counter()
    times = _validate_events(event_time_s, duration_s)
    method_id = "oa_hb_tpm_map" if prior is not None else "target_only_event_qmle"
    if len(times) < 20:
        return ObservationAwareEstimate(
            method_id=method_id,
            log_d_estimate=(float("nan"), float("nan")),
            d_estimate_um2_s=(float("nan"), float("nan")),
            covariance_log_d=((float("nan"),) * 2,) * 2,
            ci95_d_um2_s=((float("nan"),) * 2,) * 2,
            objective=float("nan"),
            success=False,
            at_search_boundary=False,
            failure_reason="too_few_photons",
            runtime_s=time.perf_counter() - started,
            n_input_photons=len(times),
            diagnostics={
                "estimator_visible_input": "event_time_s only",
                "simulation_truth_used": False,
            },
        )

    d_lower, d_upper = (float(value) for value in d_bounds_um2_s)
    if not (0 < d_lower < d_upper):
        raise ValueError("D bounds must satisfy 0 < lower < upper")
    log_bounds = (math.log(d_lower), math.log(d_upper))
    occupancy, raw_occupancy, occupancy_clipped = _plugin_mean_occupancy(
        n_events=len(times),
        duration_s=duration_s,
        molecular_brightness_cps=molecular_brightness_cps,
        background_cps=background_cps,
        occupancy_bounds=occupancy_bounds,
    )
    if not np.isfinite(occupancy):
        return ObservationAwareEstimate(
            method_id=method_id,
            log_d_estimate=(float("nan"), float("nan")),
            d_estimate_um2_s=(float("nan"), float("nan")),
            covariance_log_d=((float("nan"),) * 2,) * 2,
            ci95_d_um2_s=((float("nan"),) * 2,) * 2,
            objective=float("nan"),
            success=False,
            at_search_boundary=False,
            failure_reason="observed_rate_not_above_background",
            runtime_s=time.perf_counter() - started,
            n_input_photons=len(times),
            diagnostics={
                "estimator_visible_input": "event_time_s only",
                "simulation_truth_used": False,
                "raw_plugin_mean_occupancy": float(raw_occupancy),
            },
        )

    if prior is None:
        prior_mean = None
        prior_precision = None
    else:
        prior_mean = np.asarray(prior.mean_log_d, dtype=float)
        prior_covariance = np.asarray(prior.covariance_log_d, dtype=float)
        prior_precision = np.linalg.inv(prior_covariance)

    def negative_log_posterior(log_d: np.ndarray) -> float:
        parameters = np.asarray(log_d, dtype=float)
        d_start, d_end = np.exp(parameters)
        log_likelihood, _ = evaluate_exponential_d_event_qmle_loglikelihood(
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
        if not np.isfinite(log_likelihood):
            return float("inf")
        penalty = 0.0
        if prior_mean is not None and prior_precision is not None:
            delta = parameters - prior_mean
            penalty = 0.5 * float(delta @ prior_precision @ delta)
        return -float(log_likelihood) + penalty

    if warm_start_log_d is not None:
        warm = np.asarray(tuple(warm_start_log_d), dtype=float)
        if warm.shape != (2,) or not np.all(np.isfinite(warm)):
            raise ValueError("warm_start_log_d must contain two finite values")
    elif prior_mean is not None:
        warm = prior_mean.copy()
    else:
        midpoint = 0.5 * (log_bounds[0] + log_bounds[1])
        warm = np.asarray([midpoint, midpoint], dtype=float)
    warm = np.clip(warm, log_bounds[0], log_bounds[1])

    optimization = minimize(
        negative_log_posterior,
        x0=warm,
        bounds=[log_bounds, log_bounds],
        method="L-BFGS-B",
        options={
            "maxiter": int(optimizer_maxiter),
            "ftol": float(optimizer_ftol),
            "gtol": float(optimizer_gtol),
            "maxls": 30,
        },
    )
    estimate = np.asarray(optimization.x, dtype=float)
    boundary_tolerance = 1e-4
    at_boundary_by_parameter = [
        bool(
            value - log_bounds[0] <= boundary_tolerance
            or log_bounds[1] - value <= boundary_tolerance
        )
        for value in estimate
    ]
    hessian = finite_difference_hessian(
        negative_log_posterior,
        estimate,
        step=hessian_step,
    )
    covariance, raw_information_eigenvalues, information_regularized = (
        _regularized_inverse_information(
            hessian,
            eigenvalue_floor=information_eigenvalue_floor,
        )
    )
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    ci_log = np.column_stack(
        (estimate - 1.96 * standard_errors, estimate + 1.96 * standard_errors)
    )
    ci_d = np.exp(ci_log)
    d_estimate = np.exp(estimate)
    objective_value = float(negative_log_posterior(estimate))
    success = bool(
        optimization.success
        and np.all(np.isfinite(estimate))
        and np.isfinite(objective_value)
        and not any(at_boundary_by_parameter)
        and not occupancy_clipped
    )
    reasons: list[str] = []
    if not optimization.success:
        reasons.append("optimizer_failure")
    if any(at_boundary_by_parameter):
        reasons.append("d_search_boundary")
    if occupancy_clipped:
        reasons.append("occupancy_plugin_boundary")
    if information_regularized:
        reasons.append("observed_information_regularized")

    return ObservationAwareEstimate(
        method_id=method_id,
        log_d_estimate=(float(estimate[0]), float(estimate[1])),
        d_estimate_um2_s=(float(d_estimate[0]), float(d_estimate[1])),
        covariance_log_d=(
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        ),
        ci95_d_um2_s=(
            (float(ci_d[0, 0]), float(ci_d[0, 1])),
            (float(ci_d[1, 0]), float(ci_d[1, 1])),
        ),
        objective=objective_value,
        success=success,
        at_search_boundary=bool(any(at_boundary_by_parameter)),
        failure_reason=";".join(reasons),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        diagnostics={
            "estimator_visible_input": "event_time_s only",
            "simulation_truth_used": False,
            "trajectory_basis": "linear basis on log D at normalized time t/T",
            "observation_model": (
                "midpoint-refined time-inhomogeneous immigration-death "
                "MMPP event quasi-likelihood"
            ),
            "prior_used": prior is not None,
            "prior_learned_from": (
                "source photon event likelihood fits" if prior is not None else None
            ),
            "plugin_mean_occupancy": float(occupancy),
            "raw_plugin_mean_occupancy": float(raw_occupancy),
            "occupancy_plugin_clipped": bool(occupancy_clipped),
            "n_refinement_intervals": int(n_refinement_intervals),
            "optimizer": "bounded L-BFGS-B on (log D_start, log D_end)",
            "optimizer_success": bool(optimization.success),
            "optimizer_message": str(optimization.message),
            "optimizer_iterations": int(getattr(optimization, "nit", 0)),
            "raw_observed_information_eigenvalues": (
                raw_information_eigenvalues.tolist()
            ),
            "observed_information_regularized": bool(information_regularized),
        },
    )


def learn_prior_from_event_records(
    source_event_times_s: Sequence[np.ndarray],
    *,
    duration_s: float,
    likelihood_controls: Mapping[str, Any],
    variance_floor: float = 0.02**2,
    covariance_shrinkage: float = 0.15,
) -> tuple[GaussianTrajectoryPrior, tuple[ObservationAwareEstimate, ...]]:
    """Fit source photon records and learn the empirical trajectory prior."""

    source_fits = tuple(
        fit_log_linear_event_trajectory(
            np.asarray(events, dtype=float).copy(),
            duration_s=duration_s,
            prior=None,
            **dict(likelihood_controls),
        )
        for events in source_event_times_s
    )
    successful = [fit for fit in source_fits if fit.success]
    if len(successful) < 3:
        raise RuntimeError(
            "at least three successful source photon-likelihood fits are required"
        )
    estimates = np.asarray([fit.log_d_estimate for fit in successful], dtype=float)
    covariances = np.asarray(
        [fit.covariance_log_d for fit in successful], dtype=float
    )
    prior = estimate_gaussian_coefficient_prior(
        estimates,
        covariances,
        variance_floor=variance_floor,
        covariance_shrinkage=covariance_shrinkage,
    )
    prior = GaussianTrajectoryPrior(
        mean_log_d=prior.mean_log_d,
        covariance_log_d=prior.covariance_log_d,
        n_source_records=len(source_fits),
        n_successful_source_fits=len(successful),
        diagnostics={
            **prior.diagnostics,
            "failed_source_fits": int(len(source_fits) - len(successful)),
            "estimator_visible_input": "event_time_s only",
        },
    )
    return prior, source_fits


def fit_observation_aware_target(
    event_time_s: np.ndarray,
    *,
    prior: GaussianTrajectoryPrior,
    duration_s: float,
    likelihood_controls: Mapping[str, Any],
) -> ObservationAwareEstimate:
    """Adapt the learned source prior to one sparse target photon record."""

    return fit_log_linear_event_trajectory(
        np.asarray(event_time_s, dtype=float).copy(),
        duration_s=duration_s,
        prior=prior,
        **dict(likelihood_controls),
    )

