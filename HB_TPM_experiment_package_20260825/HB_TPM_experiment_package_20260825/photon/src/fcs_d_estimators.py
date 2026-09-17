"""Diffusion estimators that only consume estimator-visible photon times.

The functions in this module deliberately accept a one-dimensional array of
``event_time_s`` values rather than a full simulation result.  This makes it
hard to accidentally use latent trajectories, molecule identities, or source
labels while estimating ``D``.

The implemented estimators are:

``acf_equal_weight``
    The project's original fixed-baseline, equal-weight ACF fit.

``acf_free_baseline``
    The same empirical ACF with a fitted finite-window baseline.

``acf_weighted``
    A feasible diagonal weighted least-squares ACF fit.  Lag-specific
    variances are estimated from a circular moving-block bootstrap of the
    autocovariance score sequence, then stabilized by shrinkage and a weight
    ratio cap.

``immigration_death_event_qmle``
    A direct arrival-time quasi-likelihood.  It approximates the Gaussian-PSF
    Brownian fluorescence field by a stationary immigration-death molecule
    count and evaluates the resulting Markov-modulated Poisson event
    likelihood with a finite-state forward recursion.

``immigration_death_joint_mle_warm``
    A warm-started finite-state immigration-death MMPP likelihood fit that
    jointly estimates diffusion and stationary mean occupancy.  It is an MLE
    for the calibrated truncated surrogate and a QMLE for Brownian
    Gaussian-PSF photon data.

``whittle_count_qmle``
    A Gaussian Whittle quasi-likelihood for the complete binned count trace.
    It models shot noise plus the theoretical 3-D FCS covariance spectrum.

``event_pair_composite``
    A second-order composite Poisson likelihood for positive photon-pair
    lags.  It consumes exact arrival times and only bins the pair lags.

``unbinned_pair_composite``
    A continuous pair-lag composite likelihood evaluated at every selected
    photon-pair lag.  It uses neither base count bins nor pair-lag bins.

The likelihood methods are transparent, computationally light alternatives for
the benchmark.  They are *not* reproductions of the latent-trajectory MCMC
algorithms in Jazani et al. (2019) or Karimi et al. (2025), and the pairwise
objectives are composite rather than the full marginal likelihood obtained by
integrating over all latent Brownian paths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import NormalDist
import time
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln


@dataclass(frozen=True)
class DEstimate:
    """One estimator result with a small, JSON-friendly diagnostic payload."""

    method_id: str
    d_estimate_um2_s: float
    tau_d_estimate_s: float
    amplitude: float
    baseline: float
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


METHOD_LABELS = {
    "acf_equal_weight": "ACF-LS (baseline = 0)",
    "acf_free_baseline": "ACF-LS (free baseline)",
    "acf_weighted": "ACF W-LSE (bootstrap inverse variance)",
    "immigration_death_event_qmle": (
        "Immigration–death MMPP plug-in QMLE"
    ),
    "immigration_death_joint_mle_warm": (
        "Joint immigration–death MMPP surrogate QMLE"
    ),
    "whittle_count_qmle": "Whittle count QMLE",
    "event_pair_composite": "Event-pair composite",
    "unbinned_pair_composite": "Unbinned pair composite likelihood",
}

MAX_DIAGNOSTIC_LAG_FRACTION = 0.35


def _validate_event_times(event_time_s: np.ndarray, duration_s: float) -> np.ndarray:
    """Return sorted finite event times inside ``[0, duration_s)``."""

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


def _uniform_counts(
    event_time_s: np.ndarray,
    duration_s: float,
    bin_s: float,
) -> np.ndarray:
    """Bin event times into equal-width bins covering the full duration."""

    if not np.isfinite(bin_s) or bin_s <= 0:
        raise ValueError("bin_s must be finite and positive")
    n_bins_float = duration_s / bin_s
    n_bins = int(round(n_bins_float))
    if n_bins < 16 or not np.isclose(n_bins * bin_s, duration_s, rtol=0.0, atol=1e-12):
        raise ValueError("duration_s must be an integer multiple of bin_s with at least 16 bins")
    edges = np.linspace(0.0, duration_s, n_bins + 1)
    counts, _ = np.histogram(event_time_s, bins=edges)
    return counts.astype(float)


def fcs_shape(lag_s: np.ndarray, tau_d_s: float, kappa: float) -> np.ndarray:
    """Unit-amplitude 3-D Gaussian FCS correlation shape."""

    lag = np.asarray(lag_s, dtype=float)
    return 1.0 / (
        (1.0 + lag / tau_d_s)
        * np.sqrt(1.0 + lag / (kappa**2 * tau_d_s))
    )


def _profile_acf_matrix_wls(
    response: np.ndarray,
    basis: np.ndarray,
    weights: np.ndarray,
    *,
    free_baseline: bool,
) -> tuple[float, float, float, np.ndarray]:
    """Profile the linear ACF coefficients in ``r.T @ W @ r``.

    For fixed ``tau_D``, ``basis`` is the vector
    ``f_D = (f_D(tau_1), ..., f_D(tau_L))`` and ``W`` is the diagonal matrix
    with ``weights`` on its diagonal.  The design matrix is ``f_D[:, None]``
    for the fixed-zero-baseline fit and ``[f_D, 1]`` when the baseline is
    free.  The amplitude is constrained to be nonnegative.
    """

    y = np.asarray(response, dtype=float)
    f_d = np.asarray(basis, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if y.ndim != 1 or f_d.ndim != 1 or weight.ndim != 1:
        raise ValueError("response, basis, and weights must be one-dimensional")
    if not (len(y) == len(f_d) == len(weight)) or len(y) == 0:
        raise ValueError("response, basis, and weights must have equal length")
    if (
        not np.all(np.isfinite(y))
        or not np.all(np.isfinite(f_d))
        or not np.all(np.isfinite(weight))
        or np.any(weight <= 0)
    ):
        raise ValueError("W-LSE inputs must be finite and weights positive")

    design = (
        np.column_stack((f_d, np.ones_like(f_d)))
        if free_baseline
        else f_d[:, None]
    )
    square_root_weight = np.sqrt(weight)
    weighted_design = square_root_weight[:, None] * design
    weighted_response = square_root_weight * y
    coefficients = np.linalg.lstsq(
        weighted_design,
        weighted_response,
        rcond=None,
    )[0]
    amplitude = max(0.0, float(coefficients[0]))
    if free_baseline:
        baseline = float(
            np.dot(weight, y - amplitude * f_d) / np.sum(weight)
        )
    else:
        baseline = 0.0
    residual = y - amplitude * f_d - baseline
    objective = float(residual @ (weight * residual))
    return amplitude, baseline, objective, residual


def _fit_acf_matrix_wls(
    lag_s: np.ndarray,
    response: np.ndarray,
    tau_grid_s: np.ndarray,
    *,
    kappa: float,
    weights: np.ndarray,
    free_baseline: bool,
) -> tuple[float, float, float, float, int]:
    """Grid-profile ``tau_D`` under a diagonal matrix W-LSE objective."""

    lag = np.asarray(lag_s, dtype=float)
    y = np.asarray(response, dtype=float)
    tau_grid = np.asarray(tau_grid_s, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if lag.ndim != 1 or y.ndim != 1 or tau_grid.ndim != 1:
        raise ValueError("lag_s, response, and tau_grid_s must be vectors")
    if len(lag) != len(y) or len(lag) != len(weight):
        raise ValueError("lag, response, and weights must have equal length")
    if len(tau_grid) < 2 or np.any(~np.isfinite(tau_grid)) or np.any(tau_grid <= 0):
        raise ValueError("tau_grid_s must contain at least two positive values")

    best = (float("inf"), float("nan"), float("nan"), float("nan"), -1)
    for grid_index, tau_s in enumerate(tau_grid):
        basis = fcs_shape(lag, float(tau_s), kappa)
        amplitude, baseline, score, _ = _profile_acf_matrix_wls(
            y,
            basis,
            weight,
            free_baseline=free_baseline,
        )
        if score < best[0]:
            best = (
                float(score),
                float(tau_s),
                float(amplitude),
                float(baseline),
                int(grid_index),
            )
    return best


def _circular_block_bootstrap_acf(
    centered_counts: np.ndarray,
    mean_count: float,
    lag_indices: np.ndarray,
    *,
    block_length_bins: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap selected empirical ACF ordinates from lag-product scores.

    At lag ``ell``, the score sequence is

    ``Z[k, ell] = (X[k] - Xbar) (X[k + ell] - Xbar) / Xbar**2``.

    Circular moving blocks of this dependent score sequence are resampled.
    Holding the observed mean fixed makes this a fast, explicit variance
    approximation for the empirical ACF rather than a bootstrap of the entire
    photon-generating process.  The returned matrix has one row per bootstrap
    replicate and one column per selected lag.
    """

    centered = np.asarray(centered_counts, dtype=float)
    lags = np.asarray(lag_indices, dtype=int)
    if centered.ndim != 1 or lags.ndim != 1:
        raise ValueError("centered_counts and lag_indices must be vectors")
    if len(centered) < 4 or np.any(lags <= 0) or np.any(lags >= len(centered)):
        raise ValueError("lag_indices must lie in 1, ..., n_counts - 1")
    if not np.isfinite(mean_count) or mean_count <= 0:
        raise ValueError("mean_count must be finite and positive")
    if block_length_bins < 1 or n_bootstrap < 2:
        raise ValueError("bootstrap block length and replicates are too small")

    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap_acf = np.empty((int(n_bootstrap), len(lags)), dtype=float)
    effective_block_lengths = np.empty(len(lags), dtype=int)
    denominator = float(mean_count) ** 2

    for column, lag_index in enumerate(lags):
        scores = (
            centered[:-lag_index] * centered[lag_index:] / denominator
        )
        n_scores = len(scores)
        # At least two blocks are needed for nondegenerate uncertainty.  The
        # circular construction still permits a requested physical block that
        # is long relative to a short acquisition.
        effective_block = min(
            int(block_length_bins),
            max(1, n_scores // 2),
        )
        effective_block_lengths[column] = effective_block
        extended = np.concatenate(
            (scores, scores[: max(0, effective_block - 1)])
        )
        cumulative = np.concatenate(([0.0], np.cumsum(extended)))

        n_full_blocks, remainder = divmod(n_scores, effective_block)
        replicate_sum = np.zeros(int(n_bootstrap), dtype=float)
        if n_full_blocks:
            starts = rng.integers(
                0,
                n_scores,
                size=(int(n_bootstrap), n_full_blocks),
            )
            block_sums = (
                cumulative[starts + effective_block] - cumulative[starts]
            )
            replicate_sum += np.sum(block_sums, axis=1)
        if remainder:
            remainder_starts = rng.integers(
                0,
                n_scores,
                size=int(n_bootstrap),
            )
            replicate_sum += (
                cumulative[remainder_starts + remainder]
                - cumulative[remainder_starts]
            )
        bootstrap_acf[:, column] = replicate_sum / float(n_scores)

    return bootstrap_acf, effective_block_lengths


def _stabilize_inverse_variance_weights(
    variance: np.ndarray,
    *,
    shrinkage: float,
    max_weight_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool]]:
    """Shrink lag variances and return mean-one capped inverse weights."""

    raw_variance = np.asarray(variance, dtype=float)
    if raw_variance.ndim != 1 or len(raw_variance) == 0:
        raise ValueError("variance must be a nonempty vector")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    if not np.isfinite(max_weight_ratio) or max_weight_ratio < 1.0:
        raise ValueError("max_weight_ratio must be finite and at least one")

    usable = raw_variance[np.isfinite(raw_variance) & (raw_variance > 0)]
    if len(usable):
        target = float(np.median(usable))
    else:
        target = 1.0
    floor = max(target * 1e-10, np.finfo(float).tiny)
    cleaned = np.where(
        np.isfinite(raw_variance) & (raw_variance > floor),
        raw_variance,
        target,
    )
    shrunk = (1.0 - float(shrinkage)) * cleaned + float(shrinkage) * target
    shrunk = np.maximum(shrunk, floor)

    raw_weights = target / shrunk
    half_ratio = math.sqrt(float(max_weight_ratio))
    bounded_weights = np.clip(
        raw_weights,
        1.0 / half_ratio,
        half_ratio,
    )
    weight_cap_applied = bool(
        np.any(np.abs(bounded_weights - raw_weights) > 1e-12)
    )
    capped_weights = bounded_weights / float(np.mean(bounded_weights))
    diagnostics: dict[str, float | bool] = {
        "variance_shrinkage_target": target,
        "variance_floor": floor,
        "weight_cap_applied": weight_cap_applied,
        "realized_weight_ratio": float(
            np.max(capped_weights) / np.min(capped_weights)
        ),
    }
    return shrunk, capped_weights, diagnostics


def _fcs_shape_and_log_d_derivative(
    lag_s: np.ndarray,
    d_um2_s: float,
    wxy_um: float,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``f_D`` and ``d f_D / d log(D)`` for the 3-D FCS shape."""

    lag = np.asarray(lag_s, dtype=float)
    scaled_lag = 4.0 * float(d_um2_s) * lag / float(wxy_um) ** 2
    shape = 1.0 / (
        (1.0 + scaled_lag)
        * np.sqrt(1.0 + scaled_lag / float(kappa) ** 2)
    )
    log_shape_derivative = (
        -scaled_lag / (1.0 + scaled_lag)
        - 0.5 * scaled_lag / (float(kappa) ** 2 + scaled_lag)
    )
    return shape, shape * log_shape_derivative


def _truncated_poisson_stationary(
    mean_occupancy: float,
    state_max: int,
) -> np.ndarray:
    """Return the normalized Poisson weights on states ``0, ..., state_max``."""

    if not np.isfinite(mean_occupancy) or mean_occupancy <= 0:
        raise ValueError("mean_occupancy must be finite and positive")
    if state_max < 4:
        raise ValueError("state_max must be at least 4")
    states = np.arange(state_max + 1, dtype=float)
    log_weights = (
        states * math.log(float(mean_occupancy))
        - gammaln(states + 1.0)
    )
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    return weights / float(np.sum(weights))


def _immigration_death_event_loglikelihood(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    d_um2_s: float,
    wxy_um: float,
    kappa: float,
    effective_brightness_cps: float,
    background_cps: float,
    mean_occupancy: float,
    state_max: int,
) -> tuple[float, dict[str, float]]:
    """Evaluate the finite-state immigration-death MMPP event likelihood.

    The working latent process ``N(t)`` has immigration rate ``lambda`` and
    per-molecule emigration rate ``mu``.  Conditional photon intensity in state
    ``n`` is ``background_cps + effective_brightness_cps * n``.  The mapping

    ``mu(D) = (4 D / wxy^2) * (1 + 1 / (2 kappa^2))``

    matches the initial slope of the exponential immigration-death covariance
    to the calibrated 3-D Gaussian FCS covariance.

    This is an exact ordered event-time likelihood for the finite-state
    surrogate.  It is only a quasi-likelihood for the original continuously
    weighted Brownian fluorescence process.
    """

    if not np.isfinite(d_um2_s) or d_um2_s <= 0:
        raise ValueError("d_um2_s must be finite and positive")
    if not np.isfinite(wxy_um) or wxy_um <= 0:
        raise ValueError("wxy_um must be finite and positive")
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    if not np.isfinite(effective_brightness_cps) or effective_brightness_cps <= 0:
        raise ValueError("effective_brightness_cps must be finite and positive")
    if not np.isfinite(background_cps) or background_cps < 0:
        raise ValueError("background_cps must be finite and nonnegative")

    times = np.asarray(event_time_s, dtype=float)
    states = np.arange(state_max + 1, dtype=float)
    stationary = _truncated_poisson_stationary(mean_occupancy, state_max)
    stationary_sqrt = np.sqrt(stationary)

    mu_s_inv = (
        4.0
        * float(d_um2_s)
        / float(wxy_um) ** 2
        * (1.0 + 1.0 / (2.0 * float(kappa) ** 2))
    )
    immigration_rate_s_inv = float(mean_occupancy) * mu_s_inv

    generator = np.zeros((state_max + 1, state_max + 1), dtype=float)
    indices = np.arange(state_max)
    generator[indices, indices + 1] = immigration_rate_s_inv
    generator[indices + 1, indices] = (indices + 1.0) * mu_s_inv
    generator[np.arange(state_max + 1), np.arange(state_max + 1)] = (
        -generator.sum(axis=1)
    )

    photon_rates = (
        float(background_cps) + float(effective_brightness_cps) * states
    )
    killed_generator = generator - np.diag(photon_rates)

    # Detailed balance makes this similarity transform symmetric, allowing a
    # stable real eigendecomposition reused for every inter-event interval.
    symmetric_generator = (
        stationary_sqrt[:, None]
        * killed_generator
        / stationary_sqrt[None, :]
    )
    eigenvalues, eigenvectors = eigh(
        0.5 * (symmetric_generator + symmetric_generator.T)
    )

    forward = stationary.copy()
    log_likelihood = 0.0
    previous_time = 0.0
    max_filtered_top_probability = float(forward[-1])

    def propagate(probabilities: np.ndarray, gap_s: float) -> np.ndarray:
        transformed = probabilities / stationary_sqrt
        coefficients = transformed @ eigenvectors
        propagated = (
            (coefficients * np.exp(eigenvalues * float(gap_s)))
            @ eigenvectors.T
            * stationary_sqrt
        )
        # Roundoff can create values of order -1e-16 in a nonnegative vector.
        return np.maximum(propagated, 0.0)

    for event_time in times:
        forward = propagate(forward, float(event_time) - previous_time)
        forward *= photon_rates
        scale = float(np.sum(forward))
        if not np.isfinite(scale) or scale <= 0:
            return float("-inf"), {
                "mu_s_inv": mu_s_inv,
                "immigration_rate_s_inv": immigration_rate_s_inv,
                "stationary_top_probability": float(stationary[-1]),
                "max_filtered_top_probability": max_filtered_top_probability,
            }
        log_likelihood += math.log(scale)
        forward /= scale
        max_filtered_top_probability = max(
            max_filtered_top_probability,
            float(forward[-1]),
        )
        previous_time = float(event_time)

    forward = propagate(forward, float(duration_s) - previous_time)
    final_scale = float(np.sum(forward))
    if not np.isfinite(final_scale) or final_scale <= 0:
        return float("-inf"), {
            "mu_s_inv": mu_s_inv,
            "immigration_rate_s_inv": immigration_rate_s_inv,
            "stationary_top_probability": float(stationary[-1]),
            "max_filtered_top_probability": max_filtered_top_probability,
        }
    log_likelihood += math.log(final_scale)
    return log_likelihood, {
        "mu_s_inv": mu_s_inv,
        "immigration_rate_s_inv": immigration_rate_s_inv,
        "stationary_top_probability": float(stationary[-1]),
        "max_filtered_top_probability": max_filtered_top_probability,
    }


def _failure(
    method_id: str,
    started: float,
    n_photons: int,
    reason: str,
    n_analysis_points: int = 0,
    diagnostics: dict[str, Any] | None = None,
) -> DEstimate:
    return DEstimate(
        method_id=method_id,
        d_estimate_um2_s=float("nan"),
        tau_d_estimate_s=float("nan"),
        amplitude=float("nan"),
        baseline=float("nan"),
        objective=float("nan"),
        success=False,
        at_search_boundary=False,
        failure_reason=reason,
        runtime_s=time.perf_counter() - started,
        n_input_photons=int(n_photons),
        n_analysis_points=int(n_analysis_points),
        diagnostics=diagnostics or {},
    )


def estimate_acf(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    bin_s: float,
    curve_max_lag_s: float,
    fit_max_lag_s: float,
    n_lags: int,
    wxy_um: float,
    kappa: float,
    free_baseline: bool,
    tau_grid_size: int = 300,
) -> DEstimate:
    """Estimate ``D`` from the overlap-corrected empirical count ACF."""

    started = time.perf_counter()
    method_id = "acf_free_baseline" if free_baseline else "acf_equal_weight"
    times = _validate_event_times(event_time_s, duration_s)
    counts = _uniform_counts(times, duration_s, bin_s)
    if len(times) < 20 or float(counts.mean()) <= 0:
        return _failure(method_id, started, len(times), "too_few_photons")

    n = len(counts)
    centered = counts - float(counts.mean())
    n_fft = 1 << int(np.ceil(np.log2(max(2, 2 * n))))
    spectrum = np.fft.rfft(centered, n=n_fft)
    raw = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft)[:n]
    covariance = raw / np.arange(n, 0, -1, dtype=float)
    g_all = covariance / max(float(counts.mean()) ** 2, 1e-15)

    max_lag_index = max(
        2,
        min(
            n - 1,
            int(
                min(
                    curve_max_lag_s,
                    duration_s * MAX_DIAGNOSTIC_LAG_FRACTION,
                )
                / bin_s
            ),
        ),
    )
    lag_indices = np.unique(
        np.clip(
            np.round(np.logspace(0, np.log10(max_lag_index), int(n_lags))).astype(int),
            1,
            max_lag_index,
        )
    )
    lag_s = lag_indices * bin_s
    valid = np.isfinite(g_all[lag_indices]) & (lag_s > 0) & (lag_s <= fit_max_lag_s)
    lag = lag_s[valid]
    response = g_all[lag_indices][valid]
    if len(lag) < 8:
        return _failure(
            method_id,
            started,
            len(times),
            "too_few_valid_lags",
            n_analysis_points=len(lag),
        )

    tau_grid = np.logspace(
        np.log10(max(float(lag.min()) / 3.0, 1e-12)),
        np.log10(float(fit_max_lag_s) * 3.0),
        int(tau_grid_size),
    )
    best = (float("inf"), float("nan"), float("nan"), float("nan"), -1)
    for grid_index, tau_s in enumerate(tau_grid):
        basis = fcs_shape(lag, float(tau_s), kappa)
        if free_baseline:
            design = np.column_stack((basis, np.ones_like(basis)))
            amplitude_raw, baseline_raw = np.linalg.lstsq(design, response, rcond=None)[0]
            amplitude = max(0.0, float(amplitude_raw))
            baseline = float(np.mean(response - amplitude * basis))
        else:
            amplitude = max(
                0.0,
                float(np.dot(basis, response) / max(np.dot(basis, basis), 1e-15)),
            )
            baseline = 0.0
        residual = response - amplitude * basis - baseline
        score = float(np.dot(residual, residual))
        if score < best[0]:
            best = (score, float(tau_s), amplitude, baseline, grid_index)

    objective, tau_hat, amplitude_hat, baseline_hat, best_index = best
    d_hat = wxy_um**2 / (4.0 * tau_hat)
    boundary = best_index in {0, len(tau_grid) - 1}
    success = bool(
        np.isfinite(d_hat)
        and d_hat > 0
        and np.isfinite(amplitude_hat)
        and amplitude_hat > 0
        and not boundary
    )
    reasons = []
    if not np.isfinite(d_hat) or d_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_d")
    if not np.isfinite(amplitude_hat) or amplitude_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_amplitude")
    if boundary:
        reasons.append("search_boundary")

    return DEstimate(
        method_id=method_id,
        d_estimate_um2_s=float(d_hat),
        tau_d_estimate_s=float(tau_hat),
        amplitude=float(amplitude_hat),
        baseline=float(baseline_hat),
        objective=float(objective),
        success=success,
        at_search_boundary=boundary,
        failure_reason=";".join(reasons),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        n_analysis_points=len(lag),
        diagnostics={
            "count_bin_s": float(bin_s),
            "mean_count_per_bin": float(counts.mean()),
            "zero_lag_included": False,
            "zero_lag_handling": (
                "excluded because the count variance contains the Poisson "
                "self-correlation (shot-noise) term"
            ),
            "minimum_fit_lag_bins": int(round(float(lag.min()) / float(bin_s))),
            "fit_lag_min_s": float(lag.min()),
            "fit_lag_max_s": float(lag.max()),
            "tau_grid_min_s": float(tau_grid[0]),
            "tau_grid_max_s": float(tau_grid[-1]),
            "tau_grid_size": int(len(tau_grid)),
        },
    )


def estimate_acf_weighted(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    bin_s: float,
    curve_max_lag_s: float,
    fit_max_lag_s: float,
    n_lags: int,
    wxy_um: float,
    kappa: float,
    free_baseline: bool = False,
    tau_grid_size: int = 300,
    fit_grid_max_lag_s: float | None = None,
    fit_n_lags: int | None = None,
    n_bootstrap: int = 200,
    block_length_s: float = 0.004,
    weight_shrinkage: float = 0.10,
    max_weight_ratio: float = 100.0,
    bootstrap_seed: int = 20260727,
    ci_level: float = 0.95,
) -> DEstimate:
    """Estimate ``D`` by feasible diagonal matrix W-LSE of the count ACF.

    Let ``y`` collect the empirical ACF at the selected fit lags and let
    ``F(tau_D)`` collect the corresponding theoretical 3-D FCS shape.  With a
    fixed-zero baseline, this estimator profiles ``A`` and ``tau_D`` in

    ``(y - A F(tau_D)).T W (y - A F(tau_D))``.

    When ``free_baseline=True``, the residual is
    ``y - [F(tau_D), 1] [A, B].T``.  The diagonal entries of ``W`` are
    stabilized inverse lag variances.  Raw variances come from a deterministic
    circular moving-block bootstrap of each lag-product score sequence;
    shrinkage toward the median and a symmetric weight cap make the feasible
    W-LSE usable for shorter acquisition windows.
    """

    started = time.perf_counter()
    method_id = "acf_weighted"
    times = _validate_event_times(event_time_s, duration_s)
    counts = _uniform_counts(times, duration_s, bin_s)
    if len(times) < 20 or float(counts.mean()) <= 0:
        return _failure(method_id, started, len(times), "too_few_photons")
    if int(n_lags) < 8:
        raise ValueError("n_lags must be at least 8")
    if int(tau_grid_size) < 2:
        raise ValueError("tau_grid_size must be at least 2")
    if fit_grid_max_lag_s is not None and (
        not np.isfinite(fit_grid_max_lag_s)
        or float(fit_grid_max_lag_s) <= 0
    ):
        raise ValueError("fit_grid_max_lag_s must be finite and positive")
    if fit_n_lags is not None and int(fit_n_lags) < 8:
        raise ValueError("fit_n_lags must be at least 8")
    if int(n_bootstrap) < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if not np.isfinite(block_length_s) or block_length_s <= 0:
        raise ValueError("block_length_s must be finite and positive")
    if not np.isfinite(ci_level) or not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must lie strictly between zero and one")

    n = len(counts)
    mean_count = float(counts.mean())
    centered = counts - mean_count
    n_fft = 1 << int(np.ceil(np.log2(max(2, 2 * n))))
    spectrum = np.fft.rfft(centered, n=n_fft)
    raw = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft)[:n]
    covariance = raw / np.arange(n, 0, -1, dtype=float)
    g_all = covariance / max(mean_count**2, 1e-15)

    max_lag_index = max(
        2,
        min(
            n - 1,
            int(
                min(
                    curve_max_lag_s,
                    duration_s * MAX_DIAGNOSTIC_LAG_FRACTION,
                )
                / bin_s
            ),
        ),
    )
    lag_indices = np.unique(
        np.clip(
            np.round(
                np.logspace(0, np.log10(max_lag_index), int(n_lags))
            ).astype(int),
            1,
            max_lag_index,
        )
    )
    curve_lag_s = lag_indices * bin_s
    curve_response = g_all[lag_indices]
    curve_valid = np.isfinite(curve_response) & (curve_lag_s > 0)
    lag_indices = lag_indices[curve_valid]
    curve_lag_s = curve_lag_s[curve_valid]
    curve_response = curve_response[curve_valid]
    decoupled_fit_grid = (
        fit_grid_max_lag_s is not None or fit_n_lags is not None
    )
    if decoupled_fit_grid:
        requested_fit_grid_max_lag_s = float(
            curve_max_lag_s
            if fit_grid_max_lag_s is None
            else fit_grid_max_lag_s
        )
        requested_fit_n_lags = int(
            n_lags if fit_n_lags is None else fit_n_lags
        )
        fit_grid_max_lag_index = max(
            2,
            min(
                n - 1,
                int(
                    min(
                        requested_fit_grid_max_lag_s,
                        duration_s * MAX_DIAGNOSTIC_LAG_FRACTION,
                    )
                    / bin_s
                ),
            ),
        )
        fit_lag_indices = np.unique(
            np.clip(
                np.round(
                    np.logspace(
                        0,
                        np.log10(fit_grid_max_lag_index),
                        requested_fit_n_lags,
                    )
                ).astype(int),
                1,
                fit_grid_max_lag_index,
            )
        )
        fit_lag_s_all = fit_lag_indices * bin_s
        fit_valid = (
            np.isfinite(g_all[fit_lag_indices])
            & (fit_lag_s_all > 0)
            & (fit_lag_s_all <= float(fit_max_lag_s))
        )
        fit_lag_indices = fit_lag_indices[fit_valid]
        lag = fit_lag_s_all[fit_valid]
        response = g_all[fit_lag_indices]
    else:
        fit_mask = curve_lag_s <= float(fit_max_lag_s)
        fit_lag_indices = lag_indices[fit_mask]
        lag = curve_lag_s[fit_mask]
        response = curve_response[fit_mask]
    if len(lag) < 8:
        return _failure(
            method_id,
            started,
            len(times),
            "too_few_valid_lags",
            n_analysis_points=len(lag),
        )

    block_length_bins = max(1, int(round(float(block_length_s) / bin_s)))
    bootstrap_acf, effective_block_lengths = (
        _circular_block_bootstrap_acf(
            centered,
            mean_count,
            lag_indices,
            block_length_bins=block_length_bins,
            n_bootstrap=int(n_bootstrap),
            bootstrap_seed=int(bootstrap_seed),
        )
    )
    curve_variance_raw = np.var(bootstrap_acf, axis=0, ddof=1)
    curve_variance_shrunk, _, curve_stabilization = (
        _stabilize_inverse_variance_weights(
            curve_variance_raw,
            shrinkage=float(weight_shrinkage),
            max_weight_ratio=float(max_weight_ratio),
        )
    )
    if decoupled_fit_grid:
        fit_bootstrap_acf, fit_effective_block_lengths = (
            _circular_block_bootstrap_acf(
                centered,
                mean_count,
                fit_lag_indices,
                block_length_bins=block_length_bins,
                n_bootstrap=int(n_bootstrap),
                bootstrap_seed=int(bootstrap_seed),
            )
        )
        fit_variance_raw = np.var(fit_bootstrap_acf, axis=0, ddof=1)
        all_effective_block_lengths = np.concatenate(
            (effective_block_lengths, fit_effective_block_lengths)
        )
    else:
        fit_variance_raw = curve_variance_raw[fit_mask]
        all_effective_block_lengths = effective_block_lengths
    fit_variance_shrunk, fit_weights, fit_stabilization = (
        _stabilize_inverse_variance_weights(
            fit_variance_raw,
            shrinkage=float(weight_shrinkage),
            max_weight_ratio=float(max_weight_ratio),
        )
    )

    tau_grid = np.logspace(
        np.log10(max(float(lag.min()) / 3.0, 1e-12)),
        np.log10(float(fit_max_lag_s) * 3.0),
        int(tau_grid_size),
    )
    objective, tau_hat, amplitude_hat, baseline_hat, best_index = (
        _fit_acf_matrix_wls(
            lag,
            response,
            tau_grid,
            kappa=float(kappa),
            weights=fit_weights,
            free_baseline=bool(free_baseline),
        )
    )
    d_hat = float(wxy_um) ** 2 / (4.0 * tau_hat)
    boundary = best_index in {0, len(tau_grid) - 1}
    success = bool(
        np.isfinite(d_hat)
        and d_hat > 0
        and np.isfinite(amplitude_hat)
        and amplitude_hat > 0
        and not boundary
    )
    reasons = []
    if not np.isfinite(d_hat) or d_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_d")
    if not np.isfinite(amplitude_hat) or amplitude_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_amplitude")
    if boundary:
        reasons.append("search_boundary")

    normal_quantile = NormalDist().inv_cdf((1.0 + float(ci_level)) / 2.0)
    curve_standard_error = np.sqrt(curve_variance_shrunk)
    ci_half_width = normal_quantile * curve_standard_error
    curve_ci_lower = curve_response - ci_half_width
    curve_ci_upper = curve_response + ci_half_width

    return DEstimate(
        method_id=method_id,
        d_estimate_um2_s=d_hat,
        tau_d_estimate_s=float(tau_hat),
        amplitude=float(amplitude_hat),
        baseline=float(baseline_hat),
        objective=float(objective),
        success=success,
        at_search_boundary=boundary,
        failure_reason=";".join(reasons),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        n_analysis_points=len(lag),
        diagnostics={
            "estimator_visible_input": "event_time_s only",
            "simulation_truth_used": False,
            "count_bin_s": float(bin_s),
            "mean_count_per_bin": mean_count,
            "zero_lag_included": False,
            "zero_lag_handling": (
                "excluded because the count variance contains the Poisson "
                "self-correlation (shot-noise) term"
            ),
            "minimum_fit_lag_bins": int(round(float(lag.min()) / float(bin_s))),
            "free_baseline": bool(free_baseline),
            "fit_lag_min_s": float(lag.min()),
            "fit_lag_max_s": float(lag.max()),
            "tau_grid_min_s": float(tau_grid[0]),
            "tau_grid_max_s": float(tau_grid[-1]),
            "tau_grid_size": int(len(tau_grid)),
            "objective_form": "r.T @ W @ r",
            "weight_matrix": "diagonal",
            "profiled_linear_parameters": (
                ["amplitude", "baseline"]
                if free_baseline
                else ["amplitude"]
            ),
            "variance_estimator": (
                "circular moving-block bootstrap of fixed-mean "
                "lag-product score sequences"
            ),
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(bootstrap_seed),
            "requested_block_length_s": float(block_length_s),
            "requested_block_length_bins": int(block_length_bins),
            "effective_block_length_bins_min": int(
                np.min(all_effective_block_lengths)
            ),
            "effective_block_length_bins_max": int(
                np.max(all_effective_block_lengths)
            ),
            "fit_grid_decoupled_from_diagnostic_curve": bool(
                decoupled_fit_grid
            ),
            "fit_grid_max_lag_s": float(
                curve_max_lag_s
                if fit_grid_max_lag_s is None
                else fit_grid_max_lag_s
            ),
            "fit_n_lags_requested": int(
                n_lags if fit_n_lags is None else fit_n_lags
            ),
            "weight_shrinkage": float(weight_shrinkage),
            "max_weight_ratio": float(max_weight_ratio),
            "realized_weight_ratio": float(
                fit_stabilization["realized_weight_ratio"]
            ),
            "weight_cap_applied": bool(
                fit_stabilization["weight_cap_applied"]
            ),
            "variance_shrinkage_target": float(
                fit_stabilization["variance_shrinkage_target"]
            ),
            "variance_floor": float(fit_stabilization["variance_floor"]),
            "curve_variance_shrinkage_target": float(
                curve_stabilization["variance_shrinkage_target"]
            ),
            "ci_level": float(ci_level),
            "ci_type": (
                "pointwise normal interval using block-bootstrap "
                "shrunken standard errors"
            ),
            "curve_lag_s": curve_lag_s.astype(float).tolist(),
            "curve_acf": curve_response.astype(float).tolist(),
            "curve_acf_variance_raw": (
                curve_variance_raw.astype(float).tolist()
            ),
            "curve_acf_variance_shrunk": (
                curve_variance_shrunk.astype(float).tolist()
            ),
            "curve_acf_standard_error": (
                curve_standard_error.astype(float).tolist()
            ),
            "curve_acf_ci_lower": curve_ci_lower.astype(float).tolist(),
            "curve_acf_ci_upper": curve_ci_upper.astype(float).tolist(),
            "fit_lag_s": lag.astype(float).tolist(),
            "fit_acf": response.astype(float).tolist(),
            "fit_variance_raw": fit_variance_raw.astype(float).tolist(),
            "fit_variance_shrunk": (
                fit_variance_shrunk.astype(float).tolist()
            ),
            "fit_weights": fit_weights.astype(float).tolist(),
        },
    )


def estimate_immigration_death_event_qmle(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    wxy_um: float,
    kappa: float,
    molecular_brightness_cps: float,
    background_cps: float,
    d_min_um2_s: float,
    d_max_um2_s: float,
    state_max: int = 18,
    occupancy_min: float = 0.02,
    occupancy_max: float = 6.0,
    optimizer_xatol_log_d: float = 1e-5,
    optimizer_maxiter: int = 80,
    optimizer_method: str = "bounded_scalar",
    initial_d_um2_s: float | None = None,
    optimizer_start_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
    optimizer_ftol: float = 1e-12,
    optimizer_gtol: float = 1e-6,
    warm_start_method_id: str | None = None,
    warm_start_runtime_s: float = 0.0,
) -> DEstimate:
    """Estimate ``D`` by a direct immigration-death event-time QMLE.

    The observed photon times are not binned.  A 3-D Gaussian PSF with peak
    molecular brightness ``q`` is moment-matched to a hard-count surrogate:

    ``q_eff = q * integral(h^2) / integral(h) = q / 2^(3/2)``.

    The stationary mean occupancy is then plugged in from the observed mean
    photon rate, ``m_hat = (N/T - b) / q_eff``.  With calibrated ``q`` and
    background ``b``, only ``D`` is numerically optimized.  In the
    infinite-volume Poisson-bath convention, this construction matches the
    fluorescence mean, zero-lag covariance amplitude, and initial covariance
    slope.  It does not exactly match the fixed-``N`` finite periodic
    simulator or the full Brownian/Gaussian joint law; therefore the returned
    estimate is explicitly a quasi-MLE for this project's simulated data.

    ``optimizer_method="warm_start_multistart_lbfgsb"`` uses a supplied
    photon-only estimate as the first numerical starting value and evaluates
    all predeclared multiplicative starts on ``log(D)``.  The likelihood
    itself remains unbinned; any binned warm start is external initialization
    and is recorded separately in the diagnostics.
    """

    started = time.perf_counter()
    method_id = "immigration_death_event_qmle"
    times = _validate_event_times(event_time_s, duration_s)
    if len(times) < 20:
        return _failure(method_id, started, len(times), "too_few_photons")
    if not (0 < d_min_um2_s < d_max_um2_s):
        raise ValueError("D bounds must satisfy 0 < d_min < d_max")
    if not (0 < occupancy_min < occupancy_max):
        raise ValueError("occupancy bounds must satisfy 0 < min < max")
    if state_max < 4:
        raise ValueError("state_max must be at least 4")
    if optimizer_xatol_log_d <= 0 or optimizer_maxiter < 1:
        raise ValueError("optimizer controls must be positive")
    if optimizer_method not in {
        "bounded_scalar",
        "warm_start_multistart_lbfgsb",
    }:
        raise ValueError(f"unknown optimizer_method: {optimizer_method}")
    if optimizer_ftol <= 0 or optimizer_gtol <= 0:
        raise ValueError("L-BFGS-B tolerances must be positive")
    if not np.isfinite(warm_start_runtime_s) or warm_start_runtime_s < 0:
        raise ValueError("warm_start_runtime_s must be finite and nonnegative")
    start_multipliers = tuple(
        float(value) for value in optimizer_start_multipliers
    )
    if (
        not start_multipliers
        or not np.all(np.isfinite(start_multipliers))
        or np.any(np.asarray(start_multipliers) <= 0)
    ):
        raise ValueError(
            "optimizer_start_multipliers must be finite and positive"
        )
    if not np.isfinite(molecular_brightness_cps) or molecular_brightness_cps <= 0:
        raise ValueError("molecular_brightness_cps must be finite and positive")
    if not np.isfinite(background_cps) or background_cps < 0:
        raise ValueError("background_cps must be finite and nonnegative")

    effective_brightness_cps = (
        float(molecular_brightness_cps) / (2.0 ** 1.5)
    )
    mean_event_rate_cps = len(times) / float(duration_s)
    signal_rate_cps = mean_event_rate_cps - float(background_cps)
    if signal_rate_cps <= 0:
        return _failure(
            method_id,
            started,
            len(times),
            "observed_rate_not_above_background",
        )

    raw_mean_occupancy = signal_rate_cps / effective_brightness_cps
    mean_occupancy = float(
        np.clip(raw_mean_occupancy, float(occupancy_min), float(occupancy_max))
    )
    occupancy_clipped = not np.isclose(
        mean_occupancy,
        raw_mean_occupancy,
        rtol=0.0,
        atol=1e-15,
    )

    log_d_bounds = (
        math.log(float(d_min_um2_s)),
        math.log(float(d_max_um2_s)),
    )

    def objective(log_d: float) -> float:
        log_likelihood, _ = _immigration_death_event_loglikelihood(
            times,
            duration_s=duration_s,
            d_um2_s=math.exp(float(log_d)),
            wxy_um=wxy_um,
            kappa=kappa,
            effective_brightness_cps=effective_brightness_cps,
            background_cps=background_cps,
            mean_occupancy=mean_occupancy,
            state_max=state_max,
        )
        return -float(log_likelihood)

    start_d_values: list[float] = []
    start_results: list[dict[str, Any]] = []
    selected_start_index: int | None = None
    optimizer_fallback_used = False
    warm_start_valid = False

    if optimizer_method == "bounded_scalar":
        optimization = minimize_scalar(
            objective,
            bounds=log_d_bounds,
            method="bounded",
            options={
                "xatol": float(optimizer_xatol_log_d),
                "maxiter": int(optimizer_maxiter),
            },
        )
    else:
        warm_start_valid = bool(
            initial_d_um2_s is not None
            and np.isfinite(initial_d_um2_s)
            and float(initial_d_um2_s) > 0
        )
        warm_start_d = (
            float(initial_d_um2_s)
            if warm_start_valid
            else math.sqrt(float(d_min_um2_s) * float(d_max_um2_s))
        )
        proposed_starts = [
            float(
                np.clip(
                    warm_start_d * multiplier,
                    float(d_min_um2_s),
                    float(d_max_um2_s),
                )
            )
            for multiplier in start_multipliers
        ]
        for start_d in proposed_starts:
            if not any(
                np.isclose(start_d, previous, rtol=0.0, atol=1e-12)
                for previous in start_d_values
            ):
                start_d_values.append(start_d)

        optimization_candidates: list[Any] = []
        for start_index, start_d in enumerate(start_d_values):
            candidate = minimize(
                lambda value: objective(float(value[0])),
                x0=np.array([math.log(start_d)], dtype=float),
                bounds=[log_d_bounds],
                method="L-BFGS-B",
                options={
                    "ftol": float(optimizer_ftol),
                    "gtol": float(optimizer_gtol),
                    "maxiter": int(optimizer_maxiter),
                    "maxls": 30,
                },
            )
            candidate_x = float(candidate.x[0])
            candidate_objective = float(candidate.fun)
            candidate_boundary = bool(
                candidate_x - log_d_bounds[0] <= 1e-4
                or log_d_bounds[1] - candidate_x <= 1e-4
            )
            candidate_valid = bool(
                candidate.success
                and np.isfinite(candidate_x)
                and np.isfinite(candidate_objective)
                and not candidate_boundary
            )
            optimization_candidates.append(candidate)
            start_results.append(
                {
                    "start_index": int(start_index),
                    "start_d_um2_s": float(start_d),
                    "success": bool(candidate.success),
                    "valid_interior_solution": candidate_valid,
                    "d_estimate_um2_s": (
                        math.exp(candidate_x)
                        if np.isfinite(candidate_x)
                        else float("nan")
                    ),
                    "objective": candidate_objective,
                    "at_search_boundary": candidate_boundary,
                    "status": int(getattr(candidate, "status", 0)),
                    "message": str(candidate.message),
                    "iterations": int(getattr(candidate, "nit", 0)),
                    "function_evaluations": int(
                        getattr(candidate, "nfev", 0)
                    ),
                }
            )

        valid_indices = [
            index
            for index, record in enumerate(start_results)
            if record["valid_interior_solution"]
        ]
        finite_indices = [
            index
            for index, record in enumerate(start_results)
            if np.isfinite(record["objective"])
        ]
        # Select the best finite objective first.  Boundary or failed optima
        # must not be hidden merely because a worse interior candidate exists;
        # the common success gates below will flag the selected solution.
        selection_pool = finite_indices
        if selection_pool:
            selected_start_index = min(
                selection_pool,
                key=lambda index: start_results[index]["objective"],
            )
            optimization = optimization_candidates[selected_start_index]
        else:
            # This fallback is numerical only and never uses simulation truth.
            optimizer_fallback_used = True
            optimization = minimize_scalar(
                objective,
                bounds=log_d_bounds,
                method="bounded",
                options={
                    "xatol": float(optimizer_xatol_log_d),
                    "maxiter": int(optimizer_maxiter),
                },
            )

    optimization_x = (
        float(optimization.x[0])
        if np.ndim(optimization.x)
        else float(optimization.x)
    )
    optimization_success = bool(optimization.success)
    optimization_status = int(getattr(optimization, "status", 0))
    optimization_message = str(optimization.message)
    optimization_iterations = int(getattr(optimization, "nit", 0))
    optimization_function_evaluations = int(
        sum(record["function_evaluations"] for record in start_results)
        + (
            int(getattr(optimization, "nfev", 0))
            if optimizer_method == "bounded_scalar" or optimizer_fallback_used
            else 0
        )
    )
    optimizer_total_iterations = int(
        sum(record["iterations"] for record in start_results)
        + (
            int(getattr(optimization, "nit", 0))
            if optimizer_method == "bounded_scalar" or optimizer_fallback_used
            else 0
        )
    )
    selected_start_d = (
        float(start_d_values[selected_start_index])
        if selected_start_index is not None and not optimizer_fallback_used
        else float("nan")
    )

    d_hat = math.exp(optimization_x)
    tau_hat = float(wxy_um) ** 2 / (4.0 * d_hat)
    log_likelihood, likelihood_diagnostics = (
        _immigration_death_event_loglikelihood(
            times,
            duration_s=duration_s,
            d_um2_s=d_hat,
            wxy_um=wxy_um,
            kappa=kappa,
            effective_brightness_cps=effective_brightness_cps,
            background_cps=background_cps,
            mean_occupancy=mean_occupancy,
            state_max=state_max,
        )
    )
    log_range = log_d_bounds[1] - log_d_bounds[0]
    boundary_tolerance = max(
        5.0 * float(optimizer_xatol_log_d),
        1e-4 * log_range,
    )
    d_boundary = bool(
        optimization_x - log_d_bounds[0] <= boundary_tolerance
        or log_d_bounds[1] - optimization_x <= boundary_tolerance
    )
    truncation_ok = bool(
        likelihood_diagnostics["stationary_top_probability"] < 1e-8
        and likelihood_diagnostics["max_filtered_top_probability"] < 1e-6
    )
    success = bool(
        optimization_success
        and np.isfinite(d_hat)
        and d_hat > 0
        and np.isfinite(log_likelihood)
        and not d_boundary
        and not occupancy_clipped
        and truncation_ok
    )
    reasons = []
    if not optimization_success:
        reasons.append("optimizer_failure")
    if not np.isfinite(d_hat) or d_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_d")
    if not np.isfinite(log_likelihood):
        reasons.append("nonfinite_loglikelihood")
    if d_boundary:
        reasons.append("search_boundary")
    if occupancy_clipped:
        reasons.append("occupancy_plugin_boundary")
    if not truncation_ok:
        reasons.append("state_truncation_mass_too_large")

    covariance_amplitude_cps2 = (
        effective_brightness_cps**2 * mean_occupancy
    )
    return DEstimate(
        method_id=method_id,
        d_estimate_um2_s=float(d_hat),
        tau_d_estimate_s=float(tau_hat),
        amplitude=float(covariance_amplitude_cps2),
        baseline=float(background_cps),
        objective=float(-log_likelihood),
        success=success,
        at_search_boundary=d_boundary,
        failure_reason=";".join(reasons),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        n_analysis_points=len(times),
        diagnostics={
            "likelihood_type": (
                "continuous-time immigration-death MMPP quasi-likelihood"
            ),
            "exactness_scope": (
                "exact ordered event likelihood for the finite-state "
                "immigration-death MMPP conditional on plug-in occupancy"
            ),
            "brownian_gaussian_psf_status": "approximate quasi-likelihood",
            "matching_scope": (
                "signal mean, latent-rate variance, and covariance derivative "
                "at lag zero under the infinite-volume Poisson-bath "
                "convention; not an exact finite-box fixed-N match"
            ),
            "full_latent_brownian_likelihood": False,
            "finite_state_surrogate_likelihood": True,
            "arrival_time_bins_used": False,
            "likelihood_objective_uses_arrival_time_bins": False,
            "initialization_uses_binned_estimator": bool(
                optimizer_method == "warm_start_multistart_lbfgsb"
                and warm_start_valid
            ),
            "estimator_visible_field": "event_time_s",
            "psf_moment_matching": (
                "q_eff = q * integral(h^2)/integral(h) = q/2^(3/2)"
            ),
            "covariance_decay_matching": (
                "mu(D) = (4D/wxy^2) * (1 + 1/(2 kappa^2))"
            ),
            "molecular_brightness_cps": float(molecular_brightness_cps),
            "effective_brightness_cps": float(effective_brightness_cps),
            "background_cps": float(background_cps),
            "mean_event_rate_cps": float(mean_event_rate_cps),
            "signal_rate_cps": float(signal_rate_cps),
            "raw_plugin_mean_occupancy": float(raw_mean_occupancy),
            "plugin_mean_occupancy": float(mean_occupancy),
            "occupancy_plugin_clipped": occupancy_clipped,
            "occupancy_bounds": [float(occupancy_min), float(occupancy_max)],
            "state_max": int(state_max),
            **likelihood_diagnostics,
            "d_bounds_um2_s": [
                float(d_min_um2_s),
                float(d_max_um2_s),
            ],
            "optimizer": (
                "bounded scalar minimization on log(D)"
                if optimizer_method == "bounded_scalar"
                else (
                    "ACF-warm-started multi-start bounded L-BFGS-B "
                    "on log(D)"
                )
            ),
            "optimizer_method": optimizer_method,
            "optimizer_success": optimization_success,
            "optimizer_status": optimization_status,
            "optimizer_message": optimization_message,
            "optimizer_selected_iterations": optimization_iterations,
            "optimizer_total_iterations": optimizer_total_iterations,
            "optimizer_function_evaluations": (
                optimization_function_evaluations
            ),
            "warm_start_method_id": warm_start_method_id,
            "warm_start_d_um2_s": (
                float(initial_d_um2_s)
                if initial_d_um2_s is not None
                and np.isfinite(initial_d_um2_s)
                else None
            ),
            "warm_start_runtime_s": float(warm_start_runtime_s),
            "optimizer_start_multipliers": list(start_multipliers),
            "optimizer_start_d_values_um2_s": start_d_values,
            "optimizer_start_results": start_results,
            "selected_start_index": selected_start_index,
            "selected_start_d_um2_s": selected_start_d,
            "optimizer_fallback_used": optimizer_fallback_used,
        },
    )


def estimate_immigration_death_joint_mle_warm(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    wxy_um: float,
    kappa: float,
    molecular_brightness_cps: float,
    background_cps: float,
    d_min_um2_s: float,
    d_max_um2_s: float,
    state_max: int = 26,
    occupancy_min: float = 0.02,
    occupancy_max: float = 6.0,
    initial_d_um2_s: float | None = None,
    initial_mean_occupancy: float | None = None,
    optimizer_maxiter: int = 80,
    optimizer_ftol: float = 1e-9,
    optimizer_gtol: float = 1e-5,
    optimizer_finite_difference_step: float = 1e-6,
    optimizer_boundary_log_tolerance: float = 1e-4,
    warm_start_method_id: str | None = None,
    warm_start_runtime_s: float = 0.0,
) -> DEstimate:
    """Jointly fit ``D`` and occupancy by a warm-started event-time MLE.

    The objective is the exact ordered-event likelihood of the calibrated,
    finite-state immigration-death MMPP surrogate.  Unlike
    :func:`estimate_immigration_death_event_qmle`, the stationary mean
    occupancy is optimized jointly with ``D`` instead of being fixed at a
    photon-rate moment estimate.  The initial ``D`` may come from a fast
    photon-only binned estimator, but no bins enter the likelihood.

    This is a genuine maximum-likelihood estimator for the truncated surrogate
    when molecular brightness and background are treated as calibrated.  The
    Brownian Gaussian-PSF simulator is not that surrogate, so the same fit is
    still a quasi-MLE with respect to the Brownian data-generating process.
    """

    started = time.perf_counter()
    method_id = "immigration_death_joint_mle_warm"
    times = _validate_event_times(event_time_s, duration_s)
    if len(times) < 20:
        return _failure(method_id, started, len(times), "too_few_photons")
    if not (0 < d_min_um2_s < d_max_um2_s):
        raise ValueError("D bounds must satisfy 0 < d_min < d_max")
    if not (0 < occupancy_min < occupancy_max):
        raise ValueError("occupancy bounds must satisfy 0 < min < max")
    if state_max < 4:
        raise ValueError("state_max must be at least 4")
    if optimizer_maxiter < 1:
        raise ValueError("optimizer_maxiter must be positive")
    if optimizer_ftol <= 0 or optimizer_gtol <= 0:
        raise ValueError("L-BFGS-B tolerances must be positive")
    if (
        not np.isfinite(optimizer_finite_difference_step)
        or optimizer_finite_difference_step <= 0
    ):
        raise ValueError(
            "optimizer_finite_difference_step must be finite and positive"
        )
    if (
        not np.isfinite(optimizer_boundary_log_tolerance)
        or optimizer_boundary_log_tolerance <= 0
    ):
        raise ValueError(
            "optimizer_boundary_log_tolerance must be finite and positive"
        )
    if not np.isfinite(warm_start_runtime_s) or warm_start_runtime_s < 0:
        raise ValueError("warm_start_runtime_s must be finite and nonnegative")
    if (
        not np.isfinite(molecular_brightness_cps)
        or molecular_brightness_cps <= 0
    ):
        raise ValueError("molecular_brightness_cps must be finite and positive")
    if not np.isfinite(background_cps) or background_cps < 0:
        raise ValueError("background_cps must be finite and nonnegative")

    effective_brightness_cps = (
        float(molecular_brightness_cps) / (2.0 ** 1.5)
    )
    mean_event_rate_cps = len(times) / float(duration_s)
    signal_rate_cps = mean_event_rate_cps - float(background_cps)
    if signal_rate_cps <= 0:
        return _failure(
            method_id,
            started,
            len(times),
            "observed_rate_not_above_background",
        )

    moment_occupancy = signal_rate_cps / effective_brightness_cps
    occupancy_start_raw = (
        float(initial_mean_occupancy)
        if initial_mean_occupancy is not None
        and np.isfinite(initial_mean_occupancy)
        and float(initial_mean_occupancy) > 0
        else float(moment_occupancy)
    )
    occupancy_start = float(
        np.clip(
            occupancy_start_raw,
            float(occupancy_min),
            float(occupancy_max),
        )
    )
    warm_start_valid = bool(
        initial_d_um2_s is not None
        and np.isfinite(initial_d_um2_s)
        and float(initial_d_um2_s) > 0
    )
    d_start_raw = (
        float(initial_d_um2_s)
        if warm_start_valid
        else math.sqrt(float(d_min_um2_s) * float(d_max_um2_s))
    )
    d_start = float(
        np.clip(d_start_raw, float(d_min_um2_s), float(d_max_um2_s))
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
        log_d, log_occupancy = np.asarray(log_parameters, dtype=float)
        log_likelihood, _ = _immigration_death_event_loglikelihood(
            times,
            duration_s=duration_s,
            d_um2_s=math.exp(float(log_d)),
            wxy_um=wxy_um,
            kappa=kappa,
            effective_brightness_cps=effective_brightness_cps,
            background_cps=background_cps,
            mean_occupancy=math.exp(float(log_occupancy)),
            state_max=state_max,
        )
        return -float(log_likelihood)

    optimization = minimize(
        objective,
        x0=np.array(
            [math.log(d_start), math.log(occupancy_start)],
            dtype=float,
        ),
        bounds=[log_d_bounds, log_occupancy_bounds],
        method="L-BFGS-B",
        options={
            "ftol": float(optimizer_ftol),
            "gtol": float(optimizer_gtol),
            "eps": float(optimizer_finite_difference_step),
            "maxiter": int(optimizer_maxiter),
            "maxls": 30,
        },
    )
    log_d_hat, log_occupancy_hat = np.asarray(
        optimization.x,
        dtype=float,
    )
    d_hat = math.exp(float(log_d_hat))
    occupancy_hat = math.exp(float(log_occupancy_hat))
    tau_hat = float(wxy_um) ** 2 / (4.0 * d_hat)
    log_likelihood, likelihood_diagnostics = (
        _immigration_death_event_loglikelihood(
            times,
            duration_s=duration_s,
            d_um2_s=d_hat,
            wxy_um=wxy_um,
            kappa=kappa,
            effective_brightness_cps=effective_brightness_cps,
            background_cps=background_cps,
            mean_occupancy=occupancy_hat,
            state_max=state_max,
        )
    )

    d_boundary = bool(
        float(log_d_hat) - log_d_bounds[0]
        <= float(optimizer_boundary_log_tolerance)
        or log_d_bounds[1] - float(log_d_hat)
        <= float(optimizer_boundary_log_tolerance)
    )
    occupancy_boundary = bool(
        float(log_occupancy_hat) - log_occupancy_bounds[0]
        <= float(optimizer_boundary_log_tolerance)
        or log_occupancy_bounds[1] - float(log_occupancy_hat)
        <= float(optimizer_boundary_log_tolerance)
    )
    truncation_ok = bool(
        likelihood_diagnostics["stationary_top_probability"] < 1e-8
        and likelihood_diagnostics["max_filtered_top_probability"] < 1e-6
    )
    optimization_success = bool(optimization.success)
    success = bool(
        optimization_success
        and np.isfinite(d_hat)
        and d_hat > 0
        and np.isfinite(occupancy_hat)
        and occupancy_hat > 0
        and np.isfinite(log_likelihood)
        and not d_boundary
        and not occupancy_boundary
        and truncation_ok
    )
    reasons: list[str] = []
    if not optimization_success:
        reasons.append("optimizer_failure")
    if not np.isfinite(d_hat) or d_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_d")
    if not np.isfinite(occupancy_hat) or occupancy_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_occupancy")
    if not np.isfinite(log_likelihood):
        reasons.append("nonfinite_loglikelihood")
    if d_boundary:
        reasons.append("d_search_boundary")
    if occupancy_boundary:
        reasons.append("occupancy_search_boundary")
    if not truncation_ok:
        reasons.append("state_truncation_mass_too_large")

    covariance_amplitude_cps2 = (
        effective_brightness_cps**2 * occupancy_hat
    )
    return DEstimate(
        method_id=method_id,
        d_estimate_um2_s=float(d_hat),
        tau_d_estimate_s=float(tau_hat),
        amplitude=float(covariance_amplitude_cps2),
        baseline=float(background_cps),
        objective=float(-log_likelihood),
        success=success,
        at_search_boundary=bool(d_boundary or occupancy_boundary),
        failure_reason=";".join(reasons),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        n_analysis_points=len(times),
        diagnostics={
            "likelihood_type": (
                "joint finite-state immigration-death MMPP event likelihood"
            ),
            "exactness_scope": (
                "exact ordered-event likelihood for the calibrated truncated "
                "immigration-death MMPP with D and occupancy jointly fitted"
            ),
            "brownian_gaussian_psf_status": "approximate quasi-likelihood",
            "finite_state_surrogate_mle": True,
            "full_latent_brownian_likelihood": False,
            "arrival_time_bins_used": False,
            "likelihood_objective_uses_arrival_time_bins": False,
            "initialization_uses_binned_estimator": warm_start_valid,
            "estimator_visible_field": "event_time_s",
            "simulation_truth_used": False,
            "molecular_brightness_cps": float(molecular_brightness_cps),
            "effective_brightness_cps": float(effective_brightness_cps),
            "background_cps": float(background_cps),
            "mean_event_rate_cps": float(mean_event_rate_cps),
            "signal_rate_cps": float(signal_rate_cps),
            "moment_mean_occupancy": float(moment_occupancy),
            "initial_mean_occupancy": float(occupancy_start),
            "estimated_mean_occupancy": float(occupancy_hat),
            "occupancy_is_optimized": True,
            "occupancy_bounds": [
                float(occupancy_min),
                float(occupancy_max),
            ],
            "occupancy_at_search_boundary": occupancy_boundary,
            "state_max": int(state_max),
            **likelihood_diagnostics,
            "d_bounds_um2_s": [
                float(d_min_um2_s),
                float(d_max_um2_s),
            ],
            "d_at_search_boundary": d_boundary,
            "optimizer": (
                "single warm-started bounded L-BFGS-B on "
                "(log D, log mean occupancy)"
            ),
            "optimizer_method": "warm_start_joint_lbfgsb",
            "optimizer_success": optimization_success,
            "optimizer_status": int(getattr(optimization, "status", 0)),
            "optimizer_message": str(optimization.message),
            "optimizer_iterations": int(getattr(optimization, "nit", 0)),
            "optimizer_function_evaluations": int(
                getattr(optimization, "nfev", 0)
            ),
            "optimizer_gradient_evaluations": int(
                getattr(optimization, "njev", 0)
            ),
            "optimizer_finite_difference_step": float(
                optimizer_finite_difference_step
            ),
            "warm_start_method_id": warm_start_method_id,
            "warm_start_d_um2_s": (
                float(initial_d_um2_s)
                if warm_start_valid
                else None
            ),
            "warm_start_runtime_s": float(warm_start_runtime_s),
            "optimizer_start_d_um2_s": float(d_start),
            "optimizer_start_mean_occupancy": float(occupancy_start),
        },
    )


def estimate_whittle_count_qmle(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    bin_s: float,
    wxy_um: float,
    kappa: float,
    d_min_um2_s: float,
    d_max_um2_s: float,
    d_grid_size: int = 240,
) -> DEstimate:
    """Fit a shot-noise plus 3-D-FCS spectrum by Whittle quasi-likelihood.

    The centered binned counts are approximated as a stationary Gaussian
    process with spectral density

    ``S(omega) = mean_count + B * S_fcs(omega; D)``.

    The sample mean supplies the Poisson white-noise level and ``B`` is
    profiled for every candidate ``D``.  The zero-frequency ordinate is
    omitted because the trace itself is centered by its estimated mean.
    """

    started = time.perf_counter()
    method_id = "whittle_count_qmle"
    times = _validate_event_times(event_time_s, duration_s)
    counts = _uniform_counts(times, duration_s, bin_s)
    mean_count = float(counts.mean())
    if len(times) < 20 or mean_count <= 0:
        return _failure(method_id, started, len(times), "too_few_photons")

    centered = counts - mean_count
    periodogram = (np.abs(np.fft.rfft(centered)) ** 2 / len(counts))[1:]
    if len(periodogram) < 8 or not np.all(np.isfinite(periodogram)):
        return _failure(
            method_id,
            started,
            len(times),
            "invalid_periodogram",
            n_analysis_points=len(periodogram),
        )

    d_grid = np.geomspace(float(d_min_um2_s), float(d_max_um2_s), int(d_grid_size))
    circular_lag_s = (
        np.minimum(np.arange(len(counts)), len(counts) - np.arange(len(counts)))
        * float(bin_s)
    )
    best = (float("inf"), float("nan"), float("nan"), -1)
    amplitude_lower = max(1e-12, mean_count**2 * 1e-5)
    amplitude_upper = max(amplitude_lower * 10.0, mean_count**2 * 100.0)

    for grid_index, d_value in enumerate(d_grid):
        tau_s = wxy_um**2 / (4.0 * float(d_value))
        covariance_shape = fcs_shape(circular_lag_s, tau_s, kappa)
        spectral_shape = np.fft.rfft(covariance_shape).real[1:]
        spectral_shape = np.maximum(spectral_shape, 1e-12)

        def objective(log_amplitude: float) -> float:
            amplitude = math.exp(float(log_amplitude))
            spectral_density = mean_count + amplitude * spectral_shape
            return float(
                np.sum(np.log(spectral_density) + periodogram / spectral_density)
            )

        optimization = minimize_scalar(
            objective,
            bounds=(math.log(amplitude_lower), math.log(amplitude_upper)),
            method="bounded",
            options={"xatol": 1e-3},
        )
        if optimization.fun < best[0]:
            best = (
                float(optimization.fun),
                float(d_value),
                float(math.exp(float(optimization.x))),
                grid_index,
            )

    objective_value, d_hat, amplitude_hat, best_index = best
    tau_hat = wxy_um**2 / (4.0 * d_hat)
    boundary = best_index in {0, len(d_grid) - 1}
    success = bool(
        np.isfinite(d_hat)
        and d_hat > 0
        and np.isfinite(amplitude_hat)
        and amplitude_hat > 0
        and not boundary
    )
    reasons = []
    if not np.isfinite(d_hat) or d_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_d")
    if not np.isfinite(amplitude_hat) or amplitude_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_amplitude")
    if boundary:
        reasons.append("search_boundary")

    return DEstimate(
        method_id=method_id,
        d_estimate_um2_s=float(d_hat),
        tau_d_estimate_s=float(tau_hat),
        amplitude=float(amplitude_hat),
        baseline=float(mean_count),
        objective=float(objective_value),
        success=success,
        at_search_boundary=boundary,
        failure_reason=";".join(reasons),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        n_analysis_points=len(periodogram),
        diagnostics={
            "count_bin_s": float(bin_s),
            "mean_count_per_bin": mean_count,
            "white_noise_spectral_level": mean_count,
            "d_grid_min_um2_s": float(d_grid[0]),
            "d_grid_max_um2_s": float(d_grid[-1]),
            "d_grid_size": int(len(d_grid)),
            "zero_frequency_omitted": True,
            "likelihood_type": "Gaussian Whittle quasi-likelihood",
        },
    )


def _positive_pair_lags(event_time_s: np.ndarray, max_lag_s: float) -> np.ndarray:
    """Collect positive event-pair lags no larger than ``max_lag_s``."""

    chunks: list[np.ndarray] = []
    for left_index in range(max(0, len(event_time_s) - 1)):
        right_stop = int(
            np.searchsorted(
                event_time_s,
                event_time_s[left_index] + max_lag_s,
                side="right",
            )
        )
        if right_stop > left_index + 1:
            chunks.append(
                event_time_s[left_index + 1 : right_stop] - event_time_s[left_index]
            )
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=float)


def estimate_event_pair_composite(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    min_positive_lag_s: float,
    max_lag_s: float,
    n_lag_bins: int,
    wxy_um: float,
    kappa: float,
    d_min_um2_s: float,
    d_max_um2_s: float,
    d_grid_size: int = 180,
    baseline_log_bound: float = 1.0,
) -> DEstimate:
    """Estimate ``D`` from a pair-lag composite Poisson likelihood.

    For a stationary Cox process, the expected number of unordered positive
    photon pairs in a lag interval is approximated by

    ``lambda_bar^2 * integral(T-tau) d tau * [c + A f_D(tau)]``.

    ``c`` is a fitted finite-window baseline and ``A`` is the correlation
    amplitude.  Pair counts are not independent, so this is explicitly a
    composite likelihood rather than a full point-process likelihood.
    """

    started = time.perf_counter()
    method_id = "event_pair_composite"
    times = _validate_event_times(event_time_s, duration_s)
    if len(times) < 20:
        return _failure(method_id, started, len(times), "too_few_photons")
    if not (0 < min_positive_lag_s < max_lag_s < duration_s):
        raise ValueError("pair-lag bounds must satisfy 0 < min < max < duration")

    lag_edges = np.concatenate(
        (
            np.array([0.0]),
            np.geomspace(
                float(min_positive_lag_s),
                float(max_lag_s),
                int(n_lag_bins),
            ),
        )
    )
    lag_midpoints = 0.5 * (lag_edges[:-1] + lag_edges[1:])
    pair_lags = _positive_pair_lags(times, max_lag_s)
    pair_counts = np.histogram(pair_lags, bins=lag_edges)[0].astype(float)
    if int(pair_counts.sum()) < 100:
        return _failure(
            method_id,
            started,
            len(times),
            "too_few_event_pairs",
            n_analysis_points=len(pair_counts),
            diagnostics={"n_event_pairs": int(pair_counts.sum())},
        )

    exposure_integral = (
        duration_s * np.diff(lag_edges)
        - 0.5 * (lag_edges[1:] ** 2 - lag_edges[:-1] ** 2)
    )
    rate_hat = len(times) / duration_s
    independent_pair_mean = rate_hat**2 * exposure_integral
    d_grid = np.geomspace(float(d_min_um2_s), float(d_max_um2_s), int(d_grid_size))
    best = (float("inf"), float("nan"), float("nan"), float("nan"), -1)
    previous = np.array([0.0, 0.0], dtype=float)

    for grid_index, d_value in enumerate(d_grid):
        tau_s = wxy_um**2 / (4.0 * float(d_value))
        shape = fcs_shape(lag_midpoints, tau_s, kappa)

        def objective(log_parameters: np.ndarray) -> float:
            baseline = math.exp(float(log_parameters[0]))
            amplitude = math.exp(float(log_parameters[1]))
            expected = independent_pair_mean * (baseline + amplitude * shape)
            return float(
                np.sum(expected - pair_counts * np.log(expected + 1e-300))
            )

        optimization = minimize(
            objective,
            previous,
            method="L-BFGS-B",
            bounds=[
                (-float(baseline_log_bound), float(baseline_log_bound)),
                (-10.0, math.log(20.0)),
            ],
            options={"maxiter": 60, "ftol": 1e-10},
        )
        if optimization.success:
            previous = np.asarray(optimization.x, dtype=float)
        if optimization.fun < best[0]:
            best = (
                float(optimization.fun),
                float(d_value),
                float(math.exp(float(optimization.x[1]))),
                float(math.exp(float(optimization.x[0]))),
                grid_index,
            )

    objective_value, d_hat, amplitude_hat, baseline_hat, best_index = best
    tau_hat = wxy_um**2 / (4.0 * d_hat)
    boundary = best_index in {0, len(d_grid) - 1}
    success = bool(
        np.isfinite(d_hat)
        and d_hat > 0
        and np.isfinite(amplitude_hat)
        and amplitude_hat > 0
        and np.isfinite(baseline_hat)
        and baseline_hat > 0
        and not boundary
    )
    reasons = []
    if not np.isfinite(d_hat) or d_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_d")
    if not np.isfinite(amplitude_hat) or amplitude_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_amplitude")
    if not np.isfinite(baseline_hat) or baseline_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_baseline")
    if boundary:
        reasons.append("search_boundary")

    return DEstimate(
        method_id=method_id,
        d_estimate_um2_s=float(d_hat),
        tau_d_estimate_s=float(tau_hat),
        amplitude=float(amplitude_hat),
        baseline=float(baseline_hat),
        objective=float(objective_value),
        success=success,
        at_search_boundary=boundary,
        failure_reason=";".join(reasons),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        n_analysis_points=len(pair_counts),
        diagnostics={
            "n_event_pairs": int(pair_counts.sum()),
            "lag_bin_count": int(len(pair_counts)),
            "lag_min_positive_s": float(min_positive_lag_s),
            "lag_max_s": float(max_lag_s),
            "d_grid_min_um2_s": float(d_grid[0]),
            "d_grid_max_um2_s": float(d_grid[-1]),
            "d_grid_size": int(len(d_grid)),
            "likelihood_type": "second-order composite Poisson likelihood",
            "base_count_bins_used": False,
        },
    )


def estimate_unbinned_pair_composite(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    min_positive_lag_s: float,
    max_lag_s: float,
    wxy_um: float,
    kappa: float,
    d_min_um2_s: float,
    d_max_um2_s: float,
    ratio_min: float = 1e-4,
    ratio_max: float = 20.0,
    coarse_d_grid_size: int = 18,
    coarse_ratio_grid_size: int = 14,
    quadrature_order: int = 64,
    optimizer_maxiter: int = 160,
    optimizer_starts: int = 3,
    max_event_pairs: int = 2_000_000,
) -> DEstimate:
    """Fit an unbinned continuous pairwise composite likelihood.

    Let ``u = t_j - t_i`` denote every selected positive event-pair lag.  The
    finite-window pair-lag intensity of a stationary FCS Cox process is
    represented by

    ``m(u) = nu * (duration_s-u) * [1 + alpha f_D(u)]``,

    where ``f_D`` is the calibrated 3-D FCS correlation shape.  The finite
    observation window contributes exposure ``duration_s - u``.  Treating the
    dependent pair lags as a working Poisson process gives the composite log
    likelihood

    ``sum log[1 + alpha f_D(u)]``
    ``- nu * integral (duration_s-u)[1+alpha f_D(u)] du``.

    For fixed ``(D,alpha)``, the pair-scale nuisance parameter has the analytic
    profile solution

    ``nu_hat = n_pairs / [E0 + alpha E_D]``.

    The remaining smooth two-parameter objective is optimized on
    ``(log D, log alpha)`` by deterministic coarse-grid initialization followed by
    bounded multi-start L-BFGS-B.  Gauss-Legendre quadrature evaluates the
    exposure integral; it is numerical integration, not photon-count binning.

    This is an unbinned *composite* likelihood.  It is not the full marginal
    point-process likelihood, which would require integrating the latent
    multi-molecule Brownian trajectories.
    """

    started = time.perf_counter()
    method_id = "unbinned_pair_composite"
    times = _validate_event_times(event_time_s, duration_s)
    if len(times) < 20:
        return _failure(method_id, started, len(times), "too_few_photons")
    if not (0 < min_positive_lag_s < max_lag_s < duration_s):
        raise ValueError("pair-lag bounds must satisfy 0 < min < max < duration")
    if not (0 < d_min_um2_s < d_max_um2_s):
        raise ValueError("D bounds must satisfy 0 < d_min < d_max")
    if not (0 < ratio_min < ratio_max):
        raise ValueError("ratio bounds must satisfy 0 < ratio_min < ratio_max")
    if coarse_d_grid_size < 4 or coarse_ratio_grid_size < 4:
        raise ValueError("coarse likelihood grids must each contain at least 4 points")
    if quadrature_order < 16:
        raise ValueError("quadrature_order must be at least 16")
    if optimizer_starts < 1:
        raise ValueError("optimizer_starts must be positive")
    if max_event_pairs < 100:
        raise ValueError("max_event_pairs must be at least 100")

    pair_lags = _positive_pair_lags(times, max_lag_s)
    pair_lags = pair_lags[pair_lags >= float(min_positive_lag_s)]
    n_pairs = int(len(pair_lags))
    if n_pairs < 100:
        return _failure(
            method_id,
            started,
            len(times),
            "too_few_event_pairs",
            n_analysis_points=n_pairs,
            diagnostics={"n_event_pairs": n_pairs},
        )
    if n_pairs > int(max_event_pairs):
        return _failure(
            method_id,
            started,
            len(times),
            "too_many_event_pairs",
            n_analysis_points=n_pairs,
            diagnostics={
                "n_event_pairs": n_pairs,
                "max_event_pairs": int(max_event_pairs),
            },
        )

    legendre_nodes, legendre_weights = np.polynomial.legendre.leggauss(
        int(quadrature_order)
    )
    lag_half_width = 0.5 * (max_lag_s - min_positive_lag_s)
    lag_midpoint = 0.5 * (max_lag_s + min_positive_lag_s)
    integration_lags = lag_midpoint + lag_half_width * legendre_nodes
    integration_weights = lag_half_width * legendre_weights
    exposure_weights = (duration_s - integration_lags) * integration_weights
    exposure_zero = float(
        duration_s * (max_lag_s - min_positive_lag_s)
        - 0.5 * (max_lag_s**2 - min_positive_lag_s**2)
    )
    if not np.isfinite(exposure_zero) or exposure_zero <= 0:
        return _failure(
            method_id,
            started,
            len(times),
            "invalid_pair_exposure",
            n_analysis_points=n_pairs,
        )

    log_d_bounds = (math.log(float(d_min_um2_s)), math.log(float(d_max_um2_s)))
    log_ratio_bounds = (math.log(float(ratio_min)), math.log(float(ratio_max)))

    def components(
        log_d: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        d_value = math.exp(float(log_d))
        pair_shape, pair_shape_derivative = _fcs_shape_and_log_d_derivative(
            pair_lags,
            d_value,
            wxy_um,
            kappa,
        )
        integration_shape, integration_shape_derivative = (
            _fcs_shape_and_log_d_derivative(
                integration_lags,
                d_value,
                wxy_um,
                kappa,
            )
        )
        exposure_shape = float(np.dot(exposure_weights, integration_shape))
        exposure_shape_derivative = float(
            np.dot(exposure_weights, integration_shape_derivative)
        )
        return (
            pair_shape,
            pair_shape_derivative,
            exposure_shape,
            exposure_shape_derivative,
        )

    def objective_from_components(
        pair_shape: np.ndarray,
        exposure_shape: float,
        log_ratio: float,
    ) -> float:
        ratio = math.exp(float(log_ratio))
        total_exposure = exposure_zero + ratio * exposure_shape
        if not np.isfinite(total_exposure) or total_exposure <= 0:
            return float("inf")
        log_terms = np.log1p(ratio * pair_shape)
        if not np.all(np.isfinite(log_terms)):
            return float("inf")
        # Parameter-dependent part of the negative profiled composite
        # log-likelihood.  Constants common to all (D, r) are omitted.
        return float(n_pairs * math.log(total_exposure) - np.sum(log_terms))

    def objective_with_gradient(
        parameters: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        (
            pair_shape,
            pair_shape_derivative,
            exposure_shape,
            exposure_shape_derivative,
        ) = components(float(parameters[0]))
        log_ratio = float(parameters[1])
        ratio = math.exp(log_ratio)
        total_exposure = exposure_zero + ratio * exposure_shape
        denominators = 1.0 + ratio * pair_shape
        value = objective_from_components(
            pair_shape,
            exposure_shape,
            log_ratio,
        )
        gradient_log_d = (
            n_pairs * ratio * exposure_shape_derivative / total_exposure
            - np.sum(ratio * pair_shape_derivative / denominators)
        )
        gradient_log_ratio = (
            n_pairs * ratio * exposure_shape / total_exposure
            - np.sum(ratio * pair_shape / denominators)
        )
        return float(value), np.array(
            [gradient_log_d, gradient_log_ratio],
            dtype=float,
        )

    coarse_d = np.linspace(
        log_d_bounds[0],
        log_d_bounds[1],
        int(coarse_d_grid_size),
    )
    coarse_ratio = np.linspace(
        log_ratio_bounds[0],
        log_ratio_bounds[1],
        int(coarse_ratio_grid_size),
    )
    coarse_candidates: list[tuple[float, float, float]] = []
    for log_d in coarse_d:
        pair_shape, _, exposure_shape, _ = components(float(log_d))
        for log_ratio in coarse_ratio:
            score = objective_from_components(
                pair_shape,
                exposure_shape,
                float(log_ratio),
            )
            coarse_candidates.append((score, float(log_d), float(log_ratio)))
    coarse_candidates.sort(key=lambda item: item[0])

    optimizations = []
    used_starts: list[tuple[float, float]] = []
    for _, log_d_start, log_ratio_start in coarse_candidates:
        start_pair = (log_d_start, log_ratio_start)
        if any(
            abs(log_d_start - previous[0]) < 1e-9
            and abs(log_ratio_start - previous[1]) < 1e-9
            for previous in used_starts
        ):
            continue
        used_starts.append(start_pair)
        optimization = minimize(
            objective_with_gradient,
            np.array(start_pair, dtype=float),
            method="L-BFGS-B",
            jac=True,
            bounds=[log_d_bounds, log_ratio_bounds],
            options={
                "maxiter": int(optimizer_maxiter),
                "ftol": 1e-11,
                "gtol": 1e-7,
                "maxls": 30,
            },
        )
        optimizations.append(optimization)
        if len(optimizations) >= int(optimizer_starts):
            break

    finite_optimizations = [
        result for result in optimizations if np.isfinite(float(result.fun))
    ]
    if not finite_optimizations:
        return _failure(
            method_id,
            started,
            len(times),
            "optimizer_returned_no_finite_solution",
            n_analysis_points=n_pairs,
            diagnostics={"n_event_pairs": n_pairs},
        )
    best = min(finite_optimizations, key=lambda result: float(result.fun))
    log_d_hat, log_ratio_hat = (float(best.x[0]), float(best.x[1]))
    d_hat = math.exp(log_d_hat)
    ratio_hat = math.exp(log_ratio_hat)
    _, _, exposure_shape_hat, _ = components(log_d_hat)
    total_exposure_hat = exposure_zero + ratio_hat * exposure_shape_hat
    correlated_pair_fraction_hat = (
        ratio_hat * exposure_shape_hat / total_exposure_hat
    )
    rate_hat = len(times) / duration_s
    profiled_pair_scale_hat = n_pairs / max(total_exposure_hat, 1e-300)
    baseline_hat = 1.0
    amplitude_hat = ratio_hat
    tau_hat = wxy_um**2 / (4.0 * d_hat)

    boundary_tolerance = 5e-4
    d_boundary = bool(
        abs(log_d_hat - log_d_bounds[0]) <= boundary_tolerance
        or abs(log_d_hat - log_d_bounds[1]) <= boundary_tolerance
    )
    ratio_boundary = bool(
        abs(log_ratio_hat - log_ratio_bounds[0]) <= boundary_tolerance
        or abs(log_ratio_hat - log_ratio_bounds[1]) <= boundary_tolerance
    )
    boundary = d_boundary or ratio_boundary
    success = bool(
        bool(best.success)
        and np.isfinite(d_hat)
        and d_hat > 0
        and np.isfinite(amplitude_hat)
        and amplitude_hat > 0
        and np.isfinite(baseline_hat)
        and baseline_hat > 0
        and not boundary
    )
    reasons = []
    if not bool(best.success):
        reasons.append("optimizer_failed")
    if not np.isfinite(d_hat) or d_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_d")
    if not np.isfinite(amplitude_hat) or amplitude_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_amplitude")
    if not np.isfinite(baseline_hat) or baseline_hat <= 0:
        reasons.append("nonfinite_or_nonpositive_baseline")
    if d_boundary:
        reasons.append("d_search_boundary")
    if ratio_boundary:
        reasons.append("ratio_search_boundary")

    return DEstimate(
        method_id=method_id,
        d_estimate_um2_s=float(d_hat),
        tau_d_estimate_s=float(tau_hat),
        amplitude=float(amplitude_hat),
        baseline=float(baseline_hat),
        objective=float(best.fun),
        success=success,
        at_search_boundary=boundary,
        failure_reason=";".join(reasons),
        runtime_s=time.perf_counter() - started,
        n_input_photons=len(times),
        n_analysis_points=n_pairs,
        diagnostics={
            "n_event_pairs": n_pairs,
            "max_event_pairs": int(max_event_pairs),
            "lag_min_positive_s": float(min_positive_lag_s),
            "lag_max_s": float(max_lag_s),
            "pair_lag_bins_used": False,
            "base_count_bins_used": False,
            "likelihood_type": "unbinned second-order pairwise composite likelihood",
            "full_latent_brownian_likelihood": False,
            "profiled_scale": True,
            "amplitude_to_baseline_ratio": float(ratio_hat),
            "correlated_pair_fraction": float(correlated_pair_fraction_hat),
            "profiled_pair_scale": float(profiled_pair_scale_hat),
            "mean_event_rate_cps": float(rate_hat),
            "d_bounds_um2_s": [float(d_min_um2_s), float(d_max_um2_s)],
            "ratio_bounds": [float(ratio_min), float(ratio_max)],
            "coarse_d_grid_size": int(coarse_d_grid_size),
            "coarse_ratio_grid_size": int(coarse_ratio_grid_size),
            "quadrature": "Gauss-Legendre",
            "quadrature_order": int(quadrature_order),
            "optimizer": "multi-start bounded L-BFGS-B on (log D, log alpha)",
            "analytic_gradient": True,
            "optimizer_starts": int(len(optimizations)),
            "optimizer_success": bool(best.success),
            "optimizer_status": int(best.status),
            "optimizer_message": str(best.message),
            "optimizer_iterations": int(getattr(best, "nit", 0)),
            "optimizer_function_evaluations": int(getattr(best, "nfev", 0)),
            "optimizer_gradient_inf_norm": float(
                np.max(np.abs(np.asarray(getattr(best, "jac", [float("nan")]))))
            ),
            "d_at_boundary": d_boundary,
            "ratio_at_boundary": ratio_boundary,
        },
    )


def run_configured_estimators(
    event_time_s: np.ndarray,
    *,
    duration_s: float,
    wxy_um: float,
    wz_um: float,
    config: dict[str, Any],
    molecular_brightness_cps: float | None = None,
    background_cps: float | None = None,
) -> list[DEstimate]:
    """Run the methods selected in a frozen JSON-compatible configuration."""

    kappa = float(wz_um) / float(wxy_um)
    methods = list(config["methods"])
    results: list[DEstimate] = []
    for method_id in methods:
        if method_id in {"acf_equal_weight", "acf_free_baseline"}:
            method_config = config["acf"]
            estimate = estimate_acf(
                event_time_s,
                duration_s=duration_s,
                bin_s=float(method_config["bin_s"]),
                curve_max_lag_s=float(method_config["curve_max_lag_s"]),
                fit_max_lag_s=float(method_config["fit_max_lag_s"]),
                n_lags=int(method_config["n_lags"]),
                wxy_um=wxy_um,
                kappa=kappa,
                free_baseline=method_id == "acf_free_baseline",
                tau_grid_size=int(method_config.get("tau_grid_size", 300)),
            )
            results.append(estimate)
        elif method_id == "acf_weighted":
            method_config = dict(config.get("acf", {}))
            method_config.update(config.get("acf_weighted", {}))
            results.append(
                estimate_acf_weighted(
                    event_time_s,
                    duration_s=duration_s,
                    bin_s=float(method_config["bin_s"]),
                    curve_max_lag_s=float(
                        method_config["curve_max_lag_s"]
                    ),
                    fit_max_lag_s=float(method_config["fit_max_lag_s"]),
                    n_lags=int(method_config["n_lags"]),
                    wxy_um=wxy_um,
                    kappa=kappa,
                    free_baseline=bool(
                        method_config.get("free_baseline", False)
                    ),
                    tau_grid_size=int(
                        method_config.get("tau_grid_size", 300)
                    ),
                    fit_grid_max_lag_s=(
                        float(method_config["fit_grid_max_lag_s"])
                        if "fit_grid_max_lag_s" in method_config
                        else None
                    ),
                    fit_n_lags=(
                        int(method_config["fit_n_lags"])
                        if "fit_n_lags" in method_config
                        else None
                    ),
                    n_bootstrap=int(
                        method_config.get("n_bootstrap", 200)
                    ),
                    block_length_s=float(
                        method_config.get("block_length_s", 0.004)
                    ),
                    weight_shrinkage=float(
                        method_config.get("weight_shrinkage", 0.10)
                    ),
                    max_weight_ratio=float(
                        method_config.get("max_weight_ratio", 100.0)
                    ),
                    bootstrap_seed=int(
                        method_config.get("bootstrap_seed", 20260727)
                    ),
                    ci_level=float(method_config.get("ci_level", 0.95)),
                )
            )
        elif method_id == "immigration_death_event_qmle":
            method_config = config["immigration_death_event"]
            calibrated_brightness = (
                molecular_brightness_cps
                if molecular_brightness_cps is not None
                else method_config.get("molecular_brightness_cps")
            )
            calibrated_background = (
                background_cps
                if background_cps is not None
                else method_config.get("background_cps")
            )
            if calibrated_brightness is None or calibrated_background is None:
                raise ValueError(
                    "immigration_death_event_qmle requires calibrated "
                    "molecular_brightness_cps and background_cps"
                )
            optimizer_method = str(
                method_config.get("optimizer_method", "bounded_scalar")
            )
            warm_start_source_method_id = method_config.get(
                "warm_start_method_id"
            )
            warm_start_estimate = next(
                (
                    estimate
                    for estimate in results
                    if estimate.method_id == warm_start_source_method_id
                ),
                None,
            )
            if (
                optimizer_method == "warm_start_multistart_lbfgsb"
                and warm_start_estimate is None
            ):
                raise ValueError(
                    "warm-started QMLE requires its warm_start_method_id "
                    "to appear earlier in config['methods']"
                )
            initial_d_um2_s = (
                float(warm_start_estimate.d_estimate_um2_s)
                if warm_start_estimate is not None
                and warm_start_estimate.success
                and np.isfinite(warm_start_estimate.d_estimate_um2_s)
                else None
            )
            warm_start_runtime_s = (
                float(warm_start_estimate.runtime_s)
                if warm_start_estimate is not None
                else 0.0
            )
            results.append(
                estimate_immigration_death_event_qmle(
                    event_time_s,
                    duration_s=duration_s,
                    wxy_um=wxy_um,
                    kappa=kappa,
                    molecular_brightness_cps=float(calibrated_brightness),
                    background_cps=float(calibrated_background),
                    d_min_um2_s=float(method_config["d_min_um2_s"]),
                    d_max_um2_s=float(method_config["d_max_um2_s"]),
                    state_max=int(method_config.get("state_max", 18)),
                    occupancy_min=float(
                        method_config.get("occupancy_min", 0.02)
                    ),
                    occupancy_max=float(
                        method_config.get("occupancy_max", 6.0)
                    ),
                    optimizer_xatol_log_d=float(
                        method_config.get("optimizer_xatol_log_d", 1e-5)
                    ),
                    optimizer_maxiter=int(
                        method_config.get("optimizer_maxiter", 80)
                    ),
                    optimizer_method=optimizer_method,
                    initial_d_um2_s=initial_d_um2_s,
                    optimizer_start_multipliers=tuple(
                        float(value)
                        for value in method_config.get(
                            "optimizer_start_multipliers",
                            [0.5, 1.0, 2.0],
                        )
                    ),
                    optimizer_ftol=float(
                        method_config.get("optimizer_ftol", 1e-12)
                    ),
                    optimizer_gtol=float(
                        method_config.get("optimizer_gtol", 1e-6)
                    ),
                    warm_start_method_id=(
                        str(warm_start_source_method_id)
                        if warm_start_source_method_id is not None
                        else None
                    ),
                    warm_start_runtime_s=warm_start_runtime_s,
                )
            )
        elif method_id == "immigration_death_joint_mle_warm":
            method_config = config["immigration_death_joint_mle"]
            calibrated_brightness = (
                molecular_brightness_cps
                if molecular_brightness_cps is not None
                else method_config.get("molecular_brightness_cps")
            )
            calibrated_background = (
                background_cps
                if background_cps is not None
                else method_config.get("background_cps")
            )
            if calibrated_brightness is None or calibrated_background is None:
                raise ValueError(
                    "immigration_death_joint_mle_warm requires calibrated "
                    "molecular_brightness_cps and background_cps"
                )
            warm_start_source_method_id = str(
                method_config.get(
                    "warm_start_method_id",
                    "acf_equal_weight",
                )
            )
            warm_start_estimate = next(
                (
                    estimate
                    for estimate in results
                    if estimate.method_id == warm_start_source_method_id
                ),
                None,
            )
            if warm_start_estimate is None:
                raise ValueError(
                    "joint warm MLE requires its warm_start_method_id "
                    "to appear earlier in config['methods']"
                )
            initial_d_um2_s = (
                float(warm_start_estimate.d_estimate_um2_s)
                if warm_start_estimate.success
                and np.isfinite(warm_start_estimate.d_estimate_um2_s)
                and warm_start_estimate.d_estimate_um2_s > 0
                else None
            )
            results.append(
                estimate_immigration_death_joint_mle_warm(
                    event_time_s,
                    duration_s=duration_s,
                    wxy_um=wxy_um,
                    kappa=kappa,
                    molecular_brightness_cps=float(calibrated_brightness),
                    background_cps=float(calibrated_background),
                    d_min_um2_s=float(method_config["d_min_um2_s"]),
                    d_max_um2_s=float(method_config["d_max_um2_s"]),
                    state_max=int(method_config.get("state_max", 26)),
                    occupancy_min=float(
                        method_config.get("occupancy_min", 0.02)
                    ),
                    occupancy_max=float(
                        method_config.get("occupancy_max", 6.0)
                    ),
                    initial_d_um2_s=initial_d_um2_s,
                    optimizer_maxiter=int(
                        method_config.get("optimizer_maxiter", 80)
                    ),
                    optimizer_ftol=float(
                        method_config.get("optimizer_ftol", 1e-9)
                    ),
                    optimizer_gtol=float(
                        method_config.get("optimizer_gtol", 1e-5)
                    ),
                    optimizer_finite_difference_step=float(
                        method_config.get(
                            "optimizer_finite_difference_step",
                            1e-6,
                        )
                    ),
                    optimizer_boundary_log_tolerance=float(
                        method_config.get(
                            "optimizer_boundary_log_tolerance",
                            1e-4,
                        )
                    ),
                    warm_start_method_id=warm_start_source_method_id,
                    warm_start_runtime_s=float(
                        warm_start_estimate.runtime_s
                    ),
                )
            )
        elif method_id == "whittle_count_qmle":
            method_config = config["whittle"]
            results.append(
                estimate_whittle_count_qmle(
                    event_time_s,
                    duration_s=duration_s,
                    bin_s=float(method_config["bin_s"]),
                    wxy_um=wxy_um,
                    kappa=kappa,
                    d_min_um2_s=float(method_config["d_min_um2_s"]),
                    d_max_um2_s=float(method_config["d_max_um2_s"]),
                    d_grid_size=int(method_config.get("d_grid_size", 240)),
                )
            )
        elif method_id == "event_pair_composite":
            method_config = config["event_pair"]
            results.append(
                estimate_event_pair_composite(
                    event_time_s,
                    duration_s=duration_s,
                    min_positive_lag_s=float(method_config["min_positive_lag_s"]),
                    max_lag_s=float(method_config["max_lag_s"]),
                    n_lag_bins=int(method_config["n_lag_bins"]),
                    wxy_um=wxy_um,
                    kappa=kappa,
                    d_min_um2_s=float(method_config["d_min_um2_s"]),
                    d_max_um2_s=float(method_config["d_max_um2_s"]),
                    d_grid_size=int(method_config.get("d_grid_size", 180)),
                    baseline_log_bound=float(
                        method_config.get("baseline_log_bound", 1.0)
                    ),
                )
            )
        elif method_id == "unbinned_pair_composite":
            method_config = config["unbinned_pair"]
            results.append(
                estimate_unbinned_pair_composite(
                    event_time_s,
                    duration_s=duration_s,
                    min_positive_lag_s=float(method_config["min_positive_lag_s"]),
                    max_lag_s=float(method_config["max_lag_s"]),
                    wxy_um=wxy_um,
                    kappa=kappa,
                    d_min_um2_s=float(method_config["d_min_um2_s"]),
                    d_max_um2_s=float(method_config["d_max_um2_s"]),
                    ratio_min=float(method_config.get("ratio_min", 1e-4)),
                    ratio_max=float(method_config.get("ratio_max", 20.0)),
                    coarse_d_grid_size=int(
                        method_config.get("coarse_d_grid_size", 18)
                    ),
                    coarse_ratio_grid_size=int(
                        method_config.get("coarse_ratio_grid_size", 14)
                    ),
                    quadrature_order=int(method_config.get("quadrature_order", 64)),
                    optimizer_maxiter=int(
                        method_config.get("optimizer_maxiter", 160)
                    ),
                    optimizer_starts=int(method_config.get("optimizer_starts", 3)),
                    max_event_pairs=int(
                        method_config.get("max_event_pairs", 2_000_000)
                    ),
                )
            )
        else:
            raise ValueError(f"unknown estimator method_id: {method_id}")
    return results
