"""Time-varying diffusion simulations for the FCS model ladder M1--M5.

The module deliberately keeps the data-generating layers separate:

    deterministic D(t)
    -> independent three-dimensional Brownian paths
    -> Gaussian optical weights
    -> a conditional, piecewise-constant Poisson process
    -> estimator-visible ordered photon arrival times.

The Brownian transition on ``[t_k, t_{k+1}]`` is exact at its endpoints for a
deterministic diffusion schedule:

    Delta X_ki ~ N_3(0, 2 * integral_{t_k}^{t_{k+1}} D(u) du * I_3).

Photon generation is exact conditional on the rate being held constant inside
each Brownian step.  It remains a time-discretization approximation to the
continuous Brownian-driven Cox process.

Only ``observed_events`` is estimator-visible.  Latent paths, optical weights,
conditional rates, step counts, and the true diffusion schedule are stored in
the separate ``truth`` object and must not be passed to real-data estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd


SCHEMA_VERSION = "fcs-time-varying-1.0.0"

DEFAULT_DIMENSION = 3
DEFAULT_BOX_UM = (1.5, 1.5, 4.0)
DEFAULT_N_MOLECULES = 21
DEFAULT_WXY_UM = 0.25
DEFAULT_WZ_UM = 1.25
DEFAULT_MOLECULAR_BRIGHTNESS_CPS = 50_000.0
DEFAULT_BACKGROUND_CPS = 1_000.0
DEFAULT_BROWNIAN_DT_S = 2e-6


def _as_float_array(value: float | np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(value, dtype=float)
    return array, array.ndim == 0


def _restore_scalar(array: np.ndarray, was_scalar: bool) -> float | np.ndarray:
    return float(array) if was_scalar else array


def _validate_times(times_s: np.ndarray, duration_s: float) -> None:
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, duration_s)
    if np.any(~np.isfinite(times_s)):
        raise ValueError("times must be finite")
    if np.any(times_s < -tolerance) or np.any(times_s > duration_s + tolerance):
        raise ValueError(f"times must lie in [0, {duration_s}] seconds")


class DiffusionSchedule(Protocol):
    """Structural interface shared by the deterministic M1--M5 schedules."""

    model_id: str
    label: str
    duration_s: float
    cuts_s: tuple[float, ...]

    def value_um2_s(self, times_s: float | np.ndarray) -> float | np.ndarray:
        """Evaluate D(t) in um^2/s."""

    def cumulative_integral_um2(
        self, times_s: float | np.ndarray
    ) -> float | np.ndarray:
        """Return integral_0^t D(u) du in um^2."""

    def integral_um2(self, start_s: float, stop_s: float) -> float:
        """Return integral_start^stop D(u) du in um^2."""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable schedule description."""


@dataclass(frozen=True)
class PiecewiseConstantDiffusion:
    """A positive piecewise-constant deterministic diffusion schedule."""

    model_id: str
    label: str
    duration_s: float
    cuts_s: tuple[float, ...]
    values_um2_s: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be positive and finite")
        cuts = np.asarray(self.cuts_s, dtype=float)
        values = np.asarray(self.values_um2_s, dtype=float)
        if len(values) != len(cuts) + 1:
            raise ValueError("values_um2_s must contain one value per segment")
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("all diffusion values must be positive and finite")
        if np.any(~np.isfinite(cuts)):
            raise ValueError("all cuts must be finite")
        if len(cuts) and (
            np.any(np.diff(cuts) <= 0)
            or cuts[0] <= 0
            or cuts[-1] >= self.duration_s
        ):
            raise ValueError("cuts_s must be strictly increasing inside the record")

    @property
    def segment_edges_s(self) -> np.ndarray:
        return np.asarray((0.0, *self.cuts_s, self.duration_s), dtype=float)

    @property
    def cumulative_at_segment_starts_um2(self) -> np.ndarray:
        widths = np.diff(self.segment_edges_s)
        segment_integrals = widths * np.asarray(self.values_um2_s, dtype=float)
        return np.concatenate(([0.0], np.cumsum(segment_integrals)))

    def value_um2_s(self, times_s: float | np.ndarray) -> float | np.ndarray:
        times, scalar = _as_float_array(times_s)
        _validate_times(times, self.duration_s)
        indices = np.searchsorted(
            np.asarray(self.cuts_s, dtype=float), times, side="right"
        )
        values = np.asarray(self.values_um2_s, dtype=float)[indices]
        return _restore_scalar(values, scalar)

    def cumulative_integral_um2(
        self, times_s: float | np.ndarray
    ) -> float | np.ndarray:
        times, scalar = _as_float_array(times_s)
        _validate_times(times, self.duration_s)
        cuts = np.asarray(self.cuts_s, dtype=float)
        indices = np.searchsorted(cuts, times, side="right")
        starts = self.segment_edges_s[:-1][indices]
        accumulated = self.cumulative_at_segment_starts_um2[indices]
        values = np.asarray(self.values_um2_s, dtype=float)[indices]
        integral = accumulated + values * (times - starts)
        return _restore_scalar(integral, scalar)

    def integral_um2(self, start_s: float, stop_s: float) -> float:
        if stop_s < start_s:
            raise ValueError("stop_s must not precede start_s")
        return float(
            self.cumulative_integral_um2(stop_s)
            - self.cumulative_integral_um2(start_s)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "kind": "piecewise_constant",
            "duration_s": self.duration_s,
            "cuts_s": list(self.cuts_s),
            "values_um2_s": list(self.values_um2_s),
            "cut_convention": "left-closed/right-open; a cut belongs to the new segment",
        }


@dataclass(frozen=True)
class ExponentialDiffusion:
    """A positive exponential schedule joining ``start`` and ``stop``."""

    model_id: str
    label: str
    duration_s: float
    start_um2_s: float
    stop_um2_s: float
    cuts_s: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be positive and finite")
        if (
            not np.isfinite(self.start_um2_s)
            or not np.isfinite(self.stop_um2_s)
            or self.start_um2_s <= 0
            or self.stop_um2_s <= 0
        ):
            raise ValueError("endpoint diffusion values must be positive and finite")
        if self.cuts_s:
            raise ValueError("the smooth exponential schedule has no fixed cuts")

    @property
    def log_rate_per_s(self) -> float:
        return float(np.log(self.stop_um2_s / self.start_um2_s) / self.duration_s)

    def value_um2_s(self, times_s: float | np.ndarray) -> float | np.ndarray:
        times, scalar = _as_float_array(times_s)
        _validate_times(times, self.duration_s)
        values = self.start_um2_s * np.exp(self.log_rate_per_s * times)
        return _restore_scalar(values, scalar)

    def cumulative_integral_um2(
        self, times_s: float | np.ndarray
    ) -> float | np.ndarray:
        times, scalar = _as_float_array(times_s)
        _validate_times(times, self.duration_s)
        rate = self.log_rate_per_s
        if abs(rate) < 1e-14:
            integral = self.start_um2_s * times
        else:
            integral = self.start_um2_s * np.expm1(rate * times) / rate
        return _restore_scalar(integral, scalar)

    def integral_um2(self, start_s: float, stop_s: float) -> float:
        if stop_s < start_s:
            raise ValueError("stop_s must not precede start_s")
        return float(
            self.cumulative_integral_um2(stop_s)
            - self.cumulative_integral_um2(start_s)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "kind": "smooth_exponential",
            "duration_s": self.duration_s,
            "cuts_s": [],
            "start_um2_s": self.start_um2_s,
            "stop_um2_s": self.stop_um2_s,
            "formula": (
                f"D(t)={self.start_um2_s:g}"
                f"*({self.stop_um2_s / self.start_um2_s:g})^(t/T)"
            ),
        }


MODEL_SCHEDULES: Mapping[str, DiffusionSchedule] = MappingProxyType(
    {
        "M1": PiecewiseConstantDiffusion(
            model_id="M1",
            label="split-null",
            duration_s=0.100,
            cuts_s=(0.050,),
            values_um2_s=(60.0, 60.0),
        ),
        "M2": PiecewiseConstantDiffusion(
            model_id="M2",
            label="known upward step",
            duration_s=0.100,
            cuts_s=(0.050,),
            values_um2_s=(30.0, 120.0),
        ),
        "M3": PiecewiseConstantDiffusion(
            model_id="M3",
            label="known downward step",
            duration_s=0.100,
            cuts_s=(0.050,),
            values_um2_s=(120.0, 30.0),
        ),
        "M4": PiecewiseConstantDiffusion(
            model_id="M4",
            label="three fixed blocks",
            duration_s=0.150,
            cuts_s=(0.050, 0.100),
            values_um2_s=(30.0, 60.0, 120.0),
        ),
        "M5": ExponentialDiffusion(
            model_id="M5",
            label="smooth exponential increase",
            duration_s=0.200,
            start_um2_s=30.0,
            stop_um2_s=120.0,
        ),
    }
)


def get_diffusion_schedule(model_id: str) -> DiffusionSchedule:
    """Return the immutable prespecified schedule for M1--M5."""

    normalized = str(model_id).upper()
    try:
        return MODEL_SCHEDULES[normalized]
    except KeyError as exc:
        choices = ", ".join(MODEL_SCHEDULES)
        raise ValueError(f"unknown model_id {model_id!r}; choose one of {choices}") from exc


def build_time_grid(
    duration_s: float,
    brownian_dt_s: float,
    cuts_s: tuple[float, ...] = (),
) -> np.ndarray:
    """Build a deterministic grid containing every fixed schedule cut.

    If a cut is not on the nominal ``dt`` grid, it is inserted and the adjacent
    Brownian intervals become shorter.  Consequently no transition straddles a
    discontinuity in ``D(t)``.
    """

    duration = float(duration_s)
    dt = float(brownian_dt_s)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration_s must be positive and finite")
    if not np.isfinite(dt) or dt <= 0 or dt >= duration:
        raise ValueError("brownian_dt_s must lie strictly between zero and duration_s")

    cuts = np.asarray(cuts_s, dtype=float)
    if len(cuts) and (
        np.any(~np.isfinite(cuts))
        or np.any(np.diff(cuts) <= 0)
        or cuts[0] <= 0
        or cuts[-1] >= duration
    ):
        raise ValueError("cuts_s must be strictly increasing inside the record")

    n_full = int(np.floor(duration / dt))
    grid = np.arange(n_full + 1, dtype=float) * dt
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, duration, dt)
    grid = grid[grid < duration - tolerance]
    grid = np.concatenate((grid, np.array([duration], dtype=float)))

    # Replace a numerically coincident nominal point by the exact cut value;
    # otherwise insert the cut.  Exact replacement makes `cut in grid` a valid
    # diagnostic rather than a tolerance-dependent visual claim.
    for cut in cuts:
        closest = int(np.argmin(np.abs(grid - cut)))
        if abs(grid[closest] - cut) <= tolerance:
            grid[closest] = cut
        else:
            grid = np.append(grid, cut)
    grid.sort()

    differences = np.diff(grid)
    if np.any(differences <= 0):
        raise RuntimeError("time grid construction produced duplicate points")
    if np.max(differences) > dt + tolerance:
        raise RuntimeError("time grid contains a step wider than brownian_dt_s")
    return grid


@dataclass(frozen=True)
class TimeVaryingSimulationSettings:
    """Resolved physical and observation settings for one model run."""

    schedule: DiffusionSchedule
    seed: int
    dimension: int = DEFAULT_DIMENSION
    box_um: tuple[float, float, float] = DEFAULT_BOX_UM
    n_molecules: int = DEFAULT_N_MOLECULES
    wxy_um: float = DEFAULT_WXY_UM
    wz_um: float = DEFAULT_WZ_UM
    molecular_brightness_cps: float = DEFAULT_MOLECULAR_BRIGHTNESS_CPS
    background_cps: float = DEFAULT_BACKGROUND_CPS
    brownian_dt_s: float = DEFAULT_BROWNIAN_DT_S

    def __post_init__(self) -> None:
        if self.dimension != 3:
            raise ValueError("this simulator implements three-dimensional diffusion")
        box = np.asarray(self.box_um, dtype=float)
        if box.shape != (3,) or np.any(~np.isfinite(box)) or np.any(box <= 0):
            raise ValueError("box_um must contain three positive finite lengths")
        if int(self.n_molecules) != self.n_molecules or self.n_molecules <= 0:
            raise ValueError("n_molecules must be a positive integer")
        if self.wxy_um <= 0 or self.wz_um <= 0:
            raise ValueError("Gaussian PSF widths must be positive")
        if self.molecular_brightness_cps < 0 or self.background_cps < 0:
            raise ValueError("brightness and background rates must be nonnegative")
        if self.brownian_dt_s <= 0 or self.brownian_dt_s >= self.schedule.duration_s:
            raise ValueError("brownian_dt_s must lie inside the acquisition duration")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_id": self.schedule.model_id,
            "seed": int(self.seed),
            "dynamics": {
                "dimension": self.dimension,
                "diffusion_schedule": self.schedule.as_dict(),
                "boundary": "periodic",
                "box_um": list(self.box_um),
                "initialization": "uniform stationary distribution",
                "n_molecules": int(self.n_molecules),
            },
            "optics": {
                "profile": "3d_gaussian_detection",
                "psf_convention": "h=exp[-2(x^2+y^2)/wxy^2-2z^2/wz^2]",
                "wxy_um": self.wxy_um,
                "wz_um": self.wz_um,
                "molecular_brightness_cps": self.molecular_brightness_cps,
                "background_cps": self.background_cps,
            },
            "acquisition": {
                "duration_s": self.schedule.duration_s,
                "brownian_dt_s": self.brownian_dt_s,
            },
            "observation_process": (
                "conditional PPP with optical rate held piecewise constant "
                "over each Brownian interval"
            ),
            "units": {
                "space": "um",
                "time": "s",
                "diffusion": "um^2/s",
                "rate": "counts/s",
            },
            "data_access": {
                "estimator_visible": [
                    "observed_events.event_id",
                    "observed_events.event_time_s",
                ],
                "simulation_truth_only": [
                    "truth.time_grid_s",
                    "truth.positions_um",
                    "truth.brownian_increments_um",
                    "truth.step_integrated_diffusion_um2",
                    "truth.step_average_diffusion_um2_s",
                    "truth.detection_weights",
                    "truth.molecular_rate_cps",
                    "truth.background_rate_cps",
                    "truth.total_rate_cps",
                    "truth.fine_counts",
                ],
            },
        }


@dataclass
class TimeVaryingSimulationTruth:
    """Simulation-only latent variables; never estimator input."""

    time_grid_s: np.ndarray
    positions_um: np.ndarray
    brownian_increments_um: np.ndarray
    step_integrated_diffusion_um2: np.ndarray
    step_average_diffusion_um2_s: np.ndarray
    detection_weights: np.ndarray
    molecular_rate_cps: np.ndarray
    background_rate_cps: np.ndarray
    total_rate_cps: np.ndarray
    fine_counts: np.ndarray


@dataclass
class TimeVaryingSimulationResult:
    """One resolved run with observed events isolated from latent truth."""

    resolved_settings: dict[str, Any]
    observed_events: pd.DataFrame
    truth: TimeVaryingSimulationTruth
    diagnostics: dict[str, Any]

    @property
    def event_times_s(self) -> np.ndarray:
        """A defensive NumPy copy of the estimator-visible arrival times."""

        return self.observed_events["event_time_s"].to_numpy(dtype=float, copy=True)


def make_default_settings(
    model_id: str,
    *,
    seed: int,
    brownian_dt_s: float = DEFAULT_BROWNIAN_DT_S,
    n_molecules: int = DEFAULT_N_MOLECULES,
) -> TimeVaryingSimulationSettings:
    """Construct the frozen baseline settings for one prespecified model."""

    return TimeVaryingSimulationSettings(
        schedule=get_diffusion_schedule(model_id),
        seed=int(seed),
        brownian_dt_s=float(brownian_dt_s),
        n_molecules=int(n_molecules),
    )


def _periodic_wrap(positions_um: np.ndarray, box_um: np.ndarray) -> np.ndarray:
    return ((positions_um + box_um / 2.0) % box_um) - box_um / 2.0


def _gaussian_detection_profile(
    positions_um: np.ndarray, wxy_um: float, wz_um: float
) -> np.ndarray:
    radial_squared = positions_um[..., 0] ** 2 + positions_um[..., 1] ** 2
    axial_squared = positions_um[..., 2] ** 2
    return np.exp(
        -2.0 * radial_squared / wxy_um**2
        - 2.0 * axial_squared / wz_um**2
    )


def _simulate_brownian_truth(
    settings: TimeVaryingSimulationSettings,
    initial_rng: np.random.Generator,
    dynamics_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    schedule = settings.schedule
    grid = build_time_grid(
        schedule.duration_s, settings.brownian_dt_s, schedule.cuts_s
    )
    step_integrals = np.diff(
        np.asarray(schedule.cumulative_integral_um2(grid), dtype=float)
    )
    if np.any(step_integrals <= 0):
        raise RuntimeError("a positive diffusion schedule produced a nonpositive integral")

    step_widths = np.diff(grid)
    step_average_d = step_integrals / step_widths
    n_steps = len(step_widths)
    n_molecules = settings.n_molecules
    # Keep the recurrence in float64.  With 100,000 M5 steps, float32 cumsum
    # can accumulate enough rounding error that ``wrap(x_k + Delta x_k)`` no
    # longer reconstructs ``x_{k+1}`` at tight tolerance.  This is a numerical
    # state-continuity issue, so relaxing the diagnostic would hide the cause.
    box = np.asarray(settings.box_um, dtype=float)
    initial = initial_rng.uniform(
        -box / 2.0, box / 2.0, size=(n_molecules, 3)
    )

    standard_normal = dynamics_rng.standard_normal(
        size=(n_steps, n_molecules, 3)
    )
    scales = np.sqrt(2.0 * step_integrals)
    increments = standard_normal * scales[:, None, None]

    positions = np.empty((n_steps + 1, n_molecules, 3), dtype=float)
    positions[0] = initial
    np.cumsum(increments, axis=0, dtype=float, out=positions[1:])
    positions[1:] += initial[None, :, :]
    positions[:] = _periodic_wrap(positions, box)
    return grid, positions, increments, step_integrals, step_average_d


def simulate_brownian_paths(
    model_id: str,
    *,
    seed: int,
    brownian_dt_s: float,
    n_molecules: int = DEFAULT_N_MOLECULES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate only Brownian truth, primarily for deterministic QA.

    Returns ``(time_grid, wrapped_positions, pre_wrap_increments,
    step_integrated_diffusion)``.  This function is explicitly an oracle/QA
    helper; its outputs are not estimator-visible observations.
    """

    settings = make_default_settings(
        model_id,
        seed=seed,
        brownian_dt_s=brownian_dt_s,
        n_molecules=n_molecules,
    )
    root = np.random.SeedSequence(int(seed))
    initial_seed, dynamics_seed = root.spawn(2)
    grid, positions, increments, integrals, _ = _simulate_brownian_truth(
        settings,
        np.random.default_rng(initial_seed),
        np.random.default_rng(dynamics_seed),
    )
    return grid, positions, increments, integrals


def _draw_piecewise_constant_ppp(
    rng: np.random.Generator,
    grid_s: np.ndarray,
    total_rate_cps: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Draw a PPP exactly conditional on the interval-constant rates."""

    widths = np.diff(grid_s)
    means = total_rate_cps * widths
    counts = rng.poisson(means).astype(np.int32)
    event_chunks: list[np.ndarray] = []
    for step_index in np.flatnonzero(counts):
        count = int(counts[step_index])
        event_chunks.append(
            grid_s[step_index]
            + rng.random(count) * widths[step_index]
        )

    if event_chunks:
        event_times = np.sort(np.concatenate(event_chunks), kind="mergesort")
    else:
        event_times = np.empty(0, dtype=float)
    observed = pd.DataFrame(
        {
            "event_id": np.arange(1, len(event_times) + 1, dtype=np.int64),
            "event_time_s": event_times,
        }
    )
    return counts, observed


def compute_invariant_diagnostics(
    settings: TimeVaryingSimulationSettings,
    observed_events: pd.DataFrame,
    truth: TimeVaryingSimulationTruth,
) -> dict[str, Any]:
    """Evaluate deterministic structural and count-conservation invariants."""

    grid = truth.time_grid_s
    cuts = settings.schedule.cuts_s
    tolerance = 5e-11
    box = np.asarray(settings.box_um, dtype=float)
    event_times = observed_events["event_time_s"].to_numpy(dtype=float)

    cuts_on_grid = all(bool(np.any(grid == cut)) for cut in cuts)
    no_step_crosses_cut = all(
        not bool(np.any((grid[:-1] < cut) & (grid[1:] > cut)))
        for cut in cuts
    )
    reconstructed = _periodic_wrap(
        truth.positions_um[:-1] + truth.brownian_increments_um,
        box,
    )
    positions_continue = bool(
        np.allclose(
            reconstructed,
            truth.positions_um[1:],
            rtol=0.0,
            atol=tolerance,
        )
    )
    rebinned_counts = np.histogram(event_times, bins=grid)[0]
    count_conservation = bool(np.array_equal(rebinned_counts, truth.fine_counts))
    schedule_integral = float(settings.schedule.integral_um2(0.0, settings.schedule.duration_s))
    grid_integral = float(np.sum(truth.step_integrated_diffusion_um2))
    integral_consistency = bool(
        np.isclose(schedule_integral, grid_integral, rtol=2e-13, atol=2e-15)
    )
    positions_inside = bool(
        np.all(truth.positions_um >= -box[None, None, :] / 2.0 - tolerance)
        and np.all(truth.positions_um < box[None, None, :] / 2.0 + tolerance)
    )
    event_times_sorted = bool(
        len(event_times) < 2 or np.all(np.diff(event_times) >= 0)
    )
    event_times_in_range = bool(
        len(event_times) == 0
        or (
            float(event_times[0]) >= 0.0
            and float(event_times[-1]) < settings.schedule.duration_s
        )
    )
    expected_photons = float(
        np.dot(truth.total_rate_cps, np.diff(truth.time_grid_s))
    )

    invariant_flags = {
        "cuts_on_grid": cuts_on_grid,
        "no_step_crosses_cut": no_step_crosses_cut,
        "positions_continue_across_all_steps": positions_continue,
        "positions_inside_periodic_box": positions_inside,
        "event_times_sorted": event_times_sorted,
        "event_times_in_range": event_times_in_range,
        "count_conservation_ok": count_conservation,
        "schedule_integral_consistency": integral_consistency,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": settings.schedule.model_id,
        "seed": int(settings.seed),
        "n_molecules": int(settings.n_molecules),
        "n_brownian_steps": int(len(grid) - 1),
        "duration_s": float(settings.schedule.duration_s),
        "nominal_brownian_dt_s": float(settings.brownian_dt_s),
        "maximum_actual_step_s": float(np.max(np.diff(grid))),
        "schedule_integral_um2": schedule_integral,
        "expected_photons": expected_photons,
        "detected_photons": int(len(observed_events)),
        **invariant_flags,
        "all_invariants_ok": bool(all(invariant_flags.values())),
        "observation_process": (
            "exact PPP conditional on rate held constant in each Brownian interval; "
            "discretization approximation to the continuous Brownian-driven Cox process"
        ),
        "estimator_visible_columns": list(observed_events.columns),
        "truth_is_separate": True,
    }


def simulate_time_varying(
    settings: TimeVaryingSimulationSettings,
) -> TimeVaryingSimulationResult:
    """Run one reproducible M1--M5 simulation from resolved settings."""

    root = np.random.SeedSequence(int(settings.seed))
    initial_seed, dynamics_seed, photon_seed = root.spawn(3)
    grid, positions, increments, integrals, average_d = _simulate_brownian_truth(
        settings,
        np.random.default_rng(initial_seed),
        np.random.default_rng(dynamics_seed),
    )

    weights = _gaussian_detection_profile(
        positions[:-1], settings.wxy_um, settings.wz_um
    ).astype(np.float32)
    molecular_rate = (
        settings.molecular_brightness_cps
        * np.sum(weights, axis=1, dtype=np.float64)
    )
    background_rate = np.full(
        len(grid) - 1, settings.background_cps, dtype=float
    )
    total_rate = molecular_rate + background_rate
    fine_counts, observed_events = _draw_piecewise_constant_ppp(
        np.random.default_rng(photon_seed), grid, total_rate
    )

    truth = TimeVaryingSimulationTruth(
        time_grid_s=grid,
        positions_um=positions,
        brownian_increments_um=increments,
        step_integrated_diffusion_um2=integrals,
        step_average_diffusion_um2_s=average_d,
        detection_weights=weights,
        molecular_rate_cps=molecular_rate,
        background_rate_cps=background_rate,
        total_rate_cps=total_rate,
        fine_counts=fine_counts,
    )
    diagnostics = compute_invariant_diagnostics(
        settings, observed_events, truth
    )
    if not diagnostics["all_invariants_ok"]:
        failed = [
            key
            for key, value in diagnostics.items()
            if key
            in {
                "cuts_on_grid",
                "no_step_crosses_cut",
                "positions_continue_across_all_steps",
                "positions_inside_periodic_box",
                "event_times_sorted",
                "event_times_in_range",
                "count_conservation_ok",
                "schedule_integral_consistency",
            }
            and not value
        ]
        raise RuntimeError(f"simulation invariant failure: {failed}")

    return TimeVaryingSimulationResult(
        resolved_settings=settings.as_dict(),
        observed_events=observed_events,
        truth=truth,
        diagnostics=diagnostics,
    )


def run_time_varying_simulation(
    model_id: str,
    *,
    seed: int,
    brownian_dt_s: float = DEFAULT_BROWNIAN_DT_S,
    n_molecules: int = DEFAULT_N_MOLECULES,
) -> TimeVaryingSimulationResult:
    """Convenience entry point using the frozen M1--M5 baseline settings."""

    settings = make_default_settings(
        model_id,
        seed=seed,
        brownian_dt_s=brownian_dt_s,
        n_molecules=n_molecules,
    )
    return simulate_time_varying(settings)
