from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observation_aware_hb_tpm import (  # noqa: E402
    GaussianTrajectoryPrior,
    estimate_gaussian_coefficient_prior,
    finite_difference_hessian,
    fit_log_linear_event_trajectory,
)


class ObservationAwareHBTPMTests(unittest.TestCase):
    def test_finite_difference_hessian_matches_quadratic(self) -> None:
        matrix = np.asarray([[3.0, 0.7], [0.7, 2.0]])

        def objective(point: np.ndarray) -> float:
            return 0.5 * float(point @ matrix @ point)

        actual = finite_difference_hessian(
            objective, np.asarray([0.4, -0.2]), step=1e-4
        )
        np.testing.assert_allclose(actual, matrix, atol=2e-7, rtol=0.0)

    def test_empirical_prior_deconvolves_measurement_covariance(self) -> None:
        estimates = np.asarray(
            [
                [3.7, 4.3],
                [3.8, 4.5],
                [3.9, 4.6],
                [3.75, 4.4],
            ]
        )
        covariances = np.repeat(
            np.diag([0.002, 0.003])[None, :, :], len(estimates), axis=0
        )
        prior = estimate_gaussian_coefficient_prior(
            estimates,
            covariances,
            variance_floor=1e-4,
            covariance_shrinkage=0.1,
        )
        self.assertEqual(prior.n_source_records, 4)
        self.assertFalse(prior.diagnostics["simulation_truth_used"])
        self.assertTrue(
            np.all(np.linalg.eigvalsh(prior.covariance_log_d) > 0)
        )

    def test_hierarchical_fit_uses_only_events_and_moves_toward_prior(self) -> None:
        rng = np.random.default_rng(20260825)
        duration_s = 0.04
        event_times = np.sort(rng.uniform(0.0, duration_s, size=220))
        common = {
            "duration_s": duration_s,
            "wxy_um": 0.25,
            "kappa": 5.0,
            "molecular_brightness_cps": 50000.0,
            "background_cps": 1000.0,
            "d_bounds_um2_s": (10.0, 220.0),
            "state_max": 6,
            "n_refinement_intervals": 3,
            "optimizer_maxiter": 8,
            "hessian_step": 0.004,
        }
        target_only = fit_log_linear_event_trajectory(
            event_times, prior=None, **common
        )
        prior = GaussianTrajectoryPrior(
            mean_log_d=(np.log(45.0), np.log(90.0)),
            covariance_log_d=((0.01, 0.0), (0.0, 0.01)),
            n_source_records=10,
            n_successful_source_fits=10,
            diagnostics={"simulation_truth_used": False},
        )
        hierarchical = fit_log_linear_event_trajectory(
            event_times, prior=prior, **common
        )
        target_distance = np.linalg.norm(
            np.asarray(target_only.log_d_estimate) - np.asarray(prior.mean_log_d)
        )
        hierarchical_distance = np.linalg.norm(
            np.asarray(hierarchical.log_d_estimate) - np.asarray(prior.mean_log_d)
        )
        self.assertLessEqual(hierarchical_distance, target_distance + 1e-10)
        self.assertEqual(
            hierarchical.diagnostics["estimator_visible_input"],
            "event_time_s only",
        )
        self.assertFalse(hierarchical.diagnostics["simulation_truth_used"])
        self.assertTrue(hierarchical.diagnostics["prior_used"])


if __name__ == "__main__":
    unittest.main()
