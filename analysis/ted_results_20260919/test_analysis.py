"""Unit tests for the TED analysis pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analyze_ted_results import (
    build_canonical_patient_table,
    classify_interval,
    fold_stratified_bootstrap_mean,
    pair_methods,
)


class TedAnalysisTests(unittest.TestCase):
    def test_metric_directionality(self):
        self.assertEqual(classify_interval("mean_hidden_dice", 0.01, 0.03), "beneficial")
        self.assertEqual(classify_interval("mean_hidden_hd95_px", -0.8, -0.1), "beneficial")
        self.assertEqual(classify_interval("area_curve_relative_rmse", 0.01, 0.04), "harmful")
        self.assertEqual(classify_interval("mean_hidden_dice", -0.01, 0.03), "inconclusive")

    def test_replicate_averaging(self):
        rows = []
        for patient, fold in [("p1", 0), ("p2", 1)]:
            for replicate in range(5):
                rows.append({
                    "method_id": "m", "patient_id": patient, "fold": fold, "k": 2,
                    "strategy": "random", "replicate": replicate,
                    "mean_hidden_dice": float(replicate), "trust_gate_reject": replicate % 2,
                })
        canonical = build_canonical_patient_table(pd.DataFrame(rows))
        self.assertEqual(len(canonical), 2)
        self.assertTrue(np.allclose(canonical["mean_hidden_dice"], 2.0))
        self.assertTrue((canonical["n_sampling_replicates"] == 5).all())

    @staticmethod
    def _complete_pairing_frame():
        rows = []
        for method, offset in [("candidate", 1.0), ("reference", 0.0)]:
            for patient in range(98):
                fold = patient % 5
                for k in [2, 3, 5, 7, 10]:
                    for strategy in ["clustered", "random", "uniform"]:
                        rows.append({
                            "method_id": method, "patient_id": f"p{patient:03d}", "fold": fold,
                            "k": k, "strategy": strategy, "mean_hidden_dice": offset,
                        })
        return pd.DataFrame(rows)

    def test_pairing_complete_and_incomplete(self):
        frame = self._complete_pairing_frame()
        paired = pair_methods(frame, "candidate", "reference", ["mean_hidden_dice"])
        self.assertEqual(len(paired), 98 * 5 * 3)
        self.assertTrue(np.allclose(paired["mean_hidden_dice_difference"], 1.0))
        with self.assertRaises(ValueError):
            pair_methods(frame.iloc[:-1].copy(), "candidate", "reference", ["mean_hidden_dice"])

    def test_fold_stratified_resampling_preserves_blocks(self):
        frame = pd.DataFrame({
            "patient_id": ["a", "b", "c", "d"],
            "fold": [0, 0, 1, 1],
            "value": [0.0, 0.0, 10.0, 10.0],
        })
        lo, hi = fold_stratified_bootstrap_mean(frame, "value", 500, np.random.default_rng(123))
        self.assertAlmostEqual(lo, 5.0)
        self.assertAlmostEqual(hi, 5.0)

    def test_bootstrap_is_deterministic(self):
        frame = pd.DataFrame({
            "patient_id": [f"p{i}" for i in range(10)],
            "fold": [i % 2 for i in range(10)],
            "value": np.linspace(-1, 1, 10),
        })
        first = fold_stratified_bootstrap_mean(frame, "value", 1000, np.random.default_rng(20260825))
        second = fold_stratified_bootstrap_mean(frame, "value", 1000, np.random.default_rng(20260825))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
