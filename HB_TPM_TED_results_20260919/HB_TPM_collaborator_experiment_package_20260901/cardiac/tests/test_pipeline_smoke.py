from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


CARDIAC_ROOT = Path(__file__).resolve().parents[1]


class CardiacPipelineSmokeTests(unittest.TestCase):
    def test_patient_split_and_oracle_baseline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hb_tpm_cardiac_test_") as raw:
            root = Path(raw)
            rows = []
            yy, xx = np.mgrid[:16, :16]
            for patient_index in range(10):
                masks = []
                for frame_index in range(12):
                    phase = 2.0 * np.pi * frame_index / 12.0
                    radius = 3.5 + 0.8 * np.cos(phase)
                    masks.append(
                        (
                            (xx - 8.0) ** 2 + (yy - 8.0) ** 2
                            <= radius**2
                        ).astype(np.uint8)
                    )
                mask_path = root / f"patient{patient_index:04d}_mask.npy"
                image_path = root / f"patient{patient_index:04d}_image.npy"
                np.save(mask_path, np.stack(masks))
                np.save(image_path, np.stack(masks).astype(np.float32))
                rows.append(
                    {
                        "patient_id": f"patient{patient_index:04d}",
                        "image_sequence_path": image_path,
                        "mask_sequence_path": mask_path,
                        "n_frames": 12,
                        "frame_period_s": 0.033,
                        "dataset": "SYNTHETIC_QA",
                        "time_axis": 0,
                    }
                )
            manifest = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            config = json.loads(
                (CARDIAC_ROOT / "configs" / "ted_sparse_cycle.json").read_text()
            )
            config["shot_budgets"] = [3]
            config["sampling_strategies"] = ["uniform"]
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            splits_dir = root / "splits"
            baseline_dir = root / "baseline"
            subprocess.run(
                [
                    sys.executable,
                    str(CARDIAC_ROOT / "scripts" / "validate_manifest.py"),
                    "--manifest",
                    str(manifest),
                    "--check-arrays",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(CARDIAC_ROOT / "scripts" / "make_sparse_splits.py"),
                    "--manifest",
                    str(manifest),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(splits_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(
                        CARDIAC_ROOT
                        / "scripts"
                        / "run_oracle_contour_baseline.py"
                    ),
                    "--manifest",
                    str(manifest),
                    "--splits",
                    str(splits_dir / "patient_splits.csv"),
                    "--observations",
                    str(splits_dir / "sparse_observations.csv"),
                    "--fold",
                    "0",
                    "--output-dir",
                    str(baseline_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            results = pd.read_csv(baseline_dir / "patient_results.csv")
            self.assertEqual(results["patient_id"].nunique(), 2)
            self.assertTrue((results["k"] == 3).all())
            self.assertTrue((results["strategy"] == "uniform").all())
            self.assertFalse(results["hidden_frames_used_for_fitting"].any())
            self.assertTrue(results["mean_hidden_dice"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()
