from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch


CARDIAC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARDIAC_ROOT))

from src.observation_aware_cardiac import CardiacObservationAwareHBTPM  # noqa: E402


class DeepCardiacModelTests(unittest.TestCase):
    def test_slurm_array_mapping_and_bounds(self) -> None:
        if shutil.which("bash") is None:
            self.skipTest("bash is unavailable; SLURM mapping is exercised on Linux")
        script = CARDIAC_ROOT / "slurm_cardiac_array.sbatch"
        with tempfile.TemporaryDirectory(prefix="hb_tpm_slurm_test_") as raw:
            environment = os.environ.copy()
            environment.update(
                {
                    "MANIFEST": "/private/manifest.csv",
                    "SPLITS": "/private/splits.csv",
                    "OBSERVATIONS": "/private/observations.csv",
                    "RUN_ROOT": str(Path(raw) / "runs"),
                    "METHOD": "oa_hb_tpm_map",
                    "PYTHON": "/bin/echo",
                    "SLURM_ARRAY_TASK_ID": "24",
                }
            )
            mapped = subprocess.run(
                ["bash", str(script)],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(mapped.returncode, 0, msg=mapped.stderr)
            self.assertIn("--fold 4", mapped.stdout)
            self.assertIn("--k 10", mapped.stdout)
            self.assertIn("--strategy all", mapped.stdout)
            environment["SLURM_ARRAY_TASK_ID"] = "25"
            rejected = subprocess.run(
                ["bash", str(script)],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("must be in 0..24", rejected.stderr)

    def test_multiclass_mask_requires_explicit_lv_label(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hb_tpm_mask_label_test_") as raw:
            root = Path(raw)
            images = np.zeros((4, 12, 12), dtype=np.float32)
            masks = np.zeros((4, 12, 12), dtype=np.uint8)
            masks[:, 2:6, 2:6] = 1
            masks[:, 7:10, 7:10] = 2
            image_path = root / "images.npy"
            mask_path = root / "masks.npy"
            np.save(image_path, images)
            np.save(mask_path, masks)
            manifest = root / "manifest.csv"
            pd.DataFrame(
                [
                    {
                        "patient_id": "multiclass01",
                        "image_sequence_path": image_path,
                        "mask_sequence_path": mask_path,
                        "n_frames": 4,
                        "frame_period_s": 0.04,
                        "dataset": "SYNTHETIC_QA",
                        "time_axis": 0,
                    }
                ]
            ).to_csv(manifest, index=False)
            base_command = [
                sys.executable,
                str(
                    CARDIAC_ROOT
                    / "scripts"
                    / "prepare_standardized_sequences.py"
                ),
                "--manifest",
                str(manifest),
                "--frames",
                "4",
                "--height",
                "12",
                "--width",
                "12",
            ]
            rejected = subprocess.run(
                [*base_command, "--output-dir", str(root / "rejected")],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("multiple nonzero mask labels", rejected.stderr)
            accepted = subprocess.run(
                [
                    *base_command,
                    "--output-dir",
                    str(root / "accepted"),
                    "--mask-label",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                accepted.returncode,
                0,
                msg=accepted.stdout + "\n" + accepted.stderr,
            )
            prepared_manifest = root / "accepted" / "standardized_manifest.csv"
            prepared = pd.read_csv(prepared_manifest)
            selected = np.load(
                prepared_manifest.parent / prepared.iloc[0]["mask_sequence_path"]
            )
            self.assertEqual(set(np.unique(selected)), {0, 1})
            self.assertEqual(prepared.iloc[0]["foreground_mask_label"], 1)

    def test_posterior_shapes_and_positive_covariance(self) -> None:
        torch.manual_seed(7)
        model = CardiacObservationAwareHBTPM(
            latent_dim=3,
            harmonics=1,
            base_channels=4,
        )
        observed = torch.randn(2, 3, 1, 24, 24)
        observed_times = torch.tensor([[0.0, 0.25, 0.75], [0.1, 0.4, 0.8]])
        full_times = torch.arange(8, dtype=torch.float32)[None].repeat(2, 1) / 8
        output = model(
            observed,
            observed_times,
            full_times,
            output_size=(24, 24),
        )
        self.assertEqual(tuple(output["mask_logits"].shape), (2, 8, 1, 24, 24))
        self.assertEqual(tuple(output["coefficient_mean"].shape), (2, 3, 3))
        self.assertEqual(tuple(output["coefficient_covariance"].shape), (2, 3, 3, 3))
        eigenvalues = torch.linalg.eigvalsh(output["coefficient_covariance"])
        self.assertTrue(torch.all(eigenvalues > 0))
        samples = model.sample_mask_probabilities(
            output,
            output_size=(24, 24),
            n_samples=2,
        )
        self.assertEqual(tuple(samples.shape), (2, 2, 8, 1, 24, 24))

    def test_two_shot_underdetermined_posterior_is_stable(self) -> None:
        torch.manual_seed(11)
        for isotropic in (False, True):
            model = CardiacObservationAwareHBTPM(
                latent_dim=4,
                harmonics=3,
                base_channels=4,
                isotropic_prior=isotropic,
            )
            observed = torch.randn(2, 2, 1, 24, 24)
            observed_times = torch.tensor([[0.0, 0.5], [0.15, 0.70]])
            full_times = torch.arange(10, dtype=torch.float32)[None].repeat(2, 1) / 10
            for mode in ("hierarchical", "weak", "source_mean"):
                output = model(
                    observed,
                    observed_times,
                    full_times,
                    output_size=(24, 24),
                    coefficient_mode=mode,
                )
                self.assertTrue(torch.isfinite(output["mask_logits"]).all())
                self.assertTrue(
                    torch.all(
                        torch.linalg.eigvalsh(output["coefficient_covariance"])
                        > 0
                    )
                )

    def test_one_epoch_end_to_end_runner(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hb_tpm_deep_card_test_") as raw:
            root = Path(raw)
            rows = []
            yy, xx = np.mgrid[:24, :24]
            rng = np.random.default_rng(13)
            for patient_index in range(6):
                images = []
                masks = []
                raw_frames = 8 + patient_index % 3
                for frame_index in range(raw_frames):
                    phase = 2.0 * np.pi * frame_index / float(raw_frames)
                    radius = 4.8 + 1.2 * np.cos(phase + 0.08 * patient_index)
                    center_x = 12.0 + 0.3 * np.sin(phase)
                    mask = (
                        (xx - center_x) ** 2 + (yy - 12.0) ** 2 <= radius**2
                    )
                    image = 0.15 * rng.normal(size=mask.shape) + 0.9 * mask
                    images.append(image.astype(np.float32))
                    masks.append(mask.astype(np.uint8))
                image_path = root / f"patient{patient_index:02d}_image.npy"
                mask_path = root / f"patient{patient_index:02d}_mask.npy"
                np.save(image_path, np.stack(images))
                np.save(mask_path, np.stack(masks))
                rows.append(
                    {
                        "patient_id": f"patient{patient_index:02d}",
                        "image_sequence_path": image_path,
                        "mask_sequence_path": mask_path,
                        "n_frames": raw_frames,
                        "frame_period_s": 0.033,
                        "dataset": "SYNTHETIC_QA",
                        "time_axis": 0,
                    }
                )
            manifest = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            config = {
                "schema_version": "cardiac-sparse-cycle-test",
                "dataset": "SYNTHETIC_QA",
                "seed": 20260825,
                "n_folds": 3,
                "validation_fold_offset": 1,
                "shot_budgets": [3],
                "sampling_strategies": ["uniform"],
                "uniform_replicates": 1,
                "clustered_fraction_of_cycle": 0.35,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            processed = root / "processed"
            splits = root / "splits"
            output_root = root / "deep_outputs"
            subprocess.run(
                [
                    sys.executable,
                    str(
                        CARDIAC_ROOT
                        / "scripts"
                        / "prepare_standardized_sequences.py"
                    ),
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(processed),
                    "--frames",
                    "8",
                    "--height",
                    "24",
                    "--width",
                    "24",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            standardized_manifest = processed / "standardized_manifest.csv"
            prepared = pd.read_csv(standardized_manifest)
            self.assertTrue((prepared["n_frames"] == 8).all())
            self.assertTrue(
                (
                    prepared["preprocessing"]
                    == (
                        "periodic_nearest_temporal_selection+spatial_resize+"
                        "no_intensity_normalization"
                    )
                ).all()
            )
            first = prepared.iloc[0]
            source_indices = json.loads(first["temporal_source_indices_json"])
            raw_images = np.load(rows[0]["image_sequence_path"])
            processed_images = np.load(
                standardized_manifest.parent / first["image_sequence_path"]
            )
            for target_index, source_index in enumerate(source_indices):
                np.testing.assert_allclose(
                    processed_images[target_index], raw_images[source_index]
                )
            subprocess.run(
                [
                    sys.executable,
                    str(CARDIAC_ROOT / "scripts" / "make_sparse_splits.py"),
                    "--manifest",
                    str(standardized_manifest),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(splits),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            methods = (
                "oa_hb_tpm_map",
                "oa_hb_tpm_no_phase_alignment",
                "oa_hb_tpm_isotropic_prior",
                "hb_tpm_latent_mse",
                "target_only",
                "source_mean",
            )
            for method in methods:
                output = output_root / method
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(CARDIAC_ROOT / "scripts" / "train_observation_aware.py"),
                        "--manifest",
                        str(standardized_manifest),
                        "--splits",
                        str(splits / "patient_splits.csv"),
                        "--observations",
                        str(splits / "sparse_observations.csv"),
                        "--fold",
                        "0",
                        "--k",
                        "3",
                        "--strategy",
                        "all",
                        "--method",
                        method,
                        "--output-dir",
                        str(output),
                        "--epochs",
                        "1",
                        "--patience",
                        "1",
                        "--batch-size",
                        "2",
                        "--latent-dim",
                        "3",
                        "--harmonics",
                        "1",
                        "--base-channels",
                        "4",
                        "--posterior-samples",
                        "2",
                        "--device",
                        "cpu",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=f"method={method}\n{completed.stdout}\n{completed.stderr}",
                )
                self.assertIn("Checkpoint:", completed.stdout)
                results = pd.read_csv(output / "patient_results.csv")
                self.assertEqual(results["patient_id"].nunique(), 2)
                self.assertFalse(results["hidden_images_used_by_estimator"].any())
                self.assertFalse(results["hidden_masks_used_by_estimator"].any())
                self.assertTrue(
                    results["mean_hidden_dice"].between(0.0, 1.0).all()
                )
                self.assertTrue(results["trust_gate_reject"].isin([True, False]).all())
                self.assertTrue(
                    (output / "checkpoints" / "best_model.pt").is_file()
                )
                self.assertTrue((output / "run_manifest.json").is_file())
                self.assertTrue((output / "trust_gate_calibration.csv").is_file())
                visible = pd.read_csv(
                    output / "observed_inputs" / "estimator_visible_inputs.csv"
                )
                self.assertTrue(
                    visible["estimator_visible_tensors"]
                    .str.contains("observed_images")
                    .all()
                )
            aggregate = root / "aggregate"
            aggregated = subprocess.run(
                [
                    sys.executable,
                    str(CARDIAC_ROOT / "scripts" / "aggregate_gpu_runs.py"),
                    "--input-root",
                    str(output_root),
                    "--output-dir",
                    str(aggregate),
                    "--bootstrap-draws",
                    "20",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                aggregated.returncode,
                0,
                msg=aggregated.stdout + "\n" + aggregated.stderr,
            )
            paired = pd.read_csv(aggregate / "paired_differences.csv")
            self.assertEqual(paired["candidate_method"].nunique(), 5)
            self.assertTrue((paired["bootstrap_unit"] == "patient").all())


if __name__ == "__main__":
    unittest.main()
