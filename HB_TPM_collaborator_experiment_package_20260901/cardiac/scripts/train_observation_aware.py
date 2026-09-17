#!/usr/bin/env python3
"""Train and evaluate the image-observation cardiac HB-TPM implementation."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CARDIAC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARDIAC_ROOT))

from src.observation_aware_cardiac import (  # noqa: E402
    CardiacObservationAwareHBTPM,
)


METHODS = (
    "oa_hb_tpm_map",
    "oa_hb_tpm_no_phase_alignment",
    "oa_hb_tpm_isotropic_prior",
    "hb_tpm_latent_mse",
    "target_only",
    "source_mean",
)

GENERATED_OUTPUT_NAMES = {
    "logs",
    "figures",
    "checkpoints",
    "observed_inputs",
    "predictions",
    "config_resolved.json",
    "environment.json",
    "history.csv",
    "trust_gate_calibration.csv",
    "patient_results.csv",
    "patient_or_record_results.csv",
    "method_summary.csv",
    "run_manifest.json",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    """Clear only known generated artifacts when overwrite is explicit."""

    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"nonempty output directory: {path}")
        unknown = sorted(
            child.name
            for child in path.iterdir()
            if child.name not in GENERATED_OUTPUT_NAMES
        )
        if unknown:
            raise ValueError(
                "refusing to overwrite a directory containing unknown files: "
                f"{unknown}"
            )
        for child in list(path.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def load_array(path: Path) -> np.ndarray:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        return np.load(path)
    if suffixes.endswith(".npz"):
        archive = np.load(path)
        if len(archive.files) != 1:
            raise ValueError(f"NPZ must contain exactly one array: {path}")
        return archive[archive.files[0]]
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        import nibabel as nib

        return np.asarray(nib.load(str(path)).dataobj)
    raise ValueError(f"unsupported sequence format: {path}")


def as_time_first(array: np.ndarray, time_axis: int, *, name: str) -> np.ndarray:
    result = np.moveaxis(np.asarray(array), int(time_axis), 0)
    result = np.squeeze(result)
    if result.ndim != 3:
        raise ValueError(f"{name}: expected (T,H,W) after squeeze; got {result.shape}")
    return result


class CardiacDataset(Dataset[dict[str, object]]):
    def __init__(self, manifest: pd.DataFrame, patient_ids: Sequence[str]) -> None:
        by_id = manifest.set_index("patient_id", drop=False)
        self.rows = [by_id.loc[str(patient_id)] for patient_id in patient_ids]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        patient_id = str(row["patient_id"])
        time_axis = int(row["time_axis"])
        images = as_time_first(
            load_array(Path(str(row["image_sequence_path"])).expanduser()),
            time_axis,
            name=f"{patient_id} images",
        ).astype(np.float32, copy=False)
        masks = as_time_first(
            load_array(Path(str(row["mask_sequence_path"])).expanduser()),
            time_axis,
            name=f"{patient_id} masks",
        )
        if images.shape != masks.shape:
            raise ValueError(
                f"{patient_id}: image/mask shapes differ: "
                f"{images.shape} vs {masks.shape}"
            )
        phases = np.arange(len(images), dtype=np.float32) / float(len(images))
        return {
            "patient_id": patient_id,
            "images": torch.from_numpy(images[:, None]),
            "masks": torch.from_numpy((masks > 0).astype(np.float32)[:, None]),
            "times": torch.from_numpy(phases),
        }


def collate_uniform(batch: list[dict[str, object]]) -> dict[str, object]:
    shapes = {tuple(item["images"].shape) for item in batch}  # type: ignore[union-attr]
    if len(shapes) != 1:
        raise ValueError(
            "Batched sequences must share (T,1,H,W). Run "
            "prepare_standardized_sequences.py first."
        )
    return {
        "patient_id": [str(item["patient_id"]) for item in batch],
        "images": torch.stack([item["images"] for item in batch]),  # type: ignore[list-item]
        "masks": torch.stack([item["masks"] for item in batch]),  # type: ignore[list-item]
        "times": torch.stack([item["times"] for item in batch]),  # type: ignore[list-item]
    }


def choose_indices(
    patient_ids: Sequence[str],
    n_frames: int,
    k: int,
    *,
    seed: int,
    epoch: int,
    strategy: str,
) -> Tensor:
    rows: list[np.ndarray] = []
    for patient_id in patient_ids:
        rng = np.random.default_rng(stable_seed(seed, epoch, patient_id, strategy))
        local_strategy = strategy
        if strategy == "mixed":
            local_strategy = ("uniform", "random", "clustered")[
                int(rng.integers(0, 3))
            ]
        if local_strategy == "uniform":
            indices = np.rint(np.linspace(0, n_frames - 1, k)).astype(int)
        elif local_strategy == "random":
            indices = np.sort(rng.choice(n_frames, size=k, replace=False))
        elif local_strategy == "clustered":
            window = min(n_frames, max(k, int(np.ceil(0.35 * n_frames))))
            start = int(rng.integers(0, n_frames - window + 1))
            indices = start + np.rint(np.linspace(0, window - 1, k)).astype(int)
        else:
            raise ValueError(f"unknown strategy={strategy}")
        if len(np.unique(indices)) != k:
            raise RuntimeError("observation sampling produced duplicate indices")
        rows.append(indices)
    return torch.as_tensor(np.stack(rows), dtype=torch.long)


def gather_estimator_inputs(
    full_images: Tensor,
    full_times: Tensor,
    observed_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """Gather and normalize observed images without using hidden intensities."""

    observed_images = torch.stack(
        [full_images[b, observed_indices[b]] for b in range(len(full_images))]
    )
    observed_times = torch.stack(
        [full_times[b, observed_indices[b]] for b in range(len(full_times))]
    )
    mean = observed_images.mean(dim=(1, 2, 3, 4), keepdim=True)
    std = observed_images.std(dim=(1, 2, 3, 4), keepdim=True, unbiased=False)
    observed_images = (observed_images - mean) / std.clamp_min(1e-6)
    return observed_images, observed_times


def coefficient_mode(method: str) -> str:
    if method == "target_only":
        return "weak"
    if method == "source_mean":
        return "source_mean"
    return "hierarchical"


def soft_dice_loss(logits: Tensor, truth: Tensor) -> Tensor:
    probability = torch.sigmoid(logits)
    axes = tuple(range(2, probability.ndim))
    intersection = (probability * truth).sum(dim=axes)
    denominator = probability.sum(dim=axes) + truth.sum(dim=axes)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def loss_terms(
    model: CardiacObservationAwareHBTPM,
    output: dict[str, Tensor],
    truth_masks: Tensor,
    *,
    method: str,
    observation_weight: float,
    kl_weight: float,
    cycle_weight: float,
    temporal_weight: float,
) -> dict[str, Tensor]:
    logits = output["mask_logits"]
    segmentation = F.binary_cross_entropy_with_logits(logits, truth_masks)
    segmentation = segmentation + soft_dice_loss(logits, truth_masks)
    probability = torch.sigmoid(logits)
    cycle = F.mse_loss(probability[:, 0], probability[:, -1])
    if probability.shape[1] >= 3:
        second_difference = (
            probability[:, 2:] - 2.0 * probability[:, 1:-1] + probability[:, :-2]
        )
        temporal = second_difference.abs().mean()
    else:
        temporal = probability.new_zeros(())
    if method == "hb_tpm_latent_mse":
        observation = F.mse_loss(
            output["reconstructed_observations"], output["encoded_mean"]
        )
    else:
        observation = output["observation_nll"]
    if coefficient_mode(method) == "hierarchical":
        kl = model.coefficient_kl(
            output["coefficient_mean"], output["coefficient_covariance"]
        )
    else:
        kl = logits.new_zeros(())
    total = (
        segmentation
        + float(observation_weight) * observation
        + float(kl_weight) * kl
        + float(cycle_weight) * cycle
        + float(temporal_weight) * temporal
    )
    return {
        "total": total,
        "segmentation": segmentation,
        "observation": observation,
        "kl": kl,
        "cycle": cycle,
        "temporal": temporal,
    }


def dice(left: np.ndarray, right: np.ndarray) -> float:
    numerator = 2.0 * float(np.sum(left & right))
    denominator = float(np.sum(left) + np.sum(right))
    return 1.0 if denominator == 0 else numerator / denominator


def hd95_px(left: np.ndarray, right: np.ndarray) -> float:
    if not np.any(left) and not np.any(right):
        return 0.0
    if not np.any(left) or not np.any(right):
        return float(np.hypot(*left.shape))
    left_surface = left ^ binary_erosion(left)
    right_surface = right ^ binary_erosion(right)
    distance_to_right = distance_transform_edt(~right_surface)
    distance_to_left = distance_transform_edt(~left_surface)
    distances = np.concatenate(
        [distance_to_right[left_surface], distance_to_left[right_surface]]
    )
    return float(np.quantile(distances, 0.95))


def run_epoch(
    model: CardiacObservationAwareHBTPM,
    loader: DataLoader[dict[str, object]],
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    k: int,
    seed: int,
    epoch: int,
    method: str,
    strategy: str,
    weights: dict[str, float],
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, list[float]] = {
        name: []
        for name in ("total", "segmentation", "observation", "kl", "cycle", "temporal")
    }
    dice_values: list[float] = []
    for batch in loader:
        images = batch["images"].to(device)  # type: ignore[union-attr]
        masks = batch["masks"].to(device)  # type: ignore[union-attr]
        times = batch["times"].to(device)  # type: ignore[union-attr]
        patient_ids = batch["patient_id"]  # type: ignore[assignment]
        indices = choose_indices(
            patient_ids,
            images.shape[1],
            k,
            seed=seed,
            epoch=epoch if training else -1,
            strategy=strategy if training else "uniform",
        ).to(device)
        observed_images, observed_times = gather_estimator_inputs(
            images, times, indices
        )
        with torch.set_grad_enabled(training):
            output = model(
                observed_images,
                observed_times,
                times,
                output_size=tuple(images.shape[-2:]),
                coefficient_mode=coefficient_mode(method),
            )
            terms = loss_terms(
                model,
                output,
                masks,
                method=method,
                observation_weight=weights["observation"],
                kl_weight=weights["kl"],
                cycle_weight=weights["cycle"],
                temporal_weight=weights["temporal"],
            )
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                terms["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        for name, value in terms.items():
            totals[name].append(float(value.detach().cpu()))
        predicted = (torch.sigmoid(output["mask_logits"]) >= 0.5).detach().cpu().numpy()
        truth = masks.detach().cpu().numpy() > 0.5
        for b in range(len(predicted)):
            dice_values.append(
                float(np.mean([dice(predicted[b, t, 0], truth[b, t, 0]) for t in range(len(predicted[b]))]))
            )
    result = {name: float(np.mean(values)) for name, values in totals.items()}
    result["mean_full_cycle_dice"] = float(np.mean(dice_values))
    return result


def calibrate_trust_gate(
    model: CardiacObservationAwareHBTPM,
    loader: DataLoader[dict[str, object]],
    *,
    device: torch.device,
    k: int,
    seed: int,
    method: str,
    quantile: float,
) -> tuple[float, pd.DataFrame]:
    """Calibrate a source-validation residual gate without target labels."""

    if not (0.5 < quantile < 1.0):
        raise ValueError("trust quantile must lie strictly between 0.5 and 1")
    model.eval()
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)  # type: ignore[union-attr]
            times = batch["times"].to(device)  # type: ignore[union-attr]
            patient_ids = batch["patient_id"]  # type: ignore[assignment]
            indices = choose_indices(
                patient_ids,
                images.shape[1],
                k,
                seed=seed,
                epoch=-2,
                strategy="uniform",
            ).to(device)
            observed_images, observed_times = gather_estimator_inputs(
                images, times, indices
            )
            output = model(
                observed_images,
                observed_times,
                times,
                output_size=tuple(images.shape[-2:]),
                coefficient_mode=coefficient_mode(method),
            )
            standardized_squared_residual = (
                (output["encoded_mean"] - output["reconstructed_observations"]).square()
                * output["encoded_log_variance"].neg().exp()
            ).mean(dim=(1, 2))
            for patient_id, score in zip(patient_ids, standardized_squared_residual):
                rows.append(
                    {
                        "patient_id": str(patient_id),
                        "standardized_observation_residual": float(score.cpu()),
                    }
                )
    calibration = pd.DataFrame(rows)
    threshold = float(
        np.quantile(
            calibration["standardized_observation_residual"].to_numpy(),
            quantile,
        )
    )
    calibration["trust_quantile"] = float(quantile)
    calibration["trust_threshold"] = threshold
    return threshold, calibration


def parse_observed_indices(payload: object, n_frames: int, k: int) -> np.ndarray:
    indices = np.asarray(json.loads(str(payload)), dtype=int)
    if indices.shape != (int(k),):
        raise ValueError(f"expected {k} observed indices; got {indices.tolist()}")
    if np.any(indices < 0) or np.any(indices >= n_frames):
        raise ValueError(f"observed indices out of range for n_frames={n_frames}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("observed indices contain duplicates")
    return np.sort(indices)


def evaluate_targets(
    model: CardiacObservationAwareHBTPM,
    *,
    manifest: pd.DataFrame,
    selected_observations: pd.DataFrame,
    device: torch.device,
    method: str,
    posterior_samples: int,
    trust_threshold: float,
    output_dir: Path,
    save_predictions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    by_id = manifest.set_index("patient_id", drop=False)
    result_rows: list[dict[str, object]] = []
    visible_rows: list[dict[str, object]] = []
    if save_predictions:
        (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    for observation in selected_observations.itertuples(index=False):
        started = time.perf_counter()
        patient_id = str(observation.patient_id)
        row = by_id.loc[patient_id]
        images_np = as_time_first(
            load_array(Path(str(row["image_sequence_path"]))),
            int(row["time_axis"]),
            name=f"{patient_id} images",
        ).astype(np.float32, copy=False)
        masks_np = as_time_first(
            load_array(Path(str(row["mask_sequence_path"]))),
            int(row["time_axis"]),
            name=f"{patient_id} masks",
        ) > 0
        n_frames = len(images_np)
        indices_np = parse_observed_indices(
            observation.observed_frame_indices_json,
            n_frames,
            int(observation.k),
        )
        times = (
            torch.arange(n_frames, dtype=torch.float32, device=device)[None]
            / float(n_frames)
        )
        observed_images = torch.from_numpy(images_np[indices_np, None])[None].to(device)
        observed_mean = observed_images.mean(dim=(1, 2, 3, 4), keepdim=True)
        observed_std = observed_images.std(
            dim=(1, 2, 3, 4), keepdim=True, unbiased=False
        )
        observed_images = (observed_images - observed_mean) / observed_std.clamp_min(1e-6)
        observed_times = times[:, torch.from_numpy(indices_np).to(device)]
        with torch.no_grad():
            output = model(
                observed_images,
                observed_times,
                times,
                output_size=tuple(images_np.shape[-2:]),
                coefficient_mode=coefficient_mode(method),
            )
            map_probability = torch.sigmoid(output["mask_logits"])[0]
            if posterior_samples > 1 and method != "source_mean":
                probability_samples = model.sample_mask_probabilities(
                    output,
                    output_size=tuple(images_np.shape[-2:]),
                    n_samples=posterior_samples,
                )[:, 0]
                predictive_probability = probability_samples.mean(dim=0)
                predictive_variance = probability_samples.var(dim=0, unbiased=False)
            else:
                probability_samples = None
                predictive_probability = map_probability
                predictive_variance = torch.zeros_like(map_probability)
        probability_np = map_probability[:, 0].cpu().numpy()
        predictive_probability_np = predictive_probability[:, 0].cpu().numpy()
        variance_np = predictive_variance[:, 0].cpu().numpy()
        predicted = probability_np >= 0.5
        hidden = np.ones(n_frames, dtype=bool)
        hidden[indices_np] = False
        hidden_indices = np.flatnonzero(hidden)
        hidden_dice = np.asarray(
            [dice(masks_np[t], predicted[t]) for t in hidden_indices]
        )
        hidden_hd95 = np.asarray(
            [hd95_px(masks_np[t], predicted[t]) for t in hidden_indices]
        )
        truth_area = masks_np.reshape(n_frames, -1).sum(axis=1).astype(float)
        predicted_area = probability_np.reshape(n_frames, -1).sum(axis=1)
        area_scale = max(float(np.mean(truth_area)), 1.0)
        clipped = np.clip(
            predictive_probability_np[hidden], 1e-6, 1.0 - 1e-6
        )
        entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
        standardized_residual = (
            (output["encoded_mean"] - output["reconstructed_observations"]).square()
            * output["encoded_log_variance"].neg().exp()
        ).mean()
        trust_score = float(standardized_residual.cpu())
        if probability_samples is not None:
            sample_areas = (
                probability_samples[:, :, 0]
                .sum(dim=(-1, -2))
                .cpu()
                .numpy()
            )
            lower_area, upper_area = np.quantile(sample_areas, [0.025, 0.975], axis=0)
            hidden_area_coverage = float(
                np.mean(
                    (truth_area[hidden] >= lower_area[hidden])
                    & (truth_area[hidden] <= upper_area[hidden])
                )
            )
            area_interval_width_relative = float(
                np.mean(upper_area[hidden] - lower_area[hidden]) / area_scale
            )
        else:
            hidden_area_coverage = float("nan")
            area_interval_width_relative = float("nan")
        result_rows.append(
            {
                "patient_id": patient_id,
                "fold": int(observation.fold),
                "k": int(observation.k),
                "strategy": str(observation.strategy),
                "replicate": int(observation.replicate),
                "method_id": method,
                "mean_hidden_dice": float(np.mean(hidden_dice)),
                "mean_hidden_hd95_px": float(np.mean(hidden_hd95)),
                "area_curve_relative_rmse": float(
                    np.sqrt(np.mean((predicted_area - truth_area) ** 2)) / area_scale
                ),
                "cycle_closure_error": 1.0 - dice(predicted[0], predicted[-1]),
                "hidden_brier": float(
                    np.mean((probability_np[hidden] - masks_np[hidden]) ** 2)
                ),
                "posterior_predictive_hidden_brier": float(
                    np.mean(
                        (predictive_probability_np[hidden] - masks_np[hidden]) ** 2
                    )
                ),
                "mean_hidden_predictive_entropy": float(np.mean(entropy)),
                "mean_hidden_predictive_variance": float(np.mean(variance_np[hidden])),
                "hidden_area_95_coverage": hidden_area_coverage,
                "area_95_interval_width_relative": area_interval_width_relative,
                "trust_score": trust_score,
                "trust_threshold": float(trust_threshold),
                "trust_gate_reject": bool(trust_score > trust_threshold),
                "phase_offset_cycles": float(output["phase_offset"][0].cpu()),
                "runtime_s": time.perf_counter() - started,
                "hidden_images_used_by_estimator": False,
                "hidden_masks_used_by_estimator": False,
                "statistical_unit": "patient",
                "setting": "end-to-end sparse image observations",
            }
        )
        visible_rows.append(
            {
                "patient_id": patient_id,
                "fold": int(observation.fold),
                "k": int(observation.k),
                "strategy": str(observation.strategy),
                "replicate": int(observation.replicate),
                "observed_frame_indices_json": json.dumps(indices_np.tolist()),
                "estimator_visible_tensors": "observed_images,observed_times,full_time_grid",
                "scoring_only_tensors": "hidden_images,full_cycle_masks",
            }
        )
        if save_predictions:
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in patient_id)
            destination = output_dir / "predictions" / (
                f"{safe_id}_k{observation.k}_{observation.strategy}_r{observation.replicate}.npz"
            )
            np.savez_compressed(
                destination,
                map_probability=probability_np.astype(np.float32),
                posterior_predictive_probability=predictive_probability_np.astype(
                    np.float32
                ),
                predictive_variance=variance_np.astype(np.float32),
                observed_indices=indices_np,
            )
    return pd.DataFrame(result_rows), pd.DataFrame(visible_rows)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    probe = torch.ones(2, device=device) @ torch.ones(2, device=device)
    if float(probe.cpu()) != 2.0:
        raise RuntimeError(f"device probe failed on {device}")
    return device


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=CARDIAC_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def environment_record(device: torch.device) -> dict[str, object]:
    record: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": properties.total_memory,
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "cudnn_version": torch.backends.cudnn.version(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            }
        )
    return record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON defaults; explicit CLI flags take precedence.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--strategy",
        choices=("all", "uniform", "random", "clustered"),
        required=True,
    )
    parser.add_argument("--method", choices=METHODS, default="oa_hb_tpm_map")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--harmonics", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--maximum-phase-shift", type=float, default=0.25)
    parser.add_argument("--posterior-samples", type=int, default=16)
    parser.add_argument(
        "--train-strategy",
        choices=("mixed", "uniform", "random", "clustered"),
        default="mixed",
    )
    parser.add_argument("--observation-weight", type=float, default=0.10)
    parser.add_argument("--kl-weight", type=float, default=0.001)
    parser.add_argument("--cycle-weight", type=float, default=0.05)
    parser.add_argument("--temporal-weight", type=float, default=0.01)
    parser.add_argument("--trust-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    if pre_args.config is not None:
        config = json.loads(pre_args.config.read_text(encoding="utf-8"))
        valid = {action.dest for action in parser._actions}
        unknown = sorted(set(config) - valid)
        if unknown:
            raise ValueError(f"unknown training config keys: {unknown}")
        parser.set_defaults(**config)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.k < 2:
        raise ValueError("k must be at least 2")
    if min(
        args.epochs,
        args.patience,
        args.batch_size,
        args.latent_dim,
        args.harmonics,
        args.base_channels,
        args.posterior_samples,
    ) < 1:
        raise ValueError("epoch/model/sample counts must all be positive")
    if not (0.0 <= args.maximum_phase_shift <= 0.5):
        raise ValueError("maximum_phase_shift must lie in [0,0.5]")
    if not (0.5 < args.trust_quantile < 1.0):
        raise ValueError("trust_quantile must lie strictly between 0.5 and 1")
    prepare_output_directory(args.output_dir, overwrite=args.overwrite)
    for child in ("logs", "figures", "checkpoints", "observed_inputs"):
        (args.output_dir / child).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    splits = pd.read_csv(args.splits, dtype={"patient_id": str})
    observations = pd.read_csv(args.observations, dtype={"patient_id": str})
    split_fold = splits[splits["fold"] == args.fold]
    role_ids = {
        role: split_fold.loc[split_fold["role"] == role, "patient_id"].tolist()
        for role in ("train", "validation", "test")
    }
    if any(not role_ids[role] for role in role_ids):
        raise ValueError(f"fold {args.fold} has an empty role: {role_ids}")
    manifest_ids = set(manifest["patient_id"])
    missing_ids = sorted(set(sum(role_ids.values(), [])) - manifest_ids)
    if missing_ids:
        raise ValueError(f"split patients missing from manifest: {missing_ids}")
    if (manifest.set_index("patient_id").loc[sum(role_ids.values(), []), "n_frames"].astype(int) < args.k).any():
        raise ValueError("k exceeds n_frames for at least one patient")

    observation_selector = (
        (observations["fold"] == args.fold)
        & (observations["k"] == args.k)
        & observations["patient_id"].isin(role_ids["test"])
    )
    if args.strategy != "all":
        observation_selector &= observations["strategy"] == args.strategy
    selected_observations = observations[observation_selector].copy()
    if selected_observations.empty:
        raise ValueError("no frozen target observations match fold/k/strategy")

    train_loader = DataLoader(
        CardiacDataset(manifest, role_ids["train"]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_uniform,
    )
    validation_loader = DataLoader(
        CardiacDataset(manifest, role_ids["validation"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_uniform,
    )
    model = CardiacObservationAwareHBTPM(
        latent_dim=args.latent_dim,
        harmonics=args.harmonics,
        base_channels=args.base_channels,
        phase_alignment=args.method != "oa_hb_tpm_no_phase_alignment",
        isotropic_prior=args.method == "oa_hb_tpm_isotropic_prior",
        maximum_phase_shift=args.maximum_phase_shift,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    weights = {
        "observation": args.observation_weight,
        "kl": args.kl_weight,
        "cycle": args.cycle_weight,
        "temporal": args.temporal_weight,
    }

    resolved = vars(args).copy()
    resolved = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in resolved.items()
    }
    config_bytes = json.dumps(resolved, sort_keys=True).encode()
    config_hash = sha256_bytes(config_bytes)
    (args.output_dir / "config_resolved.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment = environment_record(device)
    (args.output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    history_rows: list[dict[str, object]] = []
    best_validation = -float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    started_training = time.perf_counter()
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            k=args.k,
            seed=args.seed,
            epoch=epoch,
            method=args.method,
            strategy=args.train_strategy,
            weights=weights,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            optimizer=None,
            device=device,
            k=args.k,
            seed=args.seed,
            epoch=epoch,
            method=args.method,
            strategy="uniform",
            weights=weights,
        )
        history_rows.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
        )
        score = validation_metrics["mean_full_cycle_dice"]
        if score > best_validation:
            best_validation = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['total']:.5f} "
            f"validation_dice={score:.5f} best={best_validation:.5f}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - started_training
    pd.DataFrame(history_rows).to_csv(args.output_dir / "history.csv", index=False)

    trust_threshold, trust_calibration = calibrate_trust_gate(
        model,
        validation_loader,
        device=device,
        k=args.k,
        seed=args.seed,
        method=args.method,
        quantile=args.trust_quantile,
    )
    trust_calibration.to_csv(
        args.output_dir / "trust_gate_calibration.csv", index=False
    )

    source_hash = sha256_bytes("\n".join(sorted(role_ids["train"])).encode())
    target_hash = sha256_bytes("\n".join(sorted(role_ids["test"])).encode())
    checkpoint_metadata = {
        "fold": args.fold,
        "k": args.k,
        "strategy": args.strategy,
        "replicates": sorted(
            selected_observations["replicate"].astype(int).unique().tolist()
        ),
        "method": args.method,
        "source_patient_ids_sha256": source_hash,
        "target_patient_ids_sha256": target_hash,
        "config_sha256": config_hash,
        "git_commit": git_commit(),
        "seed": args.seed,
        "encoder_pretraining_source": "none; trained on fold source patients",
        "best_validation_patient_mean_dice": best_validation,
        "best_epoch": best_epoch,
        "trust_gate_metric": "standardized_observation_residual",
        "trust_gate_quantile": args.trust_quantile,
        "trust_gate_threshold": trust_threshold,
        "estimator_visible_tensors": [
            "observed_images",
            "observed_times",
            "full_time_grid",
        ],
        "training_seconds": training_seconds,
    }
    checkpoint_path = args.output_dir / "checkpoints" / "best_model.pt"
    torch.save(
        {"model_state_dict": best_state, "metadata": checkpoint_metadata},
        checkpoint_path,
    )

    results, visible = evaluate_targets(
        model,
        manifest=manifest,
        selected_observations=selected_observations,
        device=device,
        method=args.method,
        posterior_samples=args.posterior_samples,
        trust_threshold=trust_threshold,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
    )
    results.to_csv(args.output_dir / "patient_results.csv", index=False)
    results.to_csv(args.output_dir / "patient_or_record_results.csv", index=False)
    visible.to_csv(
        args.output_dir / "observed_inputs" / "estimator_visible_inputs.csv",
        index=False,
    )
    summary = (
        results.groupby(["method_id", "k", "strategy"], as_index=False)
        .agg(
            n_patients=("patient_id", "nunique"),
            n_patient_observations=("patient_id", "size"),
            mean_hidden_dice=("mean_hidden_dice", "mean"),
            mean_hidden_hd95_px=("mean_hidden_hd95_px", "mean"),
            mean_area_curve_relative_rmse=("area_curve_relative_rmse", "mean"),
            mean_hidden_brier=("hidden_brier", "mean"),
            mean_hidden_area_95_coverage=("hidden_area_95_coverage", "mean"),
            trust_gate_rejection_rate=("trust_gate_reject", "mean"),
            mean_runtime_s=("runtime_s", "mean"),
        )
    )
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)

    output_paths = [
        args.output_dir / "config_resolved.json",
        args.output_dir / "environment.json",
        args.output_dir / "history.csv",
        args.output_dir / "trust_gate_calibration.csv",
        checkpoint_path,
        args.output_dir / "patient_results.csv",
        args.output_dir / "patient_or_record_results.csv",
        args.output_dir / "method_summary.csv",
        args.output_dir / "observed_inputs" / "estimator_visible_inputs.csv",
    ]
    run_manifest = {
        "run_id": f"cardiac_f{args.fold}_k{args.k}_{args.strategy}_{args.method}",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": sorted(manifest["dataset"].astype(str).unique().tolist()),
        "git_commit": git_commit(),
        "config_sha256": config_hash,
        "split_file_sha256": sha256_file(args.splits),
        "observation_file_sha256": sha256_file(args.observations),
        "manifest_file_sha256": sha256_file(args.manifest),
        "source_patient_ids_sha256": source_hash,
        "target_patient_ids_sha256": target_hash,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_patient_mean_dice": best_validation,
        "trust_gate_metric": "standardized_observation_residual",
        "trust_gate_quantile": args.trust_quantile,
        "trust_gate_threshold": trust_threshold,
        "n_successful_patient_observations": len(results),
        "n_failed_patient_observations": 0,
        "estimator_visible_fields": [
            "pixel values at frozen observed_frame_indices",
            "observed normalized times",
            "full-cycle time grid",
            "source-patient full images and masks during training",
        ],
        "truth_only_fields": [
            "target hidden image pixels",
            "target full-cycle masks",
        ],
        "output_sha256": {
            str(path.relative_to(args.output_dir)): sha256_file(path)
            for path in output_paths
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log_lines = [
        f"device={device}",
        f"best_epoch={best_epoch}",
        f"best_validation_patient_mean_dice={best_validation:.8f}",
        f"training_seconds={training_seconds:.3f}",
        f"patient_observations={len(results)}",
    ]
    (args.output_dir / "logs" / "training.log").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
