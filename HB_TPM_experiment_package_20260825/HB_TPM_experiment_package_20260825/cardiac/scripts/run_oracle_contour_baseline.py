#!/usr/bin/env python3
"""Periodic signed-distance baseline for the oracle sparse-contour setting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt


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
    raise ValueError(f"unsupported mask sequence: {path}")


def as_time_first(array: np.ndarray, time_axis: int) -> np.ndarray:
    result = np.moveaxis(np.asarray(array), int(time_axis), 0)
    result = np.squeeze(result)
    if result.ndim != 3:
        raise ValueError(f"expected a 3D mask sequence after squeeze; got {result.shape}")
    return result > 0


def periodic_design(phases: np.ndarray, harmonics: int) -> np.ndarray:
    columns = [np.ones_like(phases)]
    for order in range(1, harmonics + 1):
        columns.extend(
            [
                np.cos(2.0 * np.pi * order * phases),
                np.sin(2.0 * np.pi * order * phases),
            ]
        )
    return np.column_stack(columns)


def signed_distance(mask: np.ndarray) -> np.ndarray:
    return distance_transform_edt(mask) - distance_transform_edt(~mask)


def reconstruct(
    masks: np.ndarray,
    observed_indices: np.ndarray,
    *,
    max_harmonics: int,
    ridge: float,
) -> np.ndarray:
    n_frames = len(masks)
    phases = np.arange(n_frames, dtype=float) / float(n_frames)
    harmonics = min(int(max_harmonics), max(1, (len(observed_indices) - 1) // 2))
    observed_design = periodic_design(phases[observed_indices], harmonics)
    full_design = periodic_design(phases, harmonics)
    response = np.stack(
        [signed_distance(masks[index]) for index in observed_indices]
    ).reshape(len(observed_indices), -1)
    gram = observed_design.T @ observed_design
    penalty = float(ridge) * np.eye(gram.shape[0])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + penalty, observed_design.T @ response
    )
    prediction = (full_design @ coefficients).reshape(masks.shape)
    return prediction >= 0.0


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-harmonics", type=int, default=3)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = pd.read_csv(args.manifest)
    splits = pd.read_csv(args.splits)
    observations = pd.read_csv(args.observations)
    test_ids = set(
        splits.loc[
            (splits["fold"] == args.fold) & (splits["role"] == "test"),
            "patient_id",
        ].astype(str)
    )
    selected = observations[
        (observations["fold"] == args.fold)
        & observations["patient_id"].astype(str).isin(test_ids)
    ]
    manifest_by_id = manifest.set_index(manifest["patient_id"].astype(str))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        (args.output_dir / "predictions").mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    for observation in selected.itertuples(index=False):
        started = time.perf_counter()
        patient_id = str(observation.patient_id)
        patient = manifest_by_id.loc[patient_id]
        masks = as_time_first(
            load_array(Path(str(patient["mask_sequence_path"]))),
            int(patient["time_axis"]),
        )
        observed_indices = np.asarray(
            json.loads(observation.observed_frame_indices_json), dtype=int
        )
        predicted = reconstruct(
            masks,
            observed_indices,
            max_harmonics=args.max_harmonics,
            ridge=args.ridge,
        )
        hidden = np.ones(len(masks), dtype=bool)
        hidden[observed_indices] = False
        hidden_dice = np.asarray(
            [dice(masks[index], predicted[index]) for index in np.flatnonzero(hidden)]
        )
        hidden_hd95 = np.asarray(
            [
                hd95_px(masks[index], predicted[index])
                for index in np.flatnonzero(hidden)
            ]
        )
        truth_area = masks.reshape(len(masks), -1).sum(axis=1).astype(float)
        predicted_area = predicted.reshape(len(masks), -1).sum(axis=1).astype(float)
        scale = max(float(np.mean(truth_area)), 1.0)
        rows.append(
            {
                "patient_id": patient_id,
                "fold": args.fold,
                "k": int(observation.k),
                "strategy": observation.strategy,
                "replicate": int(observation.replicate),
                "method_id": "oracle_contour_periodic_fourier",
                "mean_hidden_dice": float(np.mean(hidden_dice)),
                "mean_hidden_hd95_px": float(np.mean(hidden_hd95)),
                "area_curve_relative_rmse": float(
                    np.sqrt(np.mean((predicted_area - truth_area) ** 2)) / scale
                ),
                "cycle_closure_error": 1.0 - dice(predicted[0], predicted[-1]),
                "runtime_s": time.perf_counter() - started,
                "hidden_frames_used_for_fitting": False,
                "setting": "oracle sparse contours; not end-to-end imaging",
            }
        )
        if args.save_predictions:
            slug = (
                f"{patient_id}_k{observation.k}_{observation.strategy}"
                f"_r{observation.replicate}.npz"
            )
            np.savez_compressed(
                args.output_dir / "predictions" / slug,
                predicted_mask=predicted.astype(np.uint8),
                observed_indices=observed_indices,
            )
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "patient_results.csv", index=False)
    summary = (
        results.groupby(["method_id", "k", "strategy"], as_index=False)
        .agg(
            n_patients=("patient_id", "nunique"),
            mean_hidden_dice=("mean_hidden_dice", "mean"),
            mean_hidden_hd95_px=("mean_hidden_hd95_px", "mean"),
            mean_area_curve_relative_rmse=("area_curve_relative_rmse", "mean"),
        )
    )
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
