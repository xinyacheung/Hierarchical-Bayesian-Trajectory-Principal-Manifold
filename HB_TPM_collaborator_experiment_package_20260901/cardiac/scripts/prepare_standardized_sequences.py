#!/usr/bin/env python3
"""Geometrically standardize cardiac sequences for batched PyTorch training.

This script selects nearest raw frames on a periodic target grid and performs
spatial interpolation only. It never mixes neighboring frames and deliberately
does not estimate intensity normalization statistics from a full target
sequence; the runner normalizes each target from its observed frames only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


CARDIAC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARDIAC_ROOT))

from src.array_io import (  # noqa: E402
    as_time_first,
    load_array,
    portable_manifest_path,
    resolve_manifest_path,
)


REQUIRED_COLUMNS = (
    "patient_id",
    "image_sequence_path",
    "mask_sequence_path",
    "n_frames",
    "frame_period_s",
    "dataset",
    "time_axis",
)


def periodic_nearest_indices(source_frames: int, target_frames: int) -> np.ndarray:
    """Select a periodic target grid without mixing hidden source frames."""

    phases = np.arange(int(target_frames), dtype=float) / float(target_frames)
    return np.rint(phases * int(source_frames)).astype(int) % int(source_frames)


def resize_spatial(
    sequence: np.ndarray,
    *,
    height: int,
    width: int,
    is_mask: bool,
) -> np.ndarray:
    tensor = torch.as_tensor(sequence, dtype=torch.float32)[:, None]
    if is_mask:
        resized = F.interpolate(
            tensor,
            size=(int(height), int(width)),
            mode="nearest",
        )
        return (resized[:, 0] > 0.5).to(torch.uint8).numpy()
    resized = F.interpolate(
        tensor,
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
    )
    return resized[:, 0].numpy().astype(np.float32, copy=False)


def select_binary_mask(
    masks: np.ndarray, mask_label: int | None, *, patient_id: str
) -> tuple[np.ndarray, str]:
    labels = np.unique(masks)
    nonzero = labels[labels != 0]
    if mask_label is not None:
        if mask_label not in labels:
            raise ValueError(
                f"{patient_id}: requested mask label {mask_label} is absent; "
                f"available labels={labels.tolist()}"
            )
        return masks == mask_label, str(mask_label)
    if len(nonzero) == 0:
        raise ValueError(f"{patient_id}: mask sequence contains no foreground")
    if len(nonzero) > 1:
        raise ValueError(
            f"{patient_id}: multiple nonzero mask labels {nonzero.tolist()}; "
            "pass --mask-label for the LV cavity instead of merging structures"
        )
    return masks > 0, f"auto_single_nonzero:{nonzero[0]}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--height", type=int, default=112)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument(
        "--mask-label",
        type=int,
        help="Foreground label for LV cavity; required when masks are multiclass.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if min(args.frames, args.height, args.width) < 2:
        raise ValueError("frames, height, and width must all be at least 2")
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    missing = [column for column in REQUIRED_COLUMNS if column not in manifest]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"nonempty output directory: {args.output_dir}")
    sequence_dir = args.output_dir / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output_dir / "standardized_manifest.csv"

    rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        patient_id = str(row.patient_id)
        images = as_time_first(
            load_array(resolve_manifest_path(row.image_sequence_path, args.manifest)),
            int(row.time_axis),
            name=f"{patient_id} images",
        )
        masks = as_time_first(
            load_array(resolve_manifest_path(row.mask_sequence_path, args.manifest)),
            int(row.time_axis),
            name=f"{patient_id} masks",
        )
        if images.shape[0] != masks.shape[0]:
            raise ValueError(
                f"{patient_id}: image/mask frame mismatch {images.shape[0]} vs "
                f"{masks.shape[0]}"
            )
        if args.frames > images.shape[0]:
            raise ValueError(
                f"{patient_id}: requested {args.frames} standardized frames from "
                f"only {images.shape[0]} raw frames. Temporal upsampling would "
                "duplicate observations and change the effective shot budget; "
                "choose a frame count no larger than the shortest sequence."
            )
        binary_masks, resolved_mask_label = select_binary_mask(
            masks, args.mask_label, patient_id=patient_id
        )
        source_indices = periodic_nearest_indices(len(images), args.frames)
        standardized_images = resize_spatial(
            images[source_indices],
            height=args.height,
            width=args.width,
            is_mask=False,
        )
        standardized_masks = resize_spatial(
            binary_masks[source_indices],
            height=args.height,
            width=args.width,
            is_mask=True,
        )
        image_path = (sequence_dir / f"{patient_id}_images.npy").resolve()
        mask_path = (sequence_dir / f"{patient_id}_masks.npy").resolve()
        np.save(image_path, standardized_images)
        np.save(mask_path, standardized_masks)
        item = row._asdict()
        item.update(
            {
                "patient_id": patient_id,
                "image_sequence_path": portable_manifest_path(
                    image_path, output_manifest
                ),
                "mask_sequence_path": portable_manifest_path(
                    mask_path, output_manifest
                ),
                "n_frames": int(args.frames),
                "frame_period_s": float(row.frame_period_s)
                * float(images.shape[0])
                / float(args.frames),
                "time_axis": 0,
                "original_n_frames": int(images.shape[0]),
                "temporal_source_indices_json": json.dumps(
                    source_indices.tolist(), separators=(",", ":")
                ),
                "standardized_height": int(args.height),
                "standardized_width": int(args.width),
                "foreground_mask_label": resolved_mask_label,
                "preprocessing": (
                    "periodic_nearest_temporal_selection+spatial_resize+"
                    "no_intensity_normalization"
                ),
            }
        )
        rows.append(item)

    pd.DataFrame(rows).to_csv(output_manifest, index=False)
    print(
        f"Standardized {len(rows)} patients to "
        f"({args.frames},{args.height},{args.width})."
    )
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    main()
