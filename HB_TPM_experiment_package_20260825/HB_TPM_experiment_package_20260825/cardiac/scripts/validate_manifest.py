#!/usr/bin/env python3
"""Validate a private cardiac sequence manifest and optional array shapes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "patient_id",
    "image_sequence_path",
    "mask_sequence_path",
    "n_frames",
    "frame_period_s",
    "dataset",
    "time_axis",
)


def load_array(path: Path) -> np.ndarray:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    if suffixes.endswith(".npz"):
        archive = np.load(path)
        if len(archive.files) != 1:
            raise ValueError(f"NPZ must contain exactly one array: {path}")
        return archive[archive.files[0]]
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        import nibabel as nib

        return np.asanyarray(nib.load(str(path)).dataobj)
    raise ValueError(f"unsupported sequence format: {path}")


def time_length(array: np.ndarray, time_axis: int) -> int:
    axis = int(time_axis)
    if axis < 0:
        axis += array.ndim
    if axis < 0 or axis >= array.ndim:
        raise ValueError(f"time_axis {time_axis} invalid for shape {array.shape}")
    return int(array.shape[axis])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-arrays", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    frame = pd.read_csv(args.manifest)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    if frame.empty:
        raise ValueError("manifest is empty")
    if frame["patient_id"].astype(str).duplicated().any():
        duplicates = frame.loc[
            frame["patient_id"].astype(str).duplicated(), "patient_id"
        ].tolist()
        raise ValueError(f"patient_id must be unique; duplicates: {duplicates}")
    if (pd.to_numeric(frame["n_frames"]) < 2).any():
        raise ValueError("every sequence must contain at least two frames")
    if (pd.to_numeric(frame["frame_period_s"]) <= 0).any():
        raise ValueError("frame_period_s must be positive")

    for row in frame.itertuples(index=False):
        image_path = Path(str(row.image_sequence_path)).expanduser()
        mask_path = Path(str(row.mask_sequence_path)).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image sequence: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing mask sequence: {mask_path}")
        if args.check_arrays:
            image = load_array(image_path)
            mask = load_array(mask_path)
            expected = int(row.n_frames)
            image_t = time_length(image, int(row.time_axis))
            mask_t = time_length(mask, int(row.time_axis))
            if image_t != expected or mask_t != expected:
                raise ValueError(
                    f"{row.patient_id}: manifest n_frames={expected}, "
                    f"image={image_t}, mask={mask_t}"
                )
    print(
        f"Validated {len(frame)} unique patients across "
        f"{frame['dataset'].nunique()} dataset(s)."
    )


if __name__ == "__main__":
    main()
