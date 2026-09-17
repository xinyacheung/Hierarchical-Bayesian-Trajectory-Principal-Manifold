#!/usr/bin/env python3
"""Build a private, portable TED manifest from extracted MetaImage pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np
import pandas as pd


CARDIAC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARDIAC_ROOT))

from src.array_io import (  # noqa: E402
    load_array,
    metaimage_info,
    portable_manifest_path,
)


PATIENT_PATTERN = re.compile(r"patient\d{3}$")


def parse_cfg(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_number}: expected key: value")
        key, value = (part.strip() for part in line.split(":", 1))
        values[key] = value
    return values


def normalize_patient_ids(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text.isdigit():
            text = f"patient{int(text):03d}"
        if not PATIENT_PATTERN.fullmatch(text):
            raise ValueError(f"invalid TED patient id: {value!r}")
        normalized.append(text)
    if len(set(normalized)) != len(normalized):
        raise ValueError("patient ids must be unique")
    return normalized


def build_manifest(
    database_dir: Path,
    manifest_path: Path,
    *,
    patient_ids: Sequence[str] | None = None,
    validate_pixels: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    database_dir = Path(database_dir).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    requested = normalize_patient_ids(patient_ids)
    discovered = sorted(
        child.name
        for child in database_dir.iterdir()
        if child.is_dir() and PATIENT_PATTERN.fullmatch(child.name)
    )
    selected = discovered if requested is None else requested
    missing = sorted(set(selected) - set(discovered))
    if missing:
        raise FileNotFoundError(f"TED patient directories are missing: {missing}")
    if not selected:
        raise ValueError("no TED patients selected")

    rows: list[dict[str, object]] = []
    patient_reports: list[dict[str, object]] = []
    for patient_id in selected:
        patient_dir = database_dir / patient_id
        prefix = patient_dir / f"{patient_id}_4CH"
        image_path = prefix.with_name(prefix.name + "_sequence.mhd")
        mask_path = prefix.with_name(prefix.name + "_sequence_gt.mhd")
        cfg_path = prefix.with_name(prefix.name + "_info.cfg")
        if not cfg_path.is_file():
            raise FileNotFoundError(f"missing TED config: {cfg_path}")
        image_info = metaimage_info(image_path)
        mask_info = metaimage_info(mask_path)
        if image_info.shape_time_first != mask_info.shape_time_first:
            raise ValueError(
                f"{patient_id}: image/mask shapes differ: "
                f"{image_info.shape_time_first} vs {mask_info.shape_time_first}"
            )
        config = parse_cfg(cfg_path)
        if "NbFrame" not in config:
            raise ValueError(f"{cfg_path}: missing NbFrame")
        n_frames = int(config["NbFrame"])
        if image_info.shape_time_first[0] != n_frames:
            raise ValueError(
                f"{patient_id}: cfg NbFrame={n_frames}, "
                f"MetaImage frames={image_info.shape_time_first[0]}"
            )
        labels: list[int] = []
        if validate_pixels:
            images = load_array(image_path)
            masks = load_array(mask_path)
            if not np.isfinite(np.asarray(images)).all():
                raise ValueError(f"{patient_id}: image contains NaN/Inf")
            labels = [int(value) for value in np.unique(masks)]
            unknown = sorted(set(labels) - {0, 1, 2})
            if unknown:
                raise ValueError(f"{patient_id}: unsupported TED mask labels: {unknown}")
            if 1 not in labels:
                raise ValueError(f"{patient_id}: LV-cavity label 1 is absent")
        rows.append(
            {
                "patient_id": patient_id,
                "image_sequence_path": portable_manifest_path(
                    image_path, manifest_path
                ),
                "mask_sequence_path": portable_manifest_path(mask_path, manifest_path),
                "n_frames": n_frames,
                "frame_period_s": 1.0 / float(n_frames),
                "dataset": "TED",
                "time_axis": 0,
                "timebase": "normalized_cycle",
                "source_format": "metaimage",
                "source_mask_labels_json": json.dumps(labels, separators=(",", ":")),
            }
        )
        patient_reports.append(
            {
                "patient_id": patient_id,
                "shape_time_first": list(image_info.shape_time_first),
                "mask_labels": labels,
                "raw_image_bytes": image_info.expected_bytes,
                "raw_mask_bytes": mask_info.expected_bytes,
            }
        )
    frame = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_path, index=False)
    counts = frame["n_frames"].astype(int)
    report: dict[str, object] = {
        "patient_count": len(frame),
        "patient_ids": frame["patient_id"].tolist(),
        "frame_count_min": int(counts.min()),
        "frame_count_median": float(counts.median()),
        "frame_count_max": int(counts.max()),
        "timebase": "normalized_cycle",
        "patients": patient_reports,
    }
    return frame, report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--patients", nargs="*")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    frame, report = build_manifest(
        args.database_dir,
        args.manifest,
        patient_ids=args.patients,
        validate_pixels=not args.metadata_only,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        f"Wrote {len(frame)} TED patients to {args.manifest}; "
        f"frames={report['frame_count_min']}..{report['frame_count_max']}."
    )


if __name__ == "__main__":
    main()
