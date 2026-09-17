#!/usr/bin/env python3
"""Fast real-data compatibility preflight for the split TED archive."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Sequence
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


CARDIAC_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = CARDIAC_ROOT.parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(CARDIAC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_ted_manifest import build_manifest  # noqa: E402
from src.array_io import (  # noqa: E402
    as_time_first,
    load_array,
    resolve_manifest_path,
)
from src.observation_aware_cardiac import (  # noqa: E402
    CardiacObservationAwareHBTPM,
)


EXPECTED_ARCHIVE_BYTES = 5_679_806_152
EXPECTED_ARCHIVE_ENTRIES = 495
REPRESENTATIVE_PATIENTS = (
    "patient001",
    "patient004",
    "patient074",
    "patient087",
    "patient090",
)
PART_SUFFIXES = ("aa", "ab", "ac", "ad")
PATIENT_PATTERN = re.compile(r"TED/database/(patient\d{3})/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member_destination(root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"unsafe absolute ZIP member: {member_name}")
    root = root.resolve()
    destination = (root / normalized).resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ValueError(f"unsafe ZIP member path: {member_name}") from error
    return destination


def reassemble_archive(
    parts_dir: Path, destination: Path, *, overwrite: bool
) -> dict[str, object]:
    parts = [parts_dir / f"TED.zip.part-{suffix}" for suffix in PART_SUFFIXES]
    missing = [str(path) for path in parts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing TED archive parts: {missing}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == EXPECTED_ARCHIVE_BYTES:
            return {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "reused": True,
                "sha256": sha256_file(destination),
                "part_sha256": {},
            }
        if not overwrite:
            raise FileExistsError(
                f"existing reconstructed archive is invalid or overwrite was not set: "
                f"{destination}"
            )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    archive_digest = hashlib.sha256()
    part_hashes: dict[str, str] = {}
    try:
        with temporary.open("wb") as output:
            for part in parts:
                part_digest = hashlib.sha256()
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        output.write(block)
                        archive_digest.update(block)
                        part_digest.update(block)
                part_hashes[part.name] = part_digest.hexdigest()
        if temporary.stat().st_size != EXPECTED_ARCHIVE_BYTES:
            raise ValueError(
                f"reassembled TED archive has {temporary.stat().st_size} bytes; "
                f"expected {EXPECTED_ARCHIVE_BYTES}"
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "reused": False,
        "sha256": archive_digest.hexdigest(),
        "part_sha256": part_hashes,
    }


def parse_equals_text(text: str, *, name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{name}:{line_number}: expected key = value")
        key, value = (part.strip() for part in line.split("=", 1))
        values[key] = value
    return values


def parse_colon_text(text: str, *, name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{name}:{line_number}: expected key: value")
        key, value = (part.strip() for part in line.split(":", 1))
        values[key] = value
    return values


def inspect_archive(archive_path: Path) -> dict[str, object]:
    if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("TED archive size changed after reconstruction")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) != EXPECTED_ARCHIVE_ENTRIES:
            raise ValueError(
                f"TED ZIP contains {len(infos)} entries; "
                f"expected {EXPECTED_ARCHIVE_ENTRIES}"
            )
        scratch_root = archive_path.parent / "path_safety_check"
        for info in infos:
            safe_member_destination(scratch_root, info.filename)
        names = {info.filename for info in infos}
        patients = sorted(
            {
                match.group(1)
                for name in names
                if (match := PATIENT_PATTERN.match(name)) is not None
            }
        )
        if patients != [f"patient{index:03d}" for index in range(1, 99)]:
            raise ValueError("TED ZIP does not contain the expected patient001..patient098")
        metadata_rows: list[dict[str, object]] = []
        for patient_id in patients:
            root = f"TED/database/{patient_id}/{patient_id}_4CH"
            image_header_name = root + "_sequence.mhd"
            mask_header_name = root + "_sequence_gt.mhd"
            cfg_name = root + "_info.cfg"
            required = (image_header_name, mask_header_name, cfg_name)
            missing = [name for name in required if name not in names]
            if missing:
                raise ValueError(f"{patient_id}: missing archive entries: {missing}")
            image_header = parse_equals_text(
                archive.read(image_header_name).decode("utf-8"),
                name=image_header_name,
            )
            mask_header = parse_equals_text(
                archive.read(mask_header_name).decode("utf-8"),
                name=mask_header_name,
            )
            cfg = parse_colon_text(
                archive.read(cfg_name).decode("utf-8"), name=cfg_name
            )
            image_dims = tuple(int(value) for value in image_header["DimSize"].split())
            mask_dims = tuple(int(value) for value in mask_header["DimSize"].split())
            if image_dims != mask_dims:
                raise ValueError(f"{patient_id}: image/mask DimSize mismatch")
            n_frames = int(cfg["NbFrame"])
            if len(image_dims) != 3 or image_dims[2] != n_frames:
                raise ValueError(f"{patient_id}: DimSize disagrees with NbFrame")
            for kind, header, header_name in (
                ("image", image_header, image_header_name),
                ("mask", mask_header, mask_header_name),
            ):
                if header.get("NDims") != "3":
                    raise ValueError(f"{patient_id} {kind}: NDims must be 3")
                if header.get("ElementType") != "MET_UCHAR":
                    raise ValueError(f"{patient_id} {kind}: expected MET_UCHAR")
                if header.get("CompressedData", "False").lower() != "false":
                    raise ValueError(f"{patient_id} {kind}: compressed RAW unsupported")
                if header.get("BinaryDataByteOrderMSB", "False").lower() != "false":
                    raise ValueError(f"{patient_id} {kind}: big-endian RAW unsupported")
                raw_name = str(Path(header_name).parent / header["ElementDataFile"])
                raw_name = raw_name.replace("\\", "/")
                if raw_name not in names:
                    raise ValueError(f"{patient_id} {kind}: missing RAW entry {raw_name}")
                expected_bytes = int(np.prod(image_dims, dtype=np.int64))
                if archive.getinfo(raw_name).file_size != expected_bytes:
                    raise ValueError(f"{patient_id} {kind}: RAW byte count mismatch")
            metadata_rows.append(
                {
                    "patient_id": patient_id,
                    "n_frames": n_frames,
                    "dim_size_xyz": list(image_dims),
                    "element_type": "MET_UCHAR",
                }
            )
    frame_counts = [int(row["n_frames"]) for row in metadata_rows]
    metadata_payload = json.dumps(
        metadata_rows, separators=(",", ":"), sort_keys=True
    ).encode()
    representative_metadata = [
        row for row in metadata_rows if row["patient_id"] in REPRESENTATIVE_PATIENTS
    ]
    return {
        "entry_count": len(infos),
        "patient_count": len(metadata_rows),
        "frame_count_min": min(frame_counts),
        "frame_count_median": float(np.median(frame_counts)),
        "frame_count_max": max(frame_counts),
        "metadata_sha256": hashlib.sha256(metadata_payload).hexdigest(),
        "representative_metadata": representative_metadata,
    }


def extract_selected_patients(
    archive_path: Path,
    extraction_root: Path,
    patient_ids: Sequence[str],
    *,
    overwrite: bool,
) -> list[str]:
    prefixes = tuple(f"TED/database/{patient_id}/" for patient_id in patient_ids)
    extracted: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if not info.filename.startswith(prefixes):
                continue
            destination = safe_member_destination(extraction_root, info.filename)
            if destination.exists() and not overwrite:
                if destination.stat().st_size != info.file_size:
                    raise FileExistsError(f"existing extracted file has wrong size: {destination}")
                extracted.append(info.filename)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            extracted.append(info.filename)
    expected = len(patient_ids) * 5
    if len(extracted) != expected:
        raise ValueError(f"extracted {len(extracted)} patient files; expected {expected}")
    return extracted


def run_command(command: Sequence[str], *, cwd: Path) -> dict[str, object]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr}"
        )
    return {
        "command": [str(value) for value in command],
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.strip().splitlines()[-3:],
    }


def validate_standardized(manifest_path: Path) -> dict[str, object]:
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    shapes: dict[str, list[int]] = {}
    for row in manifest.itertuples(index=False):
        images = as_time_first(
            load_array(resolve_manifest_path(row.image_sequence_path, manifest_path)),
            int(row.time_axis),
            name=f"{row.patient_id} images",
        )
        masks = as_time_first(
            load_array(resolve_manifest_path(row.mask_sequence_path, manifest_path)),
            int(row.time_axis),
            name=f"{row.patient_id} masks",
        )
        if images.shape != (24, 112, 112) or masks.shape != images.shape:
            raise ValueError(f"{row.patient_id}: unexpected standardized shape")
        if not np.isfinite(np.asarray(images)).all():
            raise ValueError(f"{row.patient_id}: standardized image contains NaN/Inf")
        labels = set(int(value) for value in np.unique(masks))
        if not labels.issubset({0, 1}) or 1 not in labels:
            raise ValueError(f"{row.patient_id}: standardized masks are not binary")
        source_indices = json.loads(row.temporal_source_indices_json)
        if len(source_indices) != 24 or len(set(source_indices)) != 24:
            raise ValueError(f"{row.patient_id}: temporal selection is not 24 unique frames")
        shapes[str(row.patient_id)] = list(images.shape)
    return {"patient_count": len(manifest), "shapes": shapes}


def real_batch_model_check(
    manifest_path: Path, splits_path: Path, observations_path: Path
) -> dict[str, object]:
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    splits = pd.read_csv(splits_path, dtype={"patient_id": str})
    observations = pd.read_csv(observations_path, dtype={"patient_id": str})
    test_id = str(
        splits.loc[
            (splits["fold"] == 0) & (splits["role"] == "test"), "patient_id"
        ].iloc[0]
    )
    observation = observations.loc[
        (observations["fold"] == 0)
        & (observations["patient_id"] == test_id)
        & (observations["k"] == 5)
        & (observations["strategy"] == "uniform")
        & (observations["replicate"] == 0)
    ].iloc[0]
    row = manifest.set_index("patient_id").loc[test_id]
    images = as_time_first(
        load_array(resolve_manifest_path(row.image_sequence_path, manifest_path)),
        int(row.time_axis),
        name=f"{test_id} images",
    ).astype(np.float32, copy=False)
    masks = as_time_first(
        load_array(resolve_manifest_path(row.mask_sequence_path, manifest_path)),
        int(row.time_axis),
        name=f"{test_id} masks",
    )
    indices = np.asarray(json.loads(observation.observed_frame_indices_json), dtype=int)
    observed = torch.from_numpy(images[indices, None].copy())[None]
    observed_mean = observed.mean(dim=(1, 2, 3, 4), keepdim=True)
    observed_std = observed.std(dim=(1, 2, 3, 4), keepdim=True, unbiased=False)
    observed = (observed - observed_mean) / observed_std.clamp_min(1e-6)
    full_times = torch.arange(24, dtype=torch.float32)[None] / 24.0
    observed_times = full_times[:, torch.from_numpy(indices)]
    truth = torch.from_numpy((np.asarray(masks) > 0).astype(np.float32)[:, None])[None]
    torch.manual_seed(20260825)
    model = CardiacObservationAwareHBTPM(
        latent_dim=4,
        harmonics=1,
        base_channels=8,
    )
    output = model(observed, observed_times, full_times, output_size=(112, 112))
    loss = F.binary_cross_entropy_with_logits(output["mask_logits"], truth)
    loss = loss + 1e-3 * model.coefficient_kl(
        output["coefficient_mean"], output["coefficient_covariance"]
    )
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    samples = model.sample_mask_probabilities(
        output, output_size=(112, 112), n_samples=2
    )
    if not bool(torch.isfinite(output["mask_logits"]).all()):
        raise RuntimeError("real TED forward pass produced non-finite logits")
    if not bool(torch.isfinite(loss)) or not finite_gradients:
        raise RuntimeError("real TED backward pass produced non-finite values")
    if not bool(torch.isfinite(samples).all()):
        raise RuntimeError("real TED posterior samples are non-finite")
    return {
        "patient_id": test_id,
        "observed_indices": indices.tolist(),
        "input_shape": list(observed.shape),
        "output_shape": list(output["mask_logits"].shape),
        "posterior_sample_shape": list(samples.shape),
        "loss": float(loss.detach()),
        "finite_gradients": finite_gradients,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPOSITORY_ROOT / "private_data" / "ted_preflight",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    parts_dir = args.parts_dir.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / "preflight_report.json"
    report: dict[str, object] = {
        "schema_version": "ted-engineering-preflight-1.0.0",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "representative_patients": list(REPRESENTATIVE_PATIENTS),
        "exceptions": [],
    }
    try:
        archive_path = work_dir / "archive" / "TED.zip"
        report["archive"] = reassemble_archive(
            parts_dir, archive_path, overwrite=args.overwrite
        )
        report["archive_inspection"] = inspect_archive(archive_path)
        extraction_root = work_dir / "raw"
        extracted_entries = extract_selected_patients(
            archive_path,
            extraction_root,
            REPRESENTATIVE_PATIENTS,
            overwrite=args.overwrite,
        )
        report["extraction"] = {
            "entry_count": len(extracted_entries),
            "patient_ids": list(REPRESENTATIVE_PATIENTS),
        }
        manifest_path = work_dir / "manifests" / "ted_preflight_manifest.csv"
        manifest_report_path = work_dir / "manifests" / "ted_manifest_report.json"
        _, manifest_report = build_manifest(
            extraction_root / "TED" / "database",
            manifest_path,
            patient_ids=REPRESENTATIVE_PATIENTS,
            validate_pixels=True,
        )
        manifest_report_path.write_text(
            json.dumps(manifest_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report["manifest"] = manifest_report
        python = sys.executable
        commands: list[dict[str, object]] = []
        commands.append(
            run_command(
                [
                    python,
                    str(CARDIAC_ROOT / "scripts" / "validate_manifest.py"),
                    "--manifest",
                    str(manifest_path),
                    "--check-arrays",
                ],
                cwd=PACKAGE_ROOT,
            )
        )
        standardized_dir = work_dir / "processed" / "ted_24x112"
        standardize_command = [
            python,
            str(CARDIAC_ROOT / "scripts" / "prepare_standardized_sequences.py"),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(standardized_dir),
            "--frames",
            "24",
            "--height",
            "112",
            "--width",
            "112",
            "--mask-label",
            "1",
        ]
        if args.overwrite:
            standardize_command.append("--overwrite")
        commands.append(run_command(standardize_command, cwd=PACKAGE_ROOT))
        standardized_manifest = standardized_dir / "standardized_manifest.csv"
        report["standardized"] = validate_standardized(standardized_manifest)
        splits_dir = work_dir / "splits_v1"
        split_command = [
            python,
            str(CARDIAC_ROOT / "scripts" / "make_sparse_splits.py"),
            "--manifest",
            str(standardized_manifest),
            "--config",
            str(CARDIAC_ROOT / "configs" / "ted_sparse_cycle.json"),
            "--output-dir",
            str(splits_dir),
        ]
        if args.overwrite:
            split_command.append("--overwrite")
        commands.append(run_command(split_command, cwd=PACKAGE_ROOT))
        splits_path = splits_dir / "patient_splits.csv"
        observations_path = splits_dir / "sparse_observations.csv"
        split_frame = pd.read_csv(splits_path)
        observation_frame = pd.read_csv(observations_path)
        if len(split_frame) != 25 or len(observation_frame) != 275:
            raise ValueError("unexpected preflight split/observation row counts")
        report["splits"] = {
            "split_rows": len(split_frame),
            "observation_rows": len(observation_frame),
        }
        oracle_dir = work_dir / "oracle" / "fold0"
        commands.append(
            run_command(
                [
                    python,
                    str(CARDIAC_ROOT / "scripts" / "run_oracle_contour_baseline.py"),
                    "--manifest",
                    str(standardized_manifest),
                    "--splits",
                    str(splits_path),
                    "--observations",
                    str(observations_path),
                    "--fold",
                    "0",
                    "--output-dir",
                    str(oracle_dir),
                ],
                cwd=PACKAGE_ROOT,
            )
        )
        oracle_results = pd.read_csv(oracle_dir / "patient_results.csv")
        oracle_summary = pd.read_csv(oracle_dir / "method_summary.csv")
        numeric = oracle_results.select_dtypes(include=[np.number])
        if len(oracle_results) != 55 or len(oracle_summary) != 15:
            raise ValueError("unexpected fold-0 oracle row counts")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("oracle results contain NaN/Inf")
        if not (~oracle_results["hidden_frames_used_for_fitting"].astype(bool)).all():
            raise ValueError("oracle used hidden frames for fitting")
        report["oracle"] = {
            "patient_result_rows": len(oracle_results),
            "summary_rows": len(oracle_summary),
            "patient_ids": sorted(oracle_results["patient_id"].astype(str).unique()),
        }
        report["model_check"] = real_batch_model_check(
            standardized_manifest, splits_path, observations_path
        )
        report["commands"] = commands
        report["environment"] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        code_paths = [
            CARDIAC_ROOT / "src" / "array_io.py",
            CARDIAC_ROOT / "src" / "observation_aware_cardiac.py",
            CARDIAC_ROOT / "scripts" / "build_ted_manifest.py",
            CARDIAC_ROOT / "scripts" / "ted_preflight.py",
            CARDIAC_ROOT / "scripts" / "prepare_standardized_sequences.py",
            CARDIAC_ROOT / "scripts" / "make_sparse_splits.py",
            CARDIAC_ROOT / "scripts" / "run_oracle_contour_baseline.py",
            CARDIAC_ROOT / "configs" / "ted_sparse_cycle.json",
        ]
        report["code_sha256"] = {
            str(path.relative_to(PACKAGE_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in code_paths
        }
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["exceptions"] = [f"{type(error).__name__}: {error}"]
        raise
    finally:
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "archive_bytes": report["archive"]["bytes"],
                "archive_sha256": report["archive"]["sha256"],
                "archive_patients": report["archive_inspection"]["patient_count"],
                "tested_patients": list(REPRESENTATIVE_PATIENTS),
                "standardized_shape": [24, 112, 112],
                "oracle_rows": report["oracle"]["patient_result_rows"],
                "finite_gradients": report["model_check"]["finite_gradients"],
                "report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"TED compatibility preflight passed: {report_path}")


if __name__ == "__main__":
    main()
