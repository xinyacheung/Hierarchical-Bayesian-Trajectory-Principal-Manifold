from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


CARDIAC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARDIAC_ROOT))
sys.path.insert(0, str(CARDIAC_ROOT / "scripts"))

from build_ted_manifest import build_manifest  # noqa: E402
from src.array_io import (  # noqa: E402
    load_array,
    metaimage_info,
    resolve_manifest_path,
)


def load_preflight_module():
    path = CARDIAC_ROOT / "scripts" / "ted_preflight.py"
    spec = importlib.util.spec_from_file_location("ted_preflight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load ted_preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_metaimage(path: Path, array: np.ndarray, *, element_type: str = "MET_UCHAR") -> None:
    raw_path = path.with_suffix(".raw")
    np.asarray(array).tofile(raw_path)
    time, height, width = array.shape
    path.write_text(
        "\n".join(
            [
                "ObjectType = Image",
                "NDims = 3",
                "BinaryData = True",
                "BinaryDataByteOrderMSB = False",
                "CompressedData = False",
                "ElementSpacing = 0.308 0.154 1.54",
                f"DimSize = {width} {height} {time}",
                "ElementNumberOfChannels = 1",
                f"ElementType = {element_type}",
                f"ElementDataFile = {raw_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_patient(database: Path, patient_id: str, *, n_frames: int = 4) -> None:
    patient_dir = database / patient_id
    patient_dir.mkdir(parents=True)
    images = np.arange(n_frames * 6 * 8, dtype=np.uint8).reshape(n_frames, 6, 8)
    masks = np.zeros_like(images)
    masks[:, 1:4, 1:4] = 1
    masks[:, 4:6, 3:7] = 2
    prefix = patient_dir / f"{patient_id}_4CH"
    write_metaimage(prefix.with_name(prefix.name + "_sequence.mhd"), images)
    write_metaimage(prefix.with_name(prefix.name + "_sequence_gt.mhd"), masks)
    prefix.with_name(prefix.name + "_info.cfg").write_text(
        f"ED: 1\nES: 3\nNbFrame: {n_frames}\n",
        encoding="utf-8",
    )


class MetaImageTests(unittest.TestCase):
    def test_valid_metaimage_loads_time_first(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ted_mhd_valid_") as raw:
            root = Path(raw)
            expected = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
            header = root / "sequence.mhd"
            write_metaimage(header, expected)
            info = metaimage_info(header)
            self.assertEqual(info.shape_time_first, (3, 4, 5))
            np.testing.assert_array_equal(load_array(header), expected)

    def test_truncated_raw_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ted_mhd_truncated_") as raw:
            root = Path(raw)
            header = root / "sequence.mhd"
            write_metaimage(header, np.zeros((3, 4, 5), dtype=np.uint8))
            header.with_suffix(".raw").write_bytes(b"too short")
            with self.assertRaisesRegex(ValueError, "RAW byte count mismatch"):
                metaimage_info(header)

    def test_unsupported_element_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ted_mhd_type_") as raw:
            root = Path(raw)
            header = root / "sequence.mhd"
            write_metaimage(
                header,
                np.zeros((3, 4, 5), dtype=np.uint8),
                element_type="MET_USHORT",
            )
            with self.assertRaisesRegex(ValueError, "unsupported ElementType"):
                metaimage_info(header)

    def test_relative_manifest_paths_are_portable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ted_manifest_paths_") as raw:
            root = Path(raw)
            manifest = root / "manifests" / "manifest.csv"
            target = root / "data" / "sequence.npy"
            target.parent.mkdir()
            np.save(target, np.zeros((2, 3, 4), dtype=np.uint8))
            resolved = resolve_manifest_path("../data/sequence.npy", manifest)
            self.assertEqual(resolved, target.resolve())


class TedManifestTests(unittest.TestCase):
    def test_builder_writes_normalized_portable_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ted_builder_") as raw:
            root = Path(raw)
            database = root / "TED" / "database"
            write_patient(database, "patient001")
            manifest_path = root / "manifests" / "ted.csv"
            frame, report = build_manifest(
                database,
                manifest_path,
                patient_ids=["patient001"],
            )
            self.assertEqual(report["patient_count"], 1)
            self.assertEqual(frame.iloc[0]["timebase"], "normalized_cycle")
            self.assertAlmostEqual(frame.iloc[0]["frame_period_s"], 0.25)
            self.assertFalse(Path(frame.iloc[0]["image_sequence_path"]).is_absolute())
            self.assertEqual(
                set(np.unique(load_array(resolve_manifest_path(
                    frame.iloc[0]["mask_sequence_path"], manifest_path
                )))),
                {0, 1, 2},
            )

    def test_cfg_frame_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ted_builder_cfg_") as raw:
            root = Path(raw)
            database = root / "TED" / "database"
            write_patient(database, "patient001")
            cfg = database / "patient001" / "patient001_4CH_info.cfg"
            cfg.write_text("ED: 1\nES: 3\nNbFrame: 5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "NbFrame"):
                build_manifest(
                    database,
                    root / "manifest.csv",
                    patient_ids=["patient001"],
                )

    def test_unknown_mask_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ted_builder_label_") as raw:
            root = Path(raw)
            database = root / "TED" / "database"
            write_patient(database, "patient001")
            mask_raw = (
                database
                / "patient001"
                / "patient001_4CH_sequence_gt.raw"
            )
            masks = np.fromfile(mask_raw, dtype=np.uint8)
            masks[0] = 7
            masks.tofile(mask_raw)
            with self.assertRaisesRegex(ValueError, "unsupported TED mask labels"):
                build_manifest(
                    database,
                    root / "manifest.csv",
                    patient_ids=["patient001"],
                )


class ZipSafetyTests(unittest.TestCase):
    def test_unsafe_zip_member_is_rejected(self) -> None:
        module = load_preflight_module()
        with tempfile.TemporaryDirectory(prefix="ted_zip_safety_") as raw:
            root = Path(raw)
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                module.safe_member_destination(root, "../escape.raw")


if __name__ == "__main__":
    unittest.main()
