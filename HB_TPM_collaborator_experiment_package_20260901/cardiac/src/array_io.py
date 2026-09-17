"""Shared cardiac array and manifest-path I/O helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MetaImageInfo:
    header_path: Path
    data_path: Path
    dim_size_xyz: tuple[int, int, int]
    element_spacing_xyz: tuple[float, float, float]
    element_type: str
    dtype: np.dtype

    @property
    def shape_time_first(self) -> tuple[int, int, int]:
        x, y, time = self.dim_size_xyz
        return time, y, x

    @property
    def expected_bytes(self) -> int:
        return int(np.prod(self.dim_size_xyz, dtype=np.int64)) * self.dtype.itemsize


def _parse_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} must be True or False; received {value!r}")


def parse_metaimage_header(path: Path) -> dict[str, str]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing MetaImage header: {path}")
    fields: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected key = value")
        key, value = (part.strip() for part in line.split("=", 1))
        if key in fields:
            raise ValueError(f"{path}:{line_number}: duplicate field {key}")
        fields[key] = value
    return fields


def metaimage_info(path: Path) -> MetaImageInfo:
    path = Path(path).expanduser().resolve()
    fields = parse_metaimage_header(path)
    required = {
        "NDims",
        "BinaryData",
        "BinaryDataByteOrderMSB",
        "CompressedData",
        "DimSize",
        "ElementNumberOfChannels",
        "ElementType",
        "ElementDataFile",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{path}: missing MetaImage fields: {missing}")
    if fields["NDims"] != "3":
        raise ValueError(f"{path}: only NDims = 3 is supported")
    if not _parse_bool(fields["BinaryData"], field="BinaryData"):
        raise ValueError(f"{path}: ASCII MetaImage data is unsupported")
    if _parse_bool(fields["BinaryDataByteOrderMSB"], field="BinaryDataByteOrderMSB"):
        raise ValueError(f"{path}: big-endian MetaImage data is unsupported")
    if _parse_bool(fields["CompressedData"], field="CompressedData"):
        raise ValueError(f"{path}: compressed MetaImage data is unsupported")
    if fields["ElementNumberOfChannels"] != "1":
        raise ValueError(f"{path}: only one-channel MetaImage data is supported")
    dtype_by_type = {"MET_UCHAR": np.dtype(np.uint8)}
    if fields["ElementType"] not in dtype_by_type:
        raise ValueError(
            f"{path}: unsupported ElementType {fields['ElementType']}; "
            "TED preflight requires MET_UCHAR"
        )
    try:
        dimensions = tuple(int(value) for value in fields["DimSize"].split())
    except ValueError as error:
        raise ValueError(f"{path}: invalid DimSize {fields['DimSize']!r}") from error
    if len(dimensions) != 3 or min(dimensions) < 1:
        raise ValueError(f"{path}: DimSize must contain three positive integers")
    spacing_text = fields.get("ElementSpacing", "1 1 1")
    try:
        spacing = tuple(float(value) for value in spacing_text.split())
    except ValueError as error:
        raise ValueError(f"{path}: invalid ElementSpacing {spacing_text!r}") from error
    if len(spacing) != 3 or min(spacing) <= 0:
        raise ValueError(f"{path}: ElementSpacing must contain three positive values")
    data_reference = Path(fields["ElementDataFile"])
    if data_reference.is_absolute():
        raise ValueError(f"{path}: ElementDataFile must be relative")
    data_path = (path.parent / data_reference).resolve()
    try:
        data_path.relative_to(path.parent.resolve())
    except ValueError as error:
        raise ValueError(f"{path}: ElementDataFile escapes the header directory") from error
    info = MetaImageInfo(
        header_path=path,
        data_path=data_path,
        dim_size_xyz=dimensions,  # type: ignore[arg-type]
        element_spacing_xyz=spacing,  # type: ignore[arg-type]
        element_type=fields["ElementType"],
        dtype=dtype_by_type[fields["ElementType"]],
    )
    if not data_path.is_file():
        raise FileNotFoundError(f"{path}: missing ElementDataFile: {data_path}")
    actual_bytes = data_path.stat().st_size
    if actual_bytes != info.expected_bytes:
        raise ValueError(
            f"{path}: RAW byte count mismatch; expected {info.expected_bytes}, "
            f"found {actual_bytes}"
        )
    return info


def load_array(path: Path) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".mhd"):
        info = metaimage_info(path)
        return np.memmap(
            info.data_path,
            dtype=info.dtype,
            mode="r",
            shape=info.shape_time_first,
            order="C",
        )
    if suffixes.endswith(".npy"):
        return np.load(path, mmap_mode="r")
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


def resolve_manifest_path(value: object, manifest_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path(manifest_path).expanduser().resolve().parent / path
    return path.resolve()


def portable_manifest_path(path: Path, manifest_path: Path) -> str:
    absolute = Path(path).expanduser().resolve()
    base = Path(manifest_path).expanduser().resolve().parent
    try:
        relative = Path(os.path.relpath(absolute, base))
    except ValueError:
        return str(absolute)
    return relative.as_posix()
