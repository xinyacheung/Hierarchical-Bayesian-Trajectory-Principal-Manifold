#!/usr/bin/env python3
"""Reproducible, patient-level analysis of the frozen 2026-09-19 TED results."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import sys
from typing import Any, Iterable, Sequence
import zipfile

import numpy as np
import pandas as pd

from figure_factory import DISPLAY_NAMES, PALETTE, DualCanvas, legend, map_value, nice_limits, panel_axes


DEFAULT_SEED = 20260825
DEFAULT_BOOTSTRAP_DRAWS = 5000
METHODS = [
    "hb_tpm_latent_mse",
    "oa_hb_tpm_isotropic_prior",
    "oa_hb_tpm_map",
    "oa_hb_tpm_no_phase_alignment",
    "source_mean",
    "target_only",
]
K_VALUES = [2, 3, 5, 7, 10]
STRATEGIES = ["clustered", "random", "uniform"]
PRIMARY_METRICS = OrderedDict(
    [
        ("mean_hidden_dice", {"label": "Hidden Dice", "higher_is_better": True, "fmt": ".4f"}),
        ("mean_hidden_hd95_px", {"label": "Hidden HD95 (px)", "higher_is_better": False, "fmt": ".3f"}),
        ("area_curve_relative_rmse", {"label": "Area relative RMSE", "higher_is_better": False, "fmt": ".4f"}),
    ]
)
SECONDARY_METRICS = OrderedDict(
    [
        ("hidden_brier", {"label": "MAP Brier", "higher_is_better": False, "fmt": ".5f"}),
        ("posterior_predictive_hidden_brier", {"label": "Posterior-predictive Brier", "higher_is_better": False, "fmt": ".5f"}),
        ("hidden_area_95_coverage", {"label": "Nominal 95% area coverage", "higher_is_better": True, "fmt": ".3f"}),
        ("area_95_interval_width_relative", {"label": "Relative interval width", "higher_is_better": None, "fmt": ".3f"}),
    ]
)
ALL_SUMMARY_METRICS = [*PRIMARY_METRICS, "hidden_brier"]
FULL_KEY = ["method_id", "patient_id", "fold", "k", "strategy", "replicate"]
PATIENT_CELL_KEY = ["method_id", "patient_id", "fold", "k", "strategy"]
CONTRASTS = OrderedDict(
    [
        ("source_prior_contribution", ("oa_hb_tpm_map", "target_only")),
        ("observation_likelihood_vs_latent_mse", ("oa_hb_tpm_map", "hb_tpm_latent_mse")),
        ("target_specific_updating", ("oa_hb_tpm_map", "source_mean")),
    ]
)
ABLATIONS = OrderedDict(
    [
        ("full_vs_isotropic_prior", ("oa_hb_tpm_map", "oa_hb_tpm_isotropic_prior")),
        ("full_vs_no_phase_alignment", ("oa_hb_tpm_map", "oa_hb_tpm_no_phase_alignment")),
    ]
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_sha256(path: Path) -> str:
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def resolve_layout(results_root: Path) -> tuple[Path, Path]:
    root = Path(results_root).resolve()
    if (root / "private_data" / "ted_full" / "aggregate").is_dir():
        return root, root / "private_data" / "ted_full"
    if (root / "aggregate").is_dir() and (root / "runs").is_dir():
        package = root.parents[1] if root.name == "ted_full" and root.parent.name == "private_data" else root
        return package, root
    raise FileNotFoundError(
        f"could not find private_data/ted_full or aggregate/runs under results root: {root}"
    )


def finite_or_fail(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    bad = ~np.isfinite(frame[list(columns)].to_numpy(dtype=float))
    if bad.any():
        row, col = np.argwhere(bad)[0]
        raise ValueError(f"non-finite {columns[col]} in {label} row {int(row)}")


def build_canonical_patient_table(results: pd.DataFrame) -> pd.DataFrame:
    """Average sampling replicates, retaining patient as the statistical unit."""
    require_columns(results, FULL_KEY, "patient results")
    numeric = [c for c in results.columns if c not in PATIENT_CELL_KEY and pd.api.types.is_numeric_dtype(results[c])]
    numeric = [c for c in numeric if c != "replicate"]
    grouped = results.groupby(PATIENT_CELL_KEY, as_index=False, sort=True)
    canonical = grouped[numeric].mean()
    counts = grouped.size().rename(columns={"size": "n_sampling_replicates"})
    canonical = canonical.merge(counts, on=PATIENT_CELL_KEY, validate="one_to_one")
    expected = canonical["strategy"].map({"uniform": 1, "random": 5, "clustered": 5})
    if expected.isna().any() or not np.array_equal(canonical["n_sampling_replicates"].to_numpy(), expected.to_numpy()):
        bad = canonical.loc[canonical["n_sampling_replicates"] != expected, PATIENT_CELL_KEY + ["n_sampling_replicates"]]
        raise ValueError(f"unexpected replicate counts after canonicalization: {bad.head().to_dict('records')}")
    return canonical.sort_values(PATIENT_CELL_KEY).reset_index(drop=True)


def pair_methods(
    canonical: pd.DataFrame,
    candidate: str,
    reference: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    keys = ["patient_id", "fold", "k", "strategy"]
    left = canonical.loc[canonical["method_id"] == candidate, keys + list(metrics)]
    right = canonical.loc[canonical["method_id"] == reference, keys + list(metrics)]
    merged = left.merge(right, on=keys, suffixes=("_candidate", "_reference"), validate="one_to_one")
    expected = 98 * len(K_VALUES) * len(STRATEGIES)
    if len(left) != expected or len(right) != expected or len(merged) != expected:
        raise ValueError(
            f"incomplete patient pairing for {candidate} vs {reference}: "
            f"candidate={len(left)}, reference={len(right)}, paired={len(merged)}, expected={expected}"
        )
    for metric in metrics:
        merged[f"{metric}_difference"] = merged[f"{metric}_candidate"] - merged[f"{metric}_reference"]
    return merged


def fold_stratified_bootstrap_mean(
    patient_values: pd.DataFrame,
    value_column: str,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap patients independently within folds, preserving fold sizes."""
    if patient_values["patient_id"].duplicated().any():
        raise ValueError("fold-stratified bootstrap requires one row per patient")
    blocks = []
    for _, group in patient_values.groupby("fold", sort=True):
        values = group[value_column].to_numpy(dtype=float)
        indices = rng.integers(0, len(values), size=(draws, len(values)))
        blocks.append(values[indices])
    means = np.concatenate(blocks, axis=1).mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def classify_interval(metric: str, lower: float, upper: float) -> str:
    if lower <= 0 <= upper:
        return "inconclusive"
    higher = PRIMARY_METRICS.get(metric, SECONDARY_METRICS.get(metric, {})).get("higher_is_better")
    if higher is None:
        return "direction_not_prespecified"
    beneficial = lower > 0 if higher else upper < 0
    return "beneficial" if beneficial else "harmful"


def analyze_contrasts(
    canonical: pd.DataFrame,
    contrasts: OrderedDict[str, tuple[str, str]],
    draws: int,
    seed: int,
    metrics: Sequence[str] = tuple(ALL_SUMMARY_METRICS),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for contrast_name, (candidate, reference) in contrasts.items():
        paired = pair_methods(canonical, candidate, reference, metrics)
        scopes: list[tuple[str, int | str, pd.DataFrame]] = [("global", "all", paired)]
        scopes.extend(("k", int(k), group) for k, group in paired.groupby("k", sort=True))
        scopes.extend(("strategy", str(strategy), group) for strategy, group in paired.groupby("strategy", sort=True))
        scopes.extend(
            ("k_strategy", f"k{int(k)}_{strategy}", group)
            for (k, strategy), group in paired.groupby(["k", "strategy"], sort=True)
        )
        for scope, level, group in scopes:
            for metric in metrics:
                diff_col = f"{metric}_difference"
                patient = group.groupby(["patient_id", "fold"], as_index=False)[diff_col].mean()
                lo, hi = fold_stratified_bootstrap_mean(patient, diff_col, draws, rng)
                cand_mean = float(group[f"{metric}_candidate"].mean())
                ref_mean = float(group[f"{metric}_reference"].mean())
                diff = float(patient[diff_col].mean())
                relative = diff / abs(ref_mean) * 100.0 if ref_mean != 0 else np.nan
                rows.append(
                    {
                        "contrast": contrast_name,
                        "candidate_method": candidate,
                        "reference_method": reference,
                        "scope": scope,
                        "level": level,
                        "metric": metric,
                        "n_paired_patients": int(patient["patient_id"].nunique()),
                        "candidate_mean": cand_mean,
                        "reference_mean": ref_mean,
                        "mean_paired_difference": diff,
                        "relative_change_percent": relative,
                        "bootstrap_95_lower": lo,
                        "bootstrap_95_upper": hi,
                        "interval_interpretation": classify_interval(metric, lo, hi),
                        "difference_definition": "candidate_minus_reference",
                        "bootstrap_unit": "patient_stratified_within_fold",
                    }
                )
    return pd.DataFrame(rows)


@dataclass
class _StorageRef:
    dtype: np.dtype
    key: str
    size: int


@dataclass
class _TensorRef:
    storage: _StorageRef
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad, backward_hooks, metadata=None):
    return _TensorRef(storage, int(offset), tuple(int(v) for v in size), tuple(int(v) for v in stride))


class _CheckpointUnpickler(pickle.Unpickler):
    STORAGE_DTYPES = {
        "FloatStorage": np.dtype("float32"),
        "DoubleStorage": np.dtype("float64"),
        "HalfStorage": np.dtype("float16"),
        "LongStorage": np.dtype("int64"),
        "IntStorage": np.dtype("int32"),
        "ShortStorage": np.dtype("int16"),
        "ByteStorage": np.dtype("uint8"),
        "BoolStorage": np.dtype("bool"),
    }

    def find_class(self, module: str, name: str):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            return _rebuild_tensor_v2
        if module == "torch._utils" and name.startswith("_rebuild_parameter"):
            return lambda tensor, *args: tensor
        if module == "torch" and name in self.STORAGE_DTYPES:
            return self.STORAGE_DTYPES[name]
        raise pickle.UnpicklingError(f"unsupported checkpoint global {module}.{name}")

    def persistent_load(self, saved_id):
        if not isinstance(saved_id, tuple) or saved_id[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id: {saved_id!r}")
        _, dtype, key, _location, size = saved_id[:5]
        return _StorageRef(np.dtype(dtype), str(key), int(size))


def load_checkpoint_state_without_torch(path: Path) -> OrderedDict[str, np.ndarray]:
    """Read tensor-only PyTorch zip checkpoints without importing or executing torch."""
    with zipfile.ZipFile(path) as archive:
        pkl_name = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        prefix = pkl_name[: -len("data.pkl")]
        state = _CheckpointUnpickler(io.BytesIO(archive.read(pkl_name))).load()
        byteorder_name = prefix + "byteorder"
        byteorder = archive.read(byteorder_name).decode("ascii").strip() if byteorder_name in archive.namelist() else "little"
        cache: dict[tuple[str, str], np.ndarray] = {}

        def resolve(value):
            if not isinstance(value, _TensorRef):
                return value
            dtype = value.storage.dtype.newbyteorder("<" if byteorder == "little" else ">")
            cache_key = (value.storage.key, dtype.str)
            if cache_key not in cache:
                raw = archive.read(prefix + "data/" + value.storage.key)
                cache[cache_key] = np.frombuffer(raw, dtype=dtype, count=value.storage.size)
            storage = cache[cache_key]
            start = value.offset
            if not value.shape:
                return np.asarray(storage[start]).copy()
            byte_strides = tuple(step * dtype.itemsize for step in value.stride)
            view = np.lib.stride_tricks.as_strided(storage[start:], shape=value.shape, strides=byte_strides)
            return np.asarray(view).copy()

        if not isinstance(state, (dict, OrderedDict)):
            raise ValueError(f"checkpoint did not contain a state dict: {path}")
        return OrderedDict((str(key), resolve(value)) for key, value in state.items())
