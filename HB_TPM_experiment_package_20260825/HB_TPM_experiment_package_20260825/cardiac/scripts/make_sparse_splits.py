#!/usr/bin/env python3
"""Create frozen patient folds and shared sparse-frame observation masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def stable_seed(base_seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        "|".join([str(base_seed), *(str(part) for part in parts)]).encode()
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def sample_indices(
    n_frames: int,
    k: int,
    strategy: str,
    *,
    rng: np.random.Generator,
    clustered_fraction: float,
) -> np.ndarray:
    if not (2 <= k <= n_frames):
        raise ValueError(f"K={k} must lie between 2 and n_frames={n_frames}")
    if strategy == "uniform":
        indices = np.rint(np.linspace(0, n_frames - 1, k)).astype(int)
    elif strategy == "random":
        indices = np.sort(rng.choice(n_frames, size=k, replace=False))
    elif strategy == "clustered":
        window = max(k, int(np.ceil(clustered_fraction * n_frames)))
        window = min(window, n_frames)
        start = int(rng.integers(0, n_frames - window + 1))
        local = np.rint(np.linspace(0, window - 1, k)).astype(int)
        indices = start + local
    else:
        raise ValueError(f"unknown sampling strategy: {strategy}")
    if len(np.unique(indices)) != k:
        raise RuntimeError("sampling produced duplicate frame indices")
    return np.sort(indices)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = pd.read_csv(args.manifest)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    patient_ids = manifest["patient_id"].astype(str).to_numpy()
    rng = np.random.default_rng(int(config["seed"]))
    shuffled = patient_ids[rng.permutation(len(patient_ids))]
    n_folds = int(config["n_folds"])
    fold_chunks = np.array_split(shuffled, n_folds)
    fold_by_patient = {
        patient_id: fold
        for fold, chunk in enumerate(fold_chunks)
        for patient_id in chunk
    }
    split_rows: list[dict[str, object]] = []
    observation_rows: list[dict[str, object]] = []
    manifest_by_id = manifest.set_index(manifest["patient_id"].astype(str))
    validation_offset = int(config["validation_fold_offset"])
    for fold in range(n_folds):
        validation_fold = (fold + validation_offset) % n_folds
        for patient_id in patient_ids:
            patient_fold = fold_by_patient[patient_id]
            role = (
                "test"
                if patient_fold == fold
                else "validation"
                if patient_fold == validation_fold
                else "train"
            )
            split_rows.append(
                {"fold": fold, "patient_id": patient_id, "role": role}
            )
            if role != "test":
                continue
            n_frames = int(manifest_by_id.loc[patient_id, "n_frames"])
            for k in config["shot_budgets"]:
                for strategy in config["sampling_strategies"]:
                    repeats = int(config[f"{strategy}_replicates"])
                    for replicate in range(repeats):
                        local_rng = np.random.default_rng(
                            stable_seed(
                                int(config["seed"]),
                                fold,
                                patient_id,
                                k,
                                strategy,
                                replicate,
                            )
                        )
                        indices = sample_indices(
                            n_frames,
                            int(k),
                            str(strategy),
                            rng=local_rng,
                            clustered_fraction=float(
                                config["clustered_fraction_of_cycle"]
                            ),
                        )
                        observation_rows.append(
                            {
                                "fold": fold,
                                "patient_id": patient_id,
                                "n_frames": n_frames,
                                "k": int(k),
                                "strategy": strategy,
                                "replicate": replicate,
                                "observed_frame_indices_json": json.dumps(
                                    indices.tolist(), separators=(",", ":")
                                ),
                            }
                        )
    splits = pd.DataFrame(split_rows)
    observations = pd.DataFrame(observation_rows)
    splits.to_csv(output / "patient_splits.csv", index=False)
    observations.to_csv(output / "sparse_observations.csv", index=False)
    (output / "config_resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = splits.groupby(["fold", "role"]).size().unstack(fill_value=0)
    print(summary.to_string())
    print(f"Wrote {len(observations)} frozen target observation masks.")


if __name__ == "__main__":
    main()
