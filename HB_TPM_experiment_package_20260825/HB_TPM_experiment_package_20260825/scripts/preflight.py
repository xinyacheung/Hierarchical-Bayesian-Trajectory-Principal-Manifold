#!/usr/bin/env python3
"""Check that the handoff package is complete without touching private data."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "TODO.md",
    "EXPERIMENT_DESIGN.md",
    "DATA_ACCESS.md",
    "RESULT_CONTRACT.md",
    "MANUSCRIPT_HANDOFF.md",
    "cardiac/README.md",
    "cardiac/configs/ted_sparse_cycle.json",
    "cardiac/scripts/validate_manifest.py",
    "cardiac/scripts/make_sparse_splits.py",
    "cardiac/scripts/run_oracle_contour_baseline.py",
    "cardiac/tests/test_pipeline_smoke.py",
    "photon/README.md",
    "photon/src/observation_aware_hb_tpm.py",
    "photon/src/run_observation_aware_hb_tpm_benchmark.py",
    "photon/tests/test_observation_aware_hb_tpm.py",
    "manuscript/hb_tpm_manuscript_preprint.pdf",
)
OPTIONAL_IMPORTS = (
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "yaml",
    "nibabel",
    "torch",
)


def main() -> None:
    missing_files = [item for item in REQUIRED if not (ROOT / item).is_file()]
    print(f"Python: {platform.python_version()}")
    print(f"Package root: {ROOT}")
    if missing_files:
        print("Missing required files:")
        for item in missing_files:
            print(f"  - {item}")
        raise SystemExit(1)
    print(f"Required files: {len(REQUIRED)}/{len(REQUIRED)} present")
    for name in OPTIONAL_IMPORTS:
        status = "available" if importlib.util.find_spec(name) else "missing"
        print(f"Dependency {name}: {status}")
    print("Preflight passed. Dataset access and scientific gates remain TODO.")


if __name__ == "__main__":
    main()
