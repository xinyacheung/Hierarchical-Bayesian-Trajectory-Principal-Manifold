#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${PACKAGE_ROOT}"

python scripts/preflight.py
python -m unittest cardiac/tests/test_pipeline_smoke.py
python -m unittest cardiac/tests/test_deep_model_smoke.py
python -m unittest photon/tests/test_observation_aware_hb_tpm.py
python -m py_compile \
  scripts/preflight.py \
  cardiac/scripts/*.py \
  cardiac/tests/*.py \
  photon/src/*.py \
  photon/tests/*.py

echo "All package tests passed."
