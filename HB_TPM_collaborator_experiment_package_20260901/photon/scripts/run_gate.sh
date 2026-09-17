#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PHOTON_ROOT="${PACKAGE_ROOT}/photon"
STAGE=${1:-smoke}

case "${STAGE}" in
  smoke)
    CONFIG="${PHOTON_ROOT}/settings/oa_hb_tpm_smoke.json"
    ;;
  pilot)
    CONFIG="${PHOTON_ROOT}/settings/oa_hb_tpm_pilot.json"
    ;;
  final)
    if ! grep -q '^\- \[x\] final config hash recorded' "${PHOTON_ROOT}/GATE_SIGNOFF.md"; then
      echo "Final gate is not signed off in photon/GATE_SIGNOFF.md" >&2
      exit 2
    fi
    CONFIG="${PHOTON_ROOT}/settings/oa_hb_tpm_final.json"
    ;;
  *)
    echo "Usage: $0 {smoke|pilot|final}" >&2
    exit 2
    ;;
esac

OUTPUT="${PHOTON_ROOT}/outputs/${STAGE}"
python -m unittest "${PHOTON_ROOT}/tests/test_observation_aware_hb_tpm.py"
python "${PHOTON_ROOT}/src/run_observation_aware_hb_tpm_benchmark.py" \
  --protocol "${CONFIG}" \
  --output-dir "${OUTPUT}" \
  --overwrite

echo "Completed ${STAGE}: ${OUTPUT}"
