#!/usr/bin/env bash
#SBATCH --job-name=hb_tpm_card
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=hb_tpm_card_%A_%a.out
#SBATCH --error=hb_tpm_card_%A_%a.err

set -euo pipefail

: "${MANIFEST:?Set MANIFEST to the private patient manifest}"
: "${SPLITS:?Set SPLITS to patient_splits.csv}"
: "${OBSERVATIONS:?Set OBSERVATIONS to sparse_observations.csv}"

CARDIAC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
ENTRYPOINT=${ENTRYPOINT:-${CARDIAC_ROOT}/scripts/train_observation_aware.py}
CONFIG=${CONFIG:-${CARDIAC_ROOT}/configs/oa_hb_tpm_main.json}
METHOD=${METHOD:-oa_hb_tpm_map}

FOLDS=(0 1 2 3 4)
KS=(2 3 5 7 10)

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if (( TASK_ID < 0 || TASK_ID >= 25 )); then
  echo "SLURM_ARRAY_TASK_ID must be in 0..24; received ${TASK_ID}" >&2
  exit 2
fi
K_INDEX=$((TASK_ID % 5))
FOLD_INDEX=$((TASK_ID / 5))

FOLD=${FOLDS[$FOLD_INDEX]}
K=${KS[$K_INDEX]}
RUN_ROOT=${RUN_ROOT:-outputs/cardiac_gpu}
RUN_DIR=${RUN_ROOT}/${METHOD}/fold${FOLD}_k${K}_all_strategies
mkdir -p "${RUN_DIR}"

"${PYTHON}" "${ENTRYPOINT}" \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --splits "${SPLITS}" \
  --observations "${OBSERVATIONS}" \
  --fold "${FOLD}" \
  --k "${K}" \
  --strategy all \
  --method "${METHOD}" \
  --device cuda \
  --seed 20260825 \
  --output-dir "${RUN_DIR}"
