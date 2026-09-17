#!/usr/bin/env bash
#SBATCH --job-name=hb_tpm_card
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hb_tpm_card_%A_%a.out
#SBATCH --error=logs/hb_tpm_card_%A_%a.err

set -euo pipefail

: "${ENTRYPOINT:?Set ENTRYPOINT to the implemented cardiac training script}"
: "${MANIFEST:?Set MANIFEST to the private patient manifest}"
: "${SPLITS:?Set SPLITS to patient_splits.csv}"
: "${OBSERVATIONS:?Set OBSERVATIONS to sparse_observations.csv}"

FOLDS=(0 1 2 3 4)
KS=(2 3 5 7 10)
STRATEGIES=(uniform random clustered)

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
STRATEGY_INDEX=$((TASK_ID % 3))
K_INDEX=$(((TASK_ID / 3) % 5))
FOLD_INDEX=$((TASK_ID / 15))

FOLD=${FOLDS[$FOLD_INDEX]}
K=${KS[$K_INDEX]}
STRATEGY=${STRATEGIES[$STRATEGY_INDEX]}
RUN_ROOT=${RUN_ROOT:-outputs/cardiac_gpu}
RUN_DIR=${RUN_ROOT}/fold${FOLD}_k${K}_${STRATEGY}
mkdir -p logs "${RUN_DIR}"

python "${ENTRYPOINT}" \
  --manifest "${MANIFEST}" \
  --splits "${SPLITS}" \
  --observations "${OBSERVATIONS}" \
  --fold "${FOLD}" \
  --k "${K}" \
  --strategy "${STRATEGY}" \
  --seed 20260825 \
  --output-dir "${RUN_DIR}"
