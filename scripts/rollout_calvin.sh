#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/rollout_calvin.sh \
    <VIDEO_MODEL_DIR> \
    <ACTION_MODEL_DIR_OR_CKPT> \
    <CLIP_MODEL_DIR_OR_NAME> \
    <CALVIN_DATASET_DIR> \
    [NUM_GPUS=1]
EOF
}

if [ "$#" -lt 4 ]; then
  usage
  exit 1
fi

VIDEO_MODEL_DIR="$1"
ACTION_MODEL_DIR_OR_CKPT="$2"
CLIP_MODEL_DIR_OR_NAME="$3"
CALVIN_DATASET_DIR="$4"
NUM_GPUS="${5:-1}"

cd "${REPO_ROOT}"
accelerate launch --num_processes "${NUM_GPUS}" policy_evaluation/calvin_evaluate.py \
  --video_model_path "${VIDEO_MODEL_DIR}" \
  --action_model_folder "${ACTION_MODEL_DIR_OR_CKPT}" \
  --clip_model_path "${CLIP_MODEL_DIR_OR_NAME}" \
  --calvin_abc_dir "${CALVIN_DATASET_DIR}"
