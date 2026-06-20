#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_calvin_stage3.sh \
    <CALVIN_DATASET_DIR> \
    <VIDEO_MODEL_DIR> \
    <CLIP_MODEL_DIR_OR_NAME> \
    <NUM_GPUS> \
    <IDM_FLOW_CKPT> \
    <IDM_DEPTH_CKPT> \
    <IDM_SEMANTIC_CKPT> \
    [USE_HALF_DATA=false] \
    [BATCH_SIZE]
EOF
}

if [ "$#" -lt 7 ]; then
  usage
  exit 1
fi

CALVIN_DATASET_DIR="$1"
VIDEO_MODEL_DIR="$2"
CLIP_MODEL_DIR_OR_NAME="$3"
NUM_GPUS="$4"
IDM_FLOW_CKPT="$5"
IDM_DEPTH_CKPT="$6"
IDM_SEMANTIC_CKPT="$7"
USE_HALF_DATA="${8:-false}"
BATCH_SIZE="${9:-}"

EXTRA_ARGS=(
  --root_data_dir "${CALVIN_DATASET_DIR}"
  --video_model_path "${VIDEO_MODEL_DIR}"
  --text_encoder_path "${CLIP_MODEL_DIR_OR_NAME}"
  --token_ckpt_path_flow "${IDM_FLOW_CKPT}"
  --token_ckpt_path_depth "${IDM_DEPTH_CKPT}"
  --token_ckpt_path_semantic "${IDM_SEMANTIC_CKPT}"
)

if [ "${USE_HALF_DATA}" = "true" ]; then
  EXTRA_ARGS+=(--use_half_data True)
fi

if [ -n "${BATCH_SIZE}" ]; then
  EXTRA_ARGS+=(--batch_size "${BATCH_SIZE}")
fi

cd "${REPO_ROOT}"
accelerate launch --num_processes "${NUM_GPUS}" step3_train_action.py "${EXTRA_ARGS[@]}"
