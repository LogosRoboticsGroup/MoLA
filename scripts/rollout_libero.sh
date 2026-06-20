#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/rollout_libero.sh \
    <ACTION_MODEL_DIR> \
    <VIDEO_MODEL_DIR> \
    <CLIP_MODEL_DIR_OR_NAME> \
    <LIBERO_REPO_OR_DATA_DIR> \
    [NUM_GPUS=1] \
    [SUITE=libero_goal] \
    [VIT_CHECKPOINT_PATH]

Valid suites: libero_spatial, libero_object, libero_goal, libero_10
EOF
}

if [ "$#" -lt 4 ]; then
  usage
  exit 1
fi

ACTION_MODEL_DIR="$1"
VIDEO_MODEL_DIR="$2"
CLIP_MODEL_DIR_OR_NAME="$3"
LIBERO_PATH="$4"
NUM_GPUS="${5:-1}"
SUITE="${LIBERO_SUITE:-libero_goal}"
VIT_CHECKPOINT_PATH=""

if [ "$#" -ge 6 ]; then
  case "$6" in
    libero_spatial|libero_object|libero_goal|libero_10)
      SUITE="$6"
      VIT_CHECKPOINT_PATH="${7:-}"
      ;;
    *)
      VIT_CHECKPOINT_PATH="$6"
      if [ "$#" -ge 7 ]; then
        SUITE="$7"
      fi
      ;;
  esac
fi

case "${SUITE}" in
  libero_spatial|libero_object|libero_goal|libero_10)
    ;;
  *)
    echo "Invalid LIBERO suite: ${SUITE}" >&2
    echo "Valid suites: libero_spatial, libero_object, libero_goal, libero_10" >&2
    exit 1
    ;;
esac

LOG_DIR="${ACTION_MODEL_DIR}/libero_eval"
mkdir -p "${LOG_DIR}"

BASE_ARGS=(
  --traj_cons
  --rgb_pad 10
  --gripper_pad 4
  --gradient_accumulation_steps 1
  --bf16_module "vision_encoder"
  --calvin_dataset ""
  --workers 16
  --lr_scheduler cosine
  --save_every_iter 50000
  --num_epochs 20
  --seed 42
  --batch_size 64
  --precision fp32
  --weight_decay 1e-4
  --num_resampler_query 6
  --run_name test
  --transformer_layers 24
  --phase "evaluate"
  --finetune_type "${SUITE}"
  --libero_path "${LIBERO_PATH}"
  --save_checkpoint_path checkpoints/
  --action_pred_steps 3
  --future_steps 3
  --sequence_length 7
  --obs_pred
  --gripper_width
  --eval_libero_ensembling
  --video_model_path "${VIDEO_MODEL_DIR}"
  --clip_model_path "${CLIP_MODEL_DIR_OR_NAME}"
)

if [ -n "${VIT_CHECKPOINT_PATH}" ]; then
  BASE_ARGS+=(--vit_checkpoint_path "${VIT_CHECKPOINT_PATH}")
fi

shopt -s nullglob
CHECKPOINTS=("${ACTION_MODEL_DIR}"/*.pt)
if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
  echo "No .pt checkpoints found under ${ACTION_MODEL_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
for checkpoint in "${CHECKPOINTS[@]}"; do
  ckpt_id="$(basename "${checkpoint}" .pt)"
  logfile="${LOG_DIR}/${ckpt_id}.log"

  python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT:-10133}" \
    eval_libero.py \
    "${BASE_ARGS[@]}" \
    --resume_from_checkpoint "${checkpoint}" \
    | tee "${logfile}"
done
