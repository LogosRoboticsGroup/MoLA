#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  cat <<'EOF'
Usage:
  bash scripts/train_tokenizer.sh <depth|flow|semantic> <calvin|libero|rtx> [MAIN_PROCESS_PORT=29501]
EOF
  exit 1
fi

MODALITY="$1"
DATASET="$2"
MAIN_PROCESS_PORT="${3:-29501}"

case "${MODALITY}" in
  depth|flow|semantic) ;;
  *)
    echo "Unknown tokenizer modality: ${MODALITY}" >&2
    exit 1
    ;;
esac

case "${DATASET}" in
  calvin|libero|rtx) ;;
  *)
    echo "Unknown dataset config: ${DATASET}" >&2
    exit 1
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TOKENIZER_DIR="${MODALITY}_tokenizer"
cd "${PROJECT_ROOT}/${TOKENIZER_DIR}/train"

accelerate launch --main_process_port "${MAIN_PROCESS_PORT}" train_latent_motion_tokenizer.py \
  --config_path "${PROJECT_ROOT}/${TOKENIZER_DIR}/configs/train/${DATASET}.yaml" \
  2>&1 | tee "${PROJECT_ROOT}/${TOKENIZER_DIR}/train_${DATASET}_${MODALITY}.log"
