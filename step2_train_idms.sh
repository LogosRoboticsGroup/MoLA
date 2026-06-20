#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash step2_train_idms.sh <calvin|libero|rtx> [MODALITIES=depth,flow,semantic] [MAIN_PROCESS_PORT=29501]

Examples:
  bash step2_train_idms.sh calvin
  bash step2_train_idms.sh libero depth,semantic
  bash step2_train_idms.sh rtx flow 29511
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 1
fi

DATASET="$1"
MODALITIES_CSV="${2:-depth,flow,semantic}"
BASE_PORT="${3:-29501}"

case "${DATASET}" in
  calvin|libero|rtx) ;;
  *)
    echo "Unknown dataset config: ${DATASET}" >&2
    usage
    exit 1
    ;;
esac

IFS=',' read -r -a MODALITIES <<< "${MODALITIES_CSV}"

idx=0
for modality in "${MODALITIES[@]}"; do
  case "${modality}" in
    depth|flow|semantic) ;;
    *)
      echo "Unknown IDM modality: ${modality}" >&2
      usage
      exit 1
      ;;
  esac

  port=$((BASE_PORT + idx))
  echo "Training ${modality} IDM on ${DATASET} with port ${port}"
  bash "${REPO_ROOT}/idms/scripts/train_tokenizer.sh" "${modality}" "${DATASET}" "${port}"
  idx=$((idx + 1))
done
