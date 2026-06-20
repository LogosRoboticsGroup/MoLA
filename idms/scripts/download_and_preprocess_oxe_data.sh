#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

if [ -z "${OUTPUT_ROOT}" ]; then
  cat <<'EOF' >&2
Please set OUTPUT_ROOT before running, for example:
  OUTPUT_ROOT=/path/to/oxe bash scripts/download_and_preprocess_oxe_data.sh
EOF
  exit 1
fi

DATASET_NAMES=(${DATASET_NAMES:-viola})

mkdir -p "${OUTPUT_ROOT}/tensorflow_datasets" "${OUTPUT_ROOT}/video_datasets"
cd "${PROJECT_ROOT}/data_process"

for dataset_name in "${DATASET_NAMES[@]}"; do
  dataset_path="${OUTPUT_ROOT}/tensorflow_datasets/${dataset_name}"
  if [ -d "${dataset_path}" ]; then
    echo "${dataset_path} exists."
  else
    echo "Downloading ${dataset_name} to ${dataset_path} ..."
    gsutil -m cp -r "gs://gresearch/robotics/${dataset_name}" "${OUTPUT_ROOT}/tensorflow_datasets/"
  fi

  video_path="${OUTPUT_ROOT}/video_datasets/${dataset_name}"
  if [ -d "${video_path}" ]; then
    echo "${video_path} exists."
  else
    echo "Writing ${dataset_name} videos to ${video_path} ..."
    python -u oxe2video.py \
      --dataset_name "${dataset_name}" \
      --input_path "${OUTPUT_ROOT}/tensorflow_datasets/" \
      --output_path "${OUTPUT_ROOT}/video_datasets/"
  fi
done
