#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"
source "${PROJECT_ROOT}/venv_universal_receiver/bin/activate"
cd "${PROJECT_ROOT}"

required_files=(
  "${PROJECT_ROOT}/outputs/twc_mild_main/metrics/curves.csv"
  "${PROJECT_ROOT}/outputs/twc_mild_clean/metrics/curves.csv"
  "${PROJECT_ROOT}/outputs/twc_mild_dmrsrich/metrics/curves.csv"
  "${PROJECT_ROOT}/outputs/twc_mild_main/artifacts/channel_example.npz"
)

for file in "${required_files[@]}"; do
  for _ in $(seq 1 120); do
    if [ -f "${file}" ]; then
      break
    fi
    sleep 30
  done
  if [ ! -f "${file}" ]; then
    echo "Missing required input for TWC plots: ${file}"
    exit 1
  fi
done

python scripts/make_twc_plots.py
