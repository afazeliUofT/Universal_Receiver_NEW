#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver_NEW"
source "${PROJECT_ROOT}/venv_universal_receiver/bin/activate"
cd "${PROJECT_ROOT}"

required_files=(
  "${PROJECT_ROOT}/TWC_plots2/csv/twc_mild_clean_curves.csv"
  "${PROJECT_ROOT}/TWC_plots2/csv/twc_mild_main_curves.csv"
  "${PROJECT_ROOT}/TWC_plots2/csv/twc_mild_dmrsrich_curves.csv"
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
