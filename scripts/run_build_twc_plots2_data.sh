#!/bin/bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/run_build_twc_plots2_data.sh <config-path>" >&2
  exit 1
fi

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver_NEW"
CONFIG_PATH="$1"

source "${PROJECT_ROOT}/venv_universal_receiver/bin/activate"
cd "${PROJECT_ROOT}"

export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

python scripts/build_twc_plots2_data.py --config "${CONFIG_PATH}"
