#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver_NEW"

source "${PROJECT_ROOT}/venv_universal_receiver/bin/activate"
cd "${PROJECT_ROOT}"

export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

python scripts/make_twc_plots.py
