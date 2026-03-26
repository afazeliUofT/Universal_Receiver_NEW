#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver_NEW"

source "${PROJECT_ROOT}/venv_universal_receiver/bin/activate"
cd "${PROJECT_ROOT}"

export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

python tests/regression_identity_init.py
python tests/regression_checkpoint_build.py
python tests/regression_baseline_builds.py --config configs/twc_mild_clean.yaml
python tests/regression_phaseaware_identity.py --config configs/twc_mild_clean.yaml
python tests/regression_paper_phasefair_build.py --config configs/twc_mild_clean.yaml
python tests/regression_paper_phasefair_warningfree.py --config configs/twc_mild_clean.yaml

python -m upair5g.cli full --config configs/twc_mild_clean.yaml
