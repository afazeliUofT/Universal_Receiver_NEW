#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"
SMOKE_CKPT="${PROJECT_ROOT}/outputs/smoke_upair_cdlc/checkpoints/best.weights.h5"

source "${PROJECT_ROOT}/venv_universal_receiver/bin/activate"
cd "${PROJECT_ROOT}"

export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

if [ ! -f "${SMOKE_CKPT}" ]; then
  echo "Missing smoke checkpoint: ${SMOKE_CKPT}"
  echo "Run sbatch slurm/smoke_test.slurm first."
  exit 1
fi

python tests/regression_identity_init.py
python tests/regression_checkpoint_build.py
python tests/regression_baseline_builds.py --config configs/smoke_phaseaware_baselines.yaml
python tests/regression_phaseaware_identity.py --config configs/smoke_phaseaware_baselines.yaml
python -m upair5g.cli eval \
  --config configs/smoke_phaseaware_baselines.yaml \
  --checkpoint "${SMOKE_CKPT}"
