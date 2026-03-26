#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"
FULL_CKPT="${PROJECT_ROOT}/outputs/upair_cdlc_highmobility/checkpoints/best.weights.h5"

source "${PROJECT_ROOT}/venv_universal_receiver/bin/activate"
cd "${PROJECT_ROOT}"

export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

if [ ! -f "${FULL_CKPT}" ]; then
  echo "Missing full checkpoint: ${FULL_CKPT}"
  echo "Run sbatch slurm/full_pipeline.slurm first."
  exit 1
fi

python tests/regression_identity_init.py
python tests/regression_checkpoint_build.py
python tests/regression_baseline_builds.py --config configs/target_cdlc_highmobility_paper_phasefair_comparison.yaml
python tests/regression_phaseaware_identity.py --config configs/target_cdlc_highmobility_paper_phasefair_comparison.yaml
python tests/regression_paper_cfgres_build.py --config configs/target_cdlc_highmobility_paper_phasefair_comparison.yaml
python tests/regression_paper_phasefair_build.py --config configs/target_cdlc_highmobility_paper_phasefair_comparison.yaml
python -m upair5g.cli eval \
  --config configs/target_cdlc_highmobility_paper_phasefair_comparison.yaml \
  --checkpoint "${FULL_CKPT}"
