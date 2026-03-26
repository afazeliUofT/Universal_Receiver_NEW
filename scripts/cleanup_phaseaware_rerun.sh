#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"

rm -rf "${PROJECT_ROOT}/outputs/smoke_upair_cdlc_phaseaware_baselines"
rm -rf "${PROJECT_ROOT}/outputs/upair_cdlc_highmobility_phaseaware_baselines"
rm -f "${PROJECT_ROOT}/logs/upair-smoke-phaseaware-"*.out
rm -f "${PROJECT_ROOT}/logs/upair-compare-phaseaware-"*.out

echo "Removed only old phase-aware logs/outputs. Older smoke/full/richer-baseline results were kept."
