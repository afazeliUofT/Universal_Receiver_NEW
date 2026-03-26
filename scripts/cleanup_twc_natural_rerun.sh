#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"

rm -rf "${PROJECT_ROOT}/TWC_plots"
rm -rf "${PROJECT_ROOT}/outputs/twc_mild_main"
rm -rf "${PROJECT_ROOT}/outputs/twc_mild_clean"
rm -rf "${PROJECT_ROOT}/outputs/twc_mild_dmrsrich"
rm -f "${PROJECT_ROOT}"/logs/upair-twc-main-*.out
rm -f "${PROJECT_ROOT}"/logs/upair-twc-clean-*.out
rm -f "${PROJECT_ROOT}"/logs/upair-twc-dmrsrich-*.out
rm -f "${PROJECT_ROOT}"/logs/upair-twc-plots-*.out
