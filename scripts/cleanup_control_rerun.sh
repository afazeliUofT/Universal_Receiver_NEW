#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"

rm -rf "${PROJECT_ROOT}/outputs/upair_cdlc_highmobility_paper_phasefair_cleancontrol"
rm -rf "${PROJECT_ROOT}/outputs/upair_cdlc_highmobility_paper_phasefair_dmrsrich"
rm -rf "${PROJECT_ROOT}/outputs/publication_robustness_summary"

rm -f "${PROJECT_ROOT}/logs/upair-compare-paper-phasefair-cleancontrol-"*.out
rm -f "${PROJECT_ROOT}/logs/upair-compare-paper-phasefair-dmrsrich-"*.out
rm -f "${PROJECT_ROOT}/logs/upair-publication-robustness-"*.out

echo "Removed only robustness-control outputs/logs. Core smoke/full/phasefair results were kept."
