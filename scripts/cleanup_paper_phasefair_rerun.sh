#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"

rm -f "${PROJECT_ROOT}"/logs/upair-smoke-paper-phasefair-*.out || true
rm -f "${PROJECT_ROOT}"/logs/upair-compare-paper-phasefair-*.out || true
rm -rf "${PROJECT_ROOT}/outputs/smoke_upair_cdlc_paper_phasefair_comparison" || true
rm -rf "${PROJECT_ROOT}/outputs/upair_cdlc_highmobility_paper_phasefair_comparison" || true

echo "Removed only stale paper-phasefair rerun artifacts."
