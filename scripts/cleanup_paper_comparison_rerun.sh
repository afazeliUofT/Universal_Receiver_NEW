#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"

rm -rf "${PROJECT_ROOT}/outputs/smoke_upair_cdlc_paper_comparison"
rm -rf "${PROJECT_ROOT}/outputs/upair_cdlc_highmobility_paper_comparison"
rm -f "${PROJECT_ROOT}/logs"/upair-smoke-paper-*.out
rm -f "${PROJECT_ROOT}/logs"/upair-compare-paper-*.out

echo "Removed stale paper-comparison outputs/logs only."
