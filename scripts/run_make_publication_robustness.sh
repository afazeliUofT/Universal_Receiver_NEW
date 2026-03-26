#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"
source "${PROJECT_ROOT}/venv_universal_receiver/bin/activate"
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1

REQUIRED_FILES=(
  "${PROJECT_ROOT}/outputs/upair_cdlc_highmobility_paper_phasefair_comparison/metrics/curves.csv"
  "${PROJECT_ROOT}/outputs/upair_cdlc_highmobility_paper_phasefair_cleancontrol/metrics/curves.csv"
  "${PROJECT_ROOT}/outputs/upair_cdlc_highmobility_paper_phasefair_dmrsrich/metrics/curves.csv"
)

MAX_WAIT_SECONDS=$((8 * 60 * 60))
SLEEP_SECONDS=60
ELAPSED=0

echo "[ROBUSTNESS-PLOTS] Waiting for required curves.csv files if needed."

while true; do
  MISSING=()
  for path in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${path}" ]]; then
      MISSING+=("${path}")
    fi
  done

  if [[ ${#MISSING[@]} -eq 0 ]]; then
    echo "[ROBUSTNESS-PLOTS] All required curves files are present."
    break
  fi

  if (( ELAPSED >= MAX_WAIT_SECONDS )); then
    echo "[ROBUSTNESS-PLOTS] Timed out waiting for required files:"
    printf '  %s\n' "${MISSING[@]}"
    exit 1
  fi

  echo "[ROBUSTNESS-PLOTS] Still waiting (${ELAPSED}s elapsed). Missing:"
  printf '  %s\n' "${MISSING[@]}"
  sleep "${SLEEP_SECONDS}"
  ELAPSED=$((ELAPSED + SLEEP_SECONDS))
done

python scripts/make_publication_robustness_figures.py
