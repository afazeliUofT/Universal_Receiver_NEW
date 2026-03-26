#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"

rm -rf "${PROJECT_ROOT}/outputs/publication_robustness_summary"
rm -f "${PROJECT_ROOT}"/logs/upair-publication-robustness-*.out

echo "Cleaned publication-robustness outputs and old publication-robustness .out logs."
