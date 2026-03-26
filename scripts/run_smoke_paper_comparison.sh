#!/bin/bash
set -euo pipefail

echo "[ALIAS] scripts/run_smoke_paper_comparison.sh delegates to scripts/run_smoke_paper_phasefair_comparison.sh"
exec bash scripts/run_smoke_paper_phasefair_comparison.sh
