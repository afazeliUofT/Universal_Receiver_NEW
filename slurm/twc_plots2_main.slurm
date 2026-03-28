#!/bin/bash
#SBATCH --job-name=upair-twc2-main
#SBATCH --account=def-rsadve
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/rsadve1/PROJECT/Universal_Receiver_NEW/logs/%x-%j.out

set -euo pipefail
module purge || true
module load StdEnv/2023 || true
module load python/3.10 || true
module load cuda/12.2 || module load cuda/12.1 || true

cd /home/rsadve1/PROJECT/Universal_Receiver_NEW
mkdir -p logs
bash scripts/run_build_twc_plots2_data.sh configs/twc_mild_main.yaml
