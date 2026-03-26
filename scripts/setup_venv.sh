#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/rsadve1/PROJECT/Universal_Receiver"
VENV_NAME="venv_universal_receiver"
VENV_PATH="${PROJECT_ROOT}/${VENV_NAME}"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Venv path   : ${VENV_PATH}"

mkdir -p "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

module --force purge || true
module load StdEnv/2023 || true
module load python/3.10 || true
module load cuda/12.2 || module load cuda/12.1 || true

python --version

if [ -d "${VENV_PATH}" ]; then
  echo "[INFO] Removing existing venv to avoid stale dependency state"
  rm -rf "${VENV_PATH}"
fi

python -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip setuptools wheel packaging
python -m pip install -r requirements.txt

# Install Sionna package only, without pulling sionna-rt / mitsuba
python -m pip install --no-deps sionna==1.2.1

python -m pip install -e .

mkdir -p logs

python - <<'PY'
import sys
print("[INFO] Python executable:", sys.executable)
print("[INFO] Python version   :", sys.version)

try:
    import tensorflow as tf
    print("[INFO] TensorFlow      :", tf.__version__)
    print("[INFO] GPUs            :", tf.config.list_physical_devices("GPU"))
except Exception as e:
    print("[WARN] TensorFlow import issue:", repr(e))

try:
    import sionna
    print("[INFO] Sionna          :", sionna.__version__)
except Exception as e:
    print("[WARN] Sionna import issue:", repr(e))
PY

echo "[INFO] Setup complete."
echo "[INFO] Activate later with:"
echo "       source ${VENV_PATH}/bin/activate"
