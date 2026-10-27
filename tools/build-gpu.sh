#!/bin/bash
# Cross-compile a .cu file to Tegra X1 PTX with the archived CUDA 10.0 nvcc (WSL)
# and deploy it to the Switch. Requires tools/setup-wsl-nvcc.sh to have been run.
#
# Usage: bash tools/build-gpu.sh file.cu [user@host]
#   e.g. bash tools/build-gpu.sh switch_gpu.cu ev1@192.168.55.1
set -e
CU="${1:?usage: bash tools/build-gpu.sh file.cu [user@host]}"
HOST="${2:-ev1@192.168.55.1}"
BASE=$(basename "$CU" .cu)
NVCC=/opt/cuda10/usr/local/cuda-10.0/bin/nvcc
STUB=-I/opt/cuda10/stubinc

# absolute /mnt/<drive>/ path of the source dir for WSL (Git-Bash on Windows)
ABS=$(cd "$(dirname "$CU")" && pwd)
WIND=$(cygpath -m "$ABS" 2>/dev/null || echo "$ABS")
if [[ "$WIND" == [A-Za-z]:* ]]; then
  WSLD="/mnt/$(echo "${WIND:0:1}" | tr 'A-Z' 'a-z')${WIND:2}"
else
  WSLD="$ABS"
fi

echo "[1/3] nvcc -> PTX (compute_53)..."
MSYS_NO_PATHCONV=1 wsl.exe -u root -- \
  "$NVCC" -ptx -arch=compute_53 "$STUB" -o "$WSLD/$BASE.ptx" "$WSLD/$BASE.cu"
echo "      wrote $ABS/$BASE.ptx"

echo "[2/3] deploying to $HOST..."
ssh "$HOST" "mkdir -p ~/gpu"
scp "$ABS/$BASE.ptx" "$HOST:gpu/"

echo "[3/3] done. On the Switch:  cd ~/gpu && ./gpu_workload $BASE.ptx"
