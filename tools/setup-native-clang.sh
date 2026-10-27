#!/bin/bash
# One-shot native CUDA toolchain setup for Switchroot L4T Ubuntu (Noble images).
# Run ON the Switch as root:  sudo bash tools/setup-native-clang.sh
#
# Installs:
#   1. clang-18 (Ubuntu apt) - a fully native CUDA compiler (NVPTX backend)
#   2. cuda-cudart-10-0 - native arm64 CUDA runtime from the image's own
#      theofficialgman/l4t-debs repo (matches the on-device libcuda 10.0 driver)
#   3. CUDA 10.0 headers + libdevice - architecture-independent, extracted from
#      NVIDIA's still-live x86_64 archive packages into /usr/local/cuda-10.0
#
# After this: ./ncc kernel.cu && ./gpu_workload kernel.ptx   (see repo README)
set -e
if [ "$(id -u)" != 0 ]; then echo "run as root: sudo bash $0"; exit 1; fi

echo "=== [1/4] apt packages (clang-18 + native cudart) ==="
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clang-18 curl ca-certificates cuda-cudart-10-0 \
  || { echo "ERROR: cuda-cudart-10-0 not found - is the l4t-debs repo enabled on this image?"; exit 1; }

echo "=== [2/4] CUDA dirs ==="
mkdir -p /usr/local/cuda-10.0/bin /usr/local/cuda-10.0/include

echo "=== [3/4] headers + libdevice from NVIDIA's x86_64 archive (arch-independent parts) ==="
TMP=$(mktemp -d); cd "$TMP"
BASE=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64
for p in cuda-cudart-dev-10-0_10.0.130-1_amd64.deb cuda-misc-headers-10-0_10.0.130-1_amd64.deb cuda-nvcc-10-0_10.0.130-1_amd64.deb; do
  echo "  $p"
  curl -fsSLO "$BASE/$p"
  dpkg-deb -x "$p" "x-${p%%_*}"
done
cp -r x-cuda-cudart-dev-10-0/usr/local/cuda-10.0/targets/x86_64-linux/include/. /usr/local/cuda-10.0/include/
cp -r x-cuda-misc-headers-10-0/usr/local/cuda-10.0/include/.            /usr/local/cuda-10.0/include/
cp -r x-cuda-nvcc-10-0/usr/local/cuda-10.0/nvvm                          /usr/local/cuda-10.0/
cd /; rm -rf "$TMP"

echo "=== [4/4] verify ==="
test -f /usr/local/cuda-10.0/include/cuda_runtime.h
test -f /usr/local/cuda-10.0/nvvm/libdevice/libdevice.10.bc
test -f /usr/local/cuda-10.0/targets/aarch64-linux/lib/libcudart.so.10.0
clang-18 --version | head -1
echo
echo "OK. Make sure your user is in the 'video' group (GPU device access), then:"
echo "  ./ncc switch_gpu.cu && ./gpu_workload switch_gpu.ptx"
