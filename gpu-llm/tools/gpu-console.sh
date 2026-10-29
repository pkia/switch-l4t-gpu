#!/bin/sh
# gpu-console.sh — OPT-IN: unlock the Switch's higher GPU/CPU clocks (console power profile).
# Run this YOURSELF with sudo (the agent does not hold sudo):
#   sudo sh gpu-console.sh on     # console profile: GPU 921.6 MHz / CPU 1785 MHz ceilings
#   sudo sh gpu-console.sh off    # back to handheld profile (thermally safer, default)
#
# Notes:
# - The current profile is printed either way.
# - FP32 GEMV decode is memory-bandwidth-bound (~17 GB/s vs ~20 GB/s ceiling), so expect
#   SMALL decode gains from this. It pays off when SM-bound kernels land (M3 FP16 / M4 GEMM
#   prefill, batched attention). Watch thermals: console profile draws more power.
# - The engine prints its GPU clock context; compare with:
#   cat /sys/class/devfreq/57000000.gpu/cur_freq   (while a run is active)
case "$1" in
  on)  sudo nvpmodel -m 0 ;;
  off) sudo nvpmodel -m 1 ;;
  *)   echo "usage: sudo $0 on|off"; exit 2 ;;
esac
echo "power profile: $(cat /sys/devices/platform/tegra-profiling-provider 2>/dev/null || echo n/a)"
echo "GPU freq now: $(cat /sys/class/devfreq/57000000.gpu/cur_freq) Hz (ramps under load)"
nvpmodel -q 2>/dev/null | head -2
