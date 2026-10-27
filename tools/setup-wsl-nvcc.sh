#!/bin/bash
# Reconstruct the archived CUDA 10.0 nvcc (x86_64) inside WSL, for cross-compiling
# maximum-performance PTX for the Switch (compute_53).
# Run on the Windows PC:  wsl -d <distro> -u root -- bash tools/setup-wsl-nvcc.sh
# Works on any modern distro; two compatibility shims for the 2018 toolchain:
#   - patches host_config.h's gcc-version guard (old EDG rejects gcc >= 8)
#   - installs stub stdio/stdlib/cstdio/... headers so EDG never parses modern
#     glibc/libstdc++ (the "__malloc__ does not take arguments" wall)
set -e
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
DEST=/opt/cuda10
BASE=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64

echo "=== [1/3] download + extract ==="
mkdir -p "$DEST/stubinc" /root/cuda-debs; cd /root/cuda-debs
for p in cuda-nvcc-10-0_10.0.130-1_amd64.deb cuda-cudart-dev-10-0_10.0.130-1_amd64.deb cuda-misc-headers-10-0_10.0.130-1_amd64.deb; do
  [ -f "$p" ] || curl -fsSLO "$BASE/$p"
  dpkg-deb -x "$p" "$DEST" 2>/dev/null || { command -v dpkg-deb >/dev/null || \
    { echo "this distro lacks dpkg-deb; run on a Debian-family distro or install dpkg"; exit 1; }; }
done
NV=$DEST/usr/local/cuda-10.0/bin/nvcc
"$NV" --version | tail -1

echo "=== [2/3] compatibility shims ==="
H=$(find "$DEST" -name host_config.h | head -1)
sed -i "s/#if __GNUC__ > 7/#if __GNUC__ > 99/" "$H"
printf '#ifndef _STDIO_H\n#define _STDIO_H\n#include <stddef.h>\n#endif\n'   > "$DEST/stubinc/stdio.h"
printf '#ifndef _STDLIB_H\n#define _STDLIB_H\n#include <stddef.h>\n#endif\n' > "$DEST/stubinc/stdlib.h"
printf '#ifndef _GLIBCXX_CSTDIO\n#define _GLIBCXX_CSTDIO\n#include <stdio.h>\n#endif\n'                  > "$DEST/stubinc/cstdio"
printf '#ifndef _GLIBCXX_CSTDLIB\n#define _GLIBCXX_CSTDLIB\n#include <stddef.h>\nnamespace std { using ::size_t; using ::ptrdiff_t; }\n#endif\n' > "$DEST/stubinc/cstdlib"
printf '#ifndef _GLIBCXX_CSTDDEF\n#define _GLIBCXX_CSTDDEF\n#include <stddef.h>\n#endif\n'                > "$DEST/stubinc/cstddef"
printf '#ifndef _GLIBCXX_CSTRING\n#define _GLIBCXX_CSTRING\nextern "C" { void* memcpy(void*, const void*, size_t); void* memset(void*, int, size_t); }\n#endif\n' > "$DEST/stubinc/cstring"
printf '#ifndef _GLIBCXX_CSTDINT\n#define _GLIBCXX_CSTDINT\n#include <stdint.h>\n#endif\n'                > "$DEST/stubinc/cstdint"

echo "=== [3/3] smoke test ==="
cat > /tmp/smoke.cu <<'EOF'
extern "C" __global__ void smoke(float* x) { x[threadIdx.x] = threadIdx.x * 2.0f; }
EOF
"$NV" -ptx -arch=compute_53 -I"$DEST/stubinc" -o /tmp/smoke.ptx /tmp/smoke.cu && echo NVCC_PTX_OK

echo
echo "Done. Compile for the Switch with:"
echo "  $NV -ptx -arch=compute_53 -I$DEST/stubinc -o kernel.ptx kernel.cu"
echo "(tools/build-gpu.sh automates compile + deploy)"
