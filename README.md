# switch-l4t-gpu

**CUDA GPU compute on Nintendo Switch Linux — reconstructed for 2026.**

Run verified, benchmarked CUDA kernels on the Switch's Tegra X1 (GM20B, Maxwell, sm_53) under
Switchroot L4T Ubuntu — with a compiler that runs **natively on the console**, no PC required.

This exists because the knowledge quietly died: NVIDIA has removed **every native arm64 CUDA 10.x
package** from its archives (the old Jetson repo now only serves x86 cross-compilers), and the
Switch's driver is CUDA 10.0 forever — too old for every modern CUDA toolkit. This repo rebuilds
the whole working path from surviving parts.

## Results (measured on hardware, all results verified against CPU golden models)

| Workload | Throughput | % of hardware ceiling |
|---|---|---|
| FP32 FMA peak (hand-written PTX) | **467.9 GFLOPS** | 99% of 472 theoretical |
| SGEMM 1024³, 4×4 register tiling (nvcc PTX) | **92.7 GFLOPS** | ~20% |
| SGEMM 1024³, same kernel compiled natively (clang) | **79.6 GFLOPS** | 86% of nvcc |
| Memory read bandwidth (float4) | **22.1–22.5 GB/s** | 87% of 25.6 theoretical |
| CPU reference (4× Cortex-A57, 4 threads) | 9.0 GFLOPS | GPU ≈ **52×** on FMA |

Full methodology and per-size numbers: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Quick start (native — everything on the Switch)

Requires: Switchroot L4T Ubuntu Noble (unity 5.1.2 image line), your user in the `video` group,
internet on the console.

```bash
git clone https://github.com/pkia/switch-l4t-gpu && cd switch-l4t-gpu
sudo bash tools/setup-native-clang.sh   # once: cudart (from the image's l4t-debs repo)
                                        # + clang-18 + CUDA 10 headers & libdevice
make native && make bench-native        # compile kernels ON the Switch, verify, benchmark
```

Write your own kernels:

```bash
./ncc mykernels.cu                      # native compile: ~0.3 s
./gpu_workload mykernels.ptx            # run through the driver-API harness
```

Kernels are plain CUDA C (`extern "C" __global__ void ...`) — see [switch_gpu.cu](switch_gpu.cu)
for working examples (tiled SGEMM family, bandwidth test) and [gpu_test.c](gpu_test.c) for a
minimal driver-API + hand-written-PTX example that needs no compiler at all.

## Quick start (cross — maximum performance)

For the last ~10–14% of performance, compile with the *actual* archived CUDA 10.0 `nvcc`
(reconstructed in WSL from NVIDIA's still-live x86_64 package pool — see
[tools/setup-wsl-nvcc.sh](tools/setup-wsl-nvcc.sh)):

```bash
bash tools/setup-wsl-nvcc.sh            # once, in WSL (any distro)
bash tools/build-gpu.sh switch_gpu.cu   # nvcc -> PTX (compute_53) -> deploys to the Switch
make bench                              # on the Switch
```

## How it works

The Switch's userspace ships `libcuda.so` (CUDA 10.0 **driver API**). The driver API JIT-compiles
PTX at module-load time, so **no nvcc is needed on the device** — you only need PTX that speaks
ISA ≤ 6.3 / `sm_53`:

- **Native path**: `clang-18` (Ubuntu apt) targets `--cuda-gpu-arch=sm_53 --cuda-feature=+ptx63`.
  Two apt packages plus the CUDA 10 headers/libdevice (architecture-independent, extracted from
  NVIDIA's surviving x86_64 debs) give a fully self-hosted toolchain.
- **Cross path**: archived `cuda-nvcc-10-0` x86_64 in WSL, with a patched gcc-version guard and
  stub `stdio/stdlib/cstdio/...` headers so the 2018 EDG frontend never touches modern glibc.
- **Harness**: a ~200-line C program ([gpu_workload.c](gpu_workload.c)) links `-lcuda`, loads PTX
  via `cuModuleLoadData`, launches kernels, checks results against CPU references, and times them.
  [cuda_drv.h](cuda_drv.h) declares the needed driver API — no toolkit headers required.

## Known limitations (all discovered the hard way)

- **Always `-O3`.** nvcc optimizes device code by default; clang does not. A native build without
  `-O3` is ~30× slower. (`ncc` and the Makefile handle this for you.)
- Runtime-API `<<<>>>` programs hit a clang `cuda_wrappers` ↔ aarch64 glibc SVE-header clash on
  this toolchain combination — see [examples/native_saxpy.cu](examples/native_saxpy.cu). The
  driver-API harness is the supported workflow.
- NVIDIA's old libdevice 10.0 bitcode cannot be parsed by LLVM 18 (`-nocudalib`); kernels here
  avoid libdevice intrinsics.
- The GPU governor (`nvhost_podgov`) is locked; first runs after idle read low until clocks ramp.
- CUDA 10.0 driver means no modern CUDA apps, no Vulkan compute (llvmpipe only), no llama.cpp GPU
  backend — CPU inference still works fine alongside this.

## Repository layout

```
README.md               this file
LICENSE                 MIT
Makefile                native + cross build/bench targets
ncc                     native .cu -> .ptx compiler wrapper (clang-18)
cuda_drv.h              minimal CUDA driver API declarations
switch_gpu.cu           SGEMM kernel family + memory bandwidth kernel
gpu_workload.c          driver-API harness: load, verify, benchmark
gpu_test.c              driver API + hand-written PTX smoke test (no compiler needed)
cpu_fma.c               CPU FMA reference benchmark
examples/               runtime-API example (documents the known limitation)
tools/                  one-shot toolchain setups (native + cross) and deploy helper
docs/                   BENCHMARKS.md, RUNBOOK.md (operations guide)
```

## Credits

- [Switchroot](https://switchroot.org) — the L4T Ubuntu Noble images and the `l4t-debs` package
  repo that still hosts native `cuda-cudart-10-0`.
- NVIDIA — the surviving x86_64 CUDA 10.0 archive packages.
- LLVM/Clang — the NVPTX backend that makes a modern native compiler possible.

## License

MIT — see [LICENSE](LICENSE). Provided as-is; you are responsible for what you flash onto your
own hardware.
