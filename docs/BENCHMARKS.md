# Benchmarks

All numbers measured 2026-10-26/27 on real hardware, every result **verified against a CPU golden
model before being timed** (naive GEMM checks compare full matrices; tolerance 5% relative —
mismatches are counted and reported, zero tolerance for silent corruption).

## Test platform

| Item | Value |
|---|---|
| Device | Nintendo Switch 2017 (T210 Erista), Switchroot L4T Ubuntu 24.04.4 Noble, kernel 4.9.140-l4t |
| GPU | NVIDIA Tegra X1 (GM20B), CC 5.3, 2 SMs / 256 cores, 921 MHz peak, 3992 MiB |
| Driver | CUDA 10.0 (libcuda from L4T 32.3.1 BSP) |
| Cross PTX | nvcc 10.0.130 (archived x86_64, WSL) |
| Native PTX | clang 18.1.3 (Ubuntu Noble apt) |
| Timing | `CLOCK_MONOTONIC` around 10 synchronized kernel launches, after 1 warm-up launch |

## FMA arithmetic peak (hand-written PTX, dependent chain)

| Kernel | GFLOPS FP32 |
|---|---|
| GPU, 256 threads × 100k FMA dependent chain | **467.9** (99% of the 472 theoretical) |
| CPU reference, 4 threads × 8 independent accumulators (`cpu_fma.c`) | 9.0 |

**GPU ≈ 52× the CPU** on raw FP32 FMA throughput.

## SGEMM (row-major C = A·B, verified vs CPU triple loop)

nvcc-compiled PTX:

| Size | naive (control) | 2×2 tiling | 4×4 tiling | float4/vectorized |
|---|---|---|---|---|
| 256³ | — | ~10 GFLOPS (cold) / 57.6 warm | 65.5 | 65.7 |
| 512³ | — | 63.6 | 76.8 | 75.1 |
| 1024³ | — | 64.6 | **92.7** | 84.8 |

Natively clang-compiled PTX (`-O3`, `+ptx63`):

| Size | 2×2 tiling | 4×4 tiling | float4/vectorized |
|---|---|---|---|
| 256³ | 57.6 | 65.5 | 65.7 |
| 512³ | 63.6 | 76.8 | 75.1 |
| 1024³ | 64.6 | **79.6** | 75.9 |

Native clang reaches **86–90% of nvcc** on compute-bound SGEMM. Compile time: ~0.3 s on-device
(the WSL nvcc path takes minutes including toolchain startup).

Notes:
- Sustained warm rates. The GPU governor (`nvhost_podgov`, locked) ramps clocks under load —
  the *first* benchmark after idle reads much lower; treat single cold samples as noise.
- The 2×2 → 4×4 register-tiling step was worth ~1.5× at 1024³ and 3.7× at 512³; the float4 /
  bank-conflict work helped most at 512³ (+55% over 4×4) but ties at 1024³. cuBLAS-class kernels
  on TX1 reach ~150–250 GFLOPS — double-buffered tiles are the known next step.
- Missing `-O3` on the native compiler costs **~30×** (scalarized loads, no unrolling).

## Memory bandwidth (64 MB float4 sweeps × 40)

| Build | GB/s | % of 25.6 GB/s theoretical |
|---|---|---|
| nvcc PTX | 22.1–22.5 | 87% |
| native clang PTX | 22.1 | 86% |

## Power & thermals (of interest to lab users)

- Runs from a Raspberry Pi USB port: battery *charges* at idle and normal load (measured to 100%
  while compiling); only sustained all-core inference/GEMM draws ~0.5–0.7 W more than a Pi port
  supplies, discharging the battery slowly. Use a PD charger for 24/7 sustained load.
- 55 °C max thermal zone under sustained full GPU load, headless (display off saves ~0.6 GiB RAM
  and meaningful power).
