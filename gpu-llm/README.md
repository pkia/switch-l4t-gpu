# gpu-llm — hand-written transformer inference engines for the Tegra X1

> Running LLMs on old hardware feels like a modern day can it run quake

No PyTorch, no llama.cpp, no CUDA toolkit. Every kernel is hand-written CUDA C, compiled to
PTX with the Switch's native `ncc` (clang-18 + CUDA 10 headers), and driven through the raw
CUDA driver API (`cuModuleLoadData` + `cuLaunchKernel`). Every kernel is verified against an
independent CPU reference and a numpy ground-truth forward of the original checkpoints.

Runs on a stock switchroot L4T Linux install — the same GPU that renders games runs the math.

## Models

| model | arch | weights | speed | status |
|---|---|---|---|---|
| GPT-2 124M | GPT-2 | FP32, 475 MB | 36 tok/s (83 % of memory-bandwidth ceiling) | verified end-to-end |
| Qwen2.5-0.5B-Instruct | Llama-family | FP16, 943 MB | 8.9 tok/s | verified end-to-end, coherent ChatML answers |

Sample (Qwen2.5-0.5B, sampled per its generation config):

> **Q: Explain what a GPU is in two short sentences.**
> A GPU (Graphics Processing Unit) is an electronic component that processes and executes
> instructions for rendering graphics and visual effects on the computer screen or other
> devices like monitors, tablets, or smartphones. GPUs excel at performing complex
> mathematical calculations required to create high-resolution images and animations with
> incredible detail and speed compared to traditional CPUs

## Layout

```
kernels/gpt2_kernels.cu   GPT-2 kernels: GEMV (warp-per-row, uniform barriers), LayerNorm,
                          GELU, attention (causal), softmax, residual — 8 entry points
kernels/qwen_kernels.cu   Llama-family kernels: FP16-weight GEMV (manual fp16 decode),
                          RMSNorm, RoPE (precomputed cos/sin tables), SwiGLU, GQA attention
engine/gpt2_engine.c      driver-API GPT-2 engine: --selftest / --gen / --debug
engine/qwen_engine.c      Qwen engine: ChatML, Qwen-sampling (temp 0.7 / top_k 20 / top_p 0.8 /
                          rep 1.1, --greedy opt-out), selftest vs independent CPU reference
tools/conv_gpt2.py        HF safetensors -> flat FP32 bin (layout documented in engine header)
tools/conv_qwen.py        HF safetensors bf16 -> flat FP16 bin (incl. RoPE tables + QKV biases)
tools/gpt2_bpe.py         pure-Python byte-level BPE (GPT-2 and Qwen vocabs via TKN_* env)
tools/referee_*.py        numpy ground-truth forwards used to catch every divergence
harness/                  per-kernel bisection harnesses (locate the wedging/lying kernel)
STATUS.md                 full results, verification evidence, and the bug autopsies
PLAN.md                   milestone plan
```

## Constraints that shaped everything

- **Always -O3.** nvcc optimizes device code by default; clang does not — a native build
  without -O3 is ~30× slower. `ncc` compiles kernels with `-O3 -S --cuda-device-only`
  (verified in the wrapper), and both engines are built `gcc -O3`: Qwen decode went
  8.9 → 10.1 tok/s from the host-side bump alone.
- **Runtime-API `<<<>>>` programs hit a clang cuda_wrappers ↔ aarch64 glibc SVE-header
  clash** on this toolchain (see `examples/native_saxpy.cu` in switch-l4t-gpu). Both
  engines use the supported workflow exclusively: driver API (`cuModuleLoadData` +
  `cuLaunchKernel`), no runtime API, no `<<<>>>`.
- **NVIDIA's libdevice 10.0 bitcode is unusable with LLVM 18 (`-nocudalib`).** All kernels
  avoid libdevice intrinsics: hand-rolled `exp` polynomial (4-term Taylor of e^(f·ln2),
  max err ~0.08 %), `tanh` via exp, `__builtin_sqrtf`, manual FP16 bit-decode, no
  `__shfl_*` — warp reductions use volatile shared memory (safe on sm_53's lockstep warps).
- **The GPU governor (nvhost_podgov) is locked; clocks ramp from idle.** First runs read
  low: 76 → 153 → 307 → 460 MHz over ~2 s after idle. Benchmarks here generate ≥60 tokens
  or warm up first; sustained runs hold 460 MHz (handheld cap) at GR3D 87-99 % utilization.
- **CUDA 10.0 driver means no modern CUDA apps, no Vulkan compute (llvmpipe only), no
  llama.cpp GPU backend.** CPU inference still works — but do NOT split a model across
  CPU+GPU on this SoC: unified memory means one ~20 GB/s bus, and pinning all four cores
  during generation *lowered* throughput 8.9 → 7.9 tok/s. The CPU handles tokenization,
  sampling and detokenization (already overlapped with async GPU work); the GPU computes.

- CUDA 10 headers only, `-nocudainc -nocudalib`: **no `cuda_fp16.h`** either — manual
  bit-level FP16 decode in kernels and on the host.
- Driver API only: PTX is JIT-compiled at load; kernels never touch the runtime API.
- 4 GB unified LPDDR4 shared with the OS: one memory bus, ~20 GB/s effective. Decode is
  bandwidth-bound — the GPU alone saturates it (GR3D 87-99 % measured). Adding CPU compute
  *lowers* throughput (8.9 → 7.9 tok/s measured with all four cores pinned): CPU and GPU
  fight over the same bus, so no CPU+GPU model splitting on this hardware.
- Old Eristo-class GPU at 460 MHz handheld: quantization (Q8/Q4) is the next big lever.

## Verify-first culture

Every stage of every engine can be diffed against: (1) a serial C reference computing the
same math, (2) a numpy forward reading the original safetensors, and (3) the real HF
transformers stack. This caught, among others: a divergent `__syncthreads` that hung the
GPU, a 30 %-wrong exp polynomial, floats-vs-bytes driver-API offsets, missing Q/K/V
attention biases, and a weight-layout macro that cascaded through every layer. See
STATUS.md for the autopsies.
