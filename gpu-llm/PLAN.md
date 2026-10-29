# GPU-LLM — a hand-written transformer inference engine for the Tegra X1 (sm_53)

**Status (2026-10-28): M0 ✅ M1 ✅ M2 ✅ GPT-2 124M FP32 @ 36 tok/s · M4 ✅ Qwen2.5-0.5B-Instruct
FP16 @ 8.9 tok/s (coherent ChatML answers, verified vs HF stack). Remaining: M3 (FP16 for
GPT-2 — superseded for Qwen which is already FP16), batched prefill + GQA kernel tuning
for Qwen (~21 tok/s ceiling), quantization (Q8/Q4). See STATUS.md.**

**Goal:** token generation for small LLMs, accelerated by the Switch's GPU, built kernel-by-kernel
with CPU golden-model verification at every step. No llama.cpp, no modern CUDA, no PC required —
plain CUDA C → `ncc` → PTX → the proven driver-API harness from `switch-l4t-gpu`.

## Why this can work (the math, measured on this device)

- GPU memory bandwidth: **22.2 GB/s** measured. Token generation (decode) reads every weight once
  per token, so `ceiling tok/s = bandwidth / model-size-in-bytes`.
- FP32 GEMV/GEMM verified working: 92.8 GFLOPS SGEMM 1024³, 22 GB/s streaming reads.
- 3.5 GiB usable RAM → models up to ~2.5 GB fit entirely in GPU memory (weights + KV cache).

| Model | Precision | Weights | Bandwidth ceiling | Notes |
|---|---|---|---|---|
| GPT-2 124M | FP32 | 500 MB | ~44 tok/s | M2 target: simplest architecture, no RoPE |
| GPT-2 124M | FP16 | 248 MB | ~88 tok/s | M3 target |
| Qwen2.5 0.5B | Q8_0 | ~0.6 GB | ~35 tok/s | M4 target (RMSNorm+RoPE+SwiGLU) |
| Qwen2.5 1.5B | Q4_K-ish | ~1.0 GB | ~22 tok/s | M4 stretch |
| 3B-class | Q4 | ~1.9 GB | ~11 tok/s | fits, but tight with KV cache |

Ceilings are for decode (memory-bound). Prefill (prompt processing) is compute-bound and can reuse
the verified 92.8 GFLOPS SGEMM.

## What is deliberately NOT this project

- Porting llama.cpp's CUDA backend (needs CUDA 11.7+, hundreds of kernels — dead end here).
- Stable-Diffusion-class image generation (needs >4 GB and modern CUDA).
- Tensor cores (sm_53 has none — plain FMA is all the hardware offers; 468 GFLOPS peak is the cap).

## Milestones

### M0 — GEMV, the decode hot path (first session)
FP32 `y[n] = Σ_k W[n][k]·x[k]` at LLM dimensions (4096×4096). Naive + float4 variants, verified
against CPU golden, measured GB/s vs the 22 GB/s ceiling. Done when ≥15 GB/s effective.

### M1 — a real decoder block, random weights
RMSNorm → GEMV chain (QKV/attn-out/MLP) → residual adds → SiLU/GELU, all fp32, all golden-verified.
An `engine.c` that runs one full transformer block on GPU tensors and matches a CPU reference
bit-close (1e-4 relative).

### M2 — end-to-end GPT-2 124M FP32 inference
Weight conversion (HF → flat .bin on the Pi, scp to Switch), token ids in → greedy token stream
out (BPE deferred: the Pi client pre-tokenizes). Logits verified against a CPU reference
inference. Measure tok/s; target ≥20 tok/s.

### M3 — FP16 + KV cache + prefill
FP16 weights (half2 loads), KV-cache growing buffers, prefill via batched SGEMM (reuses the
verified 92.8 GFLOPS work). Target: 30+ tok/s on GPT-2-class.

### M4 — modern architecture + quantization
Qwen2.5-0.5B/1.5B (RMSNorm, RoPE, SwiGLU), Q8_0 → Q4 block-dequant GEMV. Chat CLI on the Switch;
coordinator dispatch ("generate" tasks over the cluster) so the Pi can drive it.

### M5 — stretch
Better tiling (shared-memory SGEMM toward 150+ GFLOPS → faster prefill), speculative decoding with
a 0.5B draft model, llama-style HTTP endpoint on the Switch for the whole homelab.

## Verification discipline (non-negotiable, inherited from switch-l4t-gpu)

Every kernel: CPU golden reference → GPU result → relative-error check (1e-4) → only then timed.
Every milestone: end-to-end logits compared against a CPU reference inference on identical
weights. No timing numbers are reported from unverified code.

## Hardware envelope (measured)

| Item | Value |
|---|---|
| GPU | GM20B, 1 SM cluster visible, 467.9 GFLOPS FMA peak (PTX), 92.8 GFLOPS SGEMM achieved |
| Bandwidth | 22.2 GB/s measured (87 % of 25.6 theoretical) |
| Memory | 3.5 GiB usable, shared with OS — model + KV cache budget ≤2.5 GB |
| Toolchain | `ncc` (clang-18 → PTX 6.3/sm_53, ~0.3 s), `gpu_workload` harness, gcc for hosts |

## Risks

- Clang-generated PTX may need massaging for some kernels (fp16 `half2` math is the first risky
  one — M3).
- Debugging PTX JIT failures is low-information; mitigate by keeping kernels tiny and golden-tested
  one at a time.
- 22 GB/s is a hard ceiling: big models (3B+) will be slow no matter what. Design targets 0.5–2B.
- Tokenizer work is deferred by design (Pi pre-tokenizes); real BPE comes only if M4 lands and
  standalone chat is wanted.
