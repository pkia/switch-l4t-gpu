# GPU-LLM STATUS — hand-written PTX inference engine for Tegra X1

**Milestone M2 COMPLETE (2026-10-28):** GPT-2 124M FP32 runs end-to-end on the Switch's
GPU via hand-written CUDA kernels → `ncc` (clang-18 + CUDA 10 headers) → PTX → driver-API
JIT. Verified against an independent CPU reference *and* a numpy ground-truth forward.
**36.5 tok/s** greedy decode = 83 % of the 44 tok/s bandwidth ceiling (PLAN.md).

## Results (verified)
| Check | Result |
|---|---|
| Selftest (GPU vs serial-CPU, 8 pos, garbage tokens) | argmax MATCH, max logit err 0.83 |
| Selftest (GPU vs CPU, 5 pos, "The capital of France is") | argmax MATCH (262 " the"), err 3.7 |
| numpy ground truth (full safetensors forward, fp32) | top-1 identical (262), top-5 same set |
| Greedy generation | coherent English (see examples below) |
| Throughput | 36.5 tok/s (27.7 ms/tok); prefill ≈ 30 ms/tok |

Example outputs (greedy, fp32):
- "The capital of France is" → " the capital of the French Republic, and the capital of the Republic of France"
- "In a shocking finding, scientists discovered a herd of unicorns" → " that were found in a forest in the northern Indian state of Uttar Pradesh.\n\nThe researchers found that"

## The four bugs this milestone killed (and how each was found)
1. **Divergent `__syncthreads` in `gemv_warp`** — grid-stride loop trip counts differed per
   warp when rows % 8 ≠ 0 (lm_head: 50257 rows). UB → GPU hang → Wi-Fi wedge. Found by
   kernel bisection harness (`harness/gpt2_kernel_test.c`); fixed with uniform loop passes,
   `if (n < N)` guards, barriers outside the guard.
2. **proj_w upload truncation** — `dev_copy(..., C*C)` for a `[768][3072]` tensor
   (= `FC*C`). Quarter-sized upload → OOB reads. Found by maskable-bug analysis:
   random weights hide it (all tensors share statistics), real weights explode.
3. **`L_QKVB = L_QKVW + 3*C` (missing `*C`)** — every later layer offset cascaded wrong;
   the bias pointer landed inside the qkv weight matrix. Found by per-stage CPU prints vs
   a numpy replica of the same algorithm (embed+ln1 exact, qkv wrong).
4. **`cuMemcpyDtoD_v2` float-vs-byte offsets** — device pointer arithmetic is BYTES;
   `d_qkv + 2*C` copied from element 384 instead of 1536 (and `pos*C` would scatter KV
   slots). Found by GPU stage dumps: probs = 1.0 exactly at pos 0, yet attnout ≠ v.

Debug methodology that worked (keep for M3/M4): numpy referee from safetensors →
per-layer residual trace → per-stage vector prints (`--debug`) → first divergent stage
is the buggy op. Never trust a selftest that compares two implementations you wrote —
add an independent reference (numpy reads the ORIGINAL safetensors, not the flat bin).

## Files
- `kernels/gpt2_kernels.cu` — 8 kernels: gemv_warp (float4, uniform barriers), layernorm,
  gelu (tanh approx), add_bias, add2, attn_scores, attn_softmax, attn_mix. Self-contained
  device math (`dev_expf` poly, `dev_tanhf`, `__builtin_sqrtf`) because ncc links no libdevice.
- `engine/gpt2_engine.c` — driver-API engine (cuModuleLoadData + cuLaunchKernel), FP32 KV
  cache, greedy decode, `--selftest N` (GPU vs CPU logits), `--debug` (stage dumps).
- `tools/conv_gpt2.py` — HF safetensors → flat fp32 bin (124,439,808 floats; transposes
  Conv1D [in,out] → [out,in] row-major). Byte-exact vs safetensors (numpy-verified).
- `tools/gpt2_bpe.py` — pure-Python GPT-2 byte-level BPE (encode/decode).
- `tools/make_random_bin.py` — deterministic random weights for pipeline tests.
- `harness/gpt2_kernel_test.c` — per-kernel bisection harness.
- `weights/` on Pi: model.safetensors (548 MB), vocab.json, merges.txt; `gpt2_124m.bin`
  (474.7 MB) generated + shipped to Switch `~/gpu-llm/`.

## How to run
On Pi (convert + tokenize):
```
python3 tools/conv_gpt2.py weights/model.safetensors gpt2_124m.bin
python3 tools/gpt2_bpe.py encode "The capital of France is" prompt.ids
```
Ship via `ssh ev1@192.168.0.25 'cat > ~/gpu-llm/…' < local` (scp SFTP refused).
On Switch (`~/gpu-llm/engine/`):
```
gcc -O1 -g -o gpt2_engine gpt2_engine.c -lcuda -lm
cd ~/gpu-llm/engine && ./gpt2_engine --weights ~/gpu-llm/gpt2_124m.bin \
    --ptx ~/gpu-llm/kernels/gpt2.ptx --ids prompt.ids --gen 14 [--selftest 8] [--debug]
python3 (Pi) tools/gpt2_bpe.py decode out.ids
```
Compile PTX: `cd ~/switch-l4t-gpu && ./ncc ~/gpu-llm/kernels/gpt2_kernels.cu ~/gpu-llm/kernels/gpt2.ptx`

## Next
- **M3 FP16** weights → 2× bandwidth → ~70 tok/s ceiling; needs fp16 gemv kernels + conversion.
- **M4 Qwen2.5** — RMSNorm + RoPE + SwiGLU + untied embeddings; Q8/Q4 quantization after that.
- Engine housekeeping: gtrace prints still compiled in (behind `--debug`); consider mmap for
  the weights file to halve host RAM during load.

## Optimization round (2026-10-28 evening) — done
| Item | Outcome |
|---|---|
| **fast-prefill** (default) | Prompt tokens skip the final LN + 154 MB lm_head read; only the last prompt position computes logits. CPU reference skips too (3.8 s → 2.8 s for x8). 12-token prefill: ~339 ms measured (~28 ms/tok, ≈25 % under the always-logits cost). |
| **`--fused` gemv+bias** (optional flag) | 4th kernel `gemv_warp_b`; selftest PASS both modes, max err identical (0.8287); 60-token outputs byte-identical. Decode 35.8 → 34.5 tok/s = run noise: launches are async and decode is bandwidth-bound, so fusion is neutral here. Kept as an option (one launch instead of two per matmul; matters more at short contexts / future kernels). |
| **Console clocks** (opt-in script) | `tools/gpu-console.sh` — run yourself with sudo (`on` = nvpmodel mode 0, 921.6 MHz GPU ceiling; `off` = handheld). Decode is memory-bound so expect small decode gains; pays off for SM-bound M3/M4 kernels. Agent can't run sudo (policy). |
| **Batched-GEMM prefill** | **IMPLEMENTED as `--batch-prefill` (optional flag)** after this analysis: a naive multi-token kernel re-reads weights per token (no reuse), so the win needs weight-stationary batching. `gemv_multi` kernel: grid (ceil(N/8), ceil(T/16)), each warp streams one W row against a 16-token chunk held in register accumulators, warp-local smem reduction (no `__shfl_*` — unavailable under `-nocudainc`). Correct: selftest PASS (identical max err 0.8287 vs CPU). Measured prefill: **T=64: 1349→818 ms (1.65×), T=128: 2629→1661 ms (1.58×)**. Real tiled-GEMM (shared-memory blocking, cf. the 92-GFLOP SGEMM) is the path to ~5-10× and belongs with M3/M4. |
| Kernel-iteration lessons (measured, keep) | 1) Guarded 16-token unroll: 30 ms; UNguarded full unroll: 59 ms — 16 register accumulators + 16 in-flight float4 temps exceed the register budget and spill; guards were NOT the bottleneck. 2) `__shfl_down_sync` does not exist under `-nocudainc`; warp reduce with `volatile` smem + lockstep (sm_53 Pascal) instead. 3) Per-token launch loops (attention/elementwise ≈ 5000 launches at T=128) cost ~hundreds of ms — batching attention kernels is the next-biggest prefill win after weight reuse. |

Verification: selftest PASS in both modes; unfused/fused greedy streams byte-identical (73 ids,
"unicorns" prompt); "France" prompt output unchanged. Decode ~36 tok/s either way.

## GPU utilization — root-verified (2026-10-28)
tegrastats needs root on switchroot (reads /sys/kernel/debug); passwordless sudo for ev1 is
now configured (`/etc/sudoers.d/90-ev1-nopasswd`, user-authorized). During a 120-token
generation (34 tok/s, batched+fused):
- **GR3D (GPU): 87-99 % utilization at 460 MHz**, ~94 % avg across the active window
  (ramps 76 → 153 → 307 → 460 MHz in the first ~2 s, then pinned).
- EMC: 1600 MHz (full), duty-cycle 1-53 % — bursty GEMV reads, CPU mostly idle.
- devfreq `load` attribute does not exist in this kernel build (root does not add it);
  `cur_freq` and tegrastats are the working telemetry.
Interpretation: the engine saturates the GPU; the handheld 460 MHz clock cap is the only
remaining limiter (see `tools/gpu-console.sh` to lift it, opt-in).

## M4 — Qwen2.5-0.5B-Instruct COMPLETE (2026-10-28 late)
New Llama-family engine: `kernels/qwen_kernels.cu` (8 kernels) + `engine/qwen_engine.c`.
FP16 weights (manual fp16→fp32 decode, no cuda_fp16.h under -nocudainc), FP32 activations/KV.
RMSNorm (eps 1e-6) | RoPE (theta 1e6, precomputed cos/sin tables in the bin) | GQA
(NH=14, NKV=2, HD=64; kv head = q_head/7) | SwiGLU (FC=4864) | tied embeddings |
**Q/K/V ATTENTION BIASES** (this checkpoint has them — see bugs) | ChatML prompt format |
Qwen sampling (temp 0.7, top_k 20, top_p 0.8, rep_penalty 1.1, --greedy opt-out), EOT
<|endoftext|> 151643 and <|im_end|> 151645.

Results (verified): selftest GPU-vs-CPU PASS, max err **0.0018**, argmax MATCH; engine
top-5 == numpy-referee top-5 == HF-torch top-5 on identical prompts; HF generation agrees
("Paris." / "The capital of France is Paris."). **8.9 tok/s** decode (fp16, 24 layers);
34-token ChatML prefill ~2.2 s sequential (batched prefill/port of gemv_multi = future work).

Bugs found on Qwen (all via stage-trace vs HF ground truth):
1. **Missing Q/K/V attention biases** — this checkpoint carries q/k/v biases (72 extra
   tensors the first converter skipped); torch k-proj ≈ ±9 vs mine ±0.3 exposed it.
   Added qkv_b [QKVD] per layer to layout+engine+CPU ref.
2. **fp16 bias misread as fp32** — uploaded fp16 bias but kernel/CPU read float*; decode
   fp16→fp32 on host and upload fp32.
3. **dev_expf poly ~30 % wrong** at f→1 (coefficients were bogus); replaced with 4-term
   Taylor of e^(f·ln2) (max err ~0.08 %). GPT-2 masked it inside softmax; SwiGLU did not.
4. **gateup GEMV N=cFC instead of c2FC** — up half never computed (GPT-2 section bug, hit
   Qwen too; fixed everywhere).

Tools: `tools/conv_qwen.py` (safetensors bf16 → flat fp16 bin, 988,589,824 B, byte-exact),
`tools/referee_qwen.py` + `tools/l00_precise.py` + `tools/referee_test.py` (numpy/HF
bisection harnesses), `tools/hf_check.py` + venv `/home/ev/qven` (torch CPU +
transformers 5.21 for ground truth), tokenizer supports Qwen regex + special tokens +
ChatML via TKN_* env vars (see tools/gpt2_bpe.py header).

Perf note: 0.5B decode floor = 943 MB ÷ 20 GB/s ≈ 47 ms → ~21 tok/s ceiling; we're at
9 tok/s (43 %) — the gap is GQA attention kernels + per-layer launch overhead, both
optimizable. Prefill is sequential (66 ms/tok); port gemv_multi for long prompts.

