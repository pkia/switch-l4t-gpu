#!/usr/bin/env python3
"""conv_qwen.py — HF Qwen2.5-0.5B-Instruct safetensors (bf16) -> engine flat binary (fp16).

Engine layout (all fp16 unless noted; documented in qwen_engine.c):
  OFF_EMB   model.embed_tokens [V, C]              (fp16)
  OFF_COS   RoPE cos table [CTX, HD/2]             (fp32)
  OFF_SIN   RoPE sin table [CTX, HD/2]             (fp32)
  layer i (fixed stride, all fp16):
    in_ln_w [C] | qkv_w [C+2KV, C] (q|k|v rows) | o_w [C, C] | post_ln_w [C]
    gateup_w [2FC, C] (gate rows then up rows) | down_w [C, FC]
  OFF_LNFW  model.norm [C]
Qwen2 Linear = nn.Linear -> weights stored [out, in] = engine row-major gemv layout, NO transpose.
Usage: python3 conv_qwen.py [model.safetensors] [qwen_05b.bin]"""
import json
import struct
import sys

import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "weights/qwen/model.safetensors"
out = sys.argv[2] if len(sys.argv) > 2 else "qwen_05b.bin"

V, C, L, NH, NKV, HD, FC, CTX = 151936, 896, 24, 14, 2, 64, 4864, 2048
KV = NKV * HD                       # 128
QKV = C + 2 * KV                    # 1152
GATEUP = 2 * FC                     # 9728
ROPE_THETA = 1_000_000.0

with open(src, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode("utf-8"))
data_start = 8 + n
f = open(src, "rb")

def tensor(name) -> np.ndarray:
    m = hdr[name]
    a, b = m["data_offsets"]
    f.seek(data_start + a)
    raw = f.read(b - a)
    if m["dtype"] == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).reshape(m["shape"])
    if m["dtype"] == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(m["shape"])
    if m["dtype"] == "F16":
        return np.frombuffer(raw, dtype=np.float16).reshape(m["shape"]).astype(np.float32)
    raise ValueError(f"unsupported dtype {m['dtype']} for {name}")

# sanity-check dims against the actual checkpoint
assert hdr["model.embed_tokens.weight"]["shape"] == [V, C], hdr["model.embed_tokens.weight"]["shape"]
assert hdr["model.layers.0.self_attn.q_proj.weight"]["shape"] == [C, C]
assert hdr["model.layers.0.self_attn.k_proj.weight"]["shape"] == [KV, C]
assert hdr["model.layers.0.self_attn.v_proj.weight"]["shape"] == [KV, C]
assert hdr["model.layers.0.mlp.gate_proj.weight"]["shape"] == [FC, C]
assert hdr["model.layers.0.mlp.down_proj.weight"]["shape"] == [C, FC]
assert f"model.layers.{L - 1}.input_layernorm.weight" in hdr, "expected 24 layers"
def fp16(a) -> bytes:
    return np.ascontiguousarray(a, dtype=np.float16).tobytes()

def fp32(a) -> bytes:
    return np.ascontiguousarray(a, dtype=np.float32).tobytes()

half = 2
EMB_B = V * C * half
COS_B = CTX * (HD // 2) * 4
# layout (per layer, fp16): in_ln | qkv_w | qkv_b | o_w | post_ln | gateup_w | down_w
# (Qwen2.5-0.5B-Instruct HAS q/k/v attention biases; o_proj and MLP have none)
LSTRIDE_B = (C + QKV * C + QKV + C * C + C + GATEUP * C + C * FC) * half
total = EMB_B + 2 * COS_B + L * LSTRIDE_B + C * half

qkv = np.empty((QKV, C), dtype=np.float32)
gateup = np.empty((GATEUP, C), dtype=np.float32)
with open(out, "wb") as o:
    o.write(fp16(tensor("model.embed_tokens.weight")))
    # RoPE tables: freq_d = theta^(-2d/HD), angle = pos * freq_d (d in 0..HD/2-1)
    d = np.arange(HD // 2, dtype=np.float64)
    freq = ROPE_THETA ** (-2.0 * d / HD)
    ang = np.outer(np.arange(CTX, dtype=np.float64), freq)
    o.write(fp32(np.cos(ang)))
    o.write(fp32(np.sin(ang)))
    for i in range(L):
        p = f"model.layers.{i}."
        qkv[:C] = tensor(p + "self_attn.q_proj.weight")
        qkv[C:C + KV] = tensor(p + "self_attn.k_proj.weight")
        qkv[C + KV:] = tensor(p + "self_attn.v_proj.weight")
        gateup[:FC] = tensor(p + "mlp.gate_proj.weight")
        gateup[FC:] = tensor(p + "mlp.up_proj.weight")
        o.write(fp16(tensor(p + "input_layernorm.weight")))
        o.write(fp16(qkv))
        o.write(fp16(np.concatenate([
            tensor(p + "self_attn.q_proj.bias"),
            tensor(p + "self_attn.k_proj.bias"),
            tensor(p + "self_attn.v_proj.bias")])))
        o.write(fp16(tensor(p + "self_attn.o_proj.weight")))
        o.write(fp16(tensor(p + "post_attention_layernorm.weight")))
        o.write(fp16(gateup))
        o.write(fp16(tensor(p + "mlp.down_proj.weight")))
        print(f"layer {i:02d} done")
    o.write(fp16(tensor("model.norm.weight")))
f.close()

print(f"\n wrote {out}: {total} bytes ({total/1048576:.1f} MiB) expected {total} -> OK")
sys.exit(0)
