#!/usr/bin/env python3
"""referee_qwen.py — ground-truth Qwen2.5-0.5B forward straight from safetensors (bf16->fp32).
Prints TRUE top-5 next tokens for "The capital of France is" and compares with the engine's."""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, "tools")
os.environ["TKN_VOCAB"] = "weights/qwen/vocab.json"
os.environ["TKN_MERGES"] = "weights/qwen/merges.txt"
os.environ["TKN_REGEX"] = open("weights/qwen/pretok.regex").read()
from gpt2_bpe import Encoder

with open("weights/qwen/model.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode())
ds = 8 + n
f = open("weights/qwen/model.safetensors", "rb")
def T(name):
    m = hdr[name]; a, b = m["data_offsets"]
    f.seek(ds + a)
    raw = f.read(b - a)
    if m["dtype"] == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).reshape(m["shape"])
    return np.frombuffer(raw, dtype=np.float32).reshape(m["shape"])

C, NH, NKV, HD = 896, 14, 2, 64
if len(sys.argv) > 1 and sys.argv[1].endswith(".ids"):
    with open(sys.argv[1], "rb") as g:
        Tn = struct.unpack("<I", g.read(4))[0]
        ids = [struct.unpack("<I", g.read(4))[0] for _ in range(Tn)]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "qout2.ids"
else:
    ids = [785, 6722, 315, 9625, 374]
    out_file = "qout1.ids"
print(f"prompt tokens: {Tn}")
Tn = len(ids)
wte = T("model.embed_tokens.weight")
lnfw = T("model.norm.weight")
x = wte[ids].astype(np.float32)

d = np.arange(HD // 2)
freq = 1e6 ** (-2.0 * d / HD)
ang = np.outer(np.arange(Tn), freq)
cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

def rope_rows(a, cs, sn):   # a: [H, HD] half-split rotation
    a1 = a[:, :HD // 2]; a2 = a[:, HD // 2:]
    return np.concatenate([a1 * cs - a2 * sn, a2 * cs + a1 * sn], axis=-1)

for l in range(24):
    g = lambda i: T(f"model.layers.{l}.{i}").astype(np.float32)
    h = x.copy()
    ss = (h * h).mean(-1, keepdims=True)
    h = h / np.sqrt(ss + np.float32(1e-6)) * g("input_layernorm.weight")
    q = (h @ g("self_attn.q_proj.weight").T).reshape(Tn, NH, HD)
    k = (h @ g("self_attn.k_proj.weight").T).reshape(Tn, NKV, HD)
    print("numpy ln1[5][:4]:", h[5][:4].tolist())
    print("numpy kproj[5][:4]:", (h @ g("self_attn.k_proj.weight").T)[5][:4].tolist())
    print("numpy kproj[33][:4]:", (h @ g("self_attn.k_proj.weight").T)[LAST][:4].tolist())
    v = (h @ g("self_attn.v_proj.weight").T).reshape(Tn, NKV, HD)
    for t in range(Tn):
        q[t] = rope_rows(q[t], cos[t], sin[t])
        k[t] = rope_rows(k[t], cos[t], sin[t])
    k_exp = np.repeat(k, NH // NKV, axis=1)   # [T, NH, HD]; kv head gg -> q heads 7gg..7gg+6
    v_exp = np.repeat(v, NH // NKV, axis=1)
    att = np.einsum("thd,shd->hts", q, k_exp).astype(np.float32) * np.float32(0.125)
    if l == 0:
        print("shapes: att", att.shape, "q", q.shape, "v_exp", v_exp.shape)
    mask = np.triu(np.full((Tn, Tn), -1e10, dtype=np.float32), 1)
    att = att + mask[None, :, :]              # broadcast over heads: att is [NH, T, T]
    att = att - att.max(-1, keepdims=True)
    w = np.exp(att); w = w / w.sum(-1, keepdims=True)   # [NH, T(t), T(s)]
    out = np.zeros((Tn, NH, HD), dtype=np.float32)
    for hh in range(NH):
        out[:, hh] = np.einsum("ts,sd->td", w[hh], v_exp[:, hh])
    ao = out.reshape(Tn, C) @ g("self_attn.o_proj.weight").T
    x = x + ao
    h2 = x.copy()
    ss = (h2 * h2).mean(-1, keepdims=True)
    h2 = h2 / np.sqrt(ss + np.float32(1e-6)) * g("post_attention_layernorm.weight")
    gu = h2 @ g("mlp.gate_proj.weight").T
    up = h2 @ g("mlp.up_proj.weight").T
    silu = gu / (1 + np.exp(-gu))
    x = x + (silu * up) @ g("mlp.down_proj.weight").T
    vv = x[-1]
    print(f"L{l:02d} numpy last-pos: first4={np.array2string(vv[:4], precision=4)} max|.|={abs(vv).max():.4f}")

xf = x[-1]
ss = (xf * xf).mean()
xf = xf / np.sqrt(ss + np.float32(1e-6)) * lnfw
logits = xf @ wte.T
top5 = np.argsort(logits)[::-1][:5]
enc = Encoder()
print("TRUE top-5:", top5.tolist())
print("decoded:", [(int(i), repr(enc.decode([int(i)]))) for i in top5])
with open(out_file, "rb") as f:
    cnt = struct.unpack("<I", f.read(4))[0]
    eng_ids = [struct.unpack("<I", f.read(4))[0] for _ in range(cnt)]
print("engine first generated token:", eng_ids[Tn], repr(enc.decode([eng_ids[Tn]])))
print("engine top-5 (from run log): see q*.log")
