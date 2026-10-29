#!/usr/bin/env python3
"""referee_test.py — coherence probes for the true Qwen2.5-0.5B (numpy, from safetensors)."""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, "tools")
os.environ["TKN_VOCAB"] = "weights/qwen/vocab.json"
os.environ["TKN_MERGES"] = "weights/qwen/merges.txt"
os.environ["TKN_REGEX"] = open("weights/qwen/pretok.regex").read()
os.environ["TKN_SPECIALS"] = '{"<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 151643}'
from gpt2_bpe import Encoder

enc = Encoder()
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
wte = T("model.embed_tokens.weight")
lnfw = T("model.norm.weight")

def forward(ids):
    Tn = len(ids)
    x = wte[np.array(ids)].astype(np.float32)
    d = np.arange(HD // 2)
    freq = 1e6 ** (-2.0 * d / HD)
    ang = np.outer(np.arange(Tn), freq)
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    for l in range(24):
        g = lambda i: T(f"model.layers.{l}.{i}").astype(np.float32)
        h = x.copy()
        ss = (h * h).mean(-1, keepdims=True)
        h = h / np.sqrt(ss + np.float32(1e-6)) * g("input_layernorm.weight")
        q = (h @ g("self_attn.q_proj.weight").T).reshape(Tn, NH, HD)
        k = (h @ g("self_attn.k_proj.weight").T).reshape(Tn, NKV, HD)
        v = (h @ g("self_attn.v_proj.weight").T).reshape(Tn, NKV, HD)
        for t in range(Tn):
            a1 = q[t][:, :HD // 2]; a2 = q[t][:, HD // 2:]
            q[t] = np.concatenate([a1 * cos[t] - a2 * sin[t], a2 * cos[t] + a1 * sin[t]], axis=-1)
            b1 = k[t][:, :HD // 2]; b2 = k[t][:, HD // 2:]
            k[t] = np.concatenate([b1 * cos[t] - b2 * sin[t], b2 * cos[t] + b1 * sin[t]], axis=-1)
        k_exp = np.repeat(k, NH // NKV, axis=1)
        v_exp = np.repeat(v, NH // NKV, axis=1)
        att = np.einsum("thd,shd->hts", q, k_exp).astype(np.float32) * np.float32(0.125)
        mask = np.triu(np.full((Tn, Tn), -1e10, dtype=np.float32), 1)
        att = att + mask[None]
        att = att - att.max(-1, keepdims=True)
        w = np.exp(att); w = w / w.sum(-1, keepdims=True)
        out = np.zeros((Tn, NH, HD), dtype=np.float32)
        for hh in range(NH):
            out[:, hh] = np.einsum("ts,sd->td", w[hh], v_exp[:, hh])
        x = x + out.reshape(Tn, C) @ g("self_attn.o_proj.weight").T
        h2 = x.copy()
        ss = (h2 * h2).mean(-1, keepdims=True)
        h2 = h2 / np.sqrt(ss + np.float32(1e-6)) * g("post_attention_layernorm.weight")
        gu = h2 @ g("mlp.gate_proj.weight").T
        up = h2 @ g("mlp.up_proj.weight").T
        x = x + (gu / (1 + np.exp(-gu))) * up @ g("mlp.down_proj.weight").T
    xf = x[-1]
    ss = (xf * xf).mean()
    xf = xf / np.sqrt(ss + np.float32(1e-6)) * lnfw
    return xf @ wte.T

math_ids = [int(x) for x in open("qmath.ids", "rb").read()[4:].split(b"\x00")[:0]] or None
def read_ids(path):
    with open(path, "rb") as g:
        cnt = struct.unpack("<I", g.read(4))[0]
        return [struct.unpack("<I", g.read(4))[0] for _ in range(cnt)]

for name, path in [("2+2 (chat)", "qmath.ids"), ("12345 (chat)", "qseq.ids"),
                   ("France (raw)", "qprompt.ids")]:
    ids = read_ids(path)
    logits = forward(ids)
    top3 = np.argsort(logits)[::-1][:3]
    dec = [(int(i), repr(enc.decode([int(i)])), round(float(logits[i]), 2)) for i in top3]
    print(f"{name}: {dec}")
