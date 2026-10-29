#!/usr/bin/env python3
"""conv_gpt2.py — convert HF GPT-2 124M safetensors to the engine's flat fp32 layout.
Transposes Conv1D weights (HF stores [in,out]; the engine's gemv wants [out,in] row-major).
Usage: python3 conv_gpt2.py [model.safetensors] [gpt2_124m.bin]"""
import json
import struct
import sys

import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "weights/model.safetensors"
out = sys.argv[2] if len(sys.argv) > 2 else "gpt2_124m.bin"

with open(src, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode("utf-8"))
data_start = 8 + n
f = open(src, "rb")

def tensor(name) -> np.ndarray:
    m = hdr[name]
    assert m["dtype"] == "F32", (name, m["dtype"])
    a, b = m["data_offsets"]
    f.seek(data_start + a)
    raw = f.read(b - a)
    return np.frombuffer(raw, dtype=np.float32).reshape(m["shape"])

def rowmajor(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)

L = 12
names = {}
for i in range(L):
    names[f"ln1_w_{i}"] = (f"h.{i}.ln_1.weight", False)
    names[f"ln1_b_{i}"] = (f"h.{i}.ln_1.bias", False)
    names[f"qkv_w_{i}"] = (f"h.{i}.attn.c_attn.weight", True)   # [768,2304] -> [2304,768]
    names[f"qkv_b_{i}"] = (f"h.{i}.attn.c_attn.bias", False)
    names[f"ao_w_{i}"]  = (f"h.{i}.attn.c_proj.weight", True)   # [768,768] -> T
    names[f"ao_b_{i}"]  = (f"h.{i}.attn.c_proj.bias", False)
    names[f"ln2_w_{i}"] = (f"h.{i}.ln_2.weight", False)
    names[f"ln2_b_{i}"] = (f"h.{i}.ln_2.bias", False)
    names[f"fc_w_{i}"]  = (f"h.{i}.mlp.c_fc.weight", True)      # [768,3072] -> [3072,768]
    names[f"fc_b_{i}"]  = (f"h.{i}.mlp.c_fc.bias", False)
    names[f"proj_w_{i}"] = (f"h.{i}.mlp.c_proj.weight", True)   # [3072,768] -> [768,3072]
    names[f"proj_b_{i}"] = (f"h.{i}.mlp.c_proj.bias", False)

order = ["wte.weight", "wpe.weight"]
for i in range(L):
    for k in ["ln1_w_", "ln1_b_", "qkv_w_", "qkv_b_", "ao_w_", "ao_b_", "ln2_w_", "ln2_b_",
              "fc_w_", "fc_b_", "proj_w_", "proj_b_"]:
        order.append(f"{k}{i}")
order += ["ln_f.weight", "ln_f.bias"]

total = 0
with open(out, "wb") as o:
    for key in order:
        hf_name, do_T = names.get(key, (key, False))
        a = tensor(hf_name)
        if do_T:
            a = a.T
        a = rowmajor(a)
        a.tofile(o)
        total += a.size
        print(f"{key:>12}: {a.shape}  std={a.std():.4f}")
    f.close()

expect = 124439808
print(f"\n wrote {out}: {total} floats ({total*4/1048576:.1f} MiB), expected {expect} -> {'OK' if total == expect else 'MISMATCH'}")
sys.exit(0 if total == expect else 1)
