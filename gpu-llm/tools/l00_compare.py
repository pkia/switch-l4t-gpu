#!/home/ev/qven/bin/python
"""l00_compare.py — L00 stage-by-stage: torch hooks vs independent numpy parse."""
import json
import struct
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D = "/home/ev/cluster-prototype/gpu-llm/weights/qwen"
tok = AutoTokenizer.from_pretrained(D)
model = AutoModelForCausalLM.from_pretrained(D, torch_dtype=torch.float32)
model.eval()

system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
tpl = ("<|im_start|>system\n" + system + "<|im_end|>\n"
       "<|im_start|>user\nThe capital of France is<|im_end|>\n"
       "<|im_start|>assistant\n")
ids = tok(tpl)["input_ids"]
Tn = len(ids)

def sig(t, name):
    v = t.detach().float().numpy()[-1]
    print(f"  torch {name}: first4={np.array2string(v[:4], precision=4)} max={np.abs(v).max():.4f}")

l0 = model.model.layers[0]
captured = {}
def mk(name):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        captured[name] = o.detach().float()
        sig(o, name)
    return hook
l0.input_layernorm.register_forward_hook(mk("ln1_out"))
l0.self_attn.q_proj.register_forward_hook(mk("q_proj"))
l0.self_attn.k_proj.register_forward_hook(mk("k_proj"))
l0.self_attn.v_proj.register_forward_hook(mk("v_proj"))
l0.self_attn.o_proj.register_forward_hook(mk("o_proj"))
l0.post_attention_layernorm.register_forward_hook(mk("ln2_out"))
l0.mlp.gate_proj.register_forward_hook(mk("gate"))
l0.mlp.up_proj.register_forward_hook(mk("up"))
l0.mlp.down_proj.register_forward_hook(mk("down"))

with torch.no_grad():
    position_ids = torch.arange(Tn).unsqueeze(0)
    x = model.model.embed_tokens(torch.tensor([ids]))
    pos_emb = model.model.rotary_emb(x, position_ids)
    x = l0(x, position_embeddings=pos_emb)[0].squeeze(0).float()
    sig(x, "L00 residual (torch)")

# ---- numpy independent parse
with open(D + "/model.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode())
ds = 8 + n
f2 = open(D + "/model.safetensors", "rb")
def T(name):
    m = hdr[name]; a, b = m["data_offsets"]
    f2.seek(ds + a)
    raw = f2.read(b - a)
    if m["dtype"] == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).reshape(m["shape"])
    return np.frombuffer(raw, dtype=np.float32).reshape(m["shape"])

C, NH, NKV, HD = 896, 14, 2, 64
wte = T("model.embed_tokens.weight")
d = np.arange(HD // 2)
freq = 1e6 ** (-2.0 * d / HD)
ang = np.outer(np.arange(Tn), freq)
cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

def npsig(v, name):
    print(f"  numpy {name}: first4={np.array2string(v[:4], precision=4)} max={np.abs(v).max():.4f}")

x = wte[np.array(ids)].astype(np.float32)
g = lambda i: T(f"model.layers.0.{i}").astype(np.float32)
h = x.copy()
ss = (h * h).mean(-1, keepdims=True)
h = h / np.sqrt(ss + np.float32(1e-6)) * g("input_layernorm.weight")
npsig(h[-1], "ln1_out")
q = h @ g("self_attn.q_proj.weight").T
k = h @ g("self_attn.k_proj.weight").T
v = h @ g("self_attn.v_proj.weight").T
npsig(q[-1], "q_proj")
npsig(k[-1], "k_proj")
npsig(v[-1], "v_proj")
q = q.reshape(Tn, NH, HD); k = k.reshape(Tn, NKV, HD); v = v.reshape(Tn, NKV, HD)
for t in range(Tn):
    a1 = q[t][:, :HD // 2]; a2 = q[t][:, HD // 2:]
    q[t] = np.concatenate([a1 * cos[t] - a2 * sin[t], a2 * cos[t] + a1 * sin[t]], axis=-1)
    b1 = k[t][:, :HD // 2]; b2 = k[t][:, HD // 2:]
    k[t] = np.concatenate([b1 * cos[t] - b2 * sin[t], b2 * cos[t] + b1 * sin[t]], axis=-1)
qexp = q[-1].reshape(-1)   # last-position roped q
kexp = k.reshape(-1)       # all roped k
npsig(q[-1], "q_roped_last")
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
ao = out.reshape(Tn, C) @ g("self_attn.o_proj.weight").T
npsig(ao[-1], "o_proj")
x = x + ao
h2 = x.copy()
ss = (h2 * h2).mean(-1, keepdims=True)
h2 = h2 / np.sqrt(ss + np.float32(1e-6)) * g("post_attention_layernorm.weight")
npsig(h2[-1], "ln2_out")
gu = h2 @ g("mlp.gate_proj.weight").T
up = h2 @ g("mlp.up_proj.weight").T
npsig(gu[-1], "gate")
npsig(up[-1], "up")
dn = (gu / (1 + np.exp(-gu))) * up @ g("mlp.down_proj.weight").T
npsig(dn[-1], "down")
x = x + dn
npsig(x[-1], "L00 residual (numpy)")
