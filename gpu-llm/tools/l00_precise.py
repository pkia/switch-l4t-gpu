#!/home/ev/qven/bin/python
"""l00_precise.py — torch HF-submodule L00 forward vs numpy, token-33 stage prints."""
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
LAST = Tn - 1
C, NH, NKV, HD = 896, 14, 2, 64

l0 = model.model.layers[0]
cap = {}
def capk(name, mod):
    def hook(mod, inp, out):
        cap[name] = out.detach().float()
    return hook
sa = l0.self_attn
sa.k_proj.register_forward_hook(capk("kproj", sa.k_proj))
sa.q_proj.register_forward_hook(capk("qproj", sa.q_proj))
l0.input_layernorm.register_forward_hook(capk("ln1", l0.input_layernorm))

with torch.no_grad():
    x0 = model.model.embed_tokens(torch.tensor([ids]))
    position_ids = torch.arange(Tn).unsqueeze(0)
    cos, sin = model.model.rotary_emb(x0, position_ids)
    cos, sin = cos[0], sin[0]                      # [T, 32]
    print("torch cos[33][:4]:", cos[LAST][:4].tolist())
    print("torch sin[33][:4]:", sin[LAST][:4].tolist())
    print("torch cos[5][:4] :", cos[5][:4].tolist())
    h = l0.input_layernorm(x0)                     # [1, T, C]
    q = sa.q_proj(h).view(1, Tn, NH, HD).transpose(1, 2)   # [1, NH, T, HD]
    k = sa.k_proj(h).view(1, Tn, NKV, HD).transpose(1, 2)
    v = sa.v_proj(h).view(1, Tn, NKV, HD).transpose(1, 2)
    def rot(x):
        x1 = x[..., :HD // 2]; x2 = x[..., HD // 2:]
        return torch.cat([-x2, x1], dim=-1)
    cf = cos.unsqueeze(0).unsqueeze(0)             # [1, 1, T, 32]
    sf = sin.unsqueeze(0).unsqueeze(0)
    qr = q * cf + rot(q) * sf
    kr = k * cf + rot(k) * sf
    q33 = qr[0, :, LAST, :].float()
    print("torch q_roped[33, head0][:4]:", q33[0][:4].tolist())
    print("torch q_roped[33, head1][:4]:", q33[1][:4].tolist())
    k5 = kr[0, :, 5, :].float()
    print("torch k_roped[5, head0][:4]:", k5[0][:4].tolist())
    print("torch k_roped[33, head0][:4]:", kr[0, :, LAST, :][0][:4].tolist())
    print("torch kproj[5][:4]:", cap["kproj"][0, 5, :4].tolist())
    print("torch kproj[33][:4]:", cap["kproj"][0, LAST, :4].tolist())
    print("torch ln1[5][:4]:", cap["ln1"][0, 5, :4].tolist())
    krep = kr[0].repeat_interleave(NH // NKV, dim=0)      # [NH, T, HD]
    vrep = v[0].repeat_interleave(NH // NKV, dim=0)
    scores = torch.einsum("ad,asd->as", qr[0, :, LAST, :], krep) * 0.125
    print("torch scores[head0][:6]:", scores[0][:6].tolist())
    probs = torch.softmax(scores, dim=-1)
    print("torch probs[head0][:6]:", probs[0][:6].tolist())
    attn_out = torch.einsum("as,asd->ad", probs, vrep)     # [NH, HD]
    ao = sa.o_proj(attn_out.reshape(1, NH * HD)).float()
    print("torch o_proj last4:", ao[0, -4:].tolist())
    x0 = x0 + ao
    h2 = l0.post_attention_layernorm(x0)
    gu = l0.mlp.gate_proj(h2)
    up = l0.mlp.up_proj(h2)
    dn = l0.mlp.down_proj(gu / (1 + torch.exp(-gu)) * up)
    x0 = x0 + dn
    print("torch residual last4:", x0[0, LAST, -4:].tolist())
    print("torch residual max:", x0[0, LAST].abs().max().item())

# ================= numpy =================
print("---- numpy ----")
with open(D + "/model.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode())
ds = 8 + n
f = open(D + "/model.safetensors", "rb")
def T(name):
    m = hdr[name]; a, b = m["data_offsets"]
    f.seek(ds + a)
    raw = f.read(b - a)
    if m["dtype"] == "BF16":
        bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).reshape(m["shape"])
    return np.frombuffer(raw, dtype=np.float32).reshape(m["shape"])

wte = T("model.embed_tokens.weight")
d = np.arange(HD // 2)
freq = 1e6 ** (-2.0 * d / HD)
ang = np.outer(np.arange(Tn), freq)
cosn, sinn = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
print("numpy cos[33][:4]:", cosn[LAST][:4].tolist())
print("numpy sin[33][:4]:", sinn[LAST][:4].tolist())
print("numpy cos[5][:4] :", cosn[5][:4].tolist())
x = wte[np.array(ids)].astype(np.float32)
g = lambda i: T(f"model.layers.0.{i}").astype(np.float32)
h = x.copy()
ss = (h * h).mean(-1, keepdims=True)
h = h / np.sqrt(ss + np.float32(1e-6)) * g("input_layernorm.weight")
q = (h @ g("self_attn.q_proj.weight").T).reshape(Tn, NH, HD)
k = (h @ g("self_attn.k_proj.weight").T).reshape(Tn, NKV, HD)
v = (h @ g("self_attn.v_proj.weight").T).reshape(Tn, NKV, HD)
def np_rot(a):   # [T, H, HD]
    a1 = a[..., :HD // 2]; a2 = a[..., HD // 2:]
    return np.concatenate([-a2, a1], axis=-1)
def np_apply(a):  # a [T, H, HD], cos/sin [T, 32] — half-split rotation
    a1 = a[..., :HD // 2]; a2 = a[..., HD // 2:]
    return np.concatenate([a1 * cosn[:, None, :] - a2 * sinn[:, None, :],
                           a2 * cosn[:, None, :] + a1 * sinn[:, None, :]], axis=-1)
qr = np_apply(q)
kr = np_apply(k)
print("numpy q_roped[33, head0][:4]:", qr[LAST, 0][:4].tolist())
print("numpy q_roped[33, head1][:4]:", qr[LAST, 1][:4].tolist())
print("numpy k_roped[5, head0][:4]:", kr[5, 0][:4].tolist())
print("numpy k_roped[33, head0][:4]:", kr[LAST, 0][:4].tolist())
kexp = np.repeat(kr, NH // NKV, axis=1)     # [T, NH, HD]
vexp = np.repeat(v, NH // NKV, axis=1)
scores = np.einsum("hd,shd->hs", qr[LAST], kexp).astype(np.float32) * np.float32(0.125)
print("numpy scores[head0][:6]:", scores[0][:6].tolist())
probs = np.exp(scores - scores.max(-1, keepdims=True))
probs = probs / probs.sum(-1, keepdims=True)
print("numpy probs[head0][:6]:", probs[0][:6].tolist())
attn_out = np.einsum("hs,shd->hd", probs, vexp)
ao = attn_out.reshape(1, NH * HD) @ g("self_attn.o_proj.weight").T
print("numpy o_proj last4:", ao[0, -4:].tolist())
x = x + ao[0]
h2 = x.copy()
ss = (h2 * h2).mean(-1, keepdims=True)
h2 = h2 / np.sqrt(ss + np.float32(1e-6)) * g("post_attention_layernorm.weight")
gu = h2 @ g("mlp.gate_proj.weight").T
up = h2 @ g("mlp.up_proj.weight").T
dn = (gu / (1 + np.exp(-gu))) * up @ g("mlp.down_proj.weight").T
x = x + dn[0]
print("numpy residual last4:", x[LAST, -4:].tolist())
print("numpy residual max:", np.abs(x[LAST]).max())
