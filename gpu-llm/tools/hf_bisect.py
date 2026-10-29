#!/home/ev/qven/bin/python
"""hf_bisect.py — per-layer hidden states from HF torch for idsB (my template prompt)."""
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
print("ids:", ids)
x = model.model.embed_tokens(torch.tensor([ids]))[0].detach().float()
print("embed:", x[-1][:4].tolist(), "max", x.abs().max().item())

def sig(t, name):
    v = t.detach().float().numpy()
    print(f"{name}: first4={v[-1][:4].tolist()} max|.|={abs(v).max():.4f}")

with torch.no_grad():
    position_ids = torch.arange(len(ids)).unsqueeze(0)
    pos_emb = model.model.rotary_emb(x.unsqueeze(0), position_ids)
    for i, layer in enumerate(model.model.layers):
        x = layer(x.unsqueeze(0), position_embeddings=pos_emb)[0].squeeze(0).float()
        sig(x, f"L{i:02d} residual")
    x = model.model.norm(x.unsqueeze(0)).squeeze(0).float()
    sig(x, "final norm")
    logits = model.lm_head(x.unsqueeze(0)).squeeze(0).float()
    t5 = torch.topk(logits, 5).indices.tolist()
print("HF top-5:", t5)
