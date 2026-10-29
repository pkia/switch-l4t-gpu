#!/home/ev/qven/bin/python
"""hf_check.py — ground truth from the REAL HF transformers stack (transformers 5.x)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D = "/home/ev/cluster-prototype/gpu-llm/weights/qwen"
tok = AutoTokenizer.from_pretrained(D)
model = AutoModelForCausalLM.from_pretrained(D, torch_dtype=torch.float32)
model.eval()
if not getattr(tok, "chat_template", None):
    tok.chat_template = (
        "{%- for message in messages %}<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}<|im_end|>\n{%- endfor %}"
        "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}"
    )

def top5(ids):
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0, -1]
    return torch.topk(logits, 5).indices.tolist()

msgs = [{"role": "user", "content": "The capital of France is"}]

# variant A: HF-canonical template (no system message, no newline after im_end)
tplA = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
idsA = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)["input_ids"]

# variant B: MY template (system message + newline after im_end + trailing newline)
system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
tplB = ("<|im_start|>system\n" + system + "<|im_end|>\n"
        "<|im_start|>user\nThe capital of France is<|im_end|>\n"
        "<|im_start|>assistant\n")
idsB = tok(tplB)["input_ids"]

print("HF-canonical text:", repr(tplA))
print("HF-canonical ids :", idsA)
print("HF-canonical top-5:", top5(idsA))
print("MY-template ids  :", idsB)
print("MY-template top-5:", top5(idsB))

for label, ids, sample in [("A-greedy", idsA, False), ("A-sampled", idsA, True),
                           ("B-greedy", idsB, False), ("B-sampled", idsB, True)]:
    kw = {"temperature": 0.7, "top_p": 0.8, "top_k": 20} if sample else {}
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=30,
                             do_sample=sample, **kw)
    print(f"{label} text:", repr(tok.decode(out[0][len(ids):], skip_special_tokens=True)))
