#!/usr/bin/env python3
"""qwen_serve.py — OpenAI-compatible HTTP wrapper around qwen_engine --serve.
Runs on the Switch; Hermes (or anything speaking the Chat Completions dialect) is the client.

  POST /v1/chat/completions  {"model": "...", "messages": [...], "max_tokens": N}
  GET  /v1/models
  GET  /health

Auth: Authorization: Bearer <token from ./token>. Engine state: one persistent
qwen_engine --serve process (weights loaded once; requests serialized).
"""
import json
import os
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)                      # ~/gpu-llm
sys.path.insert(0, os.path.join(ROOT, "tools"))   # gpt2_bpe lives there
ROOT = os.path.dirname(BASE)                      # ~/gpu-llm
ENGINE = os.path.join(ROOT, "engine", "qwen_engine")
BIN = os.path.join(ROOT, "engine", "qwen_05b.bin")
PTX = os.path.join(ROOT, "kernels", "qwen.ptx")
TK = os.path.join(BASE, "tk")
CTX = 8192
DEFAULT_GEN = 256
MAX_GEN = 1024

os.environ["TKN_VOCAB"] = os.path.join(TK, "vocab.json")
os.environ["TKN_MERGES"] = os.path.join(TK, "merges.txt")
os.environ["TKN_REGEX"] = open(os.path.join(TK, "pretok.regex")).read()
os.environ["TKN_SPECIALS"] = json.dumps({
    "<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 151643})
from gpt2_bpe import Encoder            # noqa: E402  (imports after env is set)
enc = Encoder()

TOKEN = open(os.path.join(BASE, "token")).read().strip()
MODEL_ID = "qwen2.5-0.5b-switch"

lock = threading.Lock()
eng = None

def start_engine():
    global eng
    errlog = open("/tmp/qwen_engine.err", "ab")
    eng = subprocess.Popen(
        [ENGINE, "--weights", BIN, "--ptx", PTX, "--serve", "--gen", str(MAX_GEN)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=errlog, cwd=BASE)

def read_exact(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise RuntimeError("engine died")
        buf += chunk
    return buf

def run_engine(ids, max_gen):
    with lock:
        payload = struct.pack("<I", len(ids)) + b"".join(struct.pack("<I", i) for i in ids)
        payload += struct.pack("<I", max_gen)
        eng.stdin.write(payload)
        eng.stdin.flush()
        (outn,) = struct.unpack("<I", read_exact(eng.stdout, 4))
        raw = read_exact(eng.stdout, outn * 4)
        return [struct.unpack("<I", raw[i * 4:i * 4 + 4])[0] for i in range(outn)]

def build_prompt_ids(messages):
    """Qwen ChatML; truncates oldest turns to fit CTX (keeps the latest user turn)."""
    parts = []
    system = None
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system = content
        elif role in ("user", "assistant"):
            parts.append((role, content))
    head = "<|im_start|>system\n" + (system or "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") + "<|im_end|>\n"
    tail = "<|im_start|>assistant\n"
    ids_head = enc.encode(head)
    ids_tail = enc.encode(tail)
    budget = CTX - len(ids_head) - len(ids_tail) - 16
    body_ids = []
    kept = []
    for role, content in reversed(parts):
        block = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        bids = enc.encode(block)
        if len(bids) > budget - len(body_ids):
            break
        body_ids = bids + body_ids
        kept.insert(0, (role, content))
    return ids_head + body_ids + ids_tail, kept

class Handler(BaseHTTPRequestHandler):
    def _auth(self):
        auth = self.headers.get("Authorization", "")
        return auth == "Bearer " + TOKEN

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {"status": "ok", "model": MODEL_ID})
        if not self._auth():
            return self._json(401, {"error": "unauthorized"})
        if self.path == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "switch-l4t-gpu"}]})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": "not found"})
        if not self._auth():
            return self._json(401, {"error": "unauthorized"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length))
            messages = req.get("messages", [])
            max_tokens = min(int(req.get("max_tokens", DEFAULT_GEN)), MAX_GEN)
            if not messages:
                return self._json(400, {"error": "messages required"})
        except Exception as e:
            return self._json(400, {"error": f"bad request: {e}"})
        try:
            prompt_ids, kept = build_prompt_ids(messages)
            t0 = time.time()
            out_ids = run_engine(prompt_ids, max_tokens)
            dt = time.time() - t0
            new_ids = out_ids[len(prompt_ids):]
            text = enc.decode(new_ids)
            finish = "length"
            if new_ids and new_ids[-1] in (151643, 151645):
                new_ids = new_ids[:-1]
                finish = "stop"
                text = enc.decode(new_ids)
            return self._json(200, {
                "id": "chatcmpl-switch-%d" % int(time.time()),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                }],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(new_ids),
                    "total_tokens": len(prompt_ids) + len(new_ids),
                },
                "switch": {"seconds": round(dt, 2), "kept_turns": len(kept)},
            })
        except RuntimeError as e:
            return self._json(502, {"error": f"engine: {e}"})
        except Exception as e:
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        pass  # keep journald quiet; errors surface via responses

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8420"))
    start_engine()
    print(f"qwen_serve listening on 0.0.0.0:{port} (model {MODEL_ID}, ctx {CTX})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
