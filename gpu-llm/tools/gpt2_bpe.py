#!/usr/bin/env python3
"""gpt2_bpe.py — minimal GPT-2 byte-level BPE encoder/decoder (pure stdlib + regex).
Usage:
  python3 gpt2_bpe.py encode "The capital of France is" prompt.ids
  python3 gpt2_bpe.py decode out.ids [-]     # '-' = first id is the prompt (only decode new ones)
"""
import json
import os
import sys

import regex as re

VOCAB = __import__("os").environ.get("TKN_VOCAB", "weights/vocab.json")
MERGES = __import__("os").environ.get("TKN_MERGES", "weights/merges.txt")
PATTERN = __import__("os").environ.get("TKN_REGEX")   # Qwen sets TKN_REGEX from tokenizer.json
# special tokens bypass BPE (json map "token string" -> id), e.g. Qwen ChatML markers
SPECIALS = json.loads(__import__("os").environ.get("TKN_SPECIALS", "{}"))


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class Encoder:
    def __init__(self):
        self.encoder = json.load(open(VOCAB, encoding="utf-8"))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = [tuple(line.split()) for line in open(MERGES, encoding="utf-8").read().split("\n")[1:-1]]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(
            PATTERN or r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self.get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self.get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    @staticmethod
    def get_pairs(word):
        pairs = set()
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def encode(self, text: str):
        ids = []
        if SPECIALS:
            import re as _re
            parts = _re.split("(" + "|".join(_re.escape(s) for s in SPECIALS) + ")", text)
        else:
            parts = [text]
        for part in parts:
            if part in SPECIALS:
                ids.append(SPECIALS[part])
                continue
            for token in self.pat.findall(part):
                token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                ids.extend(self.encoder[t] for t in self.bpe(token).split(" "))
        return ids

    def decode(self, ids):
        inv = {v: k for k, v in SPECIALS.items()}
        text = "".join(inv.get(i, self.decoder.get(i, "")) for i in ids)
        return bytearray(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")


def main():
    mode = sys.argv[1]
    enc = Encoder()
    if mode == "encode":
        text = sys.argv[2]
        if mode == "encode" and os.environ.get("TKN_CHATML"):
            text = ("<|im_start|>system\n" + os.environ.get("TKN_SYSTEM", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") + "<|im_end|>\n"
                    "<|im_start|>user\n" + text + "<|im_end|>\n"
                    "<|im_start|>assistant\n")
        out = sys.argv[3] if len(sys.argv) > 3 else "prompt.ids"
        ids = enc.encode(text)
        with open(out, "wb") as f:
            import struct
            f.write(struct.pack("<I", len(ids)))
            for i in ids:
                f.write(struct.pack("<I", i))
        print(f"encoded {len(ids)} token(s) -> {out}: {ids}")
    elif mode == "decode":
        src = sys.argv[2]
        import struct
        with open(src, "rb") as f:
            cnt = struct.unpack("<I", f.read(4))[0]
            ids = [struct.unpack("<I", f.read(4))[0] for _ in range(cnt)]
        text = enc.decode(ids)
        print(f"[{len(ids)} ids] {text}")
    else:
        print("unknown mode", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
