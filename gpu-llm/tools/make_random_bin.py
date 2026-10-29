#!/usr/bin/env python3
"""make_random_bin.py — deterministic random fp32 weights file for the engine selftest."""
import sys
import numpy as np

TOTAL = 124439808  # floats in the GPT-2 124M layout (must match engine's TOTAL_FLOATS)
out = sys.argv[1] if len(sys.argv) > 1 else "random_124m.bin"
rng = np.random.default_rng(42)
a = ((rng.random(TOTAL, dtype=np.float32) - 0.5) * 0.06).astype(np.float32)
a.tofile(out)
print(f"wrote {out}: {TOTAL} floats, {TOTAL*4/1048576:.1f} MiB, range [{a.min():.4f}, {a.max():.4f}]")
