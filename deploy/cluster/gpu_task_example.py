#!/usr/bin/env python3
"""Standalone example: run + verify the GPU suite from a task-queue worker.

The consumer contract behind the author's Pi-coordinated cluster, reduced to one
copy-pasteable stdlib-only file (see README.md in this directory):

    GPU home lookup:  $CLUSTER_GPU_HOME, then ~/switch-l4t-gpu, then ~/gpu
    A home qualifies: executable ./gpu_workload + readable ./switch_gpu_native.ptx
    One unit:         gpu_workload --json <ptx>   (legacy harnesses: without --json)

Numeric truth is established on the device by the harness's CPU golden-model check;
the bounds below only catch garbage numbers, not fraud. Run this file directly on
the Switch to see one unit's result.
"""
import json
import os
import re
import subprocess
from pathlib import Path

SIZES = (256, 512, 1024)
TIMEOUT_S = 50
GFLOPS_BOUNDS = (1.0, 1000.0)   # sanity gates (Tegra X1 ceiling ~472 GFLOPS FMA)
GBPS_BOUNDS = (1.0, 1000.0)

_RE_SGEMM_VERDICT = re.compile(r"^sgemm(?:_\w+)? (\d+): (CORRECT|WRONG)\b", re.M)
_RE_SGEMM_PERF = re.compile(r"^sgemm (\d+)x\1x\1: [\d.]+ ms/iter, ([\d.]+) GFLOPS", re.M)
_RE_MEMBW = re.compile(r"^membw read .* ([\d.]+) GB/s", re.M)


class UnitError(ValueError):
    pass


def gpu_home():
    """Directory holding the built harness + PTX, or None if this machine can't run it."""
    candidates = []
    if os.environ.get("CLUSTER_GPU_HOME"):
        candidates.append(Path(os.environ["CLUSTER_GPU_HOME"]))
    candidates += [Path.home() / "switch-l4t-gpu", Path.home() / "gpu"]
    for c in candidates:
        binary, ptx = c / "gpu_workload", c / "switch_gpu_native.ptx"
        if binary.is_file() and os.access(binary, os.X_OK) and ptx.is_file():
            return c
    return None


def _check_numbers(gflops: dict, gbps) -> None:
    for s in map(str, SIZES):
        v = gflops.get(s)
        if not isinstance(v, (int, float)) or not GFLOPS_BOUNDS[0] <= v <= GFLOPS_BOUNDS[1]:
            raise UnitError(f"gflops[{s}] missing or outside sanity bounds")
    if not isinstance(gbps, (int, float)) or not GBPS_BOUNDS[0] <= gbps <= GBPS_BOUNDS[1]:
        raise UnitError("gbps missing or outside sanity bounds")


def _from_json(out: str) -> dict:
    """Parse `gpu_workload --json` output; raise UnitError on any verdict/timing problem."""
    data = json.loads(out)                                  # caller validated it starts with '{'
    if data.get("done") is not True:
        raise UnitError("harness JSON truncated (done flag missing)")
    kernels = data.get("kernels")
    if not isinstance(kernels, dict) or not kernels.get("sgemm"):
        raise UnitError("harness JSON missing kernels.sgemm verdict")
    bad = sorted(k for k, v in kernels.items() if v != "CORRECT")
    if bad:
        raise UnitError(f"kernels failed the harness CPU golden-model check: {bad}")
    gflops = {str(int(r["n"])): float(r["gflops"])
              for r in (data.get("sgemm") or [])
              if isinstance(r, dict) and isinstance(r.get("gflops"), (int, float))}
    missing = [s for s in map(str, SIZES) if s not in gflops]
    if missing:
        raise UnitError(f"no SGEMM timing for size(s) {missing}")
    gbps = data.get("membw_gbps")
    _check_numbers(gflops, gbps)
    return {"correct": True, "kernels_verified": sorted(kernels),
            "gflops": {k: round(v, 1) for k, v in gflops.items()},
            "gbps": round(float(gbps), 1)}


def _from_text(out: str) -> dict:
    """Legacy path: parse the stable human-readable lines of a pre---json harness."""
    if "WORKLOAD_DONE" not in out:
        raise UnitError("harness output truncated (no WORKLOAD_DONE sentinel)")
    verdicts = {("sgemm" + (m.group("k") or "")): m.group("v")
                for m in re.finditer(r"^sgemm(?P<k>_\w+)? \d+: (?P<v>CORRECT|WRONG)\b", out, re.M)}
    if "sgemm" not in verdicts:
        raise UnitError("harness produced no sgemm verdict line")
    bad = sorted(k for k, v in verdicts.items() if v != "CORRECT")
    if bad:
        raise UnitError(f"kernels failed the harness CPU golden-model check: {bad}")
    gflops = {m.group(1): float(m.group(2)) for m in _RE_SGEMM_PERF.finditer(out)}
    missing = [s for s in map(str, SIZES) if s not in gflops]
    if missing:
        raise UnitError(f"no SGEMM timing for size(s) {missing}")
    mb = _RE_MEMBW.search(out)
    if mb is None:
        raise UnitError("harness produced no membw line")
    gbps = float(mb.group(1))
    _check_numbers(gflops, gbps)
    return {"correct": True, "kernels_verified": sorted(verdicts),
            "gflops": {k: round(v, 1) for k, v in gflops.items()}, "gbps": round(gbps, 1)}


def run_unit() -> dict:
    """One task unit: run the suite, return a small verifiable dict (or raise UnitError)."""
    home = gpu_home()
    if home is None:
        raise UnitError("no GPU home (CLUSTER_GPU_HOME / ~/switch-l4t-gpu / ~/gpu)")
    binary, ptx = str(home / "gpu_workload"), str(home / "switch_gpu_native.ptx")
    try:
        proc = subprocess.run([binary, "--json", ptx], capture_output=True, text=True,
                              timeout=TIMEOUT_S)
        if proc.returncode != 0 and not proc.stdout.lstrip().startswith("{"):
            # pre---json harness treats --json as the PTX path; fall back to text mode
            proc = subprocess.run([binary, ptx], capture_output=True, text=True,
                                  timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise UnitError(f"harness timed out after {TIMEOUT_S}s") from exc
    if proc.returncode != 0:
        raise UnitError(f"gpu_workload exited {proc.returncode}: {proc.stderr.strip()[:300]}")
    out = proc.stdout
    if out.lstrip().startswith("{"):
        try:
            return _from_json(out)
        except json.JSONDecodeError as exc:
            raise UnitError(f"invalid --json output: {exc}") from exc
    return _from_text(out)


def verify_unit(result: dict) -> None:
    """Coordinator-side re-check of a reported result (raise UnitError to reject it)."""
    if result.get("correct") is not True:
        raise UnitError("result missing correct=true (golden-model check not reported)")
    _check_numbers(result.get("gflops", {}), result.get("gbps"))


if __name__ == "__main__":
    import sys
    try:
        print(json.dumps(run_unit(), indent=2))
    except UnitError as exc:
        sys.exit(f"unit failed: {exc}")
