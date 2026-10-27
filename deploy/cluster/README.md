# Running the suite from a task-queue worker (cluster integration)

`gpu_workload` is deliberately self-contained: one process, one PTX argument, deterministic
output. That makes it a drop-in workload for any task queue — a worker claims a unit, runs the
suite, and reports the numbers back. This directory documents the exact contract used by the
author's Pi-coordinated cluster (stdlib-Python coordinator + workers), reduced so anyone can
replicate it with any queue.

## The contract

**GPU home** — a directory where the harness has been built and a kernel PTX compiled:

```
<home>/gpu_workload              built with: make gpu_workload   (gcc, links -lcuda)
<home>/switch_gpu_native.ptx     built with: make native        (clang-18, sm_53 / PTX ISA 6.3)
```

Workers look it up in this order: `$CLUSTER_GPU_HOME`, then `~/switch-l4t-gpu`, then `~/gpu`.
A directory qualifies only with an **executable** `gpu_workload` **and** the PTX present; a
worker that finds none simply never advertises GPU work (capability detection, below).

**Invocation** — one unit = one suite run:

```bash
"$HOME/gpu_workload" --json "$HOME/switch_gpu_native.ptx"     # modern: one JSON object on stdout
"$HOME/gpu_workload" "$HOME/switch_gpu_native.ptx"            # legacy: stable human-readable lines
```

The suite is fixed by design (SGEMM tiled/naive/4×4/opt at 256³, 512³, 1024³ ×10 reps each,
CPU golden-model verification at 256³, plus a float4 read-bandwidth benchmark). One unit takes
a few seconds; the GPU governor (`nvhost_podgov`) reads low on the first run after idle.

## `--json` output schema

One single-line JSON object on stdout (errors go to stderr; CUDA errors exit non-zero):

```json
{"ptx":"switch_gpu_native.ptx","kernels":{"sgemm":"CORRECT","sgemm_naive":"CORRECT","sgemm_tiled4":"CORRECT","sgemm_opt":"CORRECT"},
 "sgemm":[{"n":256,"ms":3.5,"gflops":24.0},{"n":512,"ms":7.0,"gflops":61.4},{"n":1024,"ms":25.0,"gflops":85.9}],
 "sgemm4":[{"n":256,"ms":1.0,"gflops":33.5},{"n":512,"ms":5.0,"gflops":86.0},{"n":1024,"ms":22.0,"gflops":97.3}],
 "sgemm_opt":[{"n":256,"ms":0.9,"gflops":37.2},{"n":512,"ms":4.0,"gflops":107.5},{"n":1024,"ms":20.0,"gflops":107.4}],
 "membw_gbps":22.3,"done":true}
```

- `kernels` — verdict of the on-device CPU golden-model check per kernel (256³ only; `WRONG`
  means real numeric corruption, not noise). Any `WRONG` ⇒ the run is invalid.
- `sgemm`, `sgemm4`, `sgemm_opt` — per-size `ms` per iteration and `gflops` (10-iter average).
- `membw_gbps` — float4 read bandwidth.
- `done: true` — the sentinel: proves the object is complete, not truncated mid-run.
- `sgemm4` / `sgemm_opt` are omitted when the loaded PTX lacks those kernels.

Consumers should treat the JSON as authoritative when present and fall back to parsing the
legacy text lines (format documented in `docs/BENCHMARKS.md` and stable since day one) for
harnesses built before the `--json` flag: a legacy binary given `--json` treats it as the PTX
path, fails fast with a missing-file error, and the caller can simply re-invoke without it.

## Verification philosophy

The GPU's numeric truth is established **on the device** by the harness's CPU reference
(`rel. tol. 5% + 1e-3` at 256³). A queue coordinator cannot cheaply re-verify SGEMM results
independently, so it verifies: JSON completeness (`done`), all verdicts `CORRECT`, all sizes
timed, and sanity bounds on the reported numbers (catches garbage, not fraud). This is the
same trust model as running any benchmark binary — the harness is the verifier.

## Capability detection (worker side)

```python
def gpu_home():
    candidates = ([Path(os.environ["CLUSTER_GPU_HOME"])] if os.environ.get("CLUSTER_GPU_HOME") else [])
    candidates += [Path.home() / "switch-l4t-gpu", Path.home() / "gpu"]
    for c in candidates:
        if (c / "gpu_workload").is_file() and os.access(c / "gpu_workload", os.X_OK) \
                and (c / "switch_gpu_native.ptx").is_file():
            return c
    return None      # None => worker does not advertise GPU work to the queue
```

## Example consumer

[`gpu_task_example.py`](gpu_task_example.py) is a standalone, stdlib-only module: `run_unit()`
executes the suite (JSON preferred, legacy fallback) and returns a small dict
(`correct`, `kernels_verified`, `gflops{256,512,1024}`, `gbps`); `verify_unit()` re-checks the
result the way a coordinator would. Run it directly on the Switch to see one unit's result.

Power note: sustained 24/7 GEMM needs a PD charger on many USB ports (see RUNBOOK.md);
occasional verified runs charge fine.
