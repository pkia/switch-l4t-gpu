// CUDA driver-API GPU compute test for Nintendo Switch (Tegra X1, sm_53, L4T libcuda 10.0).
// Kernels are hand-written PTX (ISA 6.3) embedded as strings — no nvcc needed.
// Build: gcc -O2 -o gpu_test gpu_test.c -lcuda -lm
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

// --- minimal CUDA driver API declarations (matching libcuda 10.0 ABI) ---
typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext, *CUmodule, *CUfunction, *CUstream;
typedef uint64_t CUdeviceptr;
extern CUresult cuInit(unsigned int flags);
extern CUresult cuDriverGetVersion(int *v);
extern CUresult cuDeviceGetCount(int *n);
extern CUresult cuDeviceGet(CUdevice *d, int ord);
extern CUresult cuDeviceGetName(char *out, int len, CUdevice d);
extern CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice d);
extern CUresult cuDeviceGetAttribute(int *val, int attr, CUdevice d);
extern CUresult cuCtxCreate_v2(CUcontext *ctx, unsigned int flags, CUdevice d);
extern CUresult cuModuleLoadData(CUmodule *m, const void *image);
extern CUresult cuModuleGetFunction(CUfunction *f, CUmodule m, const char *name);
extern CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t bytes);
extern CUresult cuMemFree_v2(CUdeviceptr p);
extern CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytes);
extern CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytes);
extern CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                               unsigned bx, unsigned by, unsigned bz,
                               unsigned shmem, CUstream s, void **params, void *extra);
extern CUresult cuCtxSynchronize(void);
extern CUresult cuGetErrorString(CUresult r, const char **s);

#define CHK(x) do { CUresult r_ = (x); if (r_) { const char *s_; cuGetErrorString(r_, &s_); \
    fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", r_, s_ ? s_ : "?", __FILE__, __LINE__); exit(1); } } while (0)

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// PTX: d[i] = a[i] + scale * b[i]
static const char *PTX_VECADD =
".version 6.3\n.target sm_53\n.address_size 64\n"
".visible .entry vecadd(\n"
"  .param .u64 d_, .param .u64 a_, .param .u64 b_, .param .f32 scale_)\n"
"{\n"
"  .reg .b32 %r<6>; .reg .f32 %f<5>; .reg .b64 %rd<10>;\n"
"  ld.param.u64 %rd1, [d_]; ld.param.u64 %rd2, [a_]; ld.param.u64 %rd3, [b_]; ld.param.f32 %f1, [scale_];\n"
"  mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mul.lo.s32 %r3, %r1, %r2;\n"
"  mov.u32 %r4, %tid.x; add.s32 %r5, %r3, %r4;\n"
"  cvta.to.global.u64 %rd4, %rd1; cvta.to.global.u64 %rd5, %rd2; cvta.to.global.u64 %rd6, %rd3;\n"
"  mul.wide.s32 %rd7, %r5, 4;\n"
"  add.s64 %rd8, %rd5, %rd7; ld.global.f32 %f2, [%rd8];\n"
"  add.s64 %rd9, %rd6, %rd7; ld.global.f32 %f3, [%rd9];\n"
"  fma.rn.f32 %f4, %f3, %f1, %f2;\n"
"  add.s64 %rd8, %rd4, %rd7; st.global.f32 [%rd8], %f4;\n"
"  ret;\n"
"}\n";

// PTX: each thread runs a dependent FMA chain of iters_ steps -> out[tid] (ALU throughput)
static const char *PTX_FMABENCH =
".version 6.3\n.target sm_53\n.address_size 64\n"
".visible .entry fma_bench(\n"
"  .param .u64 out_, .param .f32 x_, .param .u32 iters_)\n"
"{\n"
"  .reg .b32 %r<8>; .reg .f32 %f<3>; .reg .b64 %rd<6>; .reg .pred %p<2>;\n"
"  ld.param.u64 %rd1, [out_]; ld.param.f32 %f1, [x_]; ld.param.u32 %r1, [iters_];\n"
"  mov.f32 %f2, %f1; mov.u32 %r2, 0;\n"
"$L1:\n"
"  fma.rn.f32 %f2, %f2, %f1, %f1;\n"
"  add.u32 %r2, %r2, 1; setp.lt.u32 %p1, %r2, %r1; @%p1 bra $L1;\n"
"  mov.u32 %r3, %ctaid.x; mov.u32 %r4, %ntid.x; mul.lo.s32 %r5, %r3, %r4;\n"
"  mov.u32 %r6, %tid.x; add.s32 %r7, %r5, %r6;\n"
"  cvta.to.global.u64 %rd2, %rd1; mul.wide.s32 %rd3, %r7, 4; add.s64 %rd4, %rd2, %rd3;\n"
"  st.global.f32 [%rd4], %f2;\n"
"  ret;\n"
"}\n";

int main(void) {
    CUresult r; CUcontext ctx; CUmodule mod; CUfunction f;
    int n = 0, ver = 0, attr = 0; CUdevice dev; char name[256]; size_t total = 0;

    CHK(cuInit(0));
    CHK(cuDriverGetVersion(&ver));
    CHK(cuDeviceGetCount(&n));
    printf("driver API version: %d.%d  devices: %d\n", ver / 1000, ver % 1000, n);
    if (!n) { fprintf(stderr, "no CUDA devices\n"); return 1; }
    CHK(cuDeviceGet(&dev, 0));
    CHK(cuDeviceGetName(name, sizeof name, dev));
    CHK(cuDeviceTotalMem_v2(&total, dev));
    printf("device 0: %s  total mem: %.0f MiB\n", name, total / 1048576.0);
    CHK(cuDeviceGetAttribute(&attr, 75, dev)); int cc_major = attr;
    CHK(cuDeviceGetAttribute(&attr, 76, dev)); printf("compute capability: %d.%d\n", cc_major, attr);
    CHK(cuDeviceGetAttribute(&attr, 16, dev)); printf("SM count: %d\n", attr);
    CHK(cuDeviceGetAttribute(&attr, 13, dev)); printf("core clock: %d kHz\n", attr);
    CHK(cuCtxCreate_v2(&ctx, 0, dev));

    // ---- vecadd correctness + speed ----
    CHK(cuModuleLoadData(&mod, PTX_VECADD));
    CHK(cuModuleGetFunction(&f, mod, "vecadd"));
    const int N = 1 << 22; // 4M floats
    float *a = malloc(N * 4), *b = malloc(N * 4), *d = malloc(N * 4), *ref = malloc(N * 4);
    for (int i = 0; i < N; i++) { a[i] = i * 0.001f; b[i] = (N - i) * 0.001f; ref[i] = a[i] + 2.5f * b[i]; }
    CUdeviceptr da, db, dd; float scale = 2.5f;
    CHK(cuMemAlloc_v2(&da, N * 4)); CHK(cuMemAlloc_v2(&db, N * 4)); CHK(cuMemAlloc_v2(&dd, N * 4));
    CHK(cuMemcpyHtoD_v2(da, a, N * 4)); CHK(cuMemcpyHtoD_v2(db, b, N * 4));
    void *args[] = { &dd, &da, &db, &scale };
    CHK(cuLaunchKernel(f, N / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    CHK(cuCtxSynchronize());
    CHK(cuMemcpyDtoH_v2(d, dd, N * 4));
    int bad = 0;
    for (int i = 0; i < N; i++) {
        float diff = d[i] - ref[i]; if (diff < 0) diff = -diff;
        if (diff > 1e-4f * (ref[i] < 0 ? -ref[i] : ref[i]) + 1e-6f) { if (++bad > 5) break; }
    }
    printf("vecadd 4M elems: %s (mismatches: %d)\n", bad ? "WRONG" : "CORRECT", bad);
    double t0 = now_s();
    for (int i = 0; i < 20; i++) CHK(cuLaunchKernel(f, N / 256, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
    CHK(cuCtxSynchronize());
    double dt = now_s() - t0;
    printf("vecadd: 20 iters, %.1f ms total, %.2f GElem/s, effective %.1f GB/s\n",
           dt * 1e3, 20.0 * N / dt / 1e9, 20.0 * N * 3 * 4 / dt / 1e9);

    // ---- FMA throughput ----
    CUmodule m2; CUfunction fb; CHK(cuModuleLoadData(&m2, PTX_FMABENCH));
    CHK(cuModuleGetFunction(&fb, m2, "fma_bench"));
    CUdeviceptr dout; CHK(cuMemAlloc_v2(&dout, 256 * 1024 * 4));
    uint32_t iters = 100000; float xv = 1.000001f;
    void *bargs[] = { &dout, &xv, &iters };
    CHK(cuLaunchKernel(fb, 256, 1, 1, 256, 1, 1, 0, NULL, bargs, NULL)); // warmup
    CHK(cuCtxSynchronize());
    t0 = now_s();
    CHK(cuLaunchKernel(fb, 256, 1, 1, 256, 1, 1, 0, NULL, bargs, NULL));
    CHK(cuCtxSynchronize());
    dt = now_s() - t0;
    double flops = 2.0 * 256 * 1024 * (double)iters;
    printf("fma_bench: %.3f s, %.2f GFLOPS FP32 (dependent chain)\n", dt, flops / dt / 1e9);

    printf("GPU_TEST_DONE\n");
    return 0;
}
