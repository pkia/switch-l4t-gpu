// gemv_test.c — Milestone 0 harness: verifies + times gemv/gemv4 PTX against a CPU golden model.
// Usage: ./gemv_test <kernels.ptx> [N K]     (defaults: N=4096 K=4096)
// Build: gcc -O2 -o gemv_test gemv_test.c -lcuda -lm
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext, *CUmodule, *CUfunction, *CUstream;
typedef uint64_t CUdeviceptr;
extern CUresult cuInit(unsigned int flags);
extern CUresult cuDeviceGet(CUdevice *d, int ord);
extern CUresult cuDeviceGetName(char *out, int len, CUdevice d);
extern CUresult cuCtxCreate_v2(CUcontext *ctx, unsigned int flags, CUdevice d);
extern CUresult cuModuleLoadData(CUmodule *m, const void *image);
extern CUresult cuModuleGetFunction(CUfunction *f, CUmodule m, const char *name);
extern CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t bytes);
extern CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytes);
extern CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytes);
extern CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                               unsigned bx, unsigned by, unsigned bz,
                               unsigned shmem, CUstream s, void **params, void *extra);
extern CUresult cuCtxSynchronize(void);
extern CUresult cuGetErrorString(CUresult r, const char **s);

#define CHK(x) do { CUresult r_ = (x); if (r_) { const char *s_; cuGetErrorString(r_, &s_); \
    fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", r_, s_ ? s_ : "?", __FILE__, __LINE__); exit(1); } } while (0)

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec / 1e9; }

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f); buf[n] = 0; *len = n;
    return buf;
}

static double now_s2(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec / 1e9; }

static unsigned long rng_state = 42;
static float frand(void) {  // deterministic [0,1)
    rng_state = rng_state * 6364136223846793005UL + 1442695040888963407UL;
    return (float)((rng_state >> 40) & 0xFFFFFF) / (float)0x1000000;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gemv_test <ptx> [N] [K]\n"); return 1; }
    int N = argc > 2 ? atoi(argv[2]) : 4096;
    int K = argc > 3 ? atoi(argv[3]) : 4096;
    if (K % 4) { fprintf(stderr, "K must be a multiple of 4\n"); return 1; }

    size_t plen = 0; char *ptx = read_file(argv[1], &plen);

    CHK(cuInit(0));
    CUdevice dev; CHK(cuDeviceGet(&dev, 0));
    char name[256]; CHK(cuDeviceGetName(name, sizeof name, dev));
    printf("device: %s\n", name);
    CUcontext ctx; CHK(cuCtxCreate_v2(&ctx, 0, dev));
    CUmodule mod; CHK(cuModuleLoadData(&mod, ptx));
    CUfunction f_naive, f_vec, f_warp;
    CHK(cuModuleGetFunction(&f_naive, mod, "gemv"));
    CUresult rv = cuModuleGetFunction(&f_vec, mod, "gemv4");
    int has_vec = (rv == 0);
    CUresult rw = cuModuleGetFunction(&f_warp, mod, "gemv_warp");
    int has_warp = (rw == 0);
    if (!has_vec) printf("note: gemv4 not found in PTX, skipping vec test\n");

    printf("GEMV %dx%d (fp32), weights %.1f MB\n", N, K, (double)N * K * 4 / 1048576.0);

    float *W = malloc((size_t)N * K * 4);
    float *x = malloc((size_t)K * 4);
    float *y = malloc((size_t)N * 4);
    float *yref = malloc((size_t)N * 4);
    for (int i = 0; i < N * K; i++) W[i] = frand() - 0.5f;
    for (int i = 0; i < K; i++) x[i] = frand() - 0.5f;

    // CPU golden model (serial; 16M FMA is fine single-threaded)
    double t0 = now_s();
    for (int n = 0; n < N; n++) {
        const float *row = W + (size_t)n * K;
        float acc = 0.0f;
        for (int k = 0; k < K; k++) acc += row[k] * x[k];
        yref[n] = acc;
    }
    double tcpu = now_s() - t0;
    double cpu_gflops = 2.0 * N * K / tcpu / 1e9;
    printf("CPU golden: %.1f ms (%.2f GFLOPS)\n", tcpu * 1e3, cpu_gflops);

    CUdeviceptr dW, dx, dy;
    CHK(cuMemAlloc_v2(&dW, (size_t)N * K * 4));
    CHK(cuMemAlloc_v2(&dx, (size_t)K * 4));
    CHK(cuMemAlloc_v2(&dy, (size_t)N * 4));
    CHK(cuMemcpyHtoD_v2(dW, W, (size_t)N * K * 4));
    CHK(cuMemcpyHtoD_v2(dx, x, (size_t)K * 4));

    int fails = 0;
    const char *names[3] = {"gemv", "gemv4", "gemv_warp"};
    CUfunction fns[3] = {f_naive, f_vec, f_warp};
    int present[3] = {1, has_vec, has_warp};
    for (int v = 0; v < 3; v++) {
        if (!present[v]) { printf("note: %s not found in PTX, skipping\n", names[v]); continue; }
        void *args[] = { &dW, &dx, &dy, &K, &N };
        unsigned gx = (v == 2) ? (unsigned)((N + 7) / 8) : (unsigned)((N + 255) / 256);
        CHK(cuLaunchKernel(fns[v], gx, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
        CHK(cuCtxSynchronize());
        CHK(cuMemcpyDtoH_v2(y, dy, (size_t)N * 4));
        int bad = 0; float maxabs = 0;
        for (int n = 0; n < N; n++) {
            float diff = y[n] - yref[n]; if (diff < 0) diff = -diff;
            float ref = yref[n] < 0 ? -yref[n] : yref[n];
            if (diff > maxabs) maxabs = diff;
            // combined tolerance: relative + absolute floor (outputs can cancel near zero)
            if (diff > 1e-4f * ref + 1e-4f && ++bad <= 3)
                printf("  mismatch n=%d got=%f want=%f\n", n, y[n], yref[n]);
        }
        printf("%s: %s (max abs err %.2e)\n", names[v], bad ? "WRONG" : "CORRECT", maxabs);
        if (bad) fails++;

        double best = 1e9;
        for (int rep = 0; rep < 3; rep++) {
            double s = now_s();
            for (int it = 0; it < 20; it++)
                CHK(cuLaunchKernel(fns[v], gx, 1, 1, 256, 1, 1, 0, NULL, args, NULL));
            CHK(cuCtxSynchronize());
            double d = now_s() - s;
            if (d < best) best = d;
        }
        double bytes = 20.0 * ((double)N * K * 4 + (double)K * 4 + (double)N * 4);
        double ms = best / 20 * 1e3;
        double gbps = bytes / best / 1e9;
        double tok500 = gbps * 1e9 / 0.5e9;   // tok/s for a 500 MB model at this bandwidth
        printf("%s: %.2f ms/token -> %.1f GB/s (%.1f%% of 22.2 ceiling), "
               "~%.1f tok/s for a 500 MB model at this rate\n",
               names[v], ms, gbps, gbps / 22.2 * 100, tok500);
    }

    printf("%s\n", fails ? "GEMV_TEST_FAILED" : "GEMV_TEST_DONE");
    return fails ? 1 : 0;
}
