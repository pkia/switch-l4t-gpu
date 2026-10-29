// gemv_multi timing: T=128, K=768, N=2304 (qkv shape) — isolates kernel vs launch overhead.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
typedef int CUresult; typedef int CUdevice;
typedef void *CUcontext, *CUmodule, *CUfunction, *CUstream;
typedef uint64_t CUdeviceptr;
extern CUresult cuInit(unsigned int);
extern CUresult cuDeviceGet(CUdevice *, int);
extern CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
extern CUresult cuModuleLoadData(CUmodule *, const void *);
extern CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *);
extern CUresult cuMemAlloc_v2(CUdeviceptr *, size_t);
extern CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
extern CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                               unsigned, unsigned, CUstream, void **, void **);
extern CUresult cuCtxSynchronize(void);
extern CUresult cuGetErrorString(CUresult, const char *);
#include <time.h>
static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
#define CHK(x) do { CUresult r_ = (x); if (r_) { const char *s_; cuGetErrorString(r_, &s_); \
    fprintf(stderr, "CUDA error %d (%s) at %d\n", r_, s_ ? s_ : "?", __LINE__); exit(1); } } while (0)

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    FILE *pf = fopen(argv[1], "rb");
    fseek(pf, 0, SEEK_END); long n = ftell(pf); fseek(pf, 0, SEEK_SET);
    char *ptx = malloc(n + 1);
    if (fread(ptx, 1, n, pf) != (size_t)n) return 2;
    fclose(pf); ptx[n] = 0;
    CHK(cuInit(0));
    CUdevice dev; CHK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHK(cuCtxCreate_v2(&ctx, 0, dev));
    CUmodule mod; CHK(cuModuleLoadData(&mod, ptx));
    CUfunction k; CHK(cuModuleGetFunction(&k, mod, "gemv_multi"));
    int K = 768, N = 2304, T = 128;
    CUdeviceptr dW, dX, dB, dY;
    CHK(cuMemAlloc_v2(&dW, (size_t)K * N * 4));
    CHK(cuMemAlloc_v2(&dX, (size_t)K * T * 4));
    CHK(cuMemAlloc_v2(&dB, N * 4));
    CHK(cuMemAlloc_v2(&dY, (size_t)N * T * 4));
    float *w = calloc((size_t)K * N, 4);
    for (long i = 0; i < (long)K * N; i++) w[i] = 0.01f * (i % 97) - 0.5f;
    CHK(cuMemcpyHtoD_v2(dW, w, (size_t)K * N * 4));
    CHK(cuCtxSynchronize());
    int cK = K, cN = N, cT = T;
    for (int rep = 0; rep < 5; rep++) {
        double t0 = now();
        CHK(cuLaunchKernel(k, (N + 7) / 8, (T + 15) / 16, 1, 256, 1, 1, 0, NULL,
            (void *[]){ &dW, &dX, &dB, &dY, &cK, &cN, &cT }, NULL));
        CHK(cuCtxSynchronize());
        printf("gemv_multi T=%d K=%d N=%d: %.2f ms\n", T, K, N, (now() - t0) * 1e3);
        fflush(stdout);
    }
    // bandwidth achieved: weights 7.08MB read ceil(T/16)=8 times = 56.6MB in X ms
    return 0;
}
