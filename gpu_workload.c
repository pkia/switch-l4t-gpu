// Driver-API harness: loads switch_gpu.ptx, runs tiled SGEMM + memory bandwidth on Tegra X1.
// Build: gcc -O2 -o gpu_workload gpu_workload.c -lcuda -lm
// Usage: ./gpu_workload [switch_gpu.ptx]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext, *CUmodule, *CUfunction, *CUstream;
typedef uint64_t CUdeviceptr;
extern CUresult cuInit(unsigned int);
extern CUresult cuDeviceGet(CUdevice*, int);
extern CUresult cuCtxCreate_v2(CUcontext*, unsigned, CUdevice);
extern CUresult cuModuleLoadData(CUmodule*, const void*);
extern CUresult cuModuleGetFunction(CUfunction*, CUmodule, const char*);
extern CUresult cuMemAlloc_v2(CUdeviceptr*, size_t);
extern CUresult cuMemFree_v2(CUdeviceptr);
extern CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void*, size_t);
extern CUresult cuMemcpyDtoH_v2(void*, CUdeviceptr, size_t);
extern CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned,
                               unsigned, unsigned, unsigned, unsigned,
                               CUstream, void**, void*);
extern CUresult cuCtxSynchronize(void);

#define CHK(x) do { if ((x)) { fprintf(stderr, "CUDA error at %d\n", __LINE__); exit(1); } } while (0)

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *len = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(*len + 1);
    if (fread(buf, 1, *len, f) != *len) exit(1);
    buf[*len] = 0; fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    const char *ptx_path = argc > 1 ? argv[1] : "switch_gpu.ptx";
    CUcontext ctx; CUdevice dev; CUmodule mod; CUfunction sgemm, membw;
    size_t plen; char *ptx = read_file(ptx_path, &plen);

    CHK(cuInit(0));
    CHK(cuDeviceGet(&dev, 0));
    CHK(cuCtxCreate_v2(&ctx, 0, dev));
    CHK(cuModuleLoadData(&mod, ptx));
    CHK(cuModuleGetFunction(&sgemm, mod, "sgemm_tiled"));
    CHK(cuModuleGetFunction(&membw, mod, "membw_read4"));
    CUfunction naive = NULL, sgemm4 = NULL, sgopt = NULL;
    cuModuleGetFunction(&naive, mod, "sgemm_naive");  // optional bisect kernel
    cuModuleGetFunction(&sgemm4, mod, "sgemm_tiled4"); // optional 4x4 kernel
    cuModuleGetFunction(&sgopt, mod, "sgemm_opt");     // optional float4 optimized kernel

    // ---- SGEMM ----
    int sizes[] = {256, 512, 1024};
    for (int si = 0; si < 3; si++) {
        int n = sizes[si];
        size_t bytes = (size_t)n * n * 4;
        float *A = malloc(bytes), *B = malloc(bytes), *C = malloc(bytes);
        srand(42);
        for (size_t i = 0; i < (size_t)n * n; i++) { A[i] = (rand() % 2000 - 1000) / 1000.0f; B[i] = (rand() % 2000 - 1000) / 1000.0f; }
        CUdeviceptr dA, dB, dC;
        CHK(cuMemAlloc_v2(&dA, bytes)); CHK(cuMemAlloc_v2(&dB, bytes)); CHK(cuMemAlloc_v2(&dC, bytes));
        CHK(cuMemcpyHtoD_v2(dA, A, bytes)); CHK(cuMemcpyHtoD_v2(dB, B, bytes));
        int Mn = n, Nn = n, Kn = n;
        void *args[] = { &dC, &dA, &dB, &Mn, &Nn, &Kn };
        // NOTE: kernel signature order is (A, B, C, M, N, K)
        void *args2[] = { &dA, &dB, &dC, &Mn, &Nn, &Kn };
        CHK(cuLaunchKernel(sgemm, n / 32, n / 32, 1, 16, 16, 1, 0, NULL, args2, NULL));
        CHK(cuCtxSynchronize());
        if (n == 256) {  // verify vs CPU
            CHK(cuMemcpyDtoH_v2(C, dC, bytes));
            int bad = 0;
            for (int i = 0; i < n && bad < 5; i++)
                for (int j = 0; j < n; j++) {
                    float ref = 0;
                    for (int k = 0; k < n; k++) ref += A[i * n + k] * B[k * n + j];
                    float d = C[i * n + j] - ref;
                    if (d < 0) d = -d;
                    if (d > 0.05f * (ref < 0 ? -ref : ref) + 1e-3f) {
                        if (bad == 0) {
                            float p32 = 0, p64 = 0, p96 = 0;
                            for (int k = 0; k < 32; k++)  p32 += A[i * n + k] * B[k * n + j];
                            for (int k = 0; k < 64; k++)  p64 += A[i * n + k] * B[k * n + j];
                            for (int k = 0; k < 96; k++)  p96 += A[i * n + k] * B[k * n + j];
                            printf("  mismatch [%d][%d]: got %.4f want %.4f (k<32:%.4f k<64:%.4f k<96:%.4f)\n",
                                   i, j, C[i * n + j], ref, p32, p64, p96);
                        } else if (bad < 3)
                            printf("  mismatch [%d][%d]: got %.4f want %.4f\n", i, j, C[i * n + j], ref);
                        bad++; break;
                    }
                }
            printf("sgemm %d: %s\n", n, bad ? "WRONG" : "CORRECT");
            if (naive && n == 256) {  // bisect: naive kernel against same CPU reference
                CHK(cuLaunchKernel(naive, n / 16, n / 16, 1, 16, 16, 1, 0, NULL, args2, NULL));
                CHK(cuCtxSynchronize());
                CHK(cuMemcpyDtoH_v2(C, dC, bytes));
                int nbad = 0;
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++) {
                        float ref = 0;
                        for (int k = 0; k < n; k++) ref += A[i * n + k] * B[k * n + j];
                        float d = C[i * n + j] - ref;
                        if (d < 0) d = -d;
                        if (d > 0.05f * (ref < 0 ? -ref : ref) + 1e-3f) nbad++;
                    }
                printf("sgemm_naive %d: %s (%d bad cells)\n", n, nbad ? "WRONG" : "CORRECT", nbad);
            }
        }
        double t0 = now_s();
        for (int r = 0; r < 10; r++)
            CHK(cuLaunchKernel(sgemm, n / 32, n / 32, 1, 16, 16, 1, 0, NULL, args2, NULL));
        CHK(cuCtxSynchronize());
        double dt = now_s() - t0;
        printf("sgemm %dx%dx%d: %.1f ms/iter, %.1f GFLOPS\n", n, n, n,
               dt * 100.0, 10.0 * 2.0 * n * n * n / dt / 1e9);
        if (sgemm4) {  // 4x4 register-tiled kernel: verify at 256, bench at all sizes
            CHK(cuLaunchKernel(sgemm4, n / 64, n / 64, 1, 16, 16, 1, 0, NULL, args2, NULL));
            CHK(cuCtxSynchronize());
            if (n == 256) {
                CHK(cuMemcpyDtoH_v2(C, dC, bytes));
                int qbad = 0;
                for (int i = 0; i < n && qbad < 5; i++)
                    for (int j = 0; j < n; j++) {
                        float ref = 0;
                        for (int k = 0; k < n; k++) ref += A[i * n + k] * B[k * n + j];
                        float d = C[i * n + j] - ref;
                        if (d < 0) d = -d;
                        if (d > 0.05f * (ref < 0 ? -ref : ref) + 1e-3f) { qbad++; break; }
                    }
                printf("sgemm_tiled4 %d: %s\n", n, qbad ? "WRONG" : "CORRECT");
            }
            double q0 = now_s();
            for (int r = 0; r < 10; r++)
                CHK(cuLaunchKernel(sgemm4, n / 64, n / 64, 1, 16, 16, 1, 0, NULL, args2, NULL));
            CHK(cuCtxSynchronize());
            double qdt = now_s() - q0;
            printf("sgemm4 %dx%dx%d: %.1f ms/iter, %.1f GFLOPS\n", n, n, n,
                   qdt * 100.0, 10.0 * 2.0 * n * n * n / qdt / 1e9);
        }
        if (sgopt) {  // float4/vectorized optimized kernel: verify at 256, bench all sizes
            CHK(cuLaunchKernel(sgopt, n / 64, n / 64, 1, 16, 16, 1, 0, NULL, args2, NULL));
            CHK(cuCtxSynchronize());
            if (n == 256) {
                CHK(cuMemcpyDtoH_v2(C, dC, bytes));
                int obad = 0;
                for (int i = 0; i < n && obad < 5; i++)
                    for (int j = 0; j < n; j++) {
                        float ref = 0;
                        for (int k = 0; k < n; k++) ref += A[i * n + k] * B[k * n + j];
                        float d = C[i * n + j] - ref;
                        if (d < 0) d = -d;
                        if (d > 0.05f * (ref < 0 ? -ref : ref) + 1e-3f) { obad++; break; }
                    }
                printf("sgemm_opt %d: %s\n", n, obad ? "WRONG" : "CORRECT");
            }
            double o0 = now_s();
            for (int r = 0; r < 10; r++)
                CHK(cuLaunchKernel(sgopt, n / 64, n / 64, 1, 16, 16, 1, 0, NULL, args2, NULL));
            CHK(cuCtxSynchronize());
            double odt = now_s() - o0;
            printf("sgemm_opt %dx%dx%d: %.1f ms/iter, %.1f GFLOPS\n", n, n, n,
                   odt * 100.0, 10.0 * 2.0 * n * n * n / odt / 1e9);
        }
        cuMemFree_v2(dA); cuMemFree_v2(dB); cuMemFree_v2(dC);
        free(A); free(B); free(C);
    }

    // ---- memory bandwidth (read) ----
    {
        int n4 = 16 * 1024 * 1024;             // 16M float4 = 256 MB? -> 64 MB
        n4 = 4 * 1024 * 1024;                   // 4M float4 = 64 MB
        size_t bytes = (size_t)n4 * 16;
        float *buf = malloc(bytes);
        for (int i = 0; i < n4 * 4; i++) buf[i] = i * 0.001f;
        CUdeviceptr dbuf, dout;
        CHK(cuMemAlloc_v2(&dbuf, bytes));
        CHK(cuMemAlloc_v2(&dout, (size_t)n4 * 4));
        CHK(cuMemcpyHtoD_v2(dbuf, buf, bytes));
        int iters = 40;
        void *margs[] = { &dbuf, &dout, &n4, &iters };
        CHK(cuLaunchKernel(membw, 512, 1, 1, 256, 1, 1, 0, NULL, margs, NULL));
        CHK(cuCtxSynchronize());
        double t0 = now_s();
        CHK(cuLaunchKernel(membw, 512, 1, 1, 256, 1, 1, 0, NULL, margs, NULL));
        CHK(cuCtxSynchronize());
        double dt = now_s() - t0;
        double gbs = (double)bytes * iters / dt / 1e9;
        printf("membw read (64MB x%d): %.3f s, %.1f GB/s\n", iters, dt, gbs);
        cuMemFree_v2(dbuf); cuMemFree_v2(dout); free(buf);
    }

    printf("WORKLOAD_DONE\n");
    return 0;
}
