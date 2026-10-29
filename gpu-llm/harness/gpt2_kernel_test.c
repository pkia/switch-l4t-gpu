// gpt2_kernel_test.c — bisects which kernel in gpt2.ptx wedges the GPU.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
extern CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t);
extern CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                               unsigned, unsigned, CUstream, void **, void **);
extern CUresult cuCtxSynchronize(void);
extern CUresult cuGetErrorString(CUresult, const char **);
#define CHK(x) do { CUresult r_ = (x); if (r_) { const char *s_; cuGetErrorString(r_, &s_); \
    fprintf(stderr, "CUDA error %d (%s) at %d\n", r_, s_ ? s_ : "?", __LINE__); exit(1); } } while (0)

static int cC = 768, cT, cV = 50257, cFC = 3072;
static float c_eps = 1e-5f, c_scale = 0.125f;

static void banner(const char *s) { printf("TEST %s\n", s); fflush(stdout); }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gpt2_kernel_test <ptx>\n"); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    FILE *pf = fopen(argv[1], "rb");
    if (!pf) { fprintf(stderr, "no ptx\n"); return 2; }
    fseek(pf, 0, SEEK_END); long n = ftell(pf); fseek(pf, 0, SEEK_SET);
    char *ptx = malloc(n + 1);
    if (fread(ptx, 1, n, pf) != (size_t)n) return 2;
    fclose(pf); ptx[n] = 0;

    CHK(cuInit(0));
    CUdevice dev; CHK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHK(cuCtxCreate_v2(&ctx, 0, dev));
    CUmodule mod; CHK(cuModuleLoadData(&mod, ptx));
    printf("module loaded OK\n"); fflush(stdout);

    static float xf[768], wf[768], bf[768], yf[768], qkvf[2304], scf[8 * 12], pbf[8 * 12], x2f[768];
    for (int i = 0; i < 768; i++) { xf[i] = 0.1f * i - 3.0f; wf[i] = 0.01f * i; bf[i] = -0.02f * i; x2f[i] = 0.5f; }
    for (int i = 0; i < 2304; i++) qkvf[i] = 0.005f * i - 1.0f;
    for (int i = 0; i < 8 * 12; i++) { scf[i] = 0.1f * (i % 7) - 0.3f; pbf[i] = 0.0f; }
    CUdeviceptr d_x, d_w, d_b, d_y, d_qkv, d_scores, d_probs, d_kc, d_vc, d_x2, d_big;
    d_x = 0; d_w = 0; d_b = 0; d_y = 0; d_qkv = 0; d_scores = 0;
    d_probs = 0; d_kc = 0; d_vc = 0; d_x2 = 0; d_big = 0;
    size_t zC = 768 * 4, zT12 = 8 * 12 * 4;
    CHK(cuMemAlloc_v2(&d_x, zC));   CHK(cuMemcpyHtoD_v2(d_x, xf, zC));
    CHK(cuMemAlloc_v2(&d_w, zC));   CHK(cuMemcpyHtoD_v2(d_w, wf, zC));
    CHK(cuMemAlloc_v2(&d_b, zC));   CHK(cuMemcpyHtoD_v2(d_b, bf, zC));
    CHK(cuMemAlloc_v2(&d_y, zC));
    CHK(cuMemAlloc_v2(&d_x2, zC));  CHK(cuMemcpyHtoD_v2(d_x2, x2f, zC));
    CHK(cuMemAlloc_v2(&d_qkv, 2304 * 4)); CHK(cuMemcpyHtoD_v2(d_qkv, qkvf, 2304 * 4));
    CHK(cuMemAlloc_v2(&d_scores, zT12));
    CHK(cuMemAlloc_v2(&d_probs, zT12));
    CHK(cuMemAlloc_v2(&d_kc, (size_t)8 * 768 * 4)); CHK(cuMemAlloc_v2(&d_vc, (size_t)8 * 768 * 4));
    CHK(cuMemcpyHtoD_v2(d_kc, qkvf, (size_t)8 * 768 * 4));
    CHK(cuMemcpyHtoD_v2(d_vc, qkvf, (size_t)8 * 768 * 4));
    int T = 8;

    banner("gemv_warp N=768 (uniform)");
    CUfunction kfn;
    CHK(cuModuleGetFunction(&kfn, mod, "gemv_warp"));
    CHK(cuLaunchKernel(kfn, 96, 1, 1, 256, 1, 1, 0, NULL,
        (void *[]){ &d_w, &d_x, &d_y, &cC, &cC }, NULL));
    CHK(cuCtxSynchronize());
    printf("gemv_warp OK\n"); fflush(stdout);

    banner("gemv_warp N=50257 (lm_head scale, tail exists)");
    d_big = 0; CHK(cuMemAlloc_v2(&d_big, (size_t)50257 * 768 * 4));
    printf("  allocated 154MB, launching...\n"); fflush(stdout);
    CHK(cuLaunchKernel(kfn, (50257 + 7) / 8, 1, 1, 256, 1, 1, 0, NULL,
        (void *[]){ &d_big, &d_x, &d_y, &cC, &cV }, NULL));
    CHK(cuCtxSynchronize());
    printf("lm-head-scale gemv OK\n"); fflush(stdout);

    banner("layernorm");
    CHK(cuModuleGetFunction(&kfn, mod, "layernorm"));
    CHK(cuLaunchKernel(kfn, 1, 1, 1, 256, 1, 1, 0, NULL,
        (void *[]){ &d_x, &d_w, &d_b, &d_y, &cC, &c_eps }, NULL));
    CHK(cuCtxSynchronize());
    CHK(cuMemcpyDtoH_v2(yf, d_y, zC));
    printf("layernorm OK, y[0]=%f y[767]=%f\n", yf[0], yf[767]); fflush(stdout);

    banner("gelu");
    CHK(cuModuleGetFunction(&kfn, mod, "gelu"));
    CHK(cuLaunchKernel(kfn, 3, 1, 1, 256, 1, 1, 0, NULL, (void *[]){ &d_x, &d_y, &cC }, NULL));
    CHK(cuCtxSynchronize());
    printf("gelu OK\n"); fflush(stdout);

    banner("add_bias");
    CHK(cuModuleGetFunction(&kfn, mod, "add_bias"));
    CHK(cuLaunchKernel(kfn, 3, 1, 1, 256, 1, 1, 0, NULL, (void *[]){ &d_x, &d_b, &cC }, NULL));
    CHK(cuCtxSynchronize());
    printf("add_bias OK\n"); fflush(stdout);

    banner("add2");
    CHK(cuModuleGetFunction(&kfn, mod, "add2"));
    CHK(cuLaunchKernel(kfn, 3, 1, 1, 256, 1, 1, 0, NULL,
        (void *[]){ &d_x, &d_x2, &d_y, &cC }, NULL));
    CHK(cuCtxSynchronize());
    printf("add2 OK\n"); fflush(stdout);

    banner("attn_scores T=8");
    CHK(cuModuleGetFunction(&kfn, mod, "attn_scores"));
    CHK(cuLaunchKernel(kfn, 1, 1, 1, 256, 1, 1, 0, NULL,
        (void *[]){ &d_qkv, &d_kc, &d_scores, &T, &c_scale }, NULL));
    CHK(cuCtxSynchronize());
    printf("attn_scores OK\n"); fflush(stdout);

    banner("attn_softmax T=8");
    CHK(cuModuleGetFunction(&kfn, mod, "attn_softmax"));
    CHK(cuLaunchKernel(kfn, 1, 1, 1, 12, 1, 1, 0, NULL, (void *[]){ &d_scores, &d_probs, &T }, NULL));
    CHK(cuCtxSynchronize());
    CHK(cuMemcpyDtoH_v2(pbf, d_probs, zT12));
    float s = 0; for (int i = 0; i < 12; i++) s += pbf[i];
    printf("attn_softmax OK, head0 sum=%f\n", s); fflush(stdout);

    banner("attn_mix T=8");
    CHK(cuModuleGetFunction(&kfn, mod, "attn_mix"));
    CHK(cuLaunchKernel(kfn, 3, 1, 1, 256, 1, 1, 0, NULL,
        (void *[]){ &d_probs, &d_vc, &d_y, &T }, NULL));
    CHK(cuCtxSynchronize());
    printf("attn_mix OK\n"); fflush(stdout);

    printf("ALL_KERNELS_OK\n");
    return 0;
}
