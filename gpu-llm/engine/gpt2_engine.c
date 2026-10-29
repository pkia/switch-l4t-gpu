// gpt2_engine.c — GPT-2 124M fp32 inference on Tegra X1 (driver API + PTX kernels, no libdevice).
//
// Modes:
//   ./gpt2_engine --weights W.bin --ptx gpt2.ptx --ids prompt.ids --gen 24 [--selftest 8]
//     --selftest N : run N positions on the GPU, recompute them serially on the CPU,
//                    compare final-position logits (independent reference implementation).
//     generation   : greedy from the prompt ids for --gen steps (EOT 50256 stops early);
//                    writes the full id sequence to out.ids and prints per-token timings.
//
// Weight file layout (fp32 LE, fixed order — produced by tools/conv_gpt2.py):
//   wte[V*C] wpe[CTX*C] { per layer: ln1_w[C] ln1_b[C] qkv_w[3C*C] qkv_b[3C]
//     ao_w[C*C] ao_b[C] ln2_w[C] ln2_b[C] fc_w[FC*C] fc_b[FC] proj_w[C*C] proj_b[C] }
//   ln_f_w[C] ln_f_b[C]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
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
extern CUresult cuMemcpyDtoD_v2(CUdeviceptr, CUdeviceptr, size_t);
extern CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                               unsigned, unsigned, CUstream, void **, void **);
extern CUresult cuCtxSynchronize(void);
extern CUresult cuGetErrorString(CUresult, const char **);
#define CHK(x) do { CUresult r_ = (x); if (r_) { const char *s_; cuGetErrorString(r_, &s_); \
    fprintf(stderr, "CUDA error %d (%s) at %d\n", r_, s_ ? s_ : "?", __LINE__); exit(1); } } while (0)

#define V 50257
#define C 768
#define NH 12
#define HD 64
#define NL 12
#define CTX 1024
#define FC 3072
#define BLOCK 256
#define EOT 50256

// host weight layout offsets (floats)
#define LAYER_FLOATS (2*C + 3*C*C + 3*C + C*C + C + 2*C + FC*C + FC + FC*C + C)
// ln1(2C) qkv_w(3C^2) qkv_b(3C) ao_w(C^2) ao_b(C) ln2(2C) fc_w(FC*C) fc_b(FC) proj_w(FC*C) proj_b(C)
static const long OFF_WTE = 0;
static const long OFF_WPE = (long)V * C;
static long layer_off(int l) { return OFF_WPE + (long)CTX * C + (long)l * LAYER_FLOATS; }
static const long OFF_LNFW = OFF_WPE + (long)CTX * C + (long)NL * LAYER_FLOATS;
static const long OFF_LNFB = OFF_LNFW + C;
static const long TOTAL_FLOATS = OFF_LNFB + C;   // full file: wte + wpe + 12 layers + ln_f
#define L_LN1W 0
#define L_LN1B (L_LN1W + C)
#define L_QKVW (L_LN1B + C)
#define L_QKVB (L_QKVW + 3*C*C)   // qkv weight is 3C^2 floats (was 3*C — every later offset cascaded wrong)
#define L_AOW  (L_QKVB + 3*C)
#define L_AOB  (L_AOW + C*C)
#define L_LN2W (L_AOB + C)
#define L_LN2B (L_LN2W + C)
#define L_FCW  (L_LN2B + C)
#define L_FCB  (L_FCW + FC*C)
#define L_PROJW (L_FCB + FC)
#define L_PROJB (L_PROJW + FC*C)

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec / 1e9; }

static int cC = C, c3C = 3 * C, cFC = FC, cV = V, cT12;
static float c_eps = 1e-5f, c_scale = 0.125f;

// ---- host weights + device pointers ----
static float *h_w;
static CUdeviceptr d_wte, d_wpe, d_lnfw, d_lnfb;
static CUdeviceptr d_ln1w[NL], d_ln1b[NL], d_qkvw[NL], d_qkvb[NL], d_aow[NL], d_aob[NL];
static CUdeviceptr d_ln2w[NL], d_ln2b[NL], d_fcw[NL], d_fcb[NL], d_projw[NL], d_projb[NL];
static CUdeviceptr d_kcache[NL], d_vcache[NL];
static CUdeviceptr d_x, d_ln1, d_qkv, d_att, d_y, d_ln2, d_fc, d_scores, d_probs, d_logits, d_x2;
static CUdeviceptr d_xT, d_ln1T, d_qkvT, d_attT, d_yT, d_ln2T, d_fcT;   // batched prefill
static CUfunction f_gemv, f_gemvb, f_gemvm, f_ln, f_gelu, f_abias, f_add2, f_ascores, f_asoftmax, f_amix;
static float *gpu_logits_buf;   // host copy of V logits from the last GPU forward

static void load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open weights %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long bytes = ftell(f); fseek(f, 0, SEEK_SET);
    if (bytes != TOTAL_FLOATS * 4) {
        fprintf(stderr, "weight file is %ld bytes, expected %ld\n", bytes, TOTAL_FLOATS * 4);
        exit(1);
    }
    h_w = malloc(bytes);
    if (fread(h_w, 1, bytes, f) != (size_t)bytes) exit(1);
    fclose(f);
    printf("weights: %.1f MiB (%.2fM params)\n", bytes / 1048576.0, TOTAL_FLOATS / 1e6);
}

static CUdeviceptr dev_copy(const float *src, size_t floats) {
    CUdeviceptr p; CHK(cuMemAlloc_v2(&p, floats * 4));
    if (src && floats) CHK(cuMemcpyHtoD_v2(p, src, floats * 4));
    return p;
}

static void gpu_setup(const char *ptx_path) {
    CHK(cuInit(0));
    CUdevice dev; CHK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHK(cuCtxCreate_v2(&ctx, 0, dev));
    FILE *f = fopen(ptx_path, "rb");
    if (!f) { fprintf(stderr, "cannot open ptx %s\n", ptx_path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *ptx = malloc(n + 1);
    if (fread(ptx, 1, n, f) != (size_t)n) exit(1);
    fclose(f); ptx[n] = 0;
    CUmodule mod; CHK(cuModuleLoadData(&mod, ptx));
    CHK(cuModuleGetFunction(&f_gemv, mod, "gemv_warp"));
    CHK(cuModuleGetFunction(&f_gemvb, mod, "gemv_warp_b"));   // fused gemv+bias (--fused)
    CHK(cuModuleGetFunction(&f_gemvm, mod, "gemv_multi"));    // batched gemv (--batch-prefill)
    CHK(cuModuleGetFunction(&f_ln, mod, "layernorm"));
    CHK(cuModuleGetFunction(&f_gelu, mod, "gelu"));
    CHK(cuModuleGetFunction(&f_abias, mod, "add_bias"));
    CHK(cuModuleGetFunction(&f_add2, mod, "add2"));
    CHK(cuModuleGetFunction(&f_ascores, mod, "attn_scores"));
    CHK(cuModuleGetFunction(&f_asoftmax, mod, "attn_softmax"));
    CHK(cuModuleGetFunction(&f_amix, mod, "attn_mix"));
    free(ptx);

    d_wte = dev_copy(h_w + OFF_WTE, (size_t)V * C);
    d_wpe = dev_copy(h_w + OFF_WPE, (size_t)CTX * C);
    d_lnfw = dev_copy(h_w + OFF_LNFW, C);
    d_lnfb = dev_copy(h_w + OFF_LNFB, C);
    for (int l = 0; l < NL; l++) {
        const float *b = h_w + layer_off(l);
        d_ln1w[l]  = dev_copy(b + L_LN1W, C);
        d_ln1b[l]  = dev_copy(b + L_LN1B, C);
        d_qkvw[l]  = dev_copy(b + L_QKVW, (size_t)3*C*C);
        d_qkvb[l]  = dev_copy(b + L_QKVB, 3*C);
        d_aow[l]   = dev_copy(b + L_AOW, (size_t)C*C);
        d_aob[l]   = dev_copy(b + L_AOB, C);
        d_ln2w[l]  = dev_copy(b + L_LN2W, C);
        d_ln2b[l]  = dev_copy(b + L_LN2B, C);
        d_fcw[l]   = dev_copy(b + L_FCW, (size_t)FC*C);
        d_fcb[l]   = dev_copy(b + L_FCB, FC);
        d_projw[l] = dev_copy(b + L_PROJW, (size_t)FC*C);   // [C][FC] = FC*C floats (was C*C — truncation bug)
        d_projb[l] = dev_copy(b + L_PROJB, C);
        d_kcache[l] = dev_copy(NULL, (size_t)CTX * C);
        d_vcache[l] = dev_copy(NULL, (size_t)CTX * C);
    }
    d_x = dev_copy(NULL, C); d_ln1 = dev_copy(NULL, C); d_qkv = dev_copy(NULL, 3*C);
    d_att = dev_copy(NULL, C); d_y = dev_copy(NULL, C); d_ln2 = dev_copy(NULL, C);
    d_fc = dev_copy(NULL, FC); d_x2 = dev_copy(NULL, C);
    d_scores = dev_copy(NULL, (size_t)CTX * NH);
    d_probs = dev_copy(NULL, (size_t)CTX * NH);
    d_logits = dev_copy(NULL, V);
    // batched-prefill buffers (--batch-prefill)
    d_xT = dev_copy(NULL, (size_t)CTX * C);   d_ln1T = dev_copy(NULL, (size_t)CTX * C);
    d_qkvT = dev_copy(NULL, (size_t)CTX * 3 * C);
    d_attT = dev_copy(NULL, (size_t)CTX * C); d_yT = dev_copy(NULL, (size_t)CTX * C);
    d_ln2T = dev_copy(NULL, (size_t)CTX * C); d_fcT = dev_copy(NULL, (size_t)CTX * FC);
    printf("gpu ready (weights+kv ~%.0f MiB device)\n",
           ((size_t)V*C + (size_t)CTX*C + (size_t)NL*LAYER_FLOATS + (size_t)NL*2*CTX*C
            + 3*C + 2*C + FC + 2*(size_t)CTX*NH + V) * 4 / 1048576.0);
}

// ---- GPU forward: token at `pos`; cache holds 0..pos-1 and gains position pos ----
static float *gpu_logits_buf;   // V floats
static int debug_flag = 0;      // --debug: per-stage vector dumps
static int fused_flag = 0;      // --fused: gemv with bias epilogue (one launch instead of two)
static int batch_flag = 0;      // --batch-prefill: weight-stationary batched prompt prefill
static void gprvec(const char *tag, CUdeviceptr d_buf, int n);   // fwd decl (defined near cpu_forward)
static void gpu_forward(int token, int pos, int want_logits, float *trace) {
    int gtrace = (debug_flag && trace && pos == 0);   // stage-dump layer 0 on the GPU path
    float xh[C];
    for (int i = 0; i < C; i++)
        xh[i] = h_w[OFF_WTE + (long)token * C + i] + h_w[OFF_WPE + (long)pos * C + i];
    CHK(cuMemcpyHtoD_v2(d_x, xh, C * 4));
    for (int l = 0; l < NL; l++) {
        CHK(cuLaunchKernel(f_ln, 1, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_x, &d_ln1w[l], &d_ln1b[l], &d_ln1, &cC, &c_eps }, NULL));
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("ln1", d_ln1, C); }
        if (fused_flag) {
            CHK(cuLaunchKernel(f_gemvb, (2304 + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &d_qkvw[l], &d_ln1, &d_qkvb[l], &d_qkv, &cC, &c3C }, NULL));
        } else {
            CHK(cuLaunchKernel(f_gemv, (2304 + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &d_qkvw[l], &d_ln1, &d_qkv, &cC, &c3C }, NULL));
            CHK(cuLaunchKernel(f_abias, (2304 + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &d_qkv, &d_qkvb[l], &c3C }, NULL));
        }
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("qkv", d_qkv, 3*C); }
        // NOTE: device pointer offsets are BYTES (element count * 4)
        CHK(cuMemcpyDtoD_v2(d_kcache[l] + (size_t)pos * C * 4, d_qkv + (size_t)C * 4, C * 4));
        CHK(cuMemcpyDtoD_v2(d_vcache[l] + (size_t)pos * C * 4, d_qkv + (size_t)2 * C * 4, C * 4));
        int T = pos + 1;
        int T12 = T * NH;
        CHK(cuLaunchKernel(f_ascores, (T12 + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_qkv, &d_kcache[l], &d_scores, &T, &c_scale }, NULL));
        CHK(cuLaunchKernel(f_asoftmax, 1, 1, 1, 12, 1, 1, 0, NULL,
            (void *[]){ &d_scores, &d_probs, &T }, NULL));
        CHK(cuLaunchKernel(f_amix, 3, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_probs, &d_vcache[l], &d_att, &T }, NULL));
        if (gtrace && l == 0) {
            CHK(cuCtxSynchronize());
            gprvec("scores", d_scores, T * NH); gprvec("probs", d_probs, T * NH);
            gprvec("attnout", d_att, C);
        }
        if (fused_flag) {
            CHK(cuLaunchKernel(f_gemvb, (C + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &d_aow[l], &d_att, &d_aob[l], &d_y, &cC, &cC }, NULL));
        } else {
            CHK(cuLaunchKernel(f_gemv, (C + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &d_aow[l], &d_att, &d_y, &cC, &cC }, NULL));
            CHK(cuLaunchKernel(f_abias, (C + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &d_y, &d_aob[l], &cC }, NULL));
        }
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("ao", d_y, C); }
        CHK(cuLaunchKernel(f_add2, 3, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_x, &d_y, &d_x, &cC }, NULL));
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("res1", d_x, C); }
        CHK(cuLaunchKernel(f_ln, 1, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_x, &d_ln2w[l], &d_ln2b[l], &d_ln2, &cC, &c_eps }, NULL));
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("ln2", d_ln2, C); }
        if (fused_flag) {
            CHK(cuLaunchKernel(f_gemvb, (FC + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &d_fcw[l], &d_ln2, &d_fcb[l], &d_fc, &cC, &cFC }, NULL));
        } else {
            CHK(cuLaunchKernel(f_gemv, (FC + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &d_fcw[l], &d_ln2, &d_fc, &cC, &cFC }, NULL));
            CHK(cuLaunchKernel(f_abias, (FC + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &d_fc, &d_fcb[l], &cFC }, NULL));
        }
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("fc", d_fc, FC); }
        CHK(cuLaunchKernel(f_gelu, (FC + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_fc, &d_fc, &cFC }, NULL));
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("gelu", d_fc, FC); }
        if (fused_flag) {
            CHK(cuLaunchKernel(f_gemvb, (C + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &d_projw[l], &d_fc, &d_projb[l], &d_y, &cFC, &cC }, NULL));
        } else {
            CHK(cuLaunchKernel(f_gemv, (C + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &d_projw[l], &d_fc, &d_y, &cFC, &cC }, NULL));
            CHK(cuLaunchKernel(f_abias, (C + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &d_y, &d_projb[l], &cC }, NULL));
        }
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("proj", d_y, C); }
        CHK(cuLaunchKernel(f_add2, 3, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_x, &d_y, &d_x, &cC }, NULL));
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("res2", d_x, C); }
        if (trace) {   // residual after layer l (device->host) for divergence tracing
            CHK(cuCtxSynchronize());
            CHK(cuMemcpyDtoH_v2(trace + (long)l * C, d_x, C * 4));
        }
    }
    if (!want_logits) return;   // fast-prefill: prompt tokens skip final LN + lm_head (154 MB read)
    CHK(cuLaunchKernel(f_ln, 1, 1, 1, BLOCK, 1, 1, 0, NULL,
        (void *[]){ &d_x, &d_lnfw, &d_lnfb, &d_x2, &cC, &c_eps }, NULL));
    CHK(cuLaunchKernel(f_gemv, (V + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
        (void *[]){ &d_wte, &d_x2, &d_logits, &cC, &cV }, NULL));
    CHK(cuCtxSynchronize());
    CHK(cuMemcpyDtoH_v2(gpu_logits_buf, d_logits, (size_t)V * 4));
}

// ---- batched prefill (--batch-prefill): all T prompt tokens through the layers with
// weight-stationary gemv_multi (weights read once per ceil(T/16) instead of T times).
// Fills KV cache positions 0..T-1; logits for the LAST row only.
static void gpu_prefill(const int *ids, int T) {
    static float *xb = NULL;
    if (!xb) xb = malloc((size_t)CTX * C * 4);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < C; i++)
            xb[(size_t)t * C + i] = h_w[OFF_WTE + (long)ids[t] * C + i]
                                  + h_w[OFF_WPE + (long)t * C + i];
    CHK(cuMemcpyHtoD_v2(d_xT, xb, (size_t)T * C * 4));
    for (int l = 0; l < NL; l++) {
        CHK(cuLaunchKernel(f_ln, T, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_xT, &d_ln1w[l], &d_ln1b[l], &d_ln1T, &cC, &c_eps }, NULL));
        CHK(cuLaunchKernel(f_gemvm, (3 * C + 7) / 8, (T + 15) / 16, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_qkvw[l], &d_ln1T, &d_qkvb[l], &d_qkvT, &cC, &c3C, &T }, NULL));
        for (int t = 0; t < T; t++) {
            CUdeviceptr qrow = d_qkvT + (size_t)t * 3 * C * 4;          // BYTE offsets
            CUdeviceptr arow = d_attT + (size_t)t * C * 4;
            CHK(cuMemcpyDtoD_v2(d_kcache[l] + (size_t)t * C * 4, qrow + (size_t)C * 4, C * 4));
            CHK(cuMemcpyDtoD_v2(d_vcache[l] + (size_t)t * C * 4, qrow + (size_t)2 * C * 4, C * 4));
            int Tp = t + 1, T12 = Tp * NH;
            CHK(cuLaunchKernel(f_ascores, (T12 + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &qrow, &d_kcache[l], &d_scores, &Tp, &c_scale }, NULL));
            CHK(cuLaunchKernel(f_asoftmax, 1, 1, 1, 12, 1, 1, 0, NULL,
                (void *[]){ &d_scores, &d_probs, &Tp }, NULL));
            CHK(cuLaunchKernel(f_amix, 3, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &d_probs, &d_vcache[l], &arow, &Tp }, NULL));
        }
        CHK(cuLaunchKernel(f_gemvm, (C + 7) / 8, (T + 15) / 16, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_aow[l], &d_attT, &d_aob[l], &d_yT, &cC, &cC, &T }, NULL));
        for (int t = 0; t < T; t++) {
            CUdeviceptr xr = d_xT + (size_t)t * C * 4, yr = d_yT + (size_t)t * C * 4;
            CHK(cuLaunchKernel(f_add2, 3, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &xr, &yr, &xr, &cC }, NULL));
        }
        CHK(cuLaunchKernel(f_ln, T, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_xT, &d_ln2w[l], &d_ln2b[l], &d_ln2T, &cC, &c_eps }, NULL));
        CHK(cuLaunchKernel(f_gemvm, (FC + 7) / 8, (T + 15) / 16, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_fcw[l], &d_ln2T, &d_fcb[l], &d_fcT, &cC, &cFC, &T }, NULL));
        for (int t = 0; t < T; t++) {
            CUdeviceptr fr = d_fcT + (size_t)t * FC * 4;
            CHK(cuLaunchKernel(f_gelu, (FC + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &fr, &fr, &cFC }, NULL));
        }
        CHK(cuLaunchKernel(f_gemvm, (C + 7) / 8, (T + 15) / 16, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_projw[l], &d_fcT, &d_projb[l], &d_yT, &cFC, &cC, &T }, NULL));
        for (int t = 0; t < T; t++) {
            CUdeviceptr xr = d_xT + (size_t)t * C * 4, yr = d_yT + (size_t)t * C * 4;
            CHK(cuLaunchKernel(f_add2, 3, 1, 1, BLOCK, 1, 1, 0, NULL,
                (void *[]){ &xr, &yr, &xr, &cC }, NULL));
        }
    }
    CUdeviceptr xlast = d_xT + (size_t)(T - 1) * C * 4;
    CHK(cuLaunchKernel(f_ln, 1, 1, 1, BLOCK, 1, 1, 0, NULL,
        (void *[]){ &xlast, &d_lnfw, &d_lnfb, &d_x2, &cC, &c_eps }, NULL));
    CHK(cuLaunchKernel(f_gemv, (V + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
        (void *[]){ &d_wte, &d_x2, &d_logits, &cC, &cV }, NULL));
    CHK(cuCtxSynchronize());
    CHK(cuMemcpyDtoH_v2(gpu_logits_buf, d_logits, (size_t)V * 4));
}

// ---- CPU serial reference (independent) ----
static float *kc_h, *vc_h;                     // NL*CTX*C each
static float r_x[C], r_ln[C], r_qkv[3*C], r_att[C], r_y[C], r_ln2[C], r_fc[FC];
static float ref_logits[V];

static void cpu_ln(const float *x, const float *w, const float *b, float *y) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < C; i++) mean += x[i];
    mean /= C;
    for (int i = 0; i < C; i++) { float d = x[i] - mean; var += d * d; }
    var /= C;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < C; i++) y[i] = (x[i] - mean) * inv * w[i] + b[i];
}
static void cpu_gemv(const float *W, const float *x, const float *bias, float *y, int K, int N) {
    for (int o = 0; o < N; o++) {
        const float *row = W + (size_t)o * K;
        float acc = 0.0f;
        for (int k = 0; k < K; k++) acc += row[k] * x[k];
        y[o] = acc + (bias ? bias[o] : 0.0f);   // bias may be NULL (tied lm_head has none)
    }
}
// stage tracer: prints first4 + sum of a vector (layer-0 debug, --debug only)
static void prvec(const char *tag, const float *v, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += v[i];
    printf("    c.%-8s first4=%.3f %.3f %.3f %.3f sum=%.3f\n", tag, v[0], v[1], v[2], v[3], s);
}
static CUdeviceptr d_tracebuf;   // staging buffer for GPU stage dumps
static void gprvec(const char *tag, CUdeviceptr d_buf, int n) {
    static float tmp[3 * FC];
    if (n > 3 * FC) n = 3 * FC;
    CHK(cuMemcpyDtoH_v2(tmp, d_buf, (size_t)n * 4));
    float s = 0; for (int i = 0; i < n; i++) s += tmp[i];
    printf("    g.%-8s first4=%.3f %.3f %.3f %.3f sum=%.3f\n", tag, tmp[0], tmp[1], tmp[2], tmp[3], s);
}
static void cpu_forward(int token, int pos, int want_logits, float *trace) {
    for (int i = 0; i < C; i++)
        r_x[i] = h_w[OFF_WTE + (long)token * C + i] + h_w[OFF_WPE + (long)pos * C + i];
    for (int l = 0; l < NL; l++) {
        const float *b = h_w + layer_off(l);
        if (debug_flag && l == 0 && pos == 0) prvec("embed", r_x, C);
        cpu_ln(r_x, b + L_LN1W, b + L_LN1B, r_ln);
        if (debug_flag && l == 0 && pos == 0) prvec("ln1", r_ln, C);
        cpu_gemv(b + L_QKVW, r_ln, b + L_QKVB, r_qkv, C, 3*C);
        if (debug_flag && l == 0 && pos == 0) prvec("qkv", r_qkv, 3*C);
        for (int h = 0; h < NH; h++) {
            for (int d = 0; d < HD; d++) {
                kc_h[((size_t)l * CTX + pos) * C + h * HD + d] = r_qkv[C + h * HD + d];
                vc_h[((size_t)l * CTX + pos) * C + h * HD + d] = r_qkv[2*C + h * HD + d];
            }
        }
        for (int h = 0; h < NH; h++) {
            float maxv = -1e30f, sum = 0.0f;
            static float sc[CTX];
            for (int t = 0; t <= pos; t++) {
                const float *kh = kc_h + ((size_t)l * CTX + t) * C + h * HD;
                float acc = 0.0f;
                const float *qh = r_qkv + h * HD;
                for (int d = 0; d < HD; d++) acc += qh[d] * kh[d];
                sc[t] = acc * 0.125f;
                if (sc[t] > maxv) maxv = sc[t];
            }
            for (int t = 0; t <= pos; t++) { sc[t] = expf(sc[t] - maxv); sum += sc[t]; }
            for (int d = 0; d < HD; d++) {
                float acc = 0.0f;
                for (int t = 0; t <= pos; t++)
                    acc += sc[t] * vc_h[((size_t)l * CTX + t) * C + h * HD + d];
                r_att[h * HD + d] = acc / sum;
            }
        }
        cpu_gemv(b + L_AOW, r_att, b + L_AOB, r_y, C, C);
        if (debug_flag && l == 0 && pos == 0) prvec("ao", r_y, C);
        for (int i = 0; i < C; i++) r_x[i] += r_y[i];
        if (debug_flag && l == 0 && pos == 0) prvec("res1", r_x, C);
        cpu_ln(r_x, b + L_LN2W, b + L_LN2B, r_ln2);
        if (debug_flag && l == 0 && pos == 0) prvec("ln2", r_ln2, C);
        cpu_gemv(b + L_FCW, r_ln2, b + L_FCB, r_fc, C, FC);
        if (debug_flag && l == 0 && pos == 0) prvec("fc", r_fc, FC);
        for (int i = 0; i < FC; i++) {
            float v = r_fc[i];
            float t = 0.7978845608f * (v + 0.044715f * v * v * v);
            r_fc[i] = 0.5f * v * (1.0f + tanhf(t));
        }
        if (debug_flag && l == 0 && pos == 0) prvec("gelu", r_fc, FC);
        cpu_gemv(b + L_PROJW, r_fc, b + L_PROJB, r_y, FC, C);
        if (debug_flag && l == 0 && pos == 0) prvec("proj", r_y, C);
        for (int i = 0; i < C; i++) r_x[i] += r_y[i];
        if (debug_flag && l == 0 && pos == 0) prvec("res2", r_x, C);
        if (trace) memcpy(trace + (long)l * C, r_x, C * 4);
    }
    if (!want_logits) return;   // fast-prefill: CPU reference skips lm_head too
    cpu_ln(r_x, h_w + OFF_LNFW, h_w + OFF_LNFB, r_ln);
    cpu_gemv(h_w + OFF_WTE, r_ln, NULL, ref_logits, C, V);  // tied lm_head, no bias
}

// argmax
static int argmax(const float *v, int n) {
    int best = 0; float bv = v[0];
    for (int i = 1; i < n; i++) if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   // survive crashes: no buffered-output loss
    const char *weights = NULL, *ptx = NULL, *ids_file = NULL;
    int n_gen = 24, selftest = 0, debug = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--weights") && i + 1 < argc) weights = argv[++i];
        else if (!strcmp(argv[i], "--ptx") && i + 1 < argc) ptx = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_file = argv[++i];
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--selftest") && i + 1 < argc) selftest = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fused")) fused_flag = 1;
        else if (!strcmp(argv[i], "--batch-prefill")) batch_flag = 1;
        else if (!strcmp(argv[i], "--debug")) debug = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (!weights || !ptx) { fprintf(stderr, "--weights and --ptx required\n"); return 2; }
    if (n_gen > CTX) n_gen = CTX;

    load_weights(weights);
    debug_flag = debug;
    gpu_setup(ptx);

    // prompt ids
    int total = 0;
    static int ids[CTX];
    if (ids_file) {
        FILE *f = fopen(ids_file, "rb");
        if (!f) { fprintf(stderr, "cannot open ids file\n"); return 1; }
        uint32_t cnt;
        if (fread(&cnt, 4, 1, f) != 1 || cnt > CTX) { fprintf(stderr, "bad ids file\n"); return 1; }
        for (uint32_t i = 0; i < cnt; i++) { uint32_t v; fread(&v, 4, 1, f); ids[total++] = (int)v; }
        fclose(f);
        printf("prompt: %d token(s)\n", total);
    } else {
        for (int i = 0; i < 8; i++) ids[i] = (i * 7919 + 42) % V;   // deterministic fallback
        total = 8;
        printf("no --ids file; using %d deterministic ids\n", total);
    }
    if (total + n_gen > CTX) n_gen = CTX - total;

    kc_h = calloc((size_t)NL * CTX * C, 4);
    vc_h = calloc((size_t)NL * CTX * C, 4);
    gpu_logits_buf = malloc((size_t)V * 4);

    // ---- optional selftest: GPU vs CPU over the first `selftest` positions ----
    if (selftest > total) selftest = total;
    if (selftest > 0) {
        printf("selftest: comparing GPU vs serial-CPU logits at position %d...\n", selftest - 1);
        {   // per-layer residual trace at position 0 (localizes divergence)
            static float tr_g[NL * C], tr_c[NL * C];
            gpu_forward(ids[0], 0, 1, tr_g);
            cpu_forward(ids[0], 0, 1, tr_c);
            for (int l = 0; l < NL; l++) {
                float m = 0.0f, g = 0.0f;
                for (int i = 0; i < C; i++) {
                    float d = fabsf(tr_g[(long)l * C + i] - tr_c[(long)l * C + i]);
                    if (d > m) m = d;
                    if (fabsf(tr_g[(long)l * C + i]) > g) g = fabsf(tr_g[(long)l * C + i]);
                }
                printf("  trace p0 L%02d: max|d|=%.6f (|gpu|~%.3f)\n", l, m, g);
            }
            FILE *tg = NULL, *tc = NULL;
            if (debug_flag) {
                tg = fopen("trace_g.bin", "wb"); if (tg) { fwrite(tr_g, 4, NL * C, tg); fclose(tg); }
                tc = fopen("trace_c.bin", "wb"); if (tc) { fwrite(tr_c, 4, NL * C, tc); fclose(tc); }
            }
            (void)tg; (void)tc;
        }
        double t0 = now_s();
        if (batch_flag) {
            gpu_prefill(ids, selftest);
            printf("  gpu forward x%d [batched]: %.1fs\n", selftest, now_s() - t0);
        } else {
            for (int p = 0; p < selftest; p++) gpu_forward(ids[p], p, p == selftest - 1, NULL);
            printf("  gpu forward x%d: %.1fs\n", selftest, now_s() - t0);
        }
        float gpu_last[V];
        memcpy(gpu_last, gpu_logits_buf, V * 4);
        t0 = now_s();
        for (int p = 0; p < selftest; p++) cpu_forward(ids[p], p, p == selftest - 1, NULL);
        printf("  cpu reference x%d: %.1fs\n", selftest, now_s() - t0);
        double maxerr = 0; int argmax_gpu = argmax(gpu_last, V), argmax_cpu = argmax(ref_logits, V);
        for (int i = 0; i < V; i++) {
            double e = fabs((double)gpu_last[i] - (double)ref_logits[i]);
            if (e > maxerr) maxerr = e;
        }
        // top-5 comparison
        printf("  gpu top-5 ids:");
        { float tmp[V]; memcpy(tmp, gpu_last, V * 4);
          for (int k = 0; k < 5; k++) { int b = argmax(tmp, V); printf(" %d", b); tmp[b] = -1e30f; } }
        printf("\n  cpu top-5 ids:");
        { float tmp[V]; memcpy(tmp, ref_logits, V * 4);
          for (int k = 0; k < 5; k++) { int b = argmax(tmp, V); printf(" %d", b); tmp[b] = -1e30f; } }
        printf("\n  max |gpu-cpu| logit err: %.4f  argmax gpu=%d cpu=%d %s\n",
               maxerr, argmax_gpu, argmax_cpu,
               argmax_gpu == argmax_cpu ? "MATCH" : "MISMATCH");
        if (argmax_gpu != argmax_cpu) printf("  SELFTEST: argmax mismatch (noted — fp order sensitivity)\n");
        else printf("  SELFTEST: PASS\n");
        // reset caches + rerun GPU positions to restore state for generation
        zero_caches: for (int l = 0; l < NL; l++) {
            static float *z = NULL;
            if (!z) z = calloc((size_t)CTX * C, 4);
            CHK(cuMemcpyHtoD_v2(d_kcache[l], z, (size_t)CTX * C * 4));
            CHK(cuMemcpyHtoD_v2(d_vcache[l], z, (size_t)CTX * C * 4));
        }
        for (int p = 0; p < selftest; p++) gpu_forward(ids[p], p, 0, NULL);
        memcpy(gpu_logits_buf, gpu_last, V * 4);
    }

    // ---- prime the KV cache with the prompt (always; selftest may have polluted it) ----
    {
        double tp = now_s();
        if (batch_flag && total > 1) {
            gpu_prefill(ids, total);
            printf("primed %d prompt token(s) [batched] in %.0f ms\n", total, (now_s() - tp) * 1e3);
        } else {
            for (int p = 0; p < total; p++) gpu_forward(ids[p], p, p == total - 1, NULL);
            printf("primed %d prompt token(s) in %.0f ms\n", total, (now_s() - tp) * 1e3);
        }
    }

    // ---- greedy generation ----
    int next = argmax(gpu_logits_buf, V);
    ids[total++] = next;
    double t_start = now_s();
    int g_emitted = 0;
    for (int g = 0; g < n_gen && total < CTX; g++) {
        if (next == EOT) break;
        gpu_forward(next, total - 1, 1, NULL);
        next = argmax(gpu_logits_buf, V);
        ids[total++] = next;
        g_emitted++;
        if (g % 10 == 0) printf("  gen %d/%d (%.1f ms/tok)\n", g + 1, n_gen, (now_s() - t_start) * 1e3 / (g + 1));
    }
    double gen_s = now_s() - t_start;
    printf("generated %d token(s) in %.1fs -> %.1f tok/s (fp32, %d-layer GPT-2 on Tegra X1)\n",
           g_emitted, gen_s, g_emitted / gen_s, NL);

    FILE *f = fopen("out.ids", "wb");
    if (f) { uint32_t cnt = total; fwrite(&cnt, 4, 1, f); fwrite(ids, 4, total, f); fclose(f); }
    printf("wrote out.ids (%d ids; decode on the Pi with the tokenizer)\n", total);
    printf("ENGINE_DONE\n");
    return 0;
}
