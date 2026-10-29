// qwen_engine.c — Qwen2.5-0.5B-Instruct (Llama-family) inference engine for Tegra X1.
// Same driver-API harness as gpt2_engine: cuModuleLoadData(PTX) + cuLaunchKernel.
//
// Architecture: RMSNorm (eps 1e-6) | RoPE (theta 1e6, tables in weights) | GQA attention
// (NH=14 q heads, NKV=2 kv heads, HD=64) | SwiGLU MLP (FC=4864) | tied embeddings.
// Weights FP16 (manual decode), activations/KV FP32.
//
// Weight file layout (bytes) — produced by tools/conv_qwen.py:
//   OFF_EMB  = 0                          V*C*2        fp16
//   OFF_COS  = EMB_end                    CTX*32*4     fp32
//   OFF_SIN  = COS_end                    CTX*32*4     fp32
//   L_BASE   = SIN_end, stride LSTRIDE    per layer (fp16):
//     in_ln [C] | qkv_w [QKV,C] (q|k|v) | o_w [C,C] | post_ln [C]
//     gateup_w [2FC,C] (gate|up) | down_w [C,FC]
//   OFF_LNFW = L_BASE + L*LSTRIDE         C*2          fp16
//   EOF      = OFF_LNFW + C*2
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
extern CUresult cuGetErrorString(CUresult, const char *);
#define CHK(x) do { CUresult r_ = (x); if (r_) { const char *s_; cuGetErrorString(r_, &s_); \
    fprintf(stderr, "CUDA error %d (%s) at %d\n", r_, s_ ? s_ : "?", __LINE__); exit(1); } } while (0)

#define V 151936
#define C 896
#define NL 24
#define NH 14
#define NKV 2
#define HD 64
#define KVD (NKV * HD)          // 128
#define QKVD (C + 2 * KVD)      // 1152
#define FC 4864
#define CTX 2048
#define BLOCK 256

static const float EPS = 1e-6f, SCALE = 0.125f;
static const int EOT = 151643;       // <|endoftext|>
static const int IM_END = 151645;    // <|im_end|> (instruct turn terminator)

// ---- file offsets (bytes)
#define OFF_EMB 0L
#define OFF_COS (OFF_EMB + (long)V * C * 2)
#define OFF_SIN (OFF_COS + (long)CTX * (HD / 2) * 4)
#define L_BASE (OFF_SIN + (long)CTX * (HD / 2) * 4)
#define LSTRIDE ((long)(C + QKVD * C + QKVD + C * C + C + 2 * FC * C + C * FC) * 2)
#define OFF_LNFW (L_BASE + (long)NL * LSTRIDE)
#define FILE_BYTES (OFF_LNFW + (long)C * 2)
// per-layer fp16 element offsets
#define LI_INLN 0L
#define LI_QKVW (C)
#define LI_QKVB ((long)C + (long)QKVD * C)
#define LI_OW (LI_QKVB + QKVD)
#define LI_POSTLN (LI_OW + (long)C * C)
#define LI_GATEUP (LI_POSTLN + C)
#define LI_DOWN (LI_GATEUP + (long)2 * FC * C)

static unsigned short *h_w;     // whole weight file (fp16)
static const float *h_cos, *h_sin;

static CUdeviceptr d_x, d_ln1, d_qkv, d_att, d_y, d_ln2, d_gate, d_down, d_scores, d_probs;
static CUdeviceptr d_logits, d_x2, d_cos, d_sin, d_emb, d_lnfw;
static CUdeviceptr d_ln1w[NL], d_qkvw[NL], d_qkvb[NL], d_ow[NL], d_ln2w[NL], d_gateupw[NL], d_downw[NL];
static CUdeviceptr d_kc[NL], d_vc[NL];
static CUfunction f_gemv, f_rms, f_rope, f_swiglu, f_ascores, f_asoftmax, f_amix, f_add2;
static float *gpu_logits_buf, *ref_logits;
static int debug_flag = 0;
static float *null_bias = NULL;   // gemv_h bias slot (Qwen has no biases; lm_head has none)

// ---- fp16 bits -> float
static float h2f(unsigned short h) {
    unsigned int e = (h >> 10) & 0x1fu;
    if (e == 0) return 0.0f;
    union { unsigned int u; float f; } cvt;
    cvt.u = ((unsigned int)(h & 0x8000u) << 16)
          | ((e - 15u + 127u) << 23)
          | ((unsigned int)(h & 0x3ffu) << 13);
    return cvt.f;
}

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz != FILE_BYTES) {
        fprintf(stderr, "weight file is %ld bytes, expected %ld\n", sz, FILE_BYTES);
        exit(2);
    }
    fseek(f, 0, SEEK_SET);
    h_w = malloc(sz);
    if (fread(h_w, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); exit(2); }
    fclose(f);
    h_cos = (const float *)((const char *)h_w + OFF_COS);
    h_sin = (const float *)((const char *)h_w + OFF_SIN);
    printf("weights: %.1f MiB (%d-layer Qwen2.5-0.5B, fp16)\n", sz / 1048576.0, NL);
}

static CUdeviceptr dev_copy_f(const void *src, size_t floats) {
    CUdeviceptr p = 0;
    CHK(cuMemAlloc_v2(&p, floats * 4));
    if (src) CHK(cuMemcpyHtoD_v2(p, src, floats * 4));
    return p;
}
static CUdeviceptr dev_copy_h(const void *src, size_t halves) {
    CUdeviceptr p = 0;
    CHK(cuMemAlloc_v2(&p, halves * 2));
    if (src) CHK(cuMemcpyHtoD_v2(p, src, halves * 2));
    return p;
}

static void gpu_setup(const char *ptx_path) {
    FILE *f = fopen(ptx_path, "rb");
    if (!f) { fprintf(stderr, "cannot open ptx\n"); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *ptx = malloc(n + 1);
    if (fread(ptx, 1, n, f) != (size_t)n) exit(2);
    fclose(f); ptx[n] = 0;
    CHK(cuInit(0));
    CUdevice dev; CHK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHK(cuCtxCreate_v2(&ctx, 0, dev));
    CUmodule mod; CHK(cuModuleLoadData(&mod, ptx));
    CHK(cuModuleGetFunction(&f_gemv, mod, "gemv_h"));
    CHK(cuModuleGetFunction(&f_rms, mod, "rmsnorm"));
    CHK(cuModuleGetFunction(&f_rope, mod, "rope"));
    CHK(cuModuleGetFunction(&f_swiglu, mod, "swiglu"));
    CHK(cuModuleGetFunction(&f_ascores, mod, "attn_scores_gqa"));
    CHK(cuModuleGetFunction(&f_asoftmax, mod, "attn_softmax"));
    CHK(cuModuleGetFunction(&f_amix, mod, "attn_mix_gqa"));
    CHK(cuModuleGetFunction(&f_add2, mod, "add2"));
    d_x = dev_copy_f(NULL, C); d_ln1 = dev_copy_f(NULL, C);
    d_qkv = dev_copy_f(NULL, QKVD); d_att = dev_copy_f(NULL, C);
    d_y = dev_copy_f(NULL, C); d_ln2 = dev_copy_f(NULL, C);
    d_gate = dev_copy_f(NULL, 2 * FC); d_down = dev_copy_f(NULL, FC);
    d_scores = dev_copy_f(NULL, (size_t)CTX * NH); d_probs = dev_copy_f(NULL, (size_t)CTX * NH);
    d_logits = dev_copy_f(NULL, V); d_x2 = dev_copy_f(NULL, C);
    d_cos = dev_copy_f(h_cos, (size_t)CTX * (HD / 2));
    d_sin = dev_copy_f(h_sin, (size_t)CTX * (HD / 2));
    d_emb = dev_copy_h(h_w + OFF_EMB / 2, (size_t)V * C);
    d_lnfw = dev_copy_h(h_w + OFF_LNFW / 2, C);
    for (int l = 0; l < NL; l++) {
        const unsigned short *b = h_w + L_BASE / 2 + (size_t)l * (LSTRIDE / 2);
        d_ln1w[l] = dev_copy_h(b + LI_INLN, C);
        d_qkvw[l] = dev_copy_h(b + LI_QKVW, (size_t)QKVD * C);
        {   // q/k/v bias: decode fp16 -> fp32 on host, upload fp32 (kernel reads float bias)
            static float tmp[QKVD];
            const unsigned short *src = b + LI_QKVB;
            for (int i = 0; i < QKVD; i++) tmp[i] = h2f(src[i]);
            d_qkvb[l] = dev_copy_f(tmp, QKVD);
        }
        d_ow[l] = dev_copy_h(b + LI_OW, (size_t)C * C);
        d_ln2w[l] = dev_copy_h(b + LI_POSTLN, C);
        d_gateupw[l] = dev_copy_h(b + LI_GATEUP, (size_t)2 * FC * C);
        d_downw[l] = dev_copy_h(b + LI_DOWN, (size_t)C * FC);
        d_kc[l] = dev_copy_f(NULL, (size_t)CTX * KVD);
        d_vc[l] = dev_copy_f(NULL, (size_t)CTX * KVD);
    }
    gpu_logits_buf = malloc((size_t)V * 4);
    ref_logits = malloc((size_t)V * 4);
    printf("gpu ready (weights+kv ~%.0f MiB device)\n",
           (FILE_BYTES + (size_t)NL * 2 * CTX * KVD * 4) / 1048576.0);
}

// ---- stage tracers (--debug)
static void prvec(const char *tag, const float *v, int n) {
    float s = 0; for (int i = 0; i < n; i++) s += v[i];
    printf("    c.%-8s first4=%.3f %.3f %.3f %.3f sum=%.3f\n", tag, v[0], v[1], v[2], v[3], s);
}
static void gprvec(const char *tag, CUdeviceptr d_buf, int n) {
    static float tmp[2 * FC];
    if (n > 2 * FC) n = 2 * FC;
    CHK(cuMemcpyDtoH_v2(tmp, d_buf, (size_t)n * 4));
    float s = 0; for (int i = 0; i < n; i++) s += tmp[i];
    printf("    g.%-8s first4=%.3f %.3f %.3f %.3f sum=%.3f\n", tag, tmp[0], tmp[1], tmp[2], tmp[3], s);
}

// ---- GPU decode forward: token at `pos` (cache holds 0..pos-1, gains pos)
static void gpu_forward(int token, int pos, int want_logits) {
    int gtrace = (debug_flag && pos == 0);
    static float *xb = NULL;
    if (!xb) xb = malloc((size_t)C * 4);
    const unsigned short *emb = h_w + OFF_EMB / 2;
    for (int i = 0; i < C; i++) xb[i] = h2f(emb[(size_t)token * C + i]);
    CHK(cuMemcpyHtoD_v2(d_x, xb, C * 4));
    int cC = C, cQKV = QKVD, cFC = FC, cNH = NH, cKV = KVD, cHD = HD, cV = V, cNKV = NKV;
    int c2FC = 2 * FC;
    int one = 1, Tp = pos + 1, pos0 = pos;
    for (int l = 0; l < NL; l++) {
        CHK(cuLaunchKernel(f_rms, 1, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_x, &d_ln1w[l], &d_ln1, &cC, &EPS }, NULL));
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("ln1", d_ln1, C); }
        CHK(cuLaunchKernel(f_gemv, (QKVD + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_qkvw[l], &d_ln1, &d_qkvb[l], &d_qkv, &cC, &cQKV }, NULL));
        // RoPE on q (NH heads) then k (NKV heads); v untouched
        CHK(cuLaunchKernel(f_rope, (NH * (HD / 2) + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_qkv, &d_cos, &d_sin, &one, &cNH, &cHD, &pos0 }, NULL));
        CUdeviceptr dk = d_qkv + (size_t)NH * HD * 4;                     // BYTE offsets
        CHK(cuLaunchKernel(f_rope, (NKV * (HD / 2) + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &dk, &d_cos, &d_sin, &one, &cNKV, &cHD, &pos0 }, NULL));
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("qkv", d_qkv, QKVD); }
        // cache write (BYTE offsets)
        CHK(cuMemcpyDtoD_v2(d_kc[l] + (size_t)pos * KVD * 4, d_qkv + (size_t)NH * HD * 4, KVD * 4));
        CHK(cuMemcpyDtoD_v2(d_vc[l] + (size_t)pos * KVD * 4, d_qkv + (size_t)(NH * HD + KVD) * 4, KVD * 4));
        CHK(cuLaunchKernel(f_ascores, (Tp * NH + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_qkv, &d_kc[l], &d_scores, &Tp, &SCALE, &cNH, &cKV, &cHD }, NULL));
        CHK(cuLaunchKernel(f_asoftmax, 1, 1, 1, NH, 1, 1, 0, NULL,
            (void *[]){ &d_scores, &d_probs, &Tp, &cNH }, NULL));
        CHK(cuLaunchKernel(f_amix, (NH * HD + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_probs, &d_vc[l], &d_att, &Tp, &cNH, &cKV, &cHD }, NULL));
        if (gtrace && l == 0) {
            CHK(cuCtxSynchronize());
            gprvec("scores", d_scores, Tp * NH); gprvec("probs", d_probs, Tp * NH);
            gprvec("attnout", d_att, C);
        }
        CHK(cuLaunchKernel(f_gemv, (C + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_ow[l], &d_att, &null_bias, &d_y, &cC, &cC }, NULL));
        CHK(cuLaunchKernel(f_add2, (C + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_x, &d_y, &d_x, &cC }, NULL));
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("res1", d_x, C); }
        CHK(cuLaunchKernel(f_rms, 1, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_x, &d_ln2w[l], &d_ln2, &cC, &EPS }, NULL));
        CHK(cuLaunchKernel(f_gemv, (2 * FC + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_gateupw[l], &d_ln2, &null_bias, &d_gate, &cC, &c2FC }, NULL));
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("gate", d_gate, FC); }
        CUdeviceptr du = d_gate + (size_t)FC * 4;                          // up half (BYTE offset)
        CHK(cuLaunchKernel(f_swiglu, (FC + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_gate, &du, &d_down, &cFC }, NULL));
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("swiglu", d_down, FC); }
        CHK(cuLaunchKernel(f_gemv, (C + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_downw[l], &d_down, &null_bias, &d_y, &cFC, &cC }, NULL));
        CHK(cuLaunchKernel(f_add2, (C + 255) / 256, 1, 1, BLOCK, 1, 1, 0, NULL,
            (void *[]){ &d_x, &d_y, &d_x, &cC }, NULL));
        if (gtrace && l == 0) { CHK(cuCtxSynchronize()); gprvec("res2", d_x, C); }
    }
    if (!want_logits) return;
    CHK(cuLaunchKernel(f_rms, 1, 1, 1, BLOCK, 1, 1, 0, NULL,
        (void *[]){ &d_x, &d_lnfw, &d_x2, &cC, &EPS }, NULL));
    CHK(cuLaunchKernel(f_gemv, (V + 7) / 8, 1, 1, BLOCK, 1, 1, 0, NULL,
        (void *[]){ &d_emb, &d_x2, &null_bias, &d_logits, &cC, &cV }, NULL));
    CHK(cuCtxSynchronize());
    CHK(cuMemcpyDtoH_v2(gpu_logits_buf, d_logits, (size_t)V * 4));
}

// ---- CPU serial reference (independent fp32 math over the same fp16 weights)
static float *kc_h, *vc_h;
static float sc[CTX];   // per-head score scratch
static float r_x[C], r_ln[C], r_qkv[QKVD], r_att[C], r_y[C], r_ln2[C], r_gate[2 * FC], r_down[FC];

static void cpu_gemv_h(const unsigned short *W, const float *x, const unsigned short *bias,
                       float *y, int K, int N) {
    for (int o = 0; o < N; o++) {
        const unsigned short *row = W + (size_t)o * K;
        float acc = 0.0f;
        for (int k = 0; k < K; k++) acc += h2f(row[k]) * x[k];
        y[o] = acc + (bias ? h2f(bias[o]) : 0.0f);
    }
}
static void cpu_rms(const float *x, const unsigned short *w, float *y, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + EPS);
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * h2f(w[i]);
}

static void cpu_forward(int token, int pos, int want_logits, float *trace) {
    const unsigned short *emb = h_w + OFF_EMB / 2;
    for (int i = 0; i < C; i++) r_x[i] = h2f(emb[(size_t)token * C + i]);
    for (int l = 0; l < NL; l++) {
        const unsigned short *b = h_w + L_BASE / 2 + (size_t)l * (LSTRIDE / 2);
        if (debug_flag && l == 0 && pos == 0) prvec("embed", r_x, C);
        cpu_rms(r_x, b + LI_INLN, r_ln, C);
        if (debug_flag && l == 0 && pos == 0) prvec("ln1", r_ln, C);
        cpu_gemv_h(b + LI_QKVW, r_ln, b + LI_QKVB, r_qkv, C, QKVD);   // q/k/v biases
        if (debug_flag && l == 0 && pos == 0) prvec("qkv", r_qkv, QKVD);
        // RoPE q
        for (int h = 0; h < NH; h++)
            for (int i = 0; i < HD / 2; i++) {
                float a = r_qkv[h * HD + i], b2 = r_qkv[h * HD + HD / 2 + i];
                float c = h_cos[(size_t)pos * (HD / 2) + i], s = h_sin[(size_t)pos * (HD / 2) + i];
                r_qkv[h * HD + i] = a * c - b2 * s;
                r_qkv[h * HD + HD / 2 + i] = a * s + b2 * c;
            }
        // RoPE k
        for (int g = 0; g < NKV; g++)
            for (int i = 0; i < HD / 2; i++) {
                float a = r_qkv[C + g * HD + i], b2 = r_qkv[C + g * HD + HD / 2 + i];
                float c = h_cos[(size_t)pos * (HD / 2) + i], s = h_sin[(size_t)pos * (HD / 2) + i];
                r_qkv[C + g * HD + i] = a * c - b2 * s;
                r_qkv[C + g * HD + HD / 2 + i] = a * s + b2 * c;
            }
        // cache
        for (int i = 0; i < KVD; i++) {
            kc_h[((size_t)l * CTX + pos) * KVD + i] = r_qkv[C + i];
            vc_h[((size_t)l * CTX + pos) * KVD + i] = r_qkv[C + KVD + i];
        }
        // attention (GQA)
        for (int h = 0; h < NH; h++) {
            int g = h / (NH / NKV);
            float maxv = -1e30f;
            for (int t = 0; t <= pos; t++) {
                float acc = 0.0f;
                for (int d = 0; d < HD; d++)
                    acc += r_qkv[h * HD + d] * kc_h[((size_t)l * CTX + t) * KVD + g * HD + d];
                sc[t] = acc * SCALE;
                if (sc[t] > maxv) maxv = sc[t];
            }
            float sum = 0.0f;
            for (int t = 0; t <= pos; t++) { sc[t] = expf(sc[t] - maxv); sum += sc[t]; }
            for (int d = 0; d < HD; d++) {
                float acc = 0.0f;
                for (int t = 0; t <= pos; t++)
                    acc += (sc[t] / sum) * vc_h[((size_t)l * CTX + t) * KVD + g * HD + d];
                r_att[h * HD + d] = acc;
            }
        }
        cpu_gemv_h(b + LI_OW, r_att, NULL, r_y, C, C);
        if (debug_flag && l == 0 && pos == 0) prvec("ao", r_y, C);
        for (int i = 0; i < C; i++) r_x[i] += r_y[i];
        if (debug_flag && l == 0 && pos == 0) prvec("res1", r_x, C);
        cpu_rms(r_x, b + LI_POSTLN, r_ln2, C);
        cpu_gemv_h(b + LI_GATEUP, r_ln2, NULL, r_gate, C, 2 * FC);
        for (int i = 0; i < FC; i++)
            r_down[i] = (r_gate[i] / (1.0f + expf(-r_gate[i]))) * r_gate[FC + i];
        cpu_gemv_h(b + LI_DOWN, r_down, NULL, r_y, FC, C);
        for (int i = 0; i < C; i++) r_x[i] += r_y[i];
        if (trace) memcpy(trace + (size_t)l * C, r_x, C * 4);
    }
    if (!want_logits) return;
    cpu_rms(r_x, h_w + OFF_LNFW / 2, r_ln, C);
    cpu_gemv_h(h_w + OFF_EMB / 2, r_ln, NULL, ref_logits, C, V);   // tied lm_head
}

static int argmax(const float *v, int n) {
    int b = 0; float m = v[0];
    for (int i = 1; i < n; i++) if (v[i] > m) { m = v[i]; b = i; }
    return b;
}

// ---- sampling per Qwen2.5 generation_config.json: temperature 0.7, top_k 20, top_p 0.8,
// repetition_penalty 1.1 (greedy decoding is documented-degenerate for this model family).
static int greedy_mode = 0;   // --greedy: old argmax behaviour
static unsigned int rng_state = 1234;
static float dev_expf_host(float x);
static float frand(void) {    // xorshift, deterministic per run
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return (rng_state & 0xffffff) / 16777216.0f;
}
static int sample_token(const float *logits, const int *ctx, int ctx_len) {
    static float w[V];
    memcpy(w, logits, (size_t)V * 4);
    for (int i = 0; i < ctx_len; i++) {            // repetition penalty
        int t = ctx[i];
        if (w[t] > 0) w[t] /= 1.1f; else w[t] *= 1.1f;
    }
    for (int i = 0; i < V; i++) w[i] /= 0.7f;      // temperature
    int idx[20];
    float val[20];
    for (int k = 0; k < 20; k++) {                 // top_k = 20 (partial selection)
        int b = 0; for (int i = 1; i < V; i++) if (w[i] > w[b]) b = i;
        idx[k] = b; val[k] = w[b]; w[b] = -1e30f;
    }
    float mx = val[0];
    float sum = 0.0f;
    for (int k = 0; k < 20; k++) { val[k] = dev_expf_host(val[k] - mx); sum += val[k]; }
    for (int k = 0; k < 20; k++) val[k] /= sum;    // softmax
    float cum = 0.0f; int keep = 20;               // top_p = 0.8
    for (int k = 0; k < 20; k++) { cum += val[k]; if (cum >= 0.8f) { keep = k + 1; break; } }
    float r = frand() * ((cum < 0.8f) ? cum : 0.8f);
    cum = 0.0f;
    for (int k = 0; k < keep; k++) { cum += val[k]; if (r <= cum) return idx[k]; }
    return idx[0];
}
static float dev_expf_host(float x) {
    if (x > 88.0f) x = 88.0f;
    if (x < -87.0f) return 0.0f;
    return expf(x);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *weights = NULL, *ptx = NULL, *ids_file = NULL;
    int n_gen = 24, selftest = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--weights") && i + 1 < argc) weights = argv[++i];
        else if (!strcmp(argv[i], "--ptx") && i + 1 < argc) ptx = argv[++i];
        else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_file = argv[++i];
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--selftest") && i + 1 < argc) selftest = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--debug")) debug_flag = 1;
        else if (!strcmp(argv[i], "--greedy")) greedy_mode = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (!weights || !ptx) { fprintf(stderr, "--weights and --ptx required\n"); return 2; }
    if (n_gen > CTX) n_gen = CTX;

    load_weights(weights);
    gpu_setup(ptx);
    kc_h = calloc((size_t)NL * CTX * KVD, 4);
    vc_h = calloc((size_t)NL * CTX * KVD, 4);

    // ---- prompt ids
    static int ids[CTX + CTX];
    int total = 0;
    if (ids_file) {
        FILE *f = fopen(ids_file, "rb");
        if (!f) { fprintf(stderr, "cannot open ids\n"); return 2; }
        uint32_t cnt;
        if (fread(&cnt, 4, 1, f) != 1) return 2;
        if (cnt > CTX) cnt = CTX;
        if (fread(ids, 4, cnt, f) != cnt) return 2;
        fclose(f);
        total = cnt;
        printf("prompt: %d token(s)\n", total);
    } else {
        for (int i = 0; i < 8; i++) ids[i] = (i * 7919 + 42) % V;
        total = 8;
        printf("no --ids file; using %d deterministic ids\n", total);
    }

    // ---- optional selftest (GPU vs CPU at the last selftested position)
    if (selftest > total) selftest = total;
    if (selftest > 0) {
        printf("selftest: comparing GPU vs serial-CPU logits at position %d...\n", selftest - 1);
        double t0 = now_s();
        for (int p = 0; p < selftest; p++) gpu_forward(ids[p], p, p == selftest - 1);
        printf("  gpu forward x%d: %.1fs\n", selftest, now_s() - t0);
        static float gpu_last[V];
        memcpy(gpu_last, gpu_logits_buf, V * 4);
        t0 = now_s();
        for (int p = 0; p < selftest; p++) cpu_forward(ids[p], p, p == selftest - 1, NULL);
        printf("  cpu reference x%d: %.1fs\n", selftest, now_s() - t0);
        double maxerr = 0; int ag = argmax(gpu_last, V), ac = argmax(ref_logits, V);
        for (int i = 0; i < V; i++) {
            double e = fabs((double)gpu_last[i] - (double)ref_logits[i]);
            if (e > maxerr) maxerr = e;
        }
        printf("  gpu top-5 ids:");
        { float tmp[V]; memcpy(tmp, gpu_last, V * 4);
          for (int k = 0; k < 5; k++) { int b = argmax(tmp, V); printf(" %d", b); tmp[b] = -1e30f; } }
        printf("\n  cpu top-5 ids:");
        { float tmp[V]; memcpy(tmp, ref_logits, V * 4);
          for (int k = 0; k < 5; k++) { int b = argmax(tmp, V); printf(" %d", b); tmp[b] = -1e30f; } }
        printf("\n  max |gpu-cpu| logit err: %.4f  argmax gpu=%d cpu=%d %s\n",
               maxerr, ag, ac, ag == ac ? "MATCH" : "MISMATCH");
        printf(ag == ac ? "  SELFTEST: PASS\n" : "  SELFTEST: FAIL\n");
        // restore GPU state: caches from sequential GPU pass are already at positions
        // 0..selftest-1; continue with the remaining prompt below (if any).
    }

    // ---- prime cache with the full prompt (batched path skipped for v1)
    {
        double tp = now_s();
        if (selftest >= total && selftest > 0) {
            printf("primed %d prompt token(s) in %.0f ms [via selftest pass]\n", total, (now_s() - tp) * 1e3);
        } else {
            for (int p = 0; p < total; p++) gpu_forward(ids[p], p, p == total - 1);
            printf("primed %d prompt token(s) in %.0f ms\n", total, (now_s() - tp) * 1e3);
        }
    }

    // ---- greedy generation (or sampled per Qwen generation_config)
    int next = greedy_mode ? argmax(gpu_logits_buf, V)
                           : sample_token(gpu_logits_buf, ids, total);
    ids[total++] = next;
    double t_start = now_s();
    int g_emitted = 0;
    for (int g = 0; g < n_gen && total < CTX; g++) {
        if (next == EOT || next == IM_END) break;
        gpu_forward(next, total - 1, 1);
        next = greedy_mode ? argmax(gpu_logits_buf, V)
                           : sample_token(gpu_logits_buf, ids, total);
        ids[total++] = next;
        g_emitted++;
        if (g % 10 == 0) printf("  gen %d/%d (%.1f ms/tok)\n", g + 1, n_gen, (now_s() - t_start) * 1e3 / (g + 1));
    }
    double gen_s = now_s() - t_start;
    printf("generated %d token(s) in %.1fs -> %.1f tok/s (fp16 weights, %d-layer Qwen2.5-0.5B on Tegra X1)\n",
           g_emitted, gen_s, g_emitted / gen_s, NL);

    FILE *f = fopen("out.ids", "wb");
    if (f) { uint32_t cnt = total; fwrite(&cnt, 4, 1, f); fwrite(ids, 4, total, f); fclose(f); }
    printf("wrote out.ids (%d ids; decode on the Pi with the tokenizer)\n", total);
    printf("ENGINE_DONE\n");
    return 0;
}
