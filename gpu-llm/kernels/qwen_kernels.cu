// qwen_kernels.cu — Qwen2.5 (Llama-family) kernels for Tegra X1 via ncc -> PTX.
// FP16 weights (manual fp16->fp32 decode; no cuda_fp16.h under -nocudainc), FP32 activations.
// No libdevice: self-contained math (dev_expf), no shfl intrinsics (volatile smem, lockstep warps).

// ---- fp16 bits -> float (weights are never subnormal/inf after our bf16->fp16 conversion)
static __device__ float h2f_dev(unsigned short h) {
    unsigned int e = (h >> 10) & 0x1fu;
    if (e == 0) return 0.0f;
    union { unsigned int u; float f; } cvt;
    cvt.u = ((unsigned int)(h & 0x8000u) << 16)
          | ((e - 15u + 127u) << 23)
          | ((unsigned int)(h & 0x3ffu) << 13);
    return cvt.f;
}

// ---- dev_expf: 2^(x*log2e) poly, rel err ~2e-7 (same as gpt2_kernels)
static __device__ int dev_floori(float x) { return (int)(x - (x < 0.0f ? 0.999999f : 0.0f)); }
static __device__ float dev_expf(float x) {
    if (x > 88.0f) x = 88.0f;
    if (x < -87.0f) return 0.0f;
    float y = x * 1.4426950409f;
    int e = dev_floori(y);
    float f = y - (float)e;
    union { unsigned int u; float f; } sc;
    sc.u = (unsigned int)(e + 127u) << 23;
    // 2^f via 4-term Taylor of e^(f*ln2): max err ~0.08% at f=1 (old poly was ~30% off!)
    float p = 1.0f + f * (0.6931472f + f * (0.2402265f + f * (0.0555041f + f * 0.0096181f)));
    return sc.f * p;
}

// ---- GEMV, fp16 weights: y[n] = sum_k h2f(W[n*K+k]) * x[k]  (row-major [out,in])
// warp-per-row, float4 x loads, uint2 (4 halves) w loads. K % 4 == 0 required.
// grid (ceil(N/8)); block 256 (8 warps); barriers uniform (q grid-uniform).
extern "C" __global__ void gemv_h(const unsigned short* __restrict__ W,
                                  const float* __restrict__ x,
                                  const float* __restrict__ bias,   // nullable (unused for Qwen)
                                  float* __restrict__ y,
                                  int K, int N) {
    __shared__ float smem[8 * 32];
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int nwarp = (gridDim.x * blockDim.x) >> 5;
    for (int n0 = 0; n0 < N; n0 += nwarp) {
        int n = n0 + warp;
        float acc = 0.0f;
        if (n < N) {
            const uint2* w4 = reinterpret_cast<const uint2*>(W + (size_t)n * K);
            const float4* x4 = reinterpret_cast<const float4*>(x);
            int qi = K >> 2;
            for (int i = lane; i < qi; i += 32) {
                uint2 wp = w4[i];
                float4 v = x4[i];
                acc += h2f_dev((unsigned short)(wp.x & 0xffffu)) * v.x
                     + h2f_dev((unsigned short)(wp.x >> 16))      * v.y
                     + h2f_dev((unsigned short)(wp.y & 0xffffu)) * v.z
                     + h2f_dev((unsigned short)(wp.y >> 16))      * v.w;
            }
        }
        smem[(threadIdx.x >> 5) * 32 + lane] = acc;
        __syncthreads();
        if (lane == 0) {
            float s = 0.0f;
            float* p = &smem[(threadIdx.x >> 5) * 32];
            #pragma unroll
            for (int i = 0; i < 32; i++) s += p[i];
            if (n < N) y[n] = s + (bias ? bias[n] : 0.0f);
        }
        __syncthreads();
    }
}

// ---- RMSNorm, one block (256 threads) per ROW (batched via grid=T).
// y = x * rsqrt(mean(x^2) + eps) * w   (w is FP16, decoded)
extern "C" __global__ void rmsnorm(const float* __restrict__ x,
                                   const unsigned short* __restrict__ w,
                                   float* __restrict__ y, int n, float eps) {
    __shared__ float s_ss;
    int tid = threadIdx.x;
    x += (size_t)blockIdx.x * n;
    y += (size_t)blockIdx.x * n;
    float local = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) local += x[i] * x[i];
    __shared__ float red[256];
    red[tid] = local; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) s_ss = red[0] / (float)n;
    __syncthreads();
    float inv = 1.0f / __builtin_sqrtf(s_ss + eps);
    for (int i = tid; i < n; i += blockDim.x) y[i] = x[i] * inv * h2f_dev(w[i]);
}

// ---- RoPE with precomputed cos/sin tables: grid covers T*H*(HD/2) pairs.
// x rows are [T, H*HD]; pair (i, i+HD/2) within head h of row t rotates by angle[t0+t][i].
extern "C" __global__ void rope(float* __restrict__ x,
                                const float* __restrict__ cosT,
                                const float* __restrict__ sinT,
                                int T, int H, int HD, int pos0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hd2 = HD / 2;
    int total = T * H * hd2;
    if (idx >= total) return;
    int t = idx / (H * hd2);
    int r = idx % (H * hd2);
    int h = r / hd2;
    int i = r % hd2;
    float* row = x + (size_t)t * H * HD;
    float a = row[h * HD + i];
    float b = row[h * HD + hd2 + i];
    float c = cosT[(size_t)(pos0 + t) * hd2 + i];
    float s = sinT[(size_t)(pos0 + t) * hd2 + i];
    row[h * HD + i] = a * c - b * s;
    row[h * HD + hd2 + i] = a * s + b * c;
}

// ---- SwiGLU: y[i] = silu(g[i]) * u[i], silu(x) = x / (1 + exp(-x))
extern "C" __global__ void swiglu(const float* __restrict__ g,
                                  const float* __restrict__ u,
                                  float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i];
    y[i] = gv / (1.0f + dev_expf(-gv)) * u[i];
}

// ---- GQA attention scores: scores[t*NH + h] = q[h*HD..] . kc[t*KV + g*HD..] * scale
// g = h / NG, NG = NH/NKV. threads = Tp*NH.
extern "C" __global__ void attn_scores_gqa(const float* __restrict__ q,
                                           const float* __restrict__ kc,
                                           float* __restrict__ scores,
                                           int T, float scale, int NH, int KV, int HD) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * NH) return;
    int t = idx / NH;
    int h = idx % NH;
    int g = h / (NH / (KV / HD));   // kv head group; KV/HD == NKV
    const float* qp = q + h * HD;
    const float* kp = kc + (size_t)t * KV + g * HD;
    float acc = 0.0f;
    for (int d = 0; d < HD; d++) acc += qp[d] * kp[d];
    scores[idx] = acc * scale;
}

// ---- softmax over t for each head: threads NH (one launch per position)
extern "C" __global__ void attn_softmax(const float* __restrict__ scores,
                                        float* __restrict__ probs, int T, int NH) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= NH) return;
    float maxv = -1e30f;
    for (int t = 0; t < T; t++) {
        float v = scores[(size_t)t * NH + h];
        if (v > maxv) maxv = v;
    }
    float sum = 0.0f;
    for (int t = 0; t < T; t++) {
        float ev = dev_expf(scores[(size_t)t * NH + h] - maxv);
        probs[(size_t)t * NH + h] = ev;
        sum += ev;
    }
    float inv = 1.0f / sum;
    for (int t = 0; t < T; t++) probs[(size_t)t * NH + h] *= inv;
}

// ---- GQA mix: out[h*HD + d] = sum_t probs[t*NH + h] * vc[t*KV + g*HD + d]
extern "C" __global__ void attn_mix_gqa(const float* __restrict__ probs,
                                        const float* __restrict__ vc,
                                        float* __restrict__ out, int T,
                                        int NH, int KV, int HD) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NH * HD) return;
    int h = idx / HD;
    int d = idx % HD;
    int g = h / (NH / (KV / HD));
    float acc = 0.0f;
    for (int t = 0; t < T; t++)
        acc += probs[(size_t)t * NH + h] * vc[(size_t)t * KV + g * HD + d];
    out[idx] = acc;
}

// ---- residual add: out[i] = a[i] + b[i]
extern "C" __global__ void add2(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}
