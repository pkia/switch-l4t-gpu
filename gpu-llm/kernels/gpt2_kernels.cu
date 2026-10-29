// gpt2_kernels.cu — GPT-2 fp32 decode kernels for Tegra X1 (sm_53).
// Compiled with switch-l4t-gpu's ncc (-nocudainc -nocudalib): everything here is
// self-contained — exp/tanh implemented locally, sqrt/division are native PTX ops.
//
// Fixed GPT-2 124M dims: C=768, NH=12, HD=64, FC=3072, V=50257, CTX=1024.

// ---- self-contained math (libdevice is not linked) ----
static __device__ int dev_floori(float y) {
    int i = (int)y;
    return i - ((y < 0.0f) && ((float)i != y));
}
static __device__ float dev_expf(float x) {
    // exp(x) via 2^(x*log2e), range-clamped, poly on [0,1): rel err ~2e-7
    if (x > 88.0f) x = 88.0f;
    if (x < -87.0f) return 0.0f;
    float y = x * 1.442695041f;
    int e = dev_floori(y);
    float f = y - (float)e;
    float p = 1.3780248e-3f;
    p = p * f + 8.3721884e-3f;
    p = p * f + 4.1669548e-2f;
    p = p * f + 1.6668048e-1f;
    p = p * f + 4.9998952e-1f;
    p = p * f + 1.0f;
    union { float f; int i; } u;
    u.i = (e + 127) << 23;
    return p * u.f;
}
static __device__ float dev_tanhf(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;
    float e2 = dev_expf(2.0f * x);
    return (e2 - 1.0f) / (e2 + 1.0f);
}
static __device__ float dev_fabsf(float x) { return x < 0.0f ? -x : x; }

// ---- matvec: y[n] = sum_k W[n*K + k] * x[k]  (row-major out-major; warp-per-row, coalesced)
extern "C" __global__ void gemv_warp(const float* __restrict__ W,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int K, int N) {
    __shared__ float smem[8 * 32];   // blockDim.x must be 256 (8 warps)
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int nwarp = (gridDim.x * blockDim.x) >> 5;
    // Uniform grid-stride: every thread executes every pass and every barrier.
    // (A per-warp loop condition would make __syncthreads divergent when N % nwarp != 0 — UB.)
    for (int n0 = 0; n0 < N; n0 += nwarp) {
        int n = n0 + warp;
        float acc = 0.0f;
        if (n < N) {
            const float4* row4 = reinterpret_cast<const float4*>(W + (size_t)n * K);
            const float4* x4   = reinterpret_cast<const float4*>(x);
            int q = K >> 2;
            for (int i = lane; i < q; i += 32) {
                float4 w = row4[i];
                float4 v = x4[i];
                acc += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
            }
        }
        smem[(threadIdx.x >> 5) * 32 + lane] = acc;
        __syncthreads();
        if (lane == 0) {
            float s = 0.0f;
            float* p = &smem[(threadIdx.x >> 5) * 32];
            #pragma unroll
            for (int i = 0; i < 32; i++) s += p[i];
            if (n < N) y[n] = s;
        }
        __syncthreads();
    }
}

// ---- gemv + fused bias epilogue (optional --fused path; one launch instead of two)
extern "C" __global__ void gemv_warp_b(const float* __restrict__ W,
                                       const float* __restrict__ x,
                                       const float* __restrict__ bias,
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
            const float4* row4 = reinterpret_cast<const float4*>(W + (size_t)n * K);
            const float4* x4   = reinterpret_cast<const float4*>(x);
            int q = K >> 2;
            for (int i = lane; i < q; i += 32) {
                float4 w = row4[i];
                float4 v = x4[i];
                acc += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
            }
        }
        smem[(threadIdx.x >> 5) * 32 + lane] = acc;
        __syncthreads();
        if (lane == 0) {
            float s = 0.0f;
            float* p = &smem[(threadIdx.x >> 5) * 32];
            #pragma unroll
            for (int i = 0; i < 32; i++) s += p[i];
            if (n < N) y[n] = s + bias[n];
        }
        __syncthreads();
    }
}

// ---- Batched GEMV, weight-stationary: Y[t*N + n] = sum_k W[n*K+k] * X[t*K+k] + (bias?bias[n]:0)
// grid = (ceil(N/8), ceil(T/16)); block 256 = 8 warps; each WARP owns one row n and a
// 16-token chunk: it streams the W row ONCE from DRAM while cycling 16 X rows (L1/L2),
// accumulating 16 register partials, then reduces each via volatile smem (lockstep warps
// on sm_53; no shfl intrinsic under -nocudainc). Weight traffic = ceil(T/16) sweeps.
#define TW 16
extern "C" __global__ void gemv_multi(const float* __restrict__ W,
                                      const float* __restrict__ X,
                                      const float* __restrict__ bias,
                                      float* __restrict__ Y,
                                      int K, int N, int T) {
    __shared__ volatile float wred[8][32];
    int n = blockIdx.x * 8 + (threadIdx.x >> 5);      // this warp's row
    int t0 = blockIdx.y * TW;                         // this warp's 16-token chunk
    int lane = threadIdx.x & 31;
    if (n >= N) return;
    int q = K >> 2;                                   // float4 count per row
    const float4* wrow = reinterpret_cast<const float4*>(W + (size_t)n * K);
    float acc[TW];
    #pragma unroll
    for (int j = 0; j < TW; j++) acc[j] = 0.0f;
    int tmax = T - t0;                                // tokens left in this chunk (>0)
    if (tmax > TW) tmax = TW;
    for (int i = lane; i < q; i += 32) {
        float4 w = wrow[i];
        #pragma unroll
        for (int j = 0; j < TW; j++) {
            if (j < tmax) {
                float4 v = *reinterpret_cast<const float4*>(X + (size_t)(t0 + j) * K + i * 4);
                acc[j] += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
            }
        }
    }
    int w = threadIdx.x >> 5;
    #pragma unroll
    for (int j = 0; j < TW; j++) {
        if (j >= tmax) break;                          // uniform across the warp
        wred[w][lane] = acc[j];
        if (lane < 16) wred[w][lane] += wred[w][lane + 16];
        if (lane < 8)  wred[w][lane] += wred[w][lane + 8];
        if (lane < 4)  wred[w][lane] += wred[w][lane + 4];
        if (lane < 2)  wred[w][lane] += wred[w][lane + 2];
        if (lane < 1)  wred[w][lane] += wred[w][lane + 1];
        if (lane == 0) Y[(size_t)(t0 + j) * N + n] = wred[w][0] + (bias ? bias[n] : 0.0f);
    }
}

// ---- LayerNorm over C=768: one block (256 threads) per ROW.
// Batched: launch grid=T and each block normalizes row blockIdx.x (grid=1 == single token).
// y[i] = (x[i]-mean)/sqrt(var+eps) * w[i] + b[i]
extern "C" __global__ void layernorm(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     const float* __restrict__ b,
                                     float* __restrict__ y, int n, float eps) {
    __shared__ float s_mean, s_var;
    int tid = threadIdx.x;
    x += (size_t)blockIdx.x * n;   // row select (batched prefill: grid=T)
    y += (size_t)blockIdx.x * n;
    float local_sum = 0.0f, local_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        local_sum += x[i];
        local_sq  += x[i] * x[i];
    }
    __shared__ float red[256];
    red[tid] = local_sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) s_mean = red[0] / (float)n;
    __syncthreads();
    red[tid] = local_sq; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        float mean = s_mean;
        float var = red[0] / (float)n - mean * mean;
        if (var < 0.0f) var = 0.0f;
        s_var = var;
    }
    __syncthreads();
    float inv = 1.0f / __builtin_sqrtf(s_var + eps);
    float mean = s_mean;
    for (int i = tid; i < n; i += blockDim.x) y[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// ---- GELU (GPT-2 tanh approximation), elementwise ----
extern "C" __global__ void gelu(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float c = 0.7978845608f;                       // sqrt(2/pi)
    float t = c * (v + 0.044715f * v * v * v);
    y[i] = 0.5f * v * (1.0f + dev_tanhf(t));
}

// ---- elementwise helpers ----
extern "C" __global__ void add_bias(float* __restrict__ y, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[i];
}
extern "C" __global__ void add2(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// ---- attention (decode: single query token, KV cache already includes current token) ----
// scores[t*NH + h] = (q[h*64 .. ] . kcache[t*768 + h*64 ..]) * rsqrt(HD)
extern "C" __global__ void attn_scores(const float* __restrict__ q,
                                       const float* __restrict__ kcache,
                                       float* __restrict__ scores,
                                       int T, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * 12) return;
    int t = idx / 12;
    int h = idx % 12;
    const float* qh = q + h * 64;
    const float* kh = kcache + (size_t)t * 768 + h * 64;
    float acc = 0.0f;
    #pragma unroll 4
    for (int d = 0; d < 64; d++) acc = __builtin_fmaf(qh[d], kh[d], acc);
    scores[idx] = acc * scale;
}

// softmax per head over T scores: in[t*12+h] -> out[t*12+h]; one thread per head
extern "C" __global__ void attn_softmax(const float* __restrict__ scores,
                                        float* __restrict__ probs, int T) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= 12) return;
    float maxv = scores[h];
    for (int t = 1; t < T; t++) {
        float v = scores[(size_t)t * 12 + h];
        if (v > maxv) maxv = v;
    }
    float sum = 0.0f;
    for (int t = 0; t < T; t++) {
        float e = dev_expf(scores[(size_t)t * 12 + h] - maxv);
        probs[(size_t)t * 12 + h] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (int t = 0; t < T; t++) probs[(size_t)t * 12 + h] *= inv;
}

// out[d] = sum_t probs[t*12 + d/64] * vcache[t*768 + d];  one thread per d (768 threads)
extern "C" __global__ void attn_mix(const float* __restrict__ probs,
                                    const float* __restrict__ vcache,
                                    float* __restrict__ out, int T) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= 768) return;
    int h = d / 64;
    float acc = 0.0f;
    const float* vc = vcache + d;
    for (int t = 0; t < T; t++) acc += probs[(size_t)t * 12 + h] * vc[(size_t)t * 768];
    out[d] = acc;
}
