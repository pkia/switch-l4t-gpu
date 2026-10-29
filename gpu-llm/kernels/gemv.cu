// gemv.cu — Milestone 0 kernels for GPU-LLM on Tegra X1 (sm_53).
// y[n] = sum_k W[n*K + k] * x[k], W row-major. FP32. One thread per output row.
extern "C" __global__ void gemv(const float* __restrict__ W,
                                const float* __restrict__ x,
                                float* __restrict__ y,
                                int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* row = W + (size_t)n * K;
    float acc = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < K; k++) acc = __builtin_fmaf(row[k], x[k], acc);
    y[n] = acc;
}

// Warp-per-row variant: 32 lanes stream one row with float4 (fully coalesced),
// partials reduced through shared memory. Grid-stride so any block/warp count works.
extern "C" __global__ void gemv_warp(const float* __restrict__ W,
                                     const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int K, int N) {
    __shared__ float smem[8 * 32];   // blockDim.x must be 256 (8 warps)
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = tid >> 5;
    int lane = threadIdx.x & 31;
    int nwarp = (gridDim.x * blockDim.x) >> 5;
    for (int n = warp; n < N; n += nwarp) {
        const float4* row4 = reinterpret_cast<const float4*>(W + (size_t)n * K);
        const float4* x4   = reinterpret_cast<const float4*>(x);
        int q = K >> 2;
        float acc = 0.0f;
        for (int i = lane; i < q; i += 32) {
            float4 w = row4[i];
            float4 v = x4[i];
            acc += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
        }
        smem[(threadIdx.x >> 5) * 32 + lane] = acc;
        __syncthreads();
        if (lane == 0) {
            float s = 0.0f;
            float* p = &smem[(threadIdx.x >> 5) * 32];
            #pragma unroll
            for (int i = 0; i < 32; i++) s += p[i];
            y[n] = s;
        }
        __syncthreads();   // smem reused next row iteration
    }
}


extern "C" __global__ void gemv4(const float* __restrict__ W,
                                 const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float4* row4 = reinterpret_cast<const float4*>(W + (size_t)n * K);
    const float4* x4   = reinterpret_cast<const float4*>(x);
    int q = K / 4;
    float acc = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < q; i++) {
        float4 w = row4[i];
        float4 xv = x4[i];
        acc = __builtin_fmaf(w.x, xv.x, acc);
        acc = __builtin_fmaf(w.y, xv.y, acc);
        acc = __builtin_fmaf(w.z, xv.z, acc);
        acc = __builtin_fmaf(w.w, xv.w, acc);
    }
    y[n] = acc;
}
