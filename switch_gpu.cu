// GPU kernels for Nintendo Switch (Tegra X1, sm_53) — compiled to PTX with CUDA 10.0 nvcc.
// Tiled SGEMM (shared memory, 2x2 per thread) + float4 memory-bandwidth read.
extern "C" {

__global__ void sgemm_tiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
  __shared__ float As[32][32];
  __shared__ float Bs[32][32];
  int tx = threadIdx.x, ty = threadIdx.y;
  int rowBase = blockIdx.y * 32;
  int colBase = blockIdx.x * 32;
  int row0 = rowBase + ty * 2;
  int col0 = colBase + tx * 2;
  float c00 = 0.f, c01 = 0.f, c10 = 0.f, c11 = 0.f;
  for (int t = 0; t < K; t += 32) {
    As[ty*2][tx]       = A[(size_t)(rowBase + ty*2)     * K + t + tx];
    As[ty*2][tx+16]    = A[(size_t)(rowBase + ty*2)     * K + t + tx + 16];
    As[ty*2+1][tx]     = A[(size_t)(rowBase + ty*2 + 1) * K + t + tx];
    As[ty*2+1][tx+16]  = A[(size_t)(rowBase + ty*2 + 1) * K + t + tx + 16];
    Bs[ty*2][tx]       = B[(size_t)(t + ty*2)     * N + colBase + tx];
    Bs[ty*2][tx+16]    = B[(size_t)(t + ty*2)     * N + colBase + tx + 16];
    Bs[ty*2+1][tx]     = B[(size_t)(t + ty*2 + 1) * N + colBase + tx];
    Bs[ty*2+1][tx+16]  = B[(size_t)(t + ty*2 + 1) * N + colBase + tx + 16];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
      float a0 = As[ty*2][k], a1 = As[ty*2+1][k];
      float b0 = Bs[k][tx*2], b1 = Bs[k][tx*2+1];
      c00 += a0*b0; c01 += a0*b1; c10 += a1*b0; c11 += a1*b1;
    }
    __syncthreads();
  }
  C[(size_t)row0     * N + col0]     = c00;
  C[(size_t)row0     * N + col0 + 1] = c01;
  C[(size_t)(row0+1) * N + col0]     = c10;
  C[(size_t)(row0+1) * N + col0 + 1] = c11;
}

__global__ void membw_read4(const float4* __restrict__ buf,
                            float* __restrict__ out,
                            int n4, int iters) {
  int i0 = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  float acc = 0.f;
  for (int it = 0; it < iters; ++it) {
    for (int i = i0; i < n4; i += stride) {
      float4 v = buf[i];
      acc += v.x + v.y + v.z + v.w;
    }
  }
  if (i0 < n4) out[i0] = acc;
}

// 4x4 register tiling: 16x16 threads, 64x64 block tile, K-chunk 64.
// 16 FMAs per 8 shared loads -> ~2x the arithmetic intensity of the 2x2 version.
// Requires M, N, K multiples of 64.
__global__ void sgemm_tiled4(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K) {
  __shared__ float As[64][64];
  __shared__ float Bs[64][64];
  int tx = threadIdx.x, ty = threadIdx.y;          // 16x16 threads
  int rowBase = blockIdx.y * 64 + ty * 4;          // this thread's 4 rows
  int colBase = blockIdx.x * 64 + tx * 4;          // this thread's 4 cols
  float acc[4][4] = {};
  for (int t = 0; t < K; t += 64) {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        As[ty*4 + r][tx*4 + c] = A[(size_t)(rowBase + r) * K + t + tx*4 + c];
        Bs[ty*4 + r][tx*4 + c] = B[(size_t)(t + ty*4 + r) * N + colBase + c];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 64; ++k) {
      float a0 = As[ty*4+0][k], a1 = As[ty*4+1][k], a2 = As[ty*4+2][k], a3 = As[ty*4+3][k];
      float b0 = Bs[k][tx*4+0], b1 = Bs[k][tx*4+1], b2 = Bs[k][tx*4+2], b3 = Bs[k][tx*4+3];
      acc[0][0] += a0*b0; acc[0][1] += a0*b1; acc[0][2] += a0*b2; acc[0][3] += a0*b3;
      acc[1][0] += a1*b0; acc[1][1] += a1*b1; acc[1][2] += a1*b2; acc[1][3] += a1*b3;
      acc[2][0] += a2*b0; acc[2][1] += a2*b1; acc[2][2] += a2*b2; acc[2][3] += a2*b3;
      acc[3][0] += a3*b0; acc[3][1] += a3*b1; acc[3][2] += a3*b2; acc[3][3] += a3*b3;
    }
    __syncthreads();
  }
  #pragma unroll
  for (int r = 0; r < 4; ++r)
    #pragma unroll
    for (int c = 0; c < 4; ++c)
      C[(size_t)(rowBase + r) * N + colBase + c] = acc[r][c];
}

// Optimized SGEMM v5: float4 global+shared I/O, 4-word padded tiles (bank-conflict
// free vector loads), 4x4 register tiling, 64x64 block tiles. M,N,K multiples of 64.
__global__ void sgemm_opt(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K) {
  __shared__ float As[64][68];   // +4 pad keeps &As[r][tx*4] 16B-aligned
  __shared__ float Bs[64][68];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int rowL = ty * 4, colL = tx * 4;
  const int rowB = blockIdx.y * 64 + rowL;
  const int colB = blockIdx.x * 64 + colL;
  float acc[4][4] = {{0.f}};
  const float4* A4 = reinterpret_cast<const float4*>(A);
  const float4* B4 = reinterpret_cast<const float4*>(B);
  for (int t = 0; t < K; t += 64) {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
      *reinterpret_cast<float4*>(&As[rowL + r][colL]) =
          A4[((size_t)rowB + r) * (K >> 2) + ((t + colL) >> 2)];
      *reinterpret_cast<float4*>(&Bs[rowL + r][colL]) =
          B4[((size_t)t + rowL + r) * (N >> 2) + (colB >> 2)];
    }
    __syncthreads();
    #pragma unroll 8
    for (int k = 0; k < 64; ++k) {
      const float a0 = As[rowL+0][k], a1 = As[rowL+1][k], a2 = As[rowL+2][k], a3 = As[rowL+3][k];
      const float4 b = *reinterpret_cast<const float4*>(&Bs[k][colL]);
      acc[0][0] += a0*b.x; acc[0][1] += a0*b.y; acc[0][2] += a0*b.z; acc[0][3] += a0*b.w;
      acc[1][0] += a1*b.x; acc[1][1] += a1*b.y; acc[1][2] += a1*b.z; acc[1][3] += a1*b.w;
      acc[2][0] += a2*b.x; acc[2][1] += a2*b.y; acc[2][2] += a2*b.z; acc[2][3] += a2*b.w;
      acc[3][0] += a3*b.x; acc[3][1] += a3*b.y; acc[3][2] += a3*b.z; acc[3][3] += a3*b.w;
    }
    __syncthreads();
  }
  float4* C4 = reinterpret_cast<float4*>(C);
  #pragma unroll
  for (int r = 0; r < 4; ++r)
    C4[((size_t)rowB + r) * (N >> 2) + (colB >> 2)] =
        make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
}

// Debug/bisect kernel: one output per thread, direct global reads.
__global__ void sgemm_naive(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
  int col = blockIdx.x * 16 + threadIdx.x;
  int row = blockIdx.y * 16 + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
      acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    C[(size_t)row * N + col] = acc;
  }
}

} // extern "C"
