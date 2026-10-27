// Native CUDA smoke test: compiled ON the Switch with clang-18, runtime API, <<<>>> launch.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

#define CK(x) do { cudaError_t e = (x); if (e) { \
    fprintf(stderr, "CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

int main(void) {
    const int N = 1 << 20;
    float *x = (float*)malloc(N * 4), *y = (float*)malloc(N * 4);
    for (int i = 0; i < N; i++) { x[i] = i * 0.001f; y[i] = 1.0f; }
    float *dx, *dy;
    CK(cudaMalloc(&dx, N * 4));
    CK(cudaMalloc(&dy, N * 4));
    CK(cudaMemcpy(dx, x, N * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dy, y, N * 4, cudaMemcpyHostToDevice));
    saxpy<<<N / 256, 256>>>(2.0f, dx, dy, N);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(y, dy, N * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < N; i++) if (y[i] != 2.0f * x[i] + 1.0f) bad++;
    printf("native saxpy 1M: %s (bad=%d)\n", bad ? "WRONG" : "CORRECT", bad);
    return bad ? 1 : 0;
}
