// Fair CPU FMA throughput reference for the Switch's 4x Cortex-A57.
// 4 pthreads, 8 independent accumulators each (ILP), -O3.
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

#define ITER 80000000u  // per thread

static volatile float g_sink;

void *worker(void *arg) {
    float a0 = 1.000001f, a1 = 1.000002f, a2 = 1.000003f, a3 = 1.000004f;
    float a4 = 1.000005f, a5 = 1.000006f, a6 = 1.000007f, a7 = 1.000008f;
    float b = 0.999999f, c = 0.500001f;
    for (uint32_t i = 0; i < ITER; i++) {
        a0 = a0 * b + c; a1 = a1 * b + c; a2 = a2 * b + c; a3 = a3 * b + c;
        a4 = a4 * b + c; a5 = a5 * b + c; a6 = a6 * b + c; a7 = a7 * b + c;
    }
    g_sink += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    return NULL;
}

int main(void) {
    pthread_t th[4];
    double t0 = now();
    for (int i = 0; i < 4; i++) pthread_create(&th[i], NULL, worker, NULL);
    for (int i = 0; i < 4; i++) pthread_join(th[i], NULL);
    double dt = now() - t0;
    double flops = 2.0 * 8 * ITER * 4;  // 8 FMAs/thread-iter * 4 threads
    printf("cpu fma x4 threads: %.3f s, %.2f GFLOPS FP32 (sink %.2f)\n", dt, flops / dt / 1e9, g_sink);
    return 0;
}
