// Minimal CUDA driver API header for Tegra X1 (L4T libcuda 10.0), no toolkit needed.
// Usage: #include "cuda_drv.h"  and link with -lcuda
#ifndef CUDA_DRV_MIN_H
#define CUDA_DRV_MIN_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext, *CUmodule, *CUfunction, *CUstream;
typedef uint64_t CUdeviceptr;

extern CUresult cuInit(unsigned int flags);
extern CUresult cuDriverGetVersion(int *v);
extern CUresult cuDeviceGetCount(int *n);
extern CUresult cuDeviceGet(CUdevice *d, int ord);
extern CUresult cuDeviceGetName(char *out, int len, CUdevice d);
extern CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice d);
extern CUresult cuDeviceGetAttribute(int *val, int attr, CUdevice d);
extern CUresult cuCtxCreate_v2(CUcontext *ctx, unsigned int flags, CUdevice d);
extern CUresult cuModuleLoadData(CUmodule *m, const void *image);
extern CUresult cuModuleGetFunction(CUfunction *f, CUmodule m, const char *name);
extern CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t bytes);
extern CUresult cuMemFree_v2(CUdeviceptr p);
extern CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t bytes);
extern CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t bytes);
extern CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                               unsigned bx, unsigned by, unsigned bz,
                               unsigned shmem, CUstream s, void **params, void *extra);
extern CUresult cuCtxSynchronize(void);
extern CUresult cuGetErrorString(CUresult r, const char **s);

#define CU_ATTR_SM_COUNT 16
#define CU_ATTR_CLOCK_KHZ 13
#define CU_ATTR_CC_MAJOR 75
#define CU_ATTR_CC_MINOR 76

#define CHK(x) do { CUresult r_ = (x); if (r_) { const char *s_; cuGetErrorString(r_, &s_); \
    fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", r_, s_ ? s_ : "?", __FILE__, __LINE__); exit(1); } } while (0)

#endif
