# GPU development on the Switch — native (on-device) and cross (PC nvcc) workflows
CFLAGS = -O2
PROGS = gpu_workload gpu_test
NVCC_PTX = switch_gpu.ptx
NATIVE_PTX = switch_gpu_native.ptx
NATIVE_FLAGS = -O3 -S --cuda-device-only -x cuda --cuda-gpu-arch=sm_53 --cuda-feature=+ptx63 \
  -nocudainc -nocudalib -I/usr/local/cuda-10.0/include -D__CUDACC__ \
  -include __clang_cuda_builtin_vars.h -include vector_types.h -include vector_functions.h

all: $(PROGS)

gpu_workload: gpu_workload.c cuda_drv.h
	gcc $(CFLAGS) -o $@ gpu_workload.c -lcuda -lm

gpu_test: gpu_test.c
	gcc $(CFLAGS) -o $@ gpu_test.c -lcuda -lm

# native kernel compile (clang-18, on the Switch)
%.ptx: %.cu
	clang++-18 $(NATIVE_FLAGS) -o $@ $<

$(NATIVE_PTX): switch_gpu.cu
	clang++-18 $(NATIVE_FLAGS) -o $@ $<

native: $(NATIVE_PTX)

bench: gpu_workload            # nvcc-compiled (from PC build-gpu.sh)
	./gpu_workload $(NVCC_PTX)

bench-native: gpu_workload native
	./gpu_workload $(NATIVE_PTX)

clean:
	rm -f $(PROGS) *.ptx
