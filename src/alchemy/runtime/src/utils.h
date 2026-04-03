#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << "\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void launch_init_rng(curandState* states, int n, int seed, cudaStream_t stream = 0);
void launch_sample_gaussian(float* d_out, int n, curandState* states, cudaStream_t stream = 0);
