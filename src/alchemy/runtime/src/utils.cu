#include "utils.h"

__global__ void init_rng(curandState* states, int n, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    curand_init(seed, i, 0, &states[i]);
}

__global__ void sample_gaussian(float* d_out, int n, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return; 
    d_out[i] = curand_normal(&states[i]);
}

void launch_init_rng(curandState *states, int n, int seed, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    init_rng<<<blocks, threads, 0, stream>>>(states, n, seed);
}

void launch_sample_gaussian(float *d_out, int n, curandState *states, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sample_gaussian<<<blocks, threads, 0, stream>>>(d_out, n, states);
}