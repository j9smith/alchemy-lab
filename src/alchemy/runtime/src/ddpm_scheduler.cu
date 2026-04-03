#include "scheduler.h"
#include "utils.h"

__global__ void ddpm_step(
    float* d_xt,
    const float* d_noise_pred,
    float coef1,
    float coef2, 
    float sqrt_ab,
    float sqrt_one_minus_ab,
    float std_dev,
    int n,
    curandState* rng_states
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    // One sample per thread
    float x0_pred = (d_xt[i] - sqrt_one_minus_ab * d_noise_pred[i]) / sqrt_ab;
    x0_pred = fminf(1.0f, fmaxf(-1.0f, x0_pred)); // clamp to [-1, 1]
    float mu = coef1 * x0_pred + coef2 * d_xt[i];
    d_xt[i] = mu + std_dev * curand_normal(&rng_states[i]);
}

DDPMScheduler::DDPMScheduler(int T, float beta_start, float beta_end, int max_elements) 
    : T_(T), beta_start_(beta_start), beta_end_(beta_end)
{
    // Compute diffusion coeffs once
    betas_.resize(T); alphas_.resize(T); alphas_cumprod_.resize(T);
    alphas_cumprod_prev_.resize(T); sqrt_alphas_cumprod_.resize(T);
    sqrt_one_minus_alphas_cumprod_.resize(T); posterior_variance_.resize(T);
    posterior_mean_coef1_.resize(T); posterior_mean_coef2_.resize(T);
    timestep_schedule_.resize(T);

    std::iota(timestep_schedule_.rbegin(), timestep_schedule_.rend(), 0);

    for(int i = 0; i < T; ++i) {
        betas_[i] = beta_start + (beta_end - beta_start) * i / (T - 1);
        alphas_[i] = 1.0f - betas_[i];

        alphas_cumprod_[i] = (i == 0) ? alphas_[i] : alphas_cumprod_[i-1] * alphas_[i];
        alphas_cumprod_prev_[i] = (i == 0) ? 1.0f : alphas_cumprod_[i-1];
        sqrt_alphas_cumprod_[i] = std::sqrt(alphas_cumprod_[i]);
        sqrt_one_minus_alphas_cumprod_[i] = std::sqrt(1.0f - alphas_cumprod_[i]);

        posterior_variance_[i] = betas_[i] * (1.0f - alphas_cumprod_prev_[i]) 
            / (1.0f - alphas_cumprod_[i]);
        posterior_mean_coef1_[i] = std::sqrt(alphas_cumprod_prev_[i]) * betas_[i] 
            / (1.0f - alphas_cumprod_[i]);
        posterior_mean_coef2_[i] = std::sqrt(alphas_[i]) 
            * (1.0f - alphas_cumprod_prev_[i]) / (1.0f - alphas_cumprod_[i]);
    }

    // Allocate + initialise persistent rng for use in step kernel
    cudaMalloc(&d_rng_states_, max_elements * sizeof(curandState));
    launch_init_rng(d_rng_states_, max_elements, 0);
    cudaDeviceSynchronize();
}

DDPMScheduler::~DDPMScheduler() {
    if (d_rng_states_) cudaFree(d_rng_states_);
}

void DDPMScheduler::step(
    float* d_xt, 
    const float* d_noise_pred,
    int t, 
    int total_elements, 
    cudaStream_t stream_
) {
    float coef1 = posterior_mean_coef1_[t];
    float coef2 = posterior_mean_coef2_[t];
    float sqrt_ab = sqrt_alphas_cumprod_[t];
    float sqrt_one_minus_ab = sqrt_one_minus_alphas_cumprod_[t];
    float std_dev = (t > 0) ? std::sqrt(posterior_variance_[t]) : 0.0f;

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    ddpm_step<<<blocks, threads, 0, stream_>>>(
        d_xt, d_noise_pred, coef1, coef2, sqrt_ab,
        sqrt_one_minus_ab, std_dev, total_elements, d_rng_states_
    );
}
