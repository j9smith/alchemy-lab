#include "scheduler.h"
#include "utils.h"
#include <numeric>
#include <algorithm>

__global__ void ddim_step(
    float* d_xt,
    const float* d_noise_pred,
    float sqrt_ab,
    float sqrt_one_minus_ab,
    float sqrt_ab_prev,
    float sqrt_one_minus_ab_prev,
    float sigma,
    int n,
    curandState* rng_states
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    // One element per thread
    float x0_pred = (d_xt[i] - sqrt_one_minus_ab * d_noise_pred[i]) / sqrt_ab;
    x0_pred = fminf(1.0f, fmaxf(-1.0f, x0_pred)); // clamp to [-1, 1]

    // Direction pointing to x_t, scaled by the variance left after sigma
    float dir_coef = sqrtf(fmaxf(0.0f, sqrt_one_minus_ab_prev * sqrt_one_minus_ab_prev - sigma * sigma));
    float dir = dir_coef * d_noise_pred[i];
    float x_prev = sqrt_ab_prev * x0_pred + dir;
    if (sigma > 0.0f) x_prev += sigma * curand_normal(&rng_states[i]);
    d_xt[i] = x_prev;
}

DDIMScheduler::DDIMScheduler(
    int T, int num_steps, float beta_start, float beta_end, float eta, int max_elements
)
    : T_(T), num_steps_(num_steps), beta_start_(beta_start), beta_end_(beta_end), eta_(eta)
{
    // Compute diffusion coeffs once
    betas_.resize(T); alphas_.resize(T); alphas_cumprod_.resize(T);
    sqrt_alphas_cumprod_.resize(T); sqrt_one_minus_alphas_cumprod_.resize(T);
    prev_timestep_.resize(T);

    for(int i = 0; i < T; ++i) {
        betas_[i] = beta_start + (beta_end - beta_start) * i / (T - 1);
        alphas_[i] = 1.0f - betas_[i];
        alphas_cumprod_[i] = (i == 0) ? alphas_[i] : alphas_cumprod_[i-1] * alphas_[i];
        sqrt_alphas_cumprod_[i] = std::sqrt(alphas_cumprod_[i]);
        sqrt_one_minus_alphas_cumprod_[i] = std::sqrt(1.0f - alphas_cumprod_[i]);
    }

    // Build the strided subseq of timesteps, then record each step's predecessor
    // step_ratio spacing, ascending, then reversed for the descending sampling schedule
    int step_ratio = T / num_steps;
    timestep_schedule_.resize(num_steps);
    for (int i = 0; i < num_steps; ++i) {
        timestep_schedule_[i] = i * step_ratio;
    }

    std::fill(prev_timestep_.begin(), prev_timestep_.end(), -1);
    for (int i = 0; i < num_steps; ++i) {
        int t = timestep_schedule_[i];
        prev_timestep_[t] = (i == 0) ? -1 : timestep_schedule_[i-1];
    }
    // Reverse (high noise -> low) to match the denoiser
    std::reverse(timestep_schedule_.begin(), timestep_schedule_.end());

    // Allocate + init persistent rng
    cudaMalloc(&d_rng_states_, max_elements * sizeof(curandState));
    launch_init_rng(d_rng_states_, max_elements, 0);
    cudaDeviceSynchronize();
}

DDIMScheduler::~DDIMScheduler() {
    if (d_rng_states_) cudaFree(d_rng_states_);
}

void DDIMScheduler::step(
    float* d_xt,
    const float* d_noise_pred,
    int t,
    int total_elements,
    cudaStream_t stream_
) {
    int t_prev = prev_timestep_[t];

    float sqrt_ab = sqrt_alphas_cumprod_[t];
    float sqrt_one_minus_ab = sqrt_one_minus_alphas_cumprod_[t];

    float ab_prev = (t_prev < 0) ? 1.0f : alphas_cumprod_[t_prev];
    float sqrt_ab_prev = std::sqrt(ab_prev);
    float sqrt_one_minus_ab_prev = std::sqrt(1.0f - ab_prev);

    // sigma_t = eta * sqrt((1-ab_prev)/(1-ab)) * sqrt(1 - ab/ab_prev)
    float ab = alphas_cumprod_[t];
    float sigma = 0.0f;
    if (eta_ > 0.0f && t_prev >= 0) {
        float ratio = (1.0f - ab_prev) / (1.0f - ab);
        sigma = eta_ * std::sqrt(ratio) * std::sqrt(1.0f - ab / ab_prev);
    }

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    ddim_step<<<blocks, threads, 0, stream_>>>(
        d_xt, d_noise_pred, sqrt_ab, sqrt_one_minus_ab,
        sqrt_ab_prev, sqrt_one_minus_ab_prev, sigma, total_elements, d_rng_states_
    );
}