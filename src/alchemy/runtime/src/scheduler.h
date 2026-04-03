#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

class Scheduler {
    public:
        virtual ~Scheduler() = default;

        virtual void step(
            float* d_xt, const float* d_noise_pred, 
            int t, int total_elements, cudaStream_t stream
        ) = 0;

        virtual std::vector<int> timesteps() const = 0;
};

class DDPMScheduler : public Scheduler {
    int T_;
    float beta_start_;
    float beta_end_;
    std::vector<float> betas_;
    std::vector<float> alphas_;
    std::vector<float> alphas_cumprod_;
    std::vector<float> alphas_cumprod_prev_;
    std::vector<float> sqrt_alphas_cumprod_;
    std::vector<float> sqrt_one_minus_alphas_cumprod_;
    std::vector<float> posterior_variance_;
    std::vector<float> posterior_mean_coef1_;
    std::vector<float> posterior_mean_coef2_;
    std::vector<int> timesteps_;
    curandState* d_rng_states_ = nullptr;

    public:
        DDPMScheduler(
            int T = 1000, 
            float beta_start = 1e-4f, 
            float beta_end = 0.02f,
            int max_elements = 65536
        );

        ~DDPMScheduler();

        void step(
            float* d_xt, 
            const float* d_noise_pred,
            int t, 
            int total_elements, // B*C*H*W, i.e. total size of flattened buffer
            cudaStream_t stream_
        ) override;

        std::vector<int> timesteps() const override;
};

class DDIMScheduler : public Scheduler {};