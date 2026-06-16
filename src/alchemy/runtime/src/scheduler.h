#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>

class Scheduler {
public:
  std::vector<int> timestep_schedule_;
  virtual ~Scheduler() = default;

  virtual void step(float *d_xt, const float *d_noise_pred, int t,
                    int total_elements, cudaStream_t stream) = 0;
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
  curandState *d_rng_states_ = nullptr;

public:
  DDPMScheduler(int T = 1000, float beta_start = 1e-4f, float beta_end = 0.02f,
                int max_elements = 65536);

  ~DDPMScheduler();

  void step(float *d_xt, const float *d_noise_pred, int t,
            int total_elements, // B*C*H*W, i.e. total size of flattened buffer
            cudaStream_t stream_) override;
};

class DDIMScheduler : public Scheduler {
  int T_;
  int num_steps_;
  float beta_start_;
  float beta_end_;
  float eta_;
  std::vector<float> betas_;
  std::vector<float> alphas_;
  std::vector<float> alphas_cumprod_;
  std::vector<float> sqrt_alphas_cumprod_;
  std::vector<float> sqrt_one_minus_alphas_cumprod_;
  std::vector<int> prev_timestep_;
  curandState *d_rng_states_ = nullptr;

public:
  DDIMScheduler(int T = 1000, int num_steps = 50, float beta_start = 1e-4f,
                float beta_end = 0.02f,
                float eta = 0.0f, // 0 = deterministic DDIM, 1 = DDPM
                int max_elements = 65536);
  ~DDIMScheduler();
  void step(float *d_xt, const float *d_noise_pred, int t, int total_elements,
            cudaStream_t stream_) override;
};