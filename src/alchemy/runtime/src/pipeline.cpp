#include "pipeline.h"
#include "utils.h"
#include <memory>
#include <vector>

AlchemyPipeline::AlchemyPipeline(
    const std::string& denoiser_plan,
    const std::string& decoder_plan,
    std::unique_ptr<Scheduler> scheduler,
    int max_batch,
    int C, int H, int W
) : denoiser_(denoiser_plan),
    decoder_(decoder_plan),
    scheduler_(std::move(scheduler)),
    max_batch_(max_batch),
    latent_elements_(C * H * W)
{
    int max_elements = max_batch * latent_elements_;

    CUDA_CHECK(cudaMalloc(&d_rng_states_, max_elements * sizeof(curandState)));
    launch_init_rng(d_rng_states_, max_elements, 0, denoiser_.stream());
    CUDA_CHECK(cudaStreamSynchronize(denoiser_.stream()));
}

AlchemyPipeline::~AlchemyPipeline() {
    if (d_rng_states_) cudaFree(d_rng_states_);
}

std::vector<float> AlchemyPipeline::t_to_tensor(int t, int batch_size) {
    return std::vector<float>(batch_size, static_cast<float>(t));
}

void AlchemyPipeline::sample_xT(int batch_size) {
    launch_sample_gaussian(
        denoiser_.d_xt(), 
        batch_size * latent_elements_, 
        d_rng_states_,
        denoiser_.stream());
}

std::vector<float> AlchemyPipeline::generate(
    const std::vector<std::string>& prompts,
    int batch_size
) 
{
    // Sample Gaussian noise for each request
    sample_xT(batch_size);

    denoiser_.set_shapes(batch_size);
    auto& timesteps = scheduler_->timestep_schedule_;

    for (int i = 0; i < (int)timesteps.size(); i++) {
        int t = timesteps[i];
        denoiser_.run(t_to_tensor(t, batch_size), batch_size);
        scheduler_->step(
            denoiser_.d_xt(),
            denoiser_.d_output(),
            t,
            batch_size * latent_elements_,
            denoiser_.stream()
        );
    }

    return decoder_.run(denoiser_.d_xt(), batch_size);
}