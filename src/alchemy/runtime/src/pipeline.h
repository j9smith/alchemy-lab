#pragma once
#include "engine.h"
#include "scheduler.h"
#include <memory>
#include <vector>

class AlchemyPipeline {
    DenoiserEngine denoiser_;
    DecoderEngine decoder_;
    std::unique_ptr<Scheduler> scheduler_;
    cudaStream_t stream_;

    curandState* d_rng_states_ = nullptr;

    int max_batch_;
    int latent_elements_;

    public:
        AlchemyPipeline(
            const std::string& denoiser_plan,
            const std::string& decoder_plan,
            std::unique_ptr<Scheduler> scheduler,
            int max_batch,
            int C, int H, int W
        );

        ~AlchemyPipeline();

        std::vector<float> generate(
            const std::vector<std::string>& prompts,
            int batch_size
        );

    private:
        void sample_xT(int batch_size);
        std::vector<float> t_to_tensor(int t, int batch_size);
};