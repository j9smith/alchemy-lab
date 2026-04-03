#include "infer.h"
#include "engine.h"

static TRTEngine engine("PATH");

std::vector<std::vector<float>> infer(std::vector<BatchItem>& batch, AlchemyPipeline& pipeline) {
    int B = batch.size();

    std::vector<std::string> prompts;
    prompts.reserve(B);

    // Populate prompts vector from BatchItems
    for (auto& item : batch) prompts.push_back(item.prompt);

    // Hand-off for inference
    // Result will be contiguous buffer of size B*C*H*W
    auto flat = pipeline.generate(prompts, B);

    // Calc size of each individual sample
    size_t sample_size = flat.size() / B;
    std::vector<std::vector<float>> results(B);

    // Slice flat into distinct results via indexing
    for (int i = 0; i < B; ++i) {
        results[i] = std::vector<float>(
            flat.begin() + i * sample_size,
            flat.begin() + (i + 1) * sample_size
        );
    }
    return results;
}