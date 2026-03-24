#include "engine.h"
#include <fstream>
#include "utils.h"

TRTEngine::TRTEngine(const std::string& plan_path) {
    std::ifstream file(plan_path, std::ios::binary);
    file.seekg(0, std::ios::end); // Move pointer to EOF
    size_t size = file.tellg(); // Get pointer position
    file.seekg(0, std::ios::beg); // Move ptr to beginning of file for read

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
    context_.reset(engine_->createExecutionContext());

    int max_batch = engine_ -> getProfileShape("input", 0, 
        nvinfer1::OptProfileSelector::kMAX).d[0];

    auto input_dims = engine_->getTensorShape("input");
    auto output_dims = engine_->getTensorShape("output");

    // Calculate number of floats in one sample (i.e. C*H*W)
    input_size_ = 1;
    output_size_ = 1;
    for (int i = 1; i < input_dims.nbDims; ++i) input_size_ *= input_dims.d[i];
    for (int i = 1; i < output_dims.nbDims; ++i) output_size_ *= output_dims.d[i];

    // Allocate memory on device
    CUDA_CHECK(cudaMalloc(&d_input_, max_batch * input_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_, max_batch * output_size_ * sizeof(float)));

    CUDA_CHECK(cudaStreamCreate(&stream_));
    context_->setTensorAddress("input", d_input_);
    context_->setTensorAddress("output", d_output_);
}

TRTEngine::~TRTEngine() {
    cudaFree(d_input_);
    cudaFree(d_output_);
    cudaStreamDestroy(stream_);
}

std::vector<float> TRTEngine::run(const std::vector<float>& input, int batch_size) {
    // Move input to device
    CUDA_CHECK(cudaMemcpyAsync(d_input_, input.data(), 
        batch_size * input_size_ * sizeof(float), 
        cudaMemcpyHostToDevice, stream_));

    context_->enqueueV3(stream_);

    std::vector<float> output(batch_size * output_size_);
    CUDA_CHECK(cudaMemcpyAsync(output.data(), d_output_, 
        batch_size * output_size_ * sizeof(float), 
        cudaMemcpyDeviceToHost, stream_));

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    return output;
}