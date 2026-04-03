#include "engine.h"
#include <fstream>
#include <unordered_map>
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

    // Get the maximum batch size in the onnx model
    const char* input_name = engine_->getIOTensorName(0);
    int max_batch = engine_ -> getProfileShape(input_name, 0, 
        nvinfer1::OptProfileSelector::kMAX).d[0];

    for(int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(name);

        // Store tensor sizes
        size_t n = 1;
        for(int j = 1; j < dims.nbDims; ++j) n *= dims.d[j];
        tensor_sizes_[name] = n;

        // Allocate buffer on device and store device address
        void* buf = nullptr;
        CUDA_CHECK(cudaMalloc(&buf, max_batch * n * sizeof(float)));
        device_buffers_[name] = buf;

        context_->setTensorAddress(name, buf);
    }

    CUDA_CHECK(cudaStreamCreate(&stream_));
}

TRTEngine::~TRTEngine() {
    for(auto& [name, ptr] : device_buffers_) cudaFree(ptr);
    cudaStreamDestroy(stream_);
}

void TRTEngine::run(
    const std::unordered_map<std::string, std::vector<float>>& inputs,
    int batch_size
) {
    for(auto& [name, data] : inputs) {
        // We need to set actual batch size because it's currently dynamic
        auto dims = engine_->getTensorShape(name.c_str());
        dims.d[0] = batch_size;
        context_->setInputShape(name.c_str(), dims);

        // Move input tensor to device
        CUDA_CHECK(cudaMemcpyAsync(
            device_buffers_.at(name), 
            data.data(), 
            batch_size * tensor_sizes_.at(name) * sizeof(float), 
            cudaMemcpyHostToDevice, stream_));
    }

    context_->enqueueV3(stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}
