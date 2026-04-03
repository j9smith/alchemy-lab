#pragma once
#include <iostream>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <memory>
#include <unordered_map>
#include <vector>

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class TRTEngine {
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    std::unordered_map<std::string, void*> device_buffers_;
    std::unordered_map<std::string, size_t> tensor_sizes_;
    cudaStream_t stream_;

    public:
        TRTEngine(const std::string& plan_path);
        ~TRTEngine();
        std::vector<float> run(
            const std::unordered_map<std::string, std::vector<float>>& inputs,
            int batch_size
        );
};

class DecoderEngine {
    TRTEngine engine_;

    public:
        DecoderEngine(const std::string& plan_path) : engine_(plan_path) {}
        std::vector<float> run(
            const std::vector<float>& latent,
            int batch_size
        )
        {
            return engine_.run({{"latent", latent}}, batch_size);
        }
};

class DenoiserEngine {
    TRTEngine engine_;

    public:
        DenoiserEngine(const std::string& plan_path) : engine_(plan_path) {}
        std::vector<float> run(
            const std::vector<float>& xt, // [B, C, H, W]
            const std::vector<float>& t, // [B]
            int batch_size
        )
        {
            return engine_.run({{"xt", xt}, {"t", t}}, batch_size);
        }
};