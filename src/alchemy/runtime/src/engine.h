#pragma once
#include <iostream>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <memory>
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
    
    void* d_input_; 
    void* d_output_;
    cudaStream_t stream_;
    size_t input_size_;
    size_t output_size_;

    public:
        TRTEngine(const std::string& plan_path);
        ~TRTEngine();
        std::vector<float> run(const std::vector<float>& input, int batch_size);
};

class DecoderEngine {
    TRTEngine engine_;

    public:
        DecoderEngine(const std::string& plan_path) : engine_(plan_path) {}
};

class DenoiserEngine {
    TRTEngine engine_;

    public:
        DenoiserEngine(const std::string& plan_path) : engine_(plan_path) {}
};