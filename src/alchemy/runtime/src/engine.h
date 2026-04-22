#pragma once
#include "utils.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cout << msg << std::endl;
  }
};

class TRTEngine {
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  std::unordered_map<std::string, void *> device_buffers_;
  std::unordered_map<std::string, size_t> tensor_sizes_;
  cudaStream_t stream_;

public:
  TRTEngine(const std::string &plan_path);
  ~TRTEngine();
  void run(const std::unordered_map<std::string, std::vector<float>> &inputs,
           int batch_size);

  void run_device(int batch_size) { context_->enqueueV3(stream_); }

  float *get_device_buffer(const std::string &name) {
    return static_cast<float *>(device_buffers_.at(name));
  }
  size_t get_tensor_size(const std::string &name) {
    return tensor_sizes_.at(name);
  }

  cudaStream_t stream() { return stream_; }

  void set_shapes(int batch_size) {
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char *name = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        auto dims = engine_->getTensorShape(name);
        dims.d[0] = batch_size;
        context_->setInputShape(name, dims);
      }
    }
  }
};

class DenoiserEngine {
  TRTEngine engine_;

public:
  DenoiserEngine(const std::string &plan_path) : engine_(plan_path) {}
  void run(const std::vector<float> &xt, // [B, C, H, W]
           const std::vector<float> &t,  // [B]
           int batch_size) {
    engine_.run({{"xt", xt}, {"t", t}}, batch_size);
  }

  //   void run(const std::vector<float> &t, int batch_size) {
  //     engine_.run({{"t", t}}, batch_size);
  //   }

  void run(const std::vector<float> &t_float, int batch_size) {
    std::vector<int64_t> t(t_float.begin(), t_float.end());

    engine_.set_shapes(batch_size);

    CUDA_CHECK(cudaMemcpyAsync(engine_.get_device_buffer("t"), t.data(),
                               batch_size * sizeof(int64_t),
                               cudaMemcpyHostToDevice, engine_.stream()));

    engine_.run_device(batch_size);
  }

  cudaStream_t stream() { return engine_.stream(); }

  float *d_xt() { return engine_.get_device_buffer("xt"); }
  float *d_output() { return engine_.get_device_buffer("output"); }

  void set_shapes(int batch_size) { engine_.set_shapes(batch_size); }
};

class DecoderEngine {
  TRTEngine engine_;

public:
  DecoderEngine(const std::string &plan_path) : engine_(plan_path) {}
  std::vector<float> run(float *d_xt, int batch_size) {
    engine_.set_shapes(batch_size);

    CUDA_CHECK(cudaMemcpyAsync(engine_.get_device_buffer("latent"), d_xt,
                               batch_size * engine_.get_tensor_size("latent") *
                                   sizeof(float),
                               cudaMemcpyDeviceToDevice, engine_.stream()));

    engine_.run_device(batch_size);

    size_t out_size = batch_size * engine_.get_tensor_size("output");
    std::vector<float> output(out_size);

    CUDA_CHECK(cudaMemcpyAsync(
        output.data(), engine_.get_device_buffer("output"),
        out_size * sizeof(float), cudaMemcpyDeviceToHost, engine_.stream()));

    CUDA_CHECK(cudaStreamSynchronize(engine_.stream()));
    return output;
  }
};
