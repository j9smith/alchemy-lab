#pragma once
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <vector>

struct BatchItem {
    std::string prompt;
    std::promise<std::vector<float>> p_result;
};

class BatchQueue {
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<BatchItem> queue_;
    static constexpr size_t kMaxBatch = 8;
    static constexpr auto kMaxWait = std::chrono::milliseconds(1000);
    public:
        std::future<std::vector<float>> enqueue(std::string prompt);
        void run_loop();
};