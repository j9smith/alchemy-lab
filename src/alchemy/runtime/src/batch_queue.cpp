#include "batch_queue.h"
#include "infer.h"

std::future<std::vector<float>> BatchQueue::enqueue(std::string prompt) {
    // runs on request handler thread
    std::promise<std::vector<float>> p;
    auto fut = p.get_future();
    {
        std::lock_guard lock(mu_);
        queue_.push_back({std::move(prompt), std::move(p)});
    }
    // Awaken inference thread
    cv_.notify_one();
    return fut;
}

void BatchQueue::run_loop() {
    // runs on inference thread
    while(true) {
        std::vector<BatchItem> batch;
        {
            std::unique_lock lock(mu_);

            // Sleep thread until queue is not empty
            cv_.wait(lock, [&]{ return !queue_.empty(); });

            auto deadline = std::chrono::steady_clock::now() + kMaxWait;

            // Sleep thread until deadline passes/queue is full
            cv_.wait_until(lock, deadline, [&]{ 
                return queue_.size() >= kMaxBatch;
            });

            // Swap contents of batch (empty) and queue (populated)
            batch.swap(queue_);
        }

        // Hand off batch for inference
        auto results = infer(batch);

        // Fulfill the promise of each request with corresponding result
        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i].p_result.set_value(std::move(results[i]));
        }
    }
}