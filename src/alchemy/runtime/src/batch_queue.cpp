#include "batch_queue.h"
#include "infer.h"
#include <iterator>
#include <stdio.h>
#include <string>

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
  while (true) {
    std::vector<BatchItem> batch;
    {
      std::unique_lock lock(mu_);

      // Sleep thread until queue is not empty
      cv_.wait(lock, [&] { return !queue_.empty(); });

      auto deadline = std::chrono::steady_clock::now() + kMaxWait;

      // Sleep thread until deadline passes/queue is full
      cv_.wait_until(lock, deadline,
                     [&] { return queue_.size() >= kMaxBatch; });

      size_t n = std::min(queue_.size(), kMaxBatch);

      // Move items into batch
      // We need make_move_iterator because std::promise can't be copied
      batch.assign(std::make_move_iterator(queue_.begin()),
                   std::make_move_iterator(queue_.begin() + n));

      // Remove batched items from queue
      queue_.erase(queue_.begin(), queue_.begin() + n);
    }
    std::cout << "Items in queue remaining: " + std::to_string(queue_.size()) +
                     "\n";

    // Hand off batch for inference
    auto results = infer(batch, pipeline_);

    // Fulfill the promise of each request with corresponding result
    for (size_t i = 0; i < batch.size(); ++i) {
      batch[i].p_result.set_value(std::move(results[i]));
    }
  }
}