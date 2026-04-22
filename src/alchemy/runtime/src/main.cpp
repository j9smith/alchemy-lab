#include "batch_queue.h"
#include "pipeline.h"
#include "scheduler.h"
#include <httplib.h>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

int main() {
  AlchemyPipeline pipeline("denoiser.plan", "decoder.plan",
                           std::make_unique<DDPMScheduler>(), 8, 4, 32, 32);

  BatchQueue queue(pipeline);

  std::thread inference_thread([&queue]() { queue.run_loop(); });

  httplib::Server svr;

  svr.Get("/health", [](const httplib::Request &, httplib::Response &res) {
    res.set_content("ok\n", "text/plain");
  });

  svr.Post(
      "/generate", [&queue](const httplib::Request &, httplib::Response &res) {
        std::string prompt = "Placeholder";
        auto fut = queue.enqueue(prompt);
        std::vector<float> pixels = fut.get(); // Returns [-1, 1], C*H*W

        // CelebA-HQ is 256x256
        const int C = 3, H = 256, W = 256;

        std::string ppm =
            "P6\n" + std::to_string(W) + " " + std::to_string(H) + "\n255\n";

        // [-1, 1] -> [0, 255]
        for (int y = 0; y < H; y++)
          for (int x = 0; x < W; x++)
            for (int c = 0; c < C; c++) {
              float v = (pixels[c * H * W + y * W + x] + 1.0f) *
                        0.5f; // -> [0, 2] -> [0, 1]
              ppm += static_cast<char>(
                  std::clamp(int(v * 255.0f), 0, 255)); // [0, 1] -> [0, 255]
            }

        res.set_content(ppm, "image/x-portable-pixmap");
      });

  std::cout << "listening on port 8000\n";

  svr.new_task_queue = [] { return new httplib::ThreadPool(128); };
  svr.listen("0.0.0.0", 8000);

  inference_thread.join();
}