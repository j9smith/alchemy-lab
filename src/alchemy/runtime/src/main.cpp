#include <iostream>
#include <thread>
#include <httplib.h>
#include "batch_queue.h"

int main(){
    BatchQueue queue;

    std::thread inference_thread([&queue]() {
        queue.run_loop();
    });

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok\n", "text/plain");
    });

    svr.Post("/generate", [&queue](const httplib::Request&, httplib::Response& res) {
        std::string prompt = "Placeholder";
        auto fut = queue.enqueue(prompt);
        auto result = fut.get();
        res.set_content("received result\n", "text/plain");
    });

    std::cout << "listening on port 8000\n";
    svr.listen("0.0.0.0", 8000);

    inference_thread.join();
}