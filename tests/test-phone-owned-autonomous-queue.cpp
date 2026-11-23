// Model-free Phone-owned Prefill autonomy contract: bounded FIFO,
 // input admission vs HTP execution, final-only P2C, error/stop.
#include "../ggml/src/ggml-rpc/phone-owned-autonomous-queue.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "CHECK failed: %s at %s:%d\n", \
                 #x, __FILE__, __LINE__); std::abort(); \
} } while (false)

int main() {
    {
        std::mutex order_mutex;
        std::vector<uint32_t> order;
        ggml_phone_autonomous::prefill_queue queue(
            3, 2, [&](uint32_t chunk, const std::vector<uint8_t> & input,
                       std::vector<uint8_t> & output) {
                CHECK(input.size() == 4);
                {
                    std::lock_guard<std::mutex> lock(order_mutex);
                    order.push_back(chunk);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (chunk == 2) { output = {0x11, 0x22, 0x33}; }
                return true;
            });
        CHECK(!queue.abandoned());
        CHECK(queue.push(0, {1, 2, 3, 4}));
        CHECK(queue.push(1, {5, 6, 7, 8}));
        // Backpressure will wait until a prior Chunk has completed.
        CHECK(queue.push(2, {9, 10, 11, 12}));
        std::vector<uint8_t> final;
        std::string error;
        CHECK(queue.wait_final(final, error));
        CHECK((final == std::vector<uint8_t>{0x11, 0x22, 0x33}));
        CHECK(queue.accepted() == 3);
        CHECK(queue.completed() == 3);
        {
            std::lock_guard<std::mutex> lock(order_mutex);
            CHECK((order == std::vector<uint32_t>{0, 1, 2}));
        }
        queue.stop();
    }
    {
        // No invalid/out-of-order Chunk may enter the HTP FIFO.
        ggml_phone_autonomous::prefill_queue queue(
            2, 2, [](uint32_t, const std::vector<uint8_t> &,
                      std::vector<uint8_t> &) { return true; });
        CHECK(!queue.push(1, {1}));
        CHECK(!queue.push(0, {1}));
        std::vector<uint8_t> output;
        std::string error;
        CHECK(!queue.wait_final(output, error));
        CHECK(!error.empty());
        queue.stop();
    }
    {
        // Execution failure wakes the P2C subscriber, rejects subsequent
        // input, and RESET joins the worker before freeing any Graph/KV.
        ggml_phone_autonomous::prefill_queue queue(
            2, 1, [](uint32_t, const std::vector<uint8_t> &,
                      std::vector<uint8_t> &) { return false; });
        CHECK(queue.push(0, {1}));
        std::vector<uint8_t> output;
        std::string error;
        CHECK(!queue.wait_final(output, error));
        CHECK(error == "PHONE_NATIVE_HTP_EXECUTION_FAILED");
        CHECK(queue.abandoned());
        CHECK(!queue.push(1, {2}));
        queue.stop();
    }
    {
        // Aborting a queued/in-flight Chunk is bounded: all tasks are
        // immutable and no thread or output buffer survives stop().
        ggml_phone_autonomous::prefill_queue queue(
            2, 2, [](uint32_t, const std::vector<uint8_t> &,
                      std::vector<uint8_t> &) {
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
                return true;
            });
        CHECK(queue.push(0, {1}));
        queue.stop();
        std::vector<uint8_t> output;
        std::string error;
        CHECK(!queue.wait_final(output, error));
    }
    std::puts("test-phone-owned-autonomous-queue PASS");
    return 0;
}
