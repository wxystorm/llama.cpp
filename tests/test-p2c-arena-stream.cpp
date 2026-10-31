#include "../ggml/src/ggml-rpc/p2c-arena-queue.h"
#include "../ggml/src/ggml-rpc/p2c-stream-receiver.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

// Unlike assert(), this always evaluates its expression in MSVC Release
// builds (which define NDEBUG). Test setup must never be compiled out.
#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "[P2C_TEST_FAIL] %s:%d: %s\n", \
                     __FILE__, __LINE__, #expr); \
        std::fflush(stderr); \
        std::abort(); \
    } \
} while (false)

using frame = ggml_backend_rpc_p2c_frame;
constexpr uint64_t run_id = 73;
constexpr uint32_t hidden = 4;
constexpr uint32_t tokens = 8;
constexpr uint32_t regions = 2;

static void step(const char * label) {
    std::fprintf(stderr, "[P2C_TEST_STEP] %s\n", label);
    std::fflush(stderr);
}

frame make_frame(uint32_t region, uint32_t chunk,
                 uint32_t begin, uint32_t n_tokens, uint64_t seq = 0) {
    frame f {};
    f.run_id = run_id;
    f.seq = seq;
    f.region_id = region;
    f.chunk_id = chunk;
    f.token_begin = begin;
    f.n_tokens = n_tokens;
    f.kind = GGML_BACKEND_RPC_P2C_FULL;
    f.payload_bytes = uint64_t(n_tokens) * hidden * sizeof(float);
    return f;
}

void test_arena_publication_and_send_order() {
    ggml_rpc_p2c_arena_queue arena;
    REQUIRE(arena.initialize(run_id, regions, tokens, hidden));
    step("arena initialized");
    float a[8] = { 10, 11, 12, 13, 14, 15, 16, 17 };
    float b[8] = { 20, 21, 22, 23, 24, 25, 26, 27 };

    // Chunk A has reserved its output range but has NOT completed the
    // device read. READY B must be sendable without waiting for A.
    std::mutex mutex;
    std::condition_variable cv;
    bool inside_a = false;
    bool permit_a = false;
    bool a_ok = false;
    std::thread producer_a([&]() {
        a_ok = arena.capture(make_frame(0, 0, 0, 2),
            [&](uint8_t * out, size_t bytes) {
                std::unique_lock<std::mutex> lock(mutex);
                inside_a = true;
                cv.notify_one();
                REQUIRE(cv.wait_for(lock, std::chrono::seconds(5), [&]() { return permit_a; }));
                lock.unlock();
                REQUIRE(bytes == sizeof(a));
                std::memcpy(out, a, bytes);
                return true;
            });
    });
    {
        std::unique_lock<std::mutex> lock(mutex);
        REQUIRE(cv.wait_for(lock, std::chrono::seconds(5), [&]() { return inside_a; }));
    }
    step("chunk A reserved, attempting chunk B");
    REQUIRE(arena.capture(make_frame(1, 0, 0, 2),
        [&](uint8_t * out, size_t bytes) {
            REQUIRE(bytes == sizeof(b));
            std::memcpy(out, b, bytes);
            return true;
        }));
    step("chunk B captured");
    ggml_rpc_p2c_arena_queue::item item;
    REQUIRE(arena.take_for(item, std::chrono::milliseconds(2000)));
    REQUIRE(item.frame.region_id == 1);
    REQUIRE(item.frame.seq == 1);
    REQUIRE(std::memcmp(item.data, b, sizeof(b)) == 0);
    arena.complete_send(true);
    step("chunk B dequeued; testing duplicate rejection");

    // The same token interval must not be captured twice.
    REQUIRE(!arena.capture(make_frame(1, 1, 1, 1),
        [](uint8_t *, size_t) { return true; }));

    {
        std::lock_guard<std::mutex> lock(mutex);
        permit_a = true;
    }
    cv.notify_one();
    producer_a.join();
    step("chunk A capture finished");
    REQUIRE(a_ok);
    REQUIRE(arena.take_for(item, std::chrono::milliseconds(2000)));
    REQUIRE(item.frame.region_id == 0);
    REQUIRE(item.frame.seq == 2);
    REQUIRE(std::memcmp(item.data, a, sizeof(a)) == 0);
    arena.complete_send(true);
    step("chunk A dequeued; draining stream");
    arena.finish();
    REQUIRE(!arena.take(item));
    arena.shutdown();
}

struct fake_socket {
    std::vector<uint8_t> bytes;
    size_t offset = 0;

    bool recv_data(void * out, size_t n) {
        if (n > bytes.size() - offset) {
            return false;
        }
        std::memcpy(out, bytes.data() + offset, n);
        offset += n;
        return true;
    }
};

void append(std::vector<uint8_t> & wire, const void * src, size_t n) {
    const auto * p = static_cast<const uint8_t *>(src);
    wire.insert(wire.end(), p, p + n);
}

struct receive_context {
    float outputs[regions][tokens * hidden] {};
    std::vector<uint64_t> completed;
    bool failed = false;
};

void * resolve(void * userdata, const frame * f) {
    auto * ctx = static_cast<receive_context *>(userdata);
    return ctx->outputs[f->region_id] + f->token_begin * hidden;
}

void complete(void * userdata, const frame * f, bool ok) {
    auto * ctx = static_cast<receive_context *>(userdata);
    if (ok) {
        ctx->completed.push_back(f->seq);
    } else {
        ctx->failed = true;
    }
}

void test_framing_and_partial_receive_failure() {
    float a[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    float b[8] = { 9, 10, 11, 12, 13, 14, 15, 16 };
    frame fa = make_frame(1, 0, 2, 2, 1);
    frame fb = make_frame(0, 0, 0, 2, 2);
    frame end {};
    end.run_id = run_id;
    end.seq = 3;
    end.kind = GGML_BACKEND_RPC_P2C_END;

    auto sock = std::make_shared<fake_socket>();
    append(sock->bytes, &fa, sizeof(fa));
    append(sock->bytes, a, sizeof(a));
    append(sock->bytes, &fb, sizeof(fb));
    append(sock->bytes, b, sizeof(b));
    append(sock->bytes, &end, sizeof(end));

    receive_context ctx;
    ggml_rpc_p2c_stream_receiver receiver(
        run_id, regions, tokens, hidden, resolve, complete, &ctx);
    REQUIRE(receiver.receive(sock));
    REQUIRE((ctx.completed == std::vector<uint64_t>{ 1, 2 }));
    REQUIRE(!ctx.failed);
    REQUIRE(std::memcmp(ctx.outputs[1] + 2 * hidden, a, sizeof(a)) == 0);
    REQUIRE(std::memcmp(ctx.outputs[0], b, sizeof(b)) == 0);

    auto broken = std::make_shared<fake_socket>();
    append(broken->bytes, &fa, sizeof(fa));
    append(broken->bytes, a, sizeof(a) - sizeof(float));
    receive_context failed_ctx;
    ggml_rpc_p2c_stream_receiver broken_receiver(
        run_id, regions, tokens, hidden, resolve, complete, &failed_ctx);
    REQUIRE(!broken_receiver.receive(broken));
    REQUIRE(failed_ctx.failed);
}
}

int main(int argc, char ** argv) {
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::fprintf(stderr, "[P2C_TEST_BUILD] 20261031-bounded-queue-v2\n");
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
        return 0;
    }
    std::fprintf(stderr, "[P2C_TEST] arena publication and FIFO order\n");
    test_arena_publication_and_send_order();
    std::fprintf(stderr, "[P2C_TEST] framed receive and truncated payload\n");
    test_framing_and_partial_receive_failure();
    std::fprintf(stderr, "[P2C_TEST] PASS\n");
    return 0;
}
