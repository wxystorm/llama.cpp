// Verifies the per-thread Phone Full Arena return scope without live RPC.
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <thread>

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "META_PHONE_FULL_STREAM_MODE_TEST_FAIL line=%d expr=%s\n", \
                     __LINE__, #expr); \
        std::abort(); \
    } \
} while (0)

int main() {
    REQUIRE(ggml_backend_meta_set_phone_full_stream_return_layer(-1) == -1);
    REQUIRE(!ggml_backend_meta_phone_full_stream_return_layer_matches(-1));
    REQUIRE(!ggml_backend_meta_phone_full_stream_return_layer_matches(12));

    const int outer = ggml_backend_meta_set_phone_full_stream_return_layer(12);
    REQUIRE(outer == -1);
    REQUIRE(ggml_backend_meta_phone_full_stream_return_layer_matches(12));
    REQUIRE(!ggml_backend_meta_phone_full_stream_return_layer_matches(26));

    // A parallel worker must start in legacy mode even while this worker
    // holds an active Phone Full return scope.
    std::thread worker([] {
        REQUIRE(!ggml_backend_meta_phone_full_stream_return_layer_matches(12));
        REQUIRE(ggml_backend_meta_set_phone_full_stream_return_layer(26) == -1);
        REQUIRE(ggml_backend_meta_phone_full_stream_return_layer_matches(26));
        REQUIRE(!ggml_backend_meta_phone_full_stream_return_layer_matches(12));
        REQUIRE(ggml_backend_meta_set_phone_full_stream_return_layer(-1) == 26);
        REQUIRE(!ggml_backend_meta_phone_full_stream_return_layer_matches(26));
    });
    worker.join();

    REQUIRE(ggml_backend_meta_phone_full_stream_return_layer_matches(12));
    REQUIRE(ggml_backend_meta_set_phone_full_stream_return_layer(-1) == 12);
    REQUIRE(!ggml_backend_meta_phone_full_stream_return_layer_matches(12));
    return 0;
}
