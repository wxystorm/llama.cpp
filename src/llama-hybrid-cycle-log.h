#pragma once

#include <cstdlib>
#include <cstring>

// Cycle's low-level physical workers used to print successful stages directly
// to stdout, interleaving with llama-cli generated text. Successful stage
// traces are opt-in; errors and correctness/parity diagnostics remain visible.
// Parse an explicit "1" instead of accepting any nonempty value so setting
// the variable to "0" reliably disables tracing.
inline bool llama_hybrid_cycle_verbose() {
    const char * value = std::getenv("LLAMA_HYBRID_CYCLE_VERBOSE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

// Independent opt-in for per-forward five-worker measurements. Collecting a
// full task timeline costs memory and a host timestamp for every task, so
// production inference leaves it off until performance profiling is requested.
inline bool llama_hybrid_cycle_queue_profile_enabled() {
    const char * value = std::getenv("LLAMA_HYBRID_CYCLE_QUEUE_PROFILE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}
