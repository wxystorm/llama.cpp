#include "llama-hybrid-cycle-execution-session.h"

#include <cstdint>
#include <limits>
#include <string>

bool llama_hybrid_cycle_validate_session_config(
        const llama_hybrid_cycle_execution_session_config & config,
        std::string & reason) {
    reason.clear();
    if (!config.layout.enabled() ||
        config.physical_layer_begin < 0 ||
        config.physical_layer_count <= 0 ||
        int64_t(config.physical_layer_begin) +
                config.physical_layer_count >
            int64_t(config.layout.kinds.size())) {
        reason = "INVALID_PHYSICAL_LAYER_RANGE";
        return false;
    }
    if (config.tokens <= 0 || config.chunk_tokens <= 0 ||
        config.token_position_base < 0 ||
        config.tokens > std::numeric_limits<int64_t>::max() -
                        config.token_position_base) {
        reason = "INVALID_TOKEN_WINDOW";
        return false;
    }
    const int64_t chunks =
        1 + (config.tokens - 1) / int64_t(config.chunk_tokens);
    if (chunks > 999) {
        reason = "WIRE_CHUNK_ID_OUT_OF_RANGE";
        return false;
    }
    const int64_t last_layer = int64_t(config.physical_layer_begin) +
                               config.physical_layer_count - 1;
    if (last_layer >= 99) {
        reason = "WIRE_LAYER_ID_OUT_OF_RANGE";
        return false;
    }
    const uint64_t last_suffix =
        uint64_t(last_layer + 1) * 1000 + uint64_t(chunks);
    if (config.wire_epoch >
            (std::numeric_limits<uint64_t>::max() - last_suffix) / 100000) {
        reason = "WIRE_EPOCH_OVERFLOW";
        return false;
    }
    return true;
}

uint64_t llama_hybrid_cycle_session_wire_seq(
        uint64_t epoch, int physical_layer, int chunk) {
    if (physical_layer < 0 || physical_layer >= 99 ||
        chunk < 0 || chunk >= 999 ||
        epoch > (std::numeric_limits<uint64_t>::max() -
                 (uint64_t(physical_layer + 1) * 1000 +
                  uint64_t(chunk + 1))) / 100000) {
        return 0;
    }
    return epoch * 100000 + uint64_t(physical_layer + 1) * 1000 +
           uint64_t(chunk + 1);
}
