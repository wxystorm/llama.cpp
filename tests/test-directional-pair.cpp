#include "llama-directional-pair.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "[DIRECTIONAL_PAIR_TEST] line=%d failed=%s\n", \
                     __LINE__, #cond); \
        std::abort(); \
    } \
} while (0)

static void check_layout(int layers, int c2p, int p2c) {
    const auto p = llama_directional_pair_plan::make(layers, c2p, p2c);
    REQUIRE(p.n_layers == layers);
    REQUIRE(p.phone_begin() == c2p + 1);
    REQUIRE(p.phone_end() == p2c);
    REQUIRE(p.pc_suffix_begin() == p2c + 1);
    REQUIRE(p.pc_suffix_end() == layers);
    REQUIRE(p.pc_prefix_layers() + 1 + p.phone_layers() + 1 +
            p.pc_suffix_layers() == layers);
    REQUIRE(p.pc_suffix_layers() >= 1);
    REQUIRE(p.phone_layers() >= 1);

    int counts[5] = {};
    for (int il = 0; il < layers; ++il) {
        const auto stage = p.stage_at(il);
        const auto slot = (int) stage;
        REQUIRE(slot >= 0 && slot < 5);
        ++counts[slot];
        if (stage == llama_directional_pair_stage::PC_PREFIX) {
            REQUIRE(il < c2p);
        } else if (stage == llama_directional_pair_stage::C2P_TENSOR) {
            REQUIRE(il == c2p);
        } else if (stage == llama_directional_pair_stage::PHONE_FULL) {
            REQUIRE(il > c2p && il < p2c);
        } else if (stage == llama_directional_pair_stage::P2C_TENSOR) {
            REQUIRE(il == p2c);
        } else {
            REQUIRE(il > p2c);
        }
    }
    REQUIRE(counts[(int) llama_directional_pair_stage::C2P_TENSOR] == 1);
    REQUIRE(counts[(int) llama_directional_pair_stage::P2C_TENSOR] == 1);
    REQUIRE(counts[(int) llama_directional_pair_stage::PHONE_FULL] == p2c-c2p-1);
    REQUIRE(counts[(int) llama_directional_pair_stage::PC_SUFFIX] == layers-p2c-1);
}

static void must_reject(int n, int c2p, int p2c) {
    try {
        const auto p = llama_directional_pair_plan::make(n, c2p, p2c);
        (void) p;
    } catch (const std::invalid_argument &) {
        return;
    }
    std::fprintf(stderr, "[DIRECTIONAL_PAIR_TEST] invalid accepted n=%d c2p=%d p2c=%d\n",
                 n, c2p, p2c);
    std::abort();
}

int main() {
    // Current Qwen3-30B-A3B 48-layer boundary: C2P L34, Phone L35-45,
    // P2C L46, PC tail L47.
    check_layout(48,34,46);
    // Alternative two-way topologies are still accepted at the *policy*
    // level, but may not use the specialized L34/L46 Phone pipeline.
    check_layout(48,11,27);
    check_layout(48,0,46);
    check_layout(5,0,3);
    must_reject(4,0,2);
    must_reject(48,-1,46);
    must_reject(48,34,35);
    must_reject(48,34,48);
    must_reject(48,34,47);
    must_reject(48,46,34);
    const auto p = llama_directional_pair_plan::make(48,34,46);
    bool threw = false;
    try {
        (void) p.stage_at(48);
    } catch (const std::out_of_range &) {
        threw = true;
    }
    REQUIRE(threw);
    std::puts("[DIRECTIONAL_PAIR_TEST] status=PASS layouts=4 invalid=6");
    return 0;
}
