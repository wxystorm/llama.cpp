#pragma once

// Directional hybrid topology: PC -> C2P -> Phone -> P2C -> PC.
// This is a pure topology/ownership description. The actual Tensor shard,
// residual reduction and RPC data planes are implemented by the existing
// Hybrid model policy and ggml-backend-meta.
#include <stdexcept>

enum class llama_directional_pair_stage {
    PC_PREFIX,
    C2P_TENSOR,
    PHONE_FULL,
    P2C_TENSOR,
    PC_SUFFIX,
};

struct llama_directional_pair_plan {
    int n_layers = 0;
    int c2p_layer = -1;
    int p2c_layer = -1;

    static llama_directional_pair_plan make(
            int n_layers, int c2p_layer, int p2c_layer) {
        if (n_layers < 5 || c2p_layer < 0 ||
                p2c_layer <= c2p_layer + 1 ||
                p2c_layer >= n_layers - 1) {
            throw std::invalid_argument(
                "directional pair requires 0 <= C2P < P2C-1 "
                "and P2C < n_layers-1");
        }
        return {n_layers, c2p_layer, p2c_layer};
    }

    int pc_prefix_begin() const { return 0; }
    int pc_prefix_end() const { return c2p_layer; }
    int phone_begin() const { return c2p_layer + 1; }
    int phone_end() const { return p2c_layer; }
    int pc_suffix_begin() const { return p2c_layer + 1; }
    int pc_suffix_end() const { return n_layers; }
    int phone_layers() const { return phone_end() - phone_begin(); }
    int pc_prefix_layers() const { return pc_prefix_end(); }
    int pc_suffix_layers() const { return n_layers - pc_suffix_begin(); }

    llama_directional_pair_stage stage_at(int layer) const {
        if (layer < 0 || layer >= n_layers) {
            throw std::out_of_range("directional pair layer out of range");
        }
        if (layer < c2p_layer) {
            return llama_directional_pair_stage::PC_PREFIX;
        }
        if (layer == c2p_layer) {
            return llama_directional_pair_stage::C2P_TENSOR;
        }
        if (layer < p2c_layer) {
            return llama_directional_pair_stage::PHONE_FULL;
        }
        if (layer == p2c_layer) {
            return llama_directional_pair_stage::P2C_TENSOR;
        }
        return llama_directional_pair_stage::PC_SUFFIX;
    }
};
