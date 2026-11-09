#pragma once

// Exact, process-independent cache for the existing Hybrid Profile used by
// apply-wide. It is NOT the V2 native GCNA Profile: this cache only prevents
// repeated legacy probing while per-device V2 callbacks are being integrated.
// The caller must key the path by GGUF, executable, hardware, RPC endpoint and
// thread count. The format is binary/ABI-specific and intentionally versioned.

#include "llama-hybrid.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace llama_ring_profile_cache {

inline constexpr char MAGIC[12] = "RR_HYBRID_1";
inline constexpr uint32_t VERSION = 1;
inline constexpr uint64_t MAX_POINTS = 100000;

template <class T>
bool save_field(std::ostream & out, const T & value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "profile cache scalar must be trivially copyable");
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
    return bool(out);
}

template <class T>
bool load_field(std::istream & in, T & value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "profile cache scalar must be trivially copyable");
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    return bool(in);
}

template <class T>
bool save_field(std::ostream & out, const std::vector<T> & values) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "profile cache point must be trivially copyable");
    if (values.size() > MAX_POINTS) { return false; }
    const uint64_t count = values.size();
    if (!save_field(out, count)) { return false; }
    for (const auto & item : values) {
        if (!save_field(out, item)) { return false; }
    }
    return true;
}

template <class T>
bool load_field(std::istream & in, std::vector<T> & values) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "profile cache point must be trivially copyable");
    uint64_t count = 0;
    if (!load_field(in, count) || count > MAX_POINTS) { return false; }
    values.clear();
    values.reserve((size_t) count);
    for (uint64_t i = 0; i < count; ++i) {
        T item {};
        if (!load_field(in, item)) { return false; }
        values.push_back(item);
    }
    return true;
}

// Keep one authoritative field order for both writing and reading.
// A changed layout of any POD point type invalidates the caller's binary
// fingerprint, and an explicit VERSION bump remains required.
template <class P, class F>
bool fields(P & p, F && f) {
    return
        f(p.is_moe) && f(p.dense_phone_region_capable) &&
        f(p.probe_tokens) && f(p.reference_tokens) &&
        f(p.probe_chunk_min_tokens) && f(p.profile_block_layers) &&
        f(p.n_layer) && f(p.n_embd) && f(p.n_ff) &&
        f(p.ffn_shard_granularity) &&
        f(p.cpu_ffn) && f(p.phone_ffn) &&
        f(p.cpu_attn) && f(p.phone_attn) && f(p.gpu_attn) &&
        f(p.cpu_layer_blocks) && f(p.phone_layer_blocks) &&
        f(p.gpu_layer_blocks) && f(p.decode_ratio_blocks) &&
        f(p.phone_blocks) &&
        f(p.cpu_attn_ms) && f(p.cpu_full_layer_ms) &&
        f(p.phone_attn_ms) && f(p.phone_full_layer_ms) &&
        f(p.phone_full_layer_compute_est_ms) && f(p.gpu_full_layer_ms) &&
        f(p.decode_tail_norm_ms) && f(p.decode_tail_lm_head_ms) &&
        f(p.decode_tail_ms) &&
        f(p.gpu_to_pc) && f(p.gpu_weight_h2d) &&
        f(p.pc_to_phone) && f(p.snapshot_phone_to_pc) &&
        f(p.phone_to_pc) && f(p.dual_phone_to_pc) &&
        f(p.duplex_phone_link) &&
        f(p.reduce_ms) && f(p.rpc_submit_ms) && f(p.rpc_fence_ms) &&
        f(p.pc_free_mem) && f(p.phone_free_mem) &&
        f(p.gpu_free_mem) && f(p.pc_total_mem) &&
        f(p.phone_total_mem) && f(p.gpu_total_mem) &&
        f(p.model_weight_bytes) &&
        f(p.attn_weight_bytes_per_layer) &&
        f(p.ffn_weight_bytes_per_layer) &&
        f(p.total_weight_bytes_per_layer) &&
        f(p.non_layer_weight_bytes) &&
        f(p.layer_weight_bytes) &&
        f(p.layer_attn_forced_bytes) &&
        f(p.layer_ffn_split_bytes) &&
        f(p.layer_mirrored_bytes) &&
        f(p.kv_bytes_per_token_per_layer) && f(p.n_ctx_train);
}

inline bool valid(const llama_hybrid_profile & p) {
    return p.n_layer > 0 && p.n_layer <= 1024 &&
           p.n_embd > 0 && p.n_ff > 0 &&
           !p.cpu_layer_blocks.empty() && !p.gpu_layer_blocks.empty() &&
           !p.phone_layer_blocks.empty() &&
           !p.pc_to_phone.empty() && !p.phone_to_pc.empty() &&
           p.layer_weight_bytes.size() == (size_t) p.n_layer;
}

inline bool save(const std::string & path,
                 const llama_hybrid_profile & profile) {
    if (path.empty() || !valid(profile)) { return false; }
    const std::string temporary = path + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) { return false; }
    out.write(MAGIC, sizeof(MAGIC));
    const uint32_t schema = VERSION;
    bool ok = bool(out) && save_field(out, schema) &&
        fields(profile, [&](const auto & field) {
            return save_field(out, field);
        });
    out.flush();
    ok = ok && bool(out);
    out.close();
    if (!ok) {
        std::remove(temporary.c_str());
        return false;
    }
    // On Windows std::rename cannot replace an existing destination.
    // The calling coordinator requests a new identity-specific path for
    // binary changes; explicit refresh intentionally removes the old cache.
    std::remove(path.c_str());
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        return false;
    }
    return true;
}

inline bool load(const std::string & path, llama_hybrid_profile & profile) {
    if (path.empty()) { return false; }
    std::ifstream in(path, std::ios::binary);
    if (!in) { return false; }
    char magic[sizeof(MAGIC)] {};
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0) {
        return false;
    }
    uint32_t schema = 0;
    if (!load_field(in, schema) || schema != VERSION) { return false; }
    llama_hybrid_profile candidate {};
    if (!fields(candidate, [&](auto & field) {
            return load_field(in, field);
        })) {
        return false;
    }
    if (!valid(candidate) || in.peek() != std::char_traits<char>::eof()) {
        return false;
    }
    profile = std::move(candidate);
    return true;
}

} // namespace llama_ring_profile_cache
