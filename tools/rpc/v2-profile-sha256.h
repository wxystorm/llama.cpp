#pragma once

// Small self-contained SHA-256 for exact PC/Phone GGUF fingerprint matching.
// Avoids OpenSSL, external executables, and Python on Android.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace v2_profile_hash {

class sha256 {
public:
    sha256() : state_{{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u}} {}

    void update(const uint8_t * data, size_t bytes) {
        if (finished_) { throw std::runtime_error("SHA256_ALREADY_FINAL"); }
        for (size_t n = 0; n < bytes; ++n) {
            block_[used_++] = data[n];
            if (used_ == 64) {
                compress();
                used_ = 0;
            }
        }
        total_ += bytes;
    }

    std::string finish() {
        if (finished_) { throw std::runtime_error("SHA256_ALREADY_FINAL"); }
        finished_ = true;
        const uint64_t bits = uint64_t(total_) * 8;
        block_[used_++] = 0x80;
        if (used_ > 56) {
            while (used_ != 64) { block_[used_++] = 0; }
            compress();
            used_ = 0;
        }
        while (used_ < 56) { block_[used_++] = 0; }
        for (int i = 7; i >= 0; --i) {
            block_[used_++] = static_cast<uint8_t>(bits >> (i * 8));
        }
        compress();
        const char * hex = "0123456789abcdef";
        std::string result;
        result.reserve(64);
        for (uint32_t word : state_) {
            for (int i = 7; i >= 0; --i) {
                const uint8_t digit = (word >> (i * 4)) & 15;
                result.push_back(hex[digit]);
            }
        }
        return result;
    }

private:
    static uint32_t rotr(uint32_t n, unsigned shift) {
        return (n >> shift) | (n << (32 - shift));
    }

    void compress() {
        static constexpr uint32_t k[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
            0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
            0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
            0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
            0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
            0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
            0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
            0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
            0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u,
        };
        uint32_t w[64] = {};
        for (int i = 0; i < 16; ++i) {
            const int j = i * 4;
            w[i] = (uint32_t(block_[j]) << 24) |
                   (uint32_t(block_[j + 1]) << 16) |
                   (uint32_t(block_[j + 2]) << 8) |
                   uint32_t(block_[j + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t x = w[i - 15];
            const uint32_t y = w[i - 2];
            const uint32_t s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
            const uint32_t s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a=state_[0],b=state_[1],c=state_[2],d=state_[3];
        uint32_t e=state_[4],f=state_[5],g=state_[6],h=state_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            const uint32_t choose = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + s1 + choose + k[i] + w[i];
            const uint32_t s0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h=g;g=f;f=e;e=d+temp1;d=c;c=b;b=a;a=temp1+temp2;
        }
        state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;
        state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
    }

    std::array<uint32_t, 8> state_ {};
    std::array<uint8_t, 64> block_ {};
    size_t used_ = 0;
    size_t total_ = 0;
    bool finished_ = false;
};

inline std::string string_sha256(const std::string & input) {
    sha256 hash;
    hash.update(reinterpret_cast<const uint8_t *>(input.data()), input.size());
    return hash.finish();
}

inline std::string file_sha256(const std::string & path) {
    FILE * file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        throw std::runtime_error("SHA256_FILE_OPEN_FAILED:" + path);
    }
    sha256 hash;
    std::array<uint8_t, 1024 * 1024> chunk {};
    while (true) {
        const size_t n = std::fread(chunk.data(), 1, chunk.size(), file);
        if (n) { hash.update(chunk.data(), n); }
        if (n < chunk.size()) {
            const bool failed = std::ferror(file) != 0;
            std::fclose(file);
            if (failed) { throw std::runtime_error("SHA256_FILE_READ_FAILED:" + path); }
            break;
        }
    }
    return hash.finish();
}

} // namespace v2_profile_hash
