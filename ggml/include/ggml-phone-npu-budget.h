#pragma once

// Phone HTP0 memory policy calibrated by Snapdragon 8 Elite measurements.
// Separate from the Hexagon DSP active mapping budget (GGML_HEXAGON_VMEM):
// this is the upper bound on concurrently committed NPU-owned allocations.
// Do not count Adreno weights against it; both are charged independently
// against total Phone shared RAM by the PC Planner.
#include <cstddef>

namespace ggml_phone_npu_budget {

constexpr size_t mib = size_t(1024) * 1024;
constexpr size_t total_mib = 3072;
constexpr size_t bank_mib = 1536;

// Resident: all NPU weights + NPU KV + graph/activation storage <= total.
// Streaming: two physical banks + the same non-weight overhead <= total.
// Each individual NPU stage, including its own auxiliary tensors, <= bank.
constexpr size_t total_bytes = total_mib * mib;
constexpr size_t bank_bytes = bank_mib * mib;

static_assert(bank_bytes * 2 == total_bytes,
              "HTP two-bank capacity must equal total NPU allowance");

} // namespace ggml_phone_npu_budget
