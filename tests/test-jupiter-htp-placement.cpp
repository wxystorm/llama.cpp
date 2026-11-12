#include "../src/jupiter-htp-placement.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

static void validate(const std::vector<uint64_t> & layers, uint64_t output,
                     const std::vector<uint64_t> & caps) {
    const auto plan = jupiter_htp::partition_contiguous(layers, output, caps);
    const size_t active = std::min(layers.size(), caps.size());
    assert(plan.ends.size() == active);
    assert(plan.used_bytes.size() == active);
    size_t previous = 0;
    for (size_t k = 0; k < active; ++k) {
        assert(plan.ends[k] > previous);
        assert(plan.ends[k] <= layers.size());
        uint64_t charge = (k + 1 == active ? output : 0);
        for (size_t i = previous; i < plan.ends[k]; ++i) {
            charge += layers[i];
        }
        assert(charge == plan.used_bytes[k]);
        assert(charge <= caps[k]);
        previous = plan.ends[k];
    }
    assert(previous == layers.size());
}

static bool infeasible(const std::vector<uint64_t> & layers, uint64_t output,
                       const std::vector<uint64_t> & caps) {
    try {
        (void) jupiter_htp::partition_contiguous(layers, output, caps);
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

int main() {
    // Four nonempty HTP regions with a large head on the final session.
    validate({200,200,200,200,200,200,200,200}, 600,
             {1600,1600,1600,1600});

    // The final session must reserve physical space for output.weight.
    const auto plan = jupiter_htp::partition_contiguous(
        {100,100,100,100,100,100,100,100}, 500,
        {650,650,650,650});
    assert(plan.ends.size() == 4);
    assert(plan.ends[2] == 7);
    assert(plan.ends[3] == 8);
    validate({100,100,100,100,100,100,100,100}, 500,
             {650,650,650,650});

    // Heterogeneous memory budgets, nonuniform weight sizes and a final head.
    validate({270, 280, 250, 330, 290, 190, 310}, 410,
             {1050, 800, 1000, 800});

    // For fewer layers than devices, use only the needed prefix of sessions.
    validate({320, 350}, 430, {600, 900, 800, 800});

    // Must not accidentally overcommit the total weight+KV or the output.
    assert(infeasible({800,800,800,800}, 600, {1000,1000,1000,1000}));
    assert(infeasible({200,200}, 900, {1000,1000}));

    // Input/size errors are diagnosed rather than wrapping an address.
    bool rejected = false;
    try {
        (void) jupiter_htp::partition_contiguous({0,100}, 0, {1000});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
    rejected = false;
    try {
        (void) jupiter_htp::partition_contiguous(
            {UINT64_MAX, 100}, 0, {UINT64_MAX});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
    return 0;
}
