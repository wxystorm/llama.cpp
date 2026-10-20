#include "llama-hybrid-cycle.h"

#include <cassert>
#include <stdexcept>

static void check(int n, int x, int y) {
    const auto a = llama_hybrid_build_cycle_layout(n, x, y);
    assert(a.enabled());
    assert(int(a.kinds.size()) == n);
    assert(a.cycle_size == x + y + 2);
    assert(a.cycles == n / a.cycle_size);
    assert(a.tail_begin == a.cycles * a.cycle_size);
    assert(a.remainder() == n % a.cycle_size);
    for (int l = 0; l < n; ++l) {
        auto expected = llama_hybrid_cycle_layer_kind::PC_CUDA_TAIL;
        if (l < a.tail_begin) {
            const int k = l % a.cycle_size;
            expected = k < x ? llama_hybrid_cycle_layer_kind::PC_CPU :
                       k == x ? llama_hybrid_cycle_layer_kind::TENSOR_C2P :
                       k < x + 1 + y ? llama_hybrid_cycle_layer_kind::PHONE_GPU :
                       llama_hybrid_cycle_layer_kind::TENSOR_P2C;
        }
        assert(a.kinds[l] == expected);
        if (l > 0) {
            const auto prev = llama_hybrid_cycle_output_owner(a.kinds[l-1]);
            const auto next = llama_hybrid_cycle_attn_owner(a.kinds[l]);
            assert(prev == next || (prev == llama_hybrid_cycle_owner::PC_CPU &&
                                    next == llama_hybrid_cycle_owner::PC_CUDA));
        }
    }
}

static void bad(int n, int x, int y) {
    try {
        (void) llama_hybrid_build_cycle_layout(n, x, y);
        assert(false && "invalid cycle layout accepted");
    } catch (const std::invalid_argument &) {
    }
}

int main() {
    check(48, 10, 8);
    check(40, 10, 8);
    check(41, 10, 8);
    check(4, 1, 1);
    check(48, 1, 1);
    check(48, 3, 2);
    bad(3, 1, 1);
    bad(48, 0, 8);
    bad(48, 10, -1);
    bad(48, 2147483647, 2147483647);
    bad(19, 10, 8);
    return 0;
}
