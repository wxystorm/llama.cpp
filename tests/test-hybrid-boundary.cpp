#include "llama-hybrid.h"

#include <cassert>

int main() {
    {
        const auto blocks = llama_hybrid_plan_boundary(128, 128, 128);
        assert(blocks.size() == 1);
        assert(blocks[0].action == llama_hybrid_boundary_action::PASS);
        assert(blocks[0].inputs.size() == 1);
        assert(blocks[0].output.token_begin == 0 && blocks[0].output.n_tokens == 128);
    }

    {
        const auto blocks = llama_hybrid_plan_boundary(128, 128, 64);
        assert(blocks.size() == 2);
        assert(blocks[0].action == llama_hybrid_boundary_action::SPLIT);
        assert(blocks[1].action == llama_hybrid_boundary_action::SPLIT);
        assert(blocks[0].inputs.size() == 1 && blocks[1].inputs.size() == 1);
    }

    {
        const auto blocks = llama_hybrid_plan_boundary(128, 32, 128);
        assert(blocks.size() == 1);
        assert(blocks[0].action == llama_hybrid_boundary_action::ACCUMULATE);
        assert(blocks[0].inputs.size() == 4);
    }

    {
        const auto blocks = llama_hybrid_plan_boundary(130, 128, 64);
        assert(blocks.size() == 3);
        assert(blocks[0].action == llama_hybrid_boundary_action::SPLIT);
        assert(blocks[1].action == llama_hybrid_boundary_action::SPLIT);
        assert(blocks[2].action == llama_hybrid_boundary_action::PASS);
        assert(blocks[2].output.token_begin == 128 && blocks[2].output.n_tokens == 2);
    }

    {
        const auto blocks = llama_hybrid_plan_boundary(160, 96, 64);
        assert(blocks.size() == 3);
        assert(blocks[0].action == llama_hybrid_boundary_action::SPLIT);
        assert(blocks[1].action == llama_hybrid_boundary_action::ACCUMULATE);
        assert(blocks[1].inputs.size() == 2);
        assert(blocks[2].action == llama_hybrid_boundary_action::SPLIT);
    }

    assert(llama_hybrid_plan_boundary(0, 128, 64).empty());
    assert(llama_hybrid_plan_boundary(128, 0, 64).empty());
    assert(llama_hybrid_plan_boundary(128, 128, 0).empty());

    return 0;
}
