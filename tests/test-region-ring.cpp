#include "llama-region-ring.h"

#include <array>
#include <cassert>
#include <stdexcept>
#include <vector>

int main() {
    const auto regions = llama_region_ring_parse("", 48);
    assert(regions.size() == 9);
    const auto counts = llama_region_ring_layer_counts(regions);
    assert(counts[0] == 22 && counts[1] == 11 && counts[2] == 15);
    assert(regions[0].layer_begin == 0 && regions[0].layer_end == 7);
    assert(regions[8].layer_begin == 43 && regions[8].layer_end == 48);

    for (const auto & bad : {"C7,G4,P5,C7,G4,P5,C8,G3,P6",
                            "C7,P4,G5,C7,G4,P5,C8,G3,P5",
                            "C7,G4,P5,C7,G4,P5,C8,G3",
                            "C7,G4,P5,C7,G4,P5,C8,G3,P5,"}) {
        bool rejected = false;
        try { llama_region_ring_parse(bad, 48); }
        catch (const std::runtime_error &) { rejected = true; }
        assert(rejected);
    }

    llama_region_ring_ready ready(regions, 358, 128);
    assert(ready.n_chunks() == 3);
    assert(ready.n_regions() == 9);

    std::vector<std::vector<int>> finished(
        9, std::vector<int>(3, 0));
    size_t popped = 0;
    while (ready.has_ready()) {
        const auto job = ready.take();
        assert(job);
        assert(job->token_begin == (int) job->chunk * 128);
        assert(job->n_tokens == (job->chunk == 2 ? 102 : 128));
        if (job->region > 0) {
            assert(finished[job->region - 1][job->chunk] == 1);
        }
        if (job->chunk > 0) {
            assert(finished[job->region][job->chunk - 1] == 1);
        }
        assert(finished[job->region][job->chunk] == 0);
        finished[job->region][job->chunk] = 1;
        ready.complete(*job);
        ++popped;
    }
    assert(popped == 27);
    assert(ready.all_done());
    assert(!ready.take());

    // A region must not release its second chunk before the first completes.
    llama_region_ring_ready one(regions, 256, 128);
    const auto first = one.take();
    assert(first && first->region == 0 && first->chunk == 0);
    assert(!one.has_ready());
    one.complete(*first);
    assert(one.queue_size(llama_region_ring_device::CPU) == 1);
    assert(one.queue_size(llama_region_ring_device::GPU) == 1);
    bool duplicate_rejected = false;
    try { one.complete(*first); }
    catch (const std::runtime_error &) { duplicate_rejected = true; }
    assert(duplicate_rejected);
    return 0;
}
