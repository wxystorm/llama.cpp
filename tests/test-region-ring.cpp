#include "llama-region-ring.h"

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

// Never use assert() here: CTest runs this test in Release and NDEBUG
// removes all assertions, potentially hiding broken scheduler invariants.
static void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void check_rejected(const char * layout) {
    bool rejected = false;
    try {
        llama_region_ring_parse(layout, 48);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected, "invalid layout was accepted");
}

static void test_layout() {
    const auto regions = llama_region_ring_parse("", 48);
    check(regions.size() == 9, "default region count should be 9");

    const auto counts = llama_region_ring_layer_counts(regions);
    check(counts[0] == 22 && counts[1] == 11 && counts[2] == 15,
          "default device layer counts do not match C22/G11/P15");
    check(regions.front().layer_begin == 0 &&
          regions.front().layer_end == 7, "default first region is incorrect");
    check(regions.back().layer_begin == 43 &&
          regions.back().layer_end == 48, "default last region is incorrect");

    check_rejected("C7,G4,P5,C7,G4,P5,C8,G3,P6");
    check_rejected("C7,P4,G5,C7,G4,P5,C8,G3,P5");
    check_rejected("C7,G4,P5,C7,G4,P5,C8,G3");
    check_rejected("C7,G4,P5,C7,G4,P5,C8,G3,P5,");
    check_rejected("C7,G4,P5,C7,G4,P5,C8,G3,P5, ");
    const auto spaced =
        llama_region_ring_parse(" C7, G4, P5, C7, G4, P5, C8, G3, P5  ", 48);
    check(spaced.size() == 9, "valid layout with spaces was rejected");

    // Verify the proposed two-cycle A/B layout as well.
    const auto alternative =
        llama_region_ring_parse("C11,G6,P7,C11,G5,P8", 48);
    const auto other_counts = llama_region_ring_layer_counts(alternative);
    check(alternative.size() == 6, "two-cycle layout region count");
    check(other_counts[0] == 22 && other_counts[1] == 11 &&
          other_counts[2] == 15, "two-cycle layout device totals");
}

static void test_dependencies() {
    const auto regions = llama_region_ring_parse("", 48);
    llama_region_ring_ready ready(regions, 358, 128);
    check(ready.n_chunks() == 3, "358 tokens must form 3 chunks");
    check(ready.n_regions() == 9, "expected 9 regions");

    std::vector<std::vector<int>> finished(9, std::vector<int>(3, 0));
    size_t completed = 0;
    while (ready.has_ready()) {
        const auto job = ready.take();
        check(job.has_value(), "has_ready but take returned no task");
        check(job->region < finished.size() &&
              job->chunk < finished[job->region].size(),
              "queued task has an out-of-bounds region or chunk");
        check(job->token_begin == (int) job->chunk * 128,
              "task token offset is wrong");
        check(job->n_tokens == (job->chunk == 2 ? 102 : 128),
              "task chunk size is wrong");
        if (job->region > 0) {
            check(finished[job->region - 1][job->chunk] == 1,
                  "downstream region ran before upstream completed");
        }
        if (job->chunk > 0) {
            check(finished[job->region][job->chunk - 1] == 1,
                  "KV chunk was scheduled out of order");
        }
        check(finished[job->region][job->chunk] == 0,
              "task was scheduled more than once");
        finished[job->region][job->chunk] = 1;
        ready.complete(*job);
        ++completed;
    }
    check(completed == 27, "9 regions x 3 chunks should execute 27 tasks");
    check(ready.all_done(), "scheduler drained before all tasks completed");
    check(!ready.take().has_value(), "scheduler returned a task after drain");
}

static void test_duplicate_completion() {
    const auto regions = llama_region_ring_parse("", 48);
    llama_region_ring_ready ready(regions, 256, 128);
    const auto first = ready.take();
    check(first.has_value(), "first task was not available");
    check(first->region == 0 && first->chunk == 0,
          "unexpected first task");
    check(!ready.has_ready(), "another task became ready too early");
    ready.complete(*first);
    check(ready.queue_size(llama_region_ring_device::CPU) == 1,
          "CPU chunk 1 was not queued");
    check(ready.queue_size(llama_region_ring_device::GPU) == 1,
          "GPU chunk 0 was not queued");
    bool rejected = false;
    try {
        ready.complete(*first);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected, "duplicate task completion was accepted");
}

int main() {
    std::fprintf(stderr, "[REGION_RING_TEST] begin\n");
    try {
        test_layout();
        std::fprintf(stderr, "[REGION_RING_TEST] layout=PASS\n");
        test_dependencies();
        std::fprintf(stderr, "[REGION_RING_TEST] dependencies=PASS\n");
        test_duplicate_completion();
        std::fprintf(stderr, "[REGION_RING_TEST] duplicate=PASS\n");
        std::fprintf(stderr, "[REGION_RING_TEST] status=PASS\n");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[REGION_RING_TEST] status=FAIL reason=%s\n", e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "[REGION_RING_TEST] status=FAIL reason=unknown exception\n");
        return 1;
    }
}
