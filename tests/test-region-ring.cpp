#include "llama-region-ring.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <thread>
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
          regions.front().layer_end == 4, "default first region is incorrect");
    check(regions.back().layer_begin == 43 &&
          regions.back().layer_end == 48, "default last region is incorrect");

    check_rejected("G4,C7,P5,G4,C7,P5,G3,C8,P6");
    check_rejected("G4,P7,C5,G4,C7,P5,G3,C8,P5");
    check_rejected("G4,C7,P5,G4,C7,P5,G3,C8");
    check_rejected("G4,C7,P5,G4,C7,P5,G3,C8,P5,");
    check_rejected("G4,C7,P5,G4,C7,P5,G3,C8,P5, ");
    const auto spaced =
        llama_region_ring_parse(" G4, C7, P5, G4, C7, P5, G3, C8, P5  ", 48);
    check(spaced.size() == 9, "valid layout with spaces was rejected");

    // Verify the proposed two-cycle A/B layout as well.
    const auto alternative =
        llama_region_ring_parse("G6,C11,P7,G5,C11,P8", 48);
    const auto other_counts = llama_region_ring_layer_counts(alternative);
    check(alternative.size() == 6, "two-cycle layout region count");
    check(other_counts[0] == 22 && other_counts[1] == 11 &&
          other_counts[2] == 15, "two-cycle layout device totals");
}


static void test_model_aware_default_layout() {
    const auto dense = llama_region_ring_parse("", 40);
    check(llama_region_ring_layout_string(dense) ==
          "G4,C10,C2P1,P8,P2C1,G4,C7,P5",
          "40-layer Dense baseline changed");
    check(llama_region_ring_directional_topology_compatible(dense, 40),
          "40-layer Dense baseline is not an executable directional ring");

    // A generic baseline must cover the actual model depth, end on Phone,
    // and satisfy all full-stage runtime topology constraints.
    for (int layers : {8, 12, 28, 32, 64, 80, 96}) {
        const auto regions = llama_region_ring_parse("", layers);
        const auto counts = llama_region_ring_layer_counts(regions);
        check(counts[0] + counts[1] + counts[2] == layers,
              "automatic Ring baseline did not cover all model layers");
        check(!llama_region_ring_has_directional_stages(regions),
              "generic Ring fallback must contain only full regions");
        check(llama_region_ring_apply_topology_compatible(regions, layers),
              "automatic Ring baseline violates runtime stage constraints");
        check(regions.back().device == llama_region_ring_device::PHONE,
              "automatic Ring baseline must end on Phone");
    }

    // Explicit overrides must remain strict, even if they have stale counts.
    bool rejected = false;
    try {
        (void) llama_region_ring_parse(
            "G4,C7,P5,G4,C7,P5,G3,C8,P5", 40);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected, "stale explicit layout was silently accepted");
}

static void test_directional_layout() {
    const char * one_pair =
        "G4,C6,C2P1,P4,P2C1,G4,C7,P5,G3,C8,P5";
    const auto regions =
        llama_region_ring_parse(one_pair, 48);

    check(llama_region_ring_has_directional_stages(regions),
          "directional layout did not expose boundary stages");
    check(llama_region_ring_directional_topology_compatible(regions, 48),
          "valid directional topology rejected");
    check(regions.size() == 11,
          "one-pair directional layout stage count");
    check(regions[2].kind == llama_region_ring_stage_kind::C2P &&
          regions[2].layer_begin == 10 &&
          regions[2].layer_end == 11,
          "C2P boundary placement is wrong");
    check(regions[4].kind == llama_region_ring_stage_kind::P2C &&
          regions[4].layer_begin == 15 &&
          regions[4].layer_end == 16,
          "P2C boundary placement is wrong");
    check(regions[2].device == llama_region_ring_device::PHONE &&
          regions[4].device == llama_region_ring_device::PHONE,
          "directional stages must use the existing Meta/Phone executor");

    const auto counts = llama_region_ring_layer_counts(regions);
    check(counts == std::array<int, 3>{21, 11, 14},
          "directional full-layer counts are wrong");
    check(llama_region_ring_directional_layer_count(regions) == 2,
          "directional layer count is wrong");
    check(llama_region_ring_layout_string(regions) == one_pair,
          "directional layout serialization changed");

    // Multiple independent handoff pairs are legal; each boundary remains
    // exactly one layer while the surrounding full regions choose their size.
    const char * multi_pair =
        "G4,C6,C2P1,P4,P2C1,G4,C6,C2P1,P4,P2C1,G3,C7,C2P1,P5";
    const auto multi =
        llama_region_ring_parse(multi_pair, 48);
    check(multi.size() == 14,
          "multi-pair directional layout stage count");
    check(llama_region_ring_directional_layer_count(multi) == 5,
          "multi-pair boundary layer count");
    check(llama_region_ring_layout_string(multi) == multi_pair,
          "multi-pair layout serialization changed");

    bool wide_boundary_rejected = false;
    try {
        (void) llama_region_ring_parse(
            "G4,C5,C2P2,P4,P2C1,G4,C7,P5,G3,C8,P5", 48);
    } catch (const std::runtime_error &) {
        wide_boundary_rejected = true;
    }
    check(wide_boundary_rejected,
          "multi-layer C2P boundary was accepted");

    bool bad_direction_rejected = false;
    try {
        (void) llama_region_ring_parse(
            "G4,C6,P2C1,P4,C2P1,G4,C7,P5,G3,C8,P5", 48);
    } catch (const std::runtime_error &) {
        bad_direction_rejected = true;
    }
    check(bad_direction_rejected,
          "reversed directional boundary topology was accepted");

    // The scheduler still has only CPU/GPU/PHONE physical queues. C2P and P2C
    // must be claimable from the Phone/Meta queue, not from a new worker.
    llama_region_ring_ready ready(regions, 64, 64);
    const auto g = ready.try_take(llama_region_ring_device::GPU);
    check(g && g->region == 0, "directional scheduler missing initial GPU");
    ready.complete(*g);
    const auto cpu = ready.try_take(llama_region_ring_device::CPU);
    check(cpu && cpu->region == 1, "directional scheduler missing CPU stage");
    ready.complete(*cpu);
    const auto c2p = ready.try_take(llama_region_ring_device::PHONE);
    check(c2p && c2p->region == 2,
          "C2P did not route through the Meta/Phone executor");
    ready.complete(*c2p);
    const auto phone = ready.try_take(llama_region_ring_device::PHONE);
    check(phone && phone->region == 3,
          "Phone region did not follow C2P");
    ready.complete(*phone);
    const auto p2c = ready.try_take(llama_region_ring_device::PHONE);
    check(p2c && p2c->region == 4,
          "P2C did not route through the Meta/Phone executor");
}



static void test_general_topology() {
    const auto two_cycle_tail = llama_region_ring_parse(
        "G4,C6,P8,G4,C6,P8,G4,C8",48,true);
    check(two_cycle_tail.size() == 8,
          "two-cycle PC tail stage count");
    check(llama_region_ring_apply_topology_compatible(two_cycle_tail,48),
          "two-cycle PC tail must be supported");
    check(two_cycle_tail.back().device == llama_region_ring_device::CPU,
          "GPU/CPU tail must end on CPU");
    check(llama_region_ring_layer_counts(two_cycle_tail) ==
          std::array<int,3>{20,12,16},
          "two-cycle PC tail device totals");
    check(llama_region_ring_layout_string(two_cycle_tail) ==
          "G4,C6,P8,G4,C6,P8,G4,C8",
          "two-cycle tail Plan serialization");
    const auto gpu_tail = llama_region_ring_parse(
        "G4,C6,P8,G4,C6,P8,G12",48,true);
    check(gpu_tail.back().device == llama_region_ring_device::GPU,
          "GPU-only tail owner");
    check(llama_region_ring_apply_topology_compatible(gpu_tail,48),
          "GPU-only tail compatible");
    const auto phone_last = llama_region_ring_parse(
        "G4,C7,P5,G4,C7,P5,G3,C8,P5",48,true);
    check(llama_region_ring_apply_topology_compatible(phone_last,48),
          "original topology remains compatible");
    bool legacy_rejected = false;
    try {
        (void) llama_region_ring_parse(
            "G4,C6,P8,G4,C6,P8,G4,C8",48);
    } catch (const std::exception &) {
        legacy_rejected = true;
    }
    check(legacy_rejected,
          "legacy fixed Ring must still reject PC tail");
    bool invalid_rejected = false;
    try {
        (void) llama_region_ring_parse(
            "G4,C6,P8,G4,C6,P8,C12",48,true);
    } catch (const std::exception &) {
        invalid_rejected = true;
    }
    check(invalid_rejected,
          "general Ring must reject a CPU tail without a GPU stage");
}

static void test_apply_safe_compatibility() {
    const auto baseline = llama_region_ring_parse("", 48);
    check(llama_region_ring_apply_safe_phone_shape(baseline),
          "baseline must satisfy the apply-safe Phone shape");
    check(llama_region_ring_apply_safe_compatible(baseline,baseline),
          "baseline must be compatible with itself");
    check(llama_region_ring_layout_string(baseline) ==
          "G4,C7,P5,G4,C7,P5,G3,C8,P5",
          "canonical Ring encoding has changed");

    // Reassign exactly one CUDA layer between the first two PC windows;
    // Phone L11..15, L27..31, L43..47 and aggregate CUDA11 stay unchanged.
    const auto safe = llama_region_ring_parse(
        "G3,C8,P5,G5,C6,P5,G3,C8,P5",48);
    check(llama_region_ring_apply_safe_phone_shape(safe),
          "safe candidate changed the Phone regions");
    check(llama_region_ring_apply_safe_compatible(baseline,safe),
          "safe CPU/CUDA shift rejected");
    check(llama_region_ring_layout_string(safe) ==
          "G3,C8,P5,G5,C6,P5,G3,C8,P5",
          "safe candidate serialized incorrectly");

    const auto different_gpu_total = llama_region_ring_parse(
        "G3,C8,P5,G4,C7,P5,G3,C8,P5",48);
    check(!llama_region_ring_apply_safe_compatible(
            baseline,different_gpu_total),
          "candidate silently reduced CUDA weight count");
    const auto moved_phone = llama_region_ring_parse(
        "G4,C8,P5,G4,C6,P5,G3,C8,P5",48);
    check(!llama_region_ring_apply_safe_phone_shape(moved_phone),
          "moved Phone region passed fixed-graph check");
    check(!llama_region_ring_apply_safe_compatible(baseline,moved_phone),
          "moved Phone region passed compatibility check");
    const auto tail_or_different_cycles = llama_region_ring_parse(
        "G6,C11,P7,G5,C11,P8",48);
    check(!llama_region_ring_apply_safe_compatible(
            baseline,tail_or_different_cycles),
          "two-cycle alternative passed safe-apply constraints");
    check(!llama_region_ring_apply_safe_phone_shape(
            tail_or_different_cycles),
          "two-cycle alternative passed fixed Phone shape");
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

static void test_deferred_p2c_join_scheduler() {
    // A deferred producer has completed enough causal work to let the next
    // chunk of the same Region run, but its downstream activation does not
    // exist until the asynchronous P2C join publishes complete_deferred().
    const std::vector<llama_region_ring_region> regions = {
        {llama_region_ring_device::PHONE, 0, 1},
        {llama_region_ring_device::GPU,   1, 2},
    };
    llama_region_ring_ready ready(regions, 128, 64);

    const auto p0 =
        ready.try_take(llama_region_ring_device::PHONE);
    check(p0 && p0->region == 0 && p0->chunk == 0,
          "deferred test missing Phone chunk0");
    ready.defer(*p0);
    check(ready.finished() == 0,
          "deferred producer counted as fully completed");
    check(!ready.try_take(llama_region_ring_device::GPU).has_value(),
          "deferred producer released downstream before join");

    const auto p1 =
        ready.try_take(llama_region_ring_device::PHONE);
    check(p1 && p1->region == 0 && p1->chunk == 1,
          "deferred producer did not release same-region next chunk");
    ready.defer(*p1);
    check(!ready.try_take(llama_region_ring_device::GPU).has_value(),
          "second deferred producer released downstream before join");

    // Return lane 1 may finish before lane 0. This may satisfy GPU chunk1's
    // interval input, but strict per-Region order must still keep it queued.
    ready.complete_deferred(*p1);
    check(ready.finished() == 1,
          "deferred join did not count as a completion");
    check(!ready.try_take(llama_region_ring_device::GPU).has_value(),
          "out-of-order join bypassed downstream chunk order");

    ready.complete_deferred(*p0);
    check(ready.finished() == 2,
          "first deferred join did not publish completion");

    const auto g0 =
        ready.try_take(llama_region_ring_device::GPU);
    check(g0 && g0->region == 1 && g0->chunk == 0,
          "downstream chunk0 did not release after P2C join");
    ready.complete(*g0);

    const auto g1 =
        ready.try_take(llama_region_ring_device::GPU);
    check(g1 && g1->region == 1 && g1->chunk == 1,
          "downstream chunk1 did not release after ordered predecessor");
    ready.complete(*g1);

    check(ready.all_done() && ready.finished() == 4 &&
              !ready.failed(),
          "deferred P2C scheduler did not drain cleanly");

    // Legacy bounded-slot scheduling intentionally keeps the old lifetime
    // contract and must reject split/deferred completion.
    llama_region_ring_ready bounded(regions, 128, 64, 2);
    const auto bounded_p0 =
        bounded.try_take(llama_region_ring_device::PHONE);
    check(bounded_p0.has_value(),
          "bounded deferred test missing first task");
    bool rejected = false;
    try {
        bounded.defer(*bounded_p0);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected,
          "bounded slot scheduler accepted deferred completion");
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
    check(ready.queue_size(llama_region_ring_device::GPU) == 1,
          "GPU chunk 1 was not queued");
    check(ready.queue_size(llama_region_ring_device::CPU) == 1,
          "CPU chunk 0 was not queued");
    bool rejected = false;
    try {
        ready.complete(*first);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected, "duplicate task completion was accepted");
}


static void test_device_fifo_simulation() {
    const auto regions = llama_region_ring_parse("", 48);
    llama_region_ring_ready ready(regions, 358, 64);
    check(ready.n_chunks() == 6, "358 tokens must form 6 chunks at 64");
    check(!ready.try_take(llama_region_ring_device::CPU).has_value(),
          "CPU started before GPU produced an input");
    check(!ready.try_take(llama_region_ring_device::PHONE).has_value(),
          "Phone started before CPU produced an input");

    std::vector<std::vector<int>> done(9, std::vector<int>(6, 0));
    const std::array<llama_region_ring_device, 3> devices = {
        llama_region_ring_device::CPU,
        llama_region_ring_device::GPU,
        llama_region_ring_device::PHONE,
    };
    size_t executed = 0;
    while (!ready.all_done()) {
        bool progressed = false;
        for (auto device : devices) {
            const auto job = ready.try_take(device);
            if (!job) {
                continue;
            }
            check(regions[job->region].device == device,
                  "device FIFO returned another device's task");
            check(job->chunk < 6 && job->region < 9, "task index out of range");
            check(job->token_begin == (int) job->chunk * 64,
                  "device FIFO task token offset is wrong");
            check(job->n_tokens == (job->chunk == 5 ? 38 : 64),
                  "device FIFO task token size is wrong");
            check(done[job->region][job->chunk] == 0, "duplicate FIFO claim");
            check(job->region == 0 || done[job->region - 1][job->chunk] == 1,
                  "FIFO crossed unfinished upstream region");
            check(job->chunk == 0 || done[job->region][job->chunk - 1] == 1,
                  "FIFO violated per-region KV chunk order");
            done[job->region][job->chunk] = 1;
            ready.complete(*job);
            ++executed;
            progressed = true;
        }
        check(progressed, "device FIFO simulation deadlocked");
    }
    check(executed == 54, "expected exactly 54 Region x Chunk tasks");
    check(ready.finished() == 54 && !ready.failed(), "FIFO did not drain cleanly");
    for (auto device : devices) {
        check(!ready.try_take(device).has_value(), "task available after FIFO drain");
    }
}

static void test_parallel_scheduler_only() {
    // Mock workers exercise only the header-level dependency scheduler.
    // No GGML graph is executed concurrently in Step 1.
    const auto regions = llama_region_ring_parse("", 48);
    llama_region_ring_ready ready(regions, 358, 64);
    std::array<std::atomic<int>, 54> seen{};
    std::atomic<int> completed{0};
    std::atomic<int> errors{0};
    const std::array<llama_region_ring_device, 3> devices = {
        llama_region_ring_device::CPU,
        llama_region_ring_device::GPU,
        llama_region_ring_device::PHONE,
    };
    std::array<std::thread, 3> workers;
    for (size_t i = 0; i < workers.size(); ++i) {
        const auto device = devices[i];
        workers[i] = std::thread([&, device] {
            try {
                while (const auto job = ready.wait_take(device)) {
                    const size_t index = job->region * 6 + job->chunk;
                    const bool in_range = job->region < regions.size() &&
                                          job->chunk < 6;
                    const bool correct_device =
                        in_range && regions[job->region].device == device;
                    const bool upstream_done = in_range &&
                        (job->region == 0 ||
                         seen[(job->region - 1) * 6 + job->chunk].load() == 1);
                    const bool kv_ordered = in_range &&
                        (job->chunk == 0 ||
                         seen[job->region * 6 + job->chunk - 1].load() == 1);
                    if (!in_range || !correct_device || !upstream_done ||
                        !kv_ordered || seen[index].exchange(1) != 0) {
                        ++errors;
                        ready.fail(*job);
                        return;
                    }
                    ready.complete(*job);
                    ++completed;
                }
            } catch (...) {
                ++errors;
                ready.cancel();
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    check(errors.load() == 0, "concurrent mock workers violated an invariant");
    check(completed.load() == 54, "concurrent mock workers lost tasks");
    check(ready.all_done() && !ready.failed(),
          "concurrent scheduler did not drain all tasks");
}

static void test_failure_and_wakeup() {
    const auto regions = llama_region_ring_parse("", 48);
    llama_region_ring_ready ready(regions, 128, 64);
    const auto first = ready.try_take(llama_region_ring_device::GPU);
    check(first.has_value(), "failed to claim initial GPU task");
    ready.fail(*first);
    check(ready.failed() && !ready.has_ready(),
          "failed scheduler still exposes READY tasks");
    check(!ready.try_take(llama_region_ring_device::CPU).has_value(),
          "failed scheduler allowed another task to be claimed");
    check(!ready.wait_take(llama_region_ring_device::GPU).has_value(),
          "waiter did not exit after failure");
    bool rejected = false;
    try {
        ready.complete(*first);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected, "failed task was incorrectly accepted as completed");

    llama_region_ring_ready cancelled(regions, 128, 64);
    std::atomic<bool> woke{false};
    std::thread waiter([&] {
        woke.store(!cancelled.wait_take(llama_region_ring_device::PHONE));
    });
    cancelled.cancel();
    waiter.join();
    check(woke.load() && cancelled.failed(), "cancellation did not wake waiter");
}


static void test_nonuniform_interval_ready() {
    const auto regions = llama_region_ring_parse("", 48);
    std::vector<int> chunks;
    chunks.reserve(regions.size());
    for (const auto & r : regions) {
        chunks.push_back(
            r.device == llama_region_ring_device::GPU ? 16 :
            r.device == llama_region_ring_device::CPU ? 64 : 128);
    }

    llama_region_ring_ready ready(regions, 358, chunks, 0);
    check(ready.n_chunks(0) == 23, "GPU X16 chunk count is wrong");
    check(ready.n_chunks(1) == 6, "CPU X64 chunk count is wrong");
    check(ready.n_chunks(2) == 3, "Phone X128 chunk count is wrong");

    // C[0,64) gathers four G16 intervals.
    for (int i = 0; i < 3; ++i) {
        const auto g = ready.try_take(llama_region_ring_device::GPU);
        check(g && g->region == 0 && g->chunk == (size_t) i,
              "missing early GPU interval");
        ready.complete(*g);
        check(!ready.try_take(llama_region_ring_device::CPU).has_value(),
              "CPU gather released before four G16 intervals");
    }
    const auto g3 = ready.try_take(llama_region_ring_device::GPU);
    check(g3 && g3->region == 0 && g3->chunk == 3,
          "missing fourth GPU interval");
    ready.complete(*g3);
    const auto c0 = ready.try_take(llama_region_ring_device::CPU);
    check(c0 && c0->region == 1 && c0->chunk == 0 &&
              c0->token_begin == 0 && c0->n_tokens == 64,
          "CPU X64 gather did not become READY");
    ready.complete(*c0);

    // P[0,128) needs two C64 intervals.
    check(!ready.try_take(llama_region_ring_device::PHONE).has_value(),
          "Phone gather released after only one C64 interval");
    for (int i = 4; i < 8; ++i) {
        const auto g = ready.try_take(llama_region_ring_device::GPU);
        check(g && g->region == 0 && g->chunk == (size_t) i,
              "missing GPU interval for C[64,128)");
        ready.complete(*g);
    }
    const auto c1 = ready.try_take(llama_region_ring_device::CPU);
    check(c1 && c1->region == 1 && c1->chunk == 1,
          "second CPU X64 gather missing");
    ready.complete(*c1);
    const auto p0 = ready.try_take(llama_region_ring_device::PHONE);
    check(p0 && p0->region == 2 && p0->chunk == 0 &&
              p0->token_begin == 0 && p0->n_tokens == 128,
          "Phone X128 gather did not become READY");
    ready.complete(*p0);

    // One P128 interval covers eight G16 intervals in the next cycle. KV order
    // makes only region-3 chunk 0 eligible, but the per-device GPU FIFO still
    // contains region-0 chunk 8, which was queued earlier. Fan-out therefore
    // adds a second READY GPU task; it must not jump ahead of older FIFO work.
    check(ready.queue_size(llama_region_ring_device::GPU) == 2,
          "P128 fan-out did not enqueue next-cycle G16 behind existing GPU work");

    const auto older_g = ready.try_take(llama_region_ring_device::GPU);
    check(older_g && older_g->region == 0 && older_g->chunk == 8,
          "GPU FIFO did not preserve older same-cycle work");
    ready.complete(*older_g);

    const auto next_g = ready.try_take(llama_region_ring_device::GPU);
    check(next_g && next_g->region == 3 && next_g->chunk == 0 &&
              next_g->n_tokens == 16,
          "P128 fan-out task was not reachable after older GPU FIFO work");

    // Run a fresh READY_V2 instance all the way to drain and independently
    // verify every claimed consumer sees complete upstream interval coverage.
    llama_region_ring_ready drain(regions, 358, chunks, 0);
    std::vector<std::vector<uint8_t>> done(regions.size());
    size_t expected_tasks = 0;
    for (size_t r = 0; r < regions.size(); ++r) {
        done[r].assign(drain.n_chunks(r), 0);
        expected_tasks += drain.n_chunks(r);
    }
    check(expected_tasks == drain.total_tasks(),
          "READY_V2 total task count mismatch");

    const std::array<llama_region_ring_device, 3> devices = {
        llama_region_ring_device::CPU,
        llama_region_ring_device::GPU,
        llama_region_ring_device::PHONE,
    };
    size_t completed = 0;
    while (!drain.all_done()) {
        bool progressed = false;
        for (const auto dev : devices) {
            const auto task = drain.try_take(dev);
            if (!task) {
                continue;
            }
            check(regions[task->region].device == dev,
                  "READY_V2 returned task to wrong device");
            check(task->chunk < done[task->region].size(),
                  "READY_V2 task index out of range");
            check(task->chunk == 0 ||
                      done[task->region][task->chunk - 1] == 1,
                  "READY_V2 violated strict per-region KV order");

            if (task->region > 0) {
                const size_t up = task->region - 1;
                const int db = task->token_begin;
                const int de = db + task->n_tokens;
                for (size_t uc = 0; uc < done[up].size(); ++uc) {
                    const int ub = (int) uc * chunks[up];
                    const int ue = std::min(358, ub + chunks[up]);
                    if (std::min(ue, de) > std::max(ub, db)) {
                        check(done[up][uc] == 1,
                              "READY_V2 exposed incomplete upstream interval");
                    }
                }
            }

            check(done[task->region][task->chunk] == 0,
                  "READY_V2 duplicate task claim");
            done[task->region][task->chunk] = 1;
            drain.complete(*task);
            ++completed;
            progressed = true;
        }
        check(progressed, "READY_V2 nonuniform drain deadlocked");
    }
    check(completed == expected_tasks &&
              drain.finished() == expected_tasks &&
              !drain.failed(),
          "READY_V2 nonuniform scheduler did not drain cleanly");

    bool bounded_rejected = false;
    try {
        llama_region_ring_ready invalid(regions, 358, chunks, 2);
        (void) invalid;
    } catch (const std::runtime_error &) {
        bounded_rejected = true;
    }
    check(bounded_rejected,
          "legacy bounded slots accepted nonuniform chunks");
}

static void test_dynamic_slot_pool() {
    const auto regions = llama_region_ring_parse("", 48);
    llama_region_ring_ready ready(regions, 358, 64, 2);
    check(ready.slot_pool_size() == 2, "dynamic slot pool not enabled");

    const auto first = ready.try_take(llama_region_ring_device::GPU);
    check(first && first->region == 0 && first->chunk == 0, "GPU R0C0 unavailable");
    check(first->input_slot == -1 && first->output_slot >= 0,
          "R0C0 did not receive a dynamic output lease");
    const int first_slot = first->output_slot;
    check(ready.live_output_slots() == 1, "R0C0 output lease not reserved");
    ready.complete(*first);
    const auto second = ready.try_take(llama_region_ring_device::GPU);
    check(second && second->region == 0 && second->chunk == 1, "GPU R0C1 unavailable");
    check(second->output_slot >= 0 && second->output_slot != first_slot,
          "R0C1 reused a live output slot");
    ready.complete(*second);
    check(ready.live_output_slots() == 2, "expected two R0 output slots in use");
    check(!ready.try_take(llama_region_ring_device::GPU).has_value(),
          "R0C2 overwrote R0C0 before R1 consumed it");

    const auto cpu = ready.try_take(llama_region_ring_device::CPU);
    check(cpu && cpu->region == 1 && cpu->chunk == 0, "CPU R1C0 unavailable");
    check(cpu->input_slot == first_slot && cpu->output_slot >= 0,
          "CPU did not inherit the producer's dynamic slot lease");
    check(ready.live_output_slots() == 3,
          "CPU must retain the reading upstream slot while producing its own output");
    ready.complete(*cpu);
    const auto resumed = ready.try_take(llama_region_ring_device::GPU);
    check(resumed && resumed->region == 0 && resumed->chunk == 2,
          "R0C2 did not resume after downstream released the credit");
    check(resumed->output_slot >= 0,
          "R0C2 did not acquire a freed slot dynamically");
    ready.complete(*resumed);

    // In single-thread execution a blocked queue head must not block
    // other ready regions on that device. All 54 tasks should drain.
    size_t completed = 4;
    const std::array<llama_region_ring_device, 3> devices = {
        llama_region_ring_device::CPU,
        llama_region_ring_device::GPU,
        llama_region_ring_device::PHONE,
    };
    while (!ready.all_done()) {
        bool progress = false;
        for (const auto device : devices) {
            if (auto task = ready.try_take(device)) {
                ready.complete(*task);
                ++completed;
                progress = true;
            }
        }
        check(progress, "credit blocked all devices without a running task");
    }
    check(completed == 54 && !ready.failed(), "dynamic slot pool lost tasks");
    check(ready.live_output_slots() == 0, "credits remained after full drain");
}

static void test_window_parallel_mock_workers() {
    // Tests dynamic slot leases plus READY/WRITING/READING/FREE wakeups.
    // NO concurrent llama.cpp or GGML graphs are executed here.
    const auto regions = llama_region_ring_parse("", 48);
    llama_region_ring_ready ready(regions, 358, 64, 2);
    const std::array<llama_region_ring_device, 3> devices = {
        llama_region_ring_device::CPU,
        llama_region_ring_device::GPU,
        llama_region_ring_device::PHONE,
    };
    std::array<std::atomic<int>, 54> done{};
    std::atomic<int> failures{0};
    std::array<std::thread, 3> workers;
    for (size_t i = 0; i < workers.size(); ++i) {
        workers[i] = std::thread([&, dev = devices[i]] {
            try {
                while (auto task = ready.wait_take(dev)) {
                    if (task->region >= regions.size() || task->chunk >= 6) {
                        ++failures;
                        ready.fail(*task);
                        return;
                    }
                    const size_t index = task->region * 6 + task->chunk;
                    const bool valid =
                        regions[task->region].device == dev &&
                        (task->region == 0 ||
                         done[(task->region - 1) * 6 + task->chunk].load() == 1) &&
                        (task->chunk == 0 ||
                         done[task->region * 6 + task->chunk - 1].load() == 1);
                    if (!valid || done[index].exchange(1) != 0) {
                        ++failures;
                        ready.fail(*task);
                        return;
                    }
                    ready.complete(*task);
                }
            } catch (...) {
                ++failures;
                ready.cancel();
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
    check(failures.load() == 0, "concurrent slot-pool credit mock failed");
    check(ready.all_done() && ready.finished() == 54, "window mock did not finish");
    check(ready.live_output_slots() == 0, "window mock leaked output credits");
}

static void test_window_failure_wakes_workers() {
    // A real threaded scheduler must wake blocked CPU/PHONE workers when a
    // different device fails. Keep the same 2-slot backpressure as prefill.
    const auto regions = llama_region_ring_parse("", 48);
    const std::array<llama_region_ring_device, 3> devices = {
        llama_region_ring_device::CPU,
        llama_region_ring_device::GPU,
        llama_region_ring_device::PHONE,
    };
    for (int iteration = 0; iteration < 12; ++iteration) {
        llama_region_ring_ready ready(regions, 358, 64, 2);
        std::atomic<int> injected{0};
        std::atomic<int> errors{0};
        std::array<std::thread, 3> workers;
        for (size_t i = 0; i < workers.size(); ++i) {
            const auto device = devices[i];
            workers[i] = std::thread([&, device] {
                try {
                    while (const auto task = ready.wait_take(device)) {
                        if (device == llama_region_ring_device::GPU &&
                            task->region == 0 && task->chunk == 0) {
                            ++injected;
                            ready.fail(*task);
                            return;
                        }
                        if (ready.failed()) {
                            // Simulates an already-claimed task that is
                            // cancelled before entering the backend gate.
                            ready.fail(*task);
                            return;
                        }
                        ready.complete(*task);
                        std::this_thread::yield();
                    }
                } catch (...) {
                    ++errors;
                    ready.cancel();
                }
            });
        }
        for (auto & worker : workers) {
            worker.join();
        }
        check(errors.load() == 0, "failure worker threw during cancellation");
        check(injected.load() == 1, "exactly one injected GPU failure expected");
        check(ready.failed() && !ready.all_done(),
              "injected GPU failure was not propagated");
        for (const auto device : devices) {
            check(!ready.wait_take(device).has_value(),
                  "failed scheduler did not release a blocked worker");
        }
    }
}

static void test_gpu_same_bank_lifetime_scheduler() {
    // Three GPU Regions isolate the rolling two-Bank lifetime rule:
    // rank0(A) and rank1(B) may interleave, while rank2(A) must wait for
    // rank0(A) to drain before Bank A can be reused.
    const std::vector<llama_region_ring_region> regions = {
        {llama_region_ring_device::GPU, 0, 1},
        {llama_region_ring_device::GPU, 1, 2},
        {llama_region_ring_device::GPU, 2, 3},
    };
    llama_region_ring_ready ready(regions, 32, 16, 0, true);

    const auto r0c0 =
        ready.try_take(llama_region_ring_device::GPU);
    check(r0c0 && r0c0->region == 0 && r0c0->chunk == 0,
          "rank0 chunk0 missing");
    ready.complete(*r0c0);
    check(!ready.region_done(0),
          "rank0 reported drained before all chunks completed");

    const auto r1c0 =
        ready.try_take(llama_region_ring_device::GPU);
    check(r1c0 && r1c0->region == 1 && r1c0->chunk == 0,
          "rank1/B was incorrectly blocked by rank0/A");
    ready.complete(*r1c0);

    // rank2/A is input-ready now, but Bank A still belongs to rank0 because
    // rank0 chunk1 remains unfinished. The scheduler must continue rank0.
    const auto r0c1 =
        ready.try_take(llama_region_ring_device::GPU);
    check(r0c1 && r0c1->region == 0 && r0c1->chunk == 1,
          "rank2/A reused Bank A before rank0 drained");
    ready.complete(*r0c1);
    check(ready.region_done(0),
          "rank0 did not report drained after all chunks completed");

    // Bank A is now free. rank2 may start even though rank1/B still has its
    // second chunk queued.
    const auto r2c0 =
        ready.try_take(llama_region_ring_device::GPU);
    check(r2c0 && r2c0->region == 2 && r2c0->chunk == 0,
          "rank2/A did not become claimable after rank0 drained");
}

static void test_gpu_weight_two_bank_policy() {
    const std::array<size_t, 4> gpu_regions = {0, 3, 6, 9};

    check(llama_region_ring_gpu_weight_bank_for_rank(0) == 0,
          "rank0 must use Bank A");
    check(llama_region_ring_gpu_weight_bank_for_rank(1) == 1,
          "rank1 must use Bank B");
    check(llama_region_ring_gpu_weight_bank_for_rank(2) == 0,
          "rank2 must reuse Bank A");
    check(llama_region_ring_gpu_weight_bank_for_rank(3) == 1,
          "rank3 must reuse Bank B");

    const auto next0 =
        llama_region_ring_gpu_weight_next_rank(0, gpu_regions.size());
    const auto next1 =
        llama_region_ring_gpu_weight_next_rank(1, gpu_regions.size());
    const auto next2 =
        llama_region_ring_gpu_weight_next_rank(2, gpu_regions.size());
    const auto next3 =
        llama_region_ring_gpu_weight_next_rank(3, gpu_regions.size());

    check(next0 && *next0 == 2,
          "rank0 drain must refill rank2");
    check(next1 && *next1 == 3,
          "rank1 drain must refill rank3");
    check(!next2 && !next3,
          "last two GPU ranks must not refill past the Region list");
}

static void test_pc_phone_overlap_intervals() {
    const std::vector<std::pair<int64_t, int64_t>> pc {
        {0, 20}, {30, 60}, {70, 90}
    };
    const std::vector<std::pair<int64_t, int64_t>> phone {
        {10, 40}, {50, 75}
    };
    // [10,20]=10 + [30,40]=10 + [50,60]=10 + [70,75]=5.
    check(llama_region_ring_interval_overlap_us(pc, phone) == 35,
          "PC/Phone interval intersection is wrong");
    check(llama_region_ring_interval_overlap_us(pc, {{100, 120}}) == 0,
          "disjoint intervals counted as overlap");
}

int main() {
    std::fprintf(stderr, "[REGION_RING_TEST] begin\n");
    try {
        test_layout();
        std::fprintf(stderr, "[REGION_RING_TEST] layout=PASS\n");
        test_model_aware_default_layout();
        std::fprintf(stderr, "[REGION_RING_TEST] model_aware_layout=PASS\n");
        test_directional_layout();
        std::fprintf(stderr, "[REGION_RING_TEST] directional_layout=PASS\n");
        test_general_topology();
        std::fprintf(stderr, "[REGION_RING_TEST] general_topology=PASS\n");
        test_apply_safe_compatibility();
        std::fprintf(stderr, "[REGION_RING_TEST] apply_safe_compatibility=PASS\n");
        test_dependencies();
        std::fprintf(stderr, "[REGION_RING_TEST] dependencies=PASS\n");
        test_deferred_p2c_join_scheduler();
        std::fprintf(stderr, "[REGION_RING_TEST] deferred_p2c_join=PASS\n");
        test_duplicate_completion();
        std::fprintf(stderr, "[REGION_RING_TEST] duplicate=PASS\n");
        test_device_fifo_simulation();
        std::fprintf(stderr, "[REGION_RING_TEST] device_fifo=PASS\n");
        test_parallel_scheduler_only();
        std::fprintf(stderr, "[REGION_RING_TEST] parallel_scheduler=PASS\n");
        test_failure_and_wakeup();
        std::fprintf(stderr, "[REGION_RING_TEST] failure=PASS\n");
        test_nonuniform_interval_ready();
        std::fprintf(stderr, "[REGION_RING_TEST] nonuniform_interval_ready=PASS\n");
        test_dynamic_slot_pool();
        std::fprintf(stderr, "[REGION_RING_TEST] dynamic_slot_pool=PASS\n");
        test_window_parallel_mock_workers();
        std::fprintf(stderr, "[REGION_RING_TEST] window_parallel=PASS\n");
        test_window_failure_wakes_workers();
        std::fprintf(stderr, "[REGION_RING_TEST] window_failure_wakeup=PASS\n");
        test_gpu_same_bank_lifetime_scheduler();
        std::fprintf(stderr, "[REGION_RING_TEST] gpu_same_bank_lifetime_scheduler=PASS\n");
        test_gpu_weight_two_bank_policy();
        std::fprintf(stderr, "[REGION_RING_TEST] gpu_weight_two_bank_policy=PASS\n");
        test_pc_phone_overlap_intervals();
        std::fprintf(stderr, "[REGION_RING_TEST] pc_phone_overlap_intervals=PASS\n");
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
