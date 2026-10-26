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


static void test_general_topology() {
    const auto two_cycle_tail = llama_region_ring_parse(
        "C6,G4,P8,C6,G4,P8,C8,G4",48,true);
    check(two_cycle_tail.size() == 8,
          "two-cycle PC tail stage count");
    check(llama_region_ring_apply_topology_compatible(two_cycle_tail,48),
          "two-cycle PC tail must be supported");
    check(two_cycle_tail.back().device == llama_region_ring_device::GPU,
          "PC GPU tail must be final owner");
    check(llama_region_ring_layer_counts(two_cycle_tail) ==
          std::array<int,3>{20,12,16},
          "two-cycle PC tail device totals");
    check(llama_region_ring_layout_string(two_cycle_tail) ==
          "C6,G4,P8,C6,G4,P8,C8,G4",
          "two-cycle tail Plan serialization");
    const auto cpu_tail = llama_region_ring_parse(
        "C6,G4,P8,C6,G4,P8,C12",48,true);
    check(cpu_tail.back().device == llama_region_ring_device::CPU,
          "CPU tail owner");
    check(llama_region_ring_apply_topology_compatible(cpu_tail,48),
          "CPU tail compatible");
    const auto phone_last = llama_region_ring_parse(
        "C7,G4,P5,C7,G4,P5,C8,G3,P5",48,true);
    check(llama_region_ring_apply_topology_compatible(phone_last,48),
          "original topology remains compatible");
    bool legacy_rejected = false;
    try {
        (void) llama_region_ring_parse(
            "C6,G4,P8,C6,G4,P8,C8,G4",48);
    } catch (const std::exception &) {
        legacy_rejected = true;
    }
    check(legacy_rejected,
          "legacy fixed Ring must still reject PC tail");
    bool invalid_rejected = false;
    try {
        (void) llama_region_ring_parse(
            "C6,G4,P8,C6,G4,P8,G12",48,true);
    } catch (const std::exception &) {
        invalid_rejected = true;
    }
    check(invalid_rejected,
          "general Ring must reject a tail without a CPU stage");
}

static void test_apply_safe_compatibility() {
    const auto baseline = llama_region_ring_parse("", 48);
    check(llama_region_ring_apply_safe_phone_shape(baseline),
          "baseline must satisfy the apply-safe Phone shape");
    check(llama_region_ring_apply_safe_compatible(baseline,baseline),
          "baseline must be compatible with itself");
    check(llama_region_ring_layout_string(baseline) ==
          "C7,G4,P5,C7,G4,P5,C8,G3,P5",
          "canonical Ring encoding has changed");

    // Reassign exactly one CUDA layer between the first two PC windows;
    // Phone L11..15, L27..31, L43..47 and aggregate CUDA11 stay unchanged.
    const auto safe = llama_region_ring_parse(
        "C8,G3,P5,C6,G5,P5,C8,G3,P5",48);
    check(llama_region_ring_apply_safe_phone_shape(safe),
          "safe candidate changed the Phone regions");
    check(llama_region_ring_apply_safe_compatible(baseline,safe),
          "safe CPU/CUDA shift rejected");
    check(llama_region_ring_layout_string(safe) ==
          "C8,G3,P5,C6,G5,P5,C8,G3,P5",
          "safe candidate serialized incorrectly");

    const auto different_gpu_total = llama_region_ring_parse(
        "C8,G3,P5,C7,G4,P5,C8,G3,P5",48);
    check(!llama_region_ring_apply_safe_compatible(
            baseline,different_gpu_total),
          "candidate silently reduced CUDA weight count");
    const auto moved_phone = llama_region_ring_parse(
        "C8,G4,P5,C6,G4,P5,C8,G3,P5",48);
    check(!llama_region_ring_apply_safe_phone_shape(moved_phone),
          "moved Phone region passed fixed-graph check");
    check(!llama_region_ring_apply_safe_compatible(baseline,moved_phone),
          "moved Phone region passed compatibility check");
    const auto tail_or_different_cycles = llama_region_ring_parse(
        "C11,G6,P7,C11,G5,P8",48);
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


static void test_device_fifo_simulation() {
    const auto regions = llama_region_ring_parse("", 48);
    llama_region_ring_ready ready(regions, 358, 64);
    check(ready.n_chunks() == 6, "358 tokens must form 6 chunks at 64");
    check(!ready.try_take(llama_region_ring_device::GPU).has_value(),
          "GPU started before CPU produced an input");
    check(!ready.try_take(llama_region_ring_device::PHONE).has_value(),
          "Phone started before GPU produced an input");

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
    const auto first = ready.try_take(llama_region_ring_device::CPU);
    check(first.has_value(), "failed to claim initial CPU task");
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


static void test_output_window_credits() {
    const auto regions = llama_region_ring_parse("", 48);
    llama_region_ring_ready ready(regions, 358, 64, 2);
    check(ready.output_window() == 2, "window credit mode not enabled");

    const auto first = ready.try_take(llama_region_ring_device::CPU);
    check(first && first->region == 0 && first->chunk == 0, "CPU R0C0 unavailable");
    check(ready.live_output_slots() == 1, "R0C0 output lease not reserved");
    ready.complete(*first);
    const auto second = ready.try_take(llama_region_ring_device::CPU);
    check(second && second->region == 0 && second->chunk == 1, "CPU R0C1 unavailable");
    ready.complete(*second);
    check(ready.live_output_slots() == 2, "expected two R0 output slots in use");
    check(!ready.try_take(llama_region_ring_device::CPU).has_value(),
          "R0C2 overwrote R0C0 before R1 consumed it");

    const auto gpu = ready.try_take(llama_region_ring_device::GPU);
    check(gpu && gpu->region == 1 && gpu->chunk == 0, "GPU R1C0 unavailable");
    check(ready.live_output_slots() == 3,
          "GPU must retain the reading upstream slot while producing its own output");
    ready.complete(*gpu);
    const auto resumed = ready.try_take(llama_region_ring_device::CPU);
    check(resumed && resumed->region == 0 && resumed->chunk == 2,
          "R0C2 did not resume after downstream released the credit");
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
    check(completed == 54 && !ready.failed(), "window credits lost tasks");
    check(ready.live_output_slots() == 0, "credits remained after full drain");
}

static void test_window_parallel_mock_workers() {
    // Tests READY, WRITING, READING, FREE and wakeups concurrently.
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
    check(failures.load() == 0, "concurrent window credit mock failed");
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
                            task->region == 1 && task->chunk == 0) {
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
        test_general_topology();
        std::fprintf(stderr, "[REGION_RING_TEST] general_topology=PASS\n");
        test_apply_safe_compatibility();
        std::fprintf(stderr, "[REGION_RING_TEST] apply_safe_compatibility=PASS\n");
        test_dependencies();
        std::fprintf(stderr, "[REGION_RING_TEST] dependencies=PASS\n");
        test_duplicate_completion();
        std::fprintf(stderr, "[REGION_RING_TEST] duplicate=PASS\n");
        test_device_fifo_simulation();
        std::fprintf(stderr, "[REGION_RING_TEST] device_fifo=PASS\n");
        test_parallel_scheduler_only();
        std::fprintf(stderr, "[REGION_RING_TEST] parallel_scheduler=PASS\n");
        test_failure_and_wakeup();
        std::fprintf(stderr, "[REGION_RING_TEST] failure=PASS\n");
        test_output_window_credits();
        std::fprintf(stderr, "[REGION_RING_TEST] window_credits=PASS\n");
        test_window_parallel_mock_workers();
        std::fprintf(stderr, "[REGION_RING_TEST] window_parallel=PASS\n");
        test_window_failure_wakes_workers();
        std::fprintf(stderr, "[REGION_RING_TEST] window_failure_wakeup=PASS\n");
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
