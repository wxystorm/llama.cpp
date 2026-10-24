// Opt-in REAL 48-layer resource preflight, not a full-model forward pass.
// Loads ONLY the physical PC/Phone Cycle model: no 30B CPU oracle, no
// additional RPC connection, no generated logits, no llama_decode().
//
// Usage (PowerShell, after setting the Cycle X/Y placement variables):
//   test-hybrid-cycle-real-resources.exe <model.gguf> <phone:rpc-port>
//                                        [kv_tokens=64] [cuda_budget_mib=0]
//
// CUDA budget=0 is intentional: all PC full layers fall back to CPU.
// Larger budgets mirror only as many PC full layers as fit after reserving
// 512 MiB CUDA headroom. The effective per-physical-layer layout is printed.
#include "llama.h"
#include "llama-model.h"
#include "llama-hybrid-cycle-resources.h"
#include "llama-hybrid-cycle-execution-session.h"
#include "llama-hybrid-cycle-phone-regions.h"

#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <exception>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>

static bool cycle_resource_parse_uint(
        const char * str, uint64_t maximum, uint64_t & value) {
    if (!str || !*str || *str == '-') return false;
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(str, &end, 10);
    if (errno || end == str || !end || *end != '\0' ||
        parsed > maximum) {
        return false;
    }
    value = uint64_t(parsed);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 5 || !llama_hybrid_cycle_requested()) {
        std::fprintf(stderr,
            "Usage: test-hybrid-cycle-real-resources <model.gguf> "
            "<phone:rpc-port> [kv_tokens=64] [cuda_budget_mib=0]\n"
            "Set LLAMA_HYBRID_CYCLE_PC_LAYERS=4 and "
            "LLAMA_HYBRID_CYCLE_PHONE_LAYERS=2 before loading.\n");
        return 2;
    }
    uint64_t kv_tokens = 64, budget_mib = 0;
    if ((argc >= 4 &&
         !cycle_resource_parse_uint(argv[3], 512, kv_tokens)) ||
        (argc >= 5 &&
         !cycle_resource_parse_uint(argv[4], 16384, budget_mib)) ||
        kv_tokens == 0 || kv_tokens > 512) {
        std::fprintf(stderr,
            "[CYCLE_FULL_RESOURCE_PROBE] status=FAIL "
            "reason=BAD_KV_TOKENS_OR_CUDA_BUDGET\n");
        return 2;
    }

    llama_backend_init();
    const auto cpu_dev =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        std::fprintf(stderr,
            "[CYCLE_FULL_RESOURCE_PROBE] status=FAIL reason=NO_CPU_DEVICE\n");
        return 1;
    }
    ggml_backend_dev_t cuda_dev = ggml_backend_dev_by_name("CUDA0");
    if (!cuda_dev) {
        ggml_backend_load_all();
        cuda_dev = ggml_backend_dev_by_name("CUDA0");
    }
    if (!cuda_dev ||
        ggml_backend_dev_type(cuda_dev) !=
            GGML_BACKEND_DEVICE_TYPE_GPU) {
        std::fprintf(stderr,
            "[CYCLE_FULL_RESOURCE_PROBE] status=FAIL "
            "reason=PC_FULL_REQUIRES_CUDA0\n");
        return 1;
    }
    const auto rpc_reg = ggml_backend_rpc_add_server(argv[2]);
    if (!rpc_reg || ggml_backend_reg_dev_count(rpc_reg) != 1) {
        std::fprintf(stderr,
            "[CYCLE_FULL_RESOURCE_PROBE] status=FAIL reason=RPC_DEVICE_COUNT\n");
        return 1;
    }
    const auto phone_dev = ggml_backend_reg_dev_get(rpc_reg, 0);
    if (!phone_dev ||
        ggml_backend_dev_type(phone_dev) !=
            GGML_BACKEND_DEVICE_TYPE_GPU) {
        std::fprintf(stderr,
            "[CYCLE_FULL_RESOURCE_PROBE] status=FAIL "
            "reason=PHONE_GPU_UNAVAILABLE\n");
        return 1;
    }
    ggml_backend_dev_t devices[] = {cpu_dev, phone_dev, nullptr};
    llama_model_params mp = llama_model_default_params();
    mp.devices = devices;
    mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    mp.hybrid_auto = false;
    mp.use_extra_bufts = false; // Avoid a second, CPU_REPACK model.
    std::printf("[CYCLE_FULL_RESOURCE_PROBE] phase=MODEL_LOAD "
                "split_mode=TENSOR hybrid_auto=0 status=BEGIN\n");
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), &llama_model_free);
    if (!model) {
        std::fprintf(stderr,
            "[CYCLE_FULL_RESOURCE_PROBE] status=FAIL "
            "reason=PHYSICAL_MODEL_LOAD_FAILED\n");
        return 1;
    }
    const int n_layers = llama_model_n_layer(model.get());
    llama_hybrid_cycle_layout layout;
    try {
        if (!llama_hybrid_cycle_layout_from_env(n_layers, layout) ||
            n_layers != 48 || layout.cycle_size != 8 ||
            layout.cycles != 6 || layout.x != 4 || layout.y != 2) {
            std::fprintf(stderr,
                "[CYCLE_FULL_RESOURCE_PROBE] status=FAIL "
                "reason=EXPECTED_QWEN3_48_LAYER_X4_Y2 "
                "model_layers=%d\n", n_layers);
            return 1;
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr,
            "[CYCLE_FULL_RESOURCE_PROBE] status=FAIL "
            "reason=BAD_CYCLE_PLACEMENT detail=%s\n", error.what());
        return 1;
    }

    const auto cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    const auto phone = ggml_backend_dev_init(phone_dev, nullptr);
    const auto cuda = cuda_dev ?
        ggml_backend_dev_init(cuda_dev, nullptr) : nullptr;
    if (!cpu || !phone || !cuda) {
        std::fprintf(stderr,
            "[CYCLE_FULL_RESOURCE_PROBE] status=FAIL "
            "reason=BACKEND_INITIALIZATION\n");
        if (cuda) ggml_backend_free(cuda);
        if (phone) ggml_backend_free(phone);
        if (cpu) ggml_backend_free(cpu);
        return 1;
    }

    bool ok = false;
    {
        llama_hybrid_cycle_physical_resource_pool resource_pool;
        llama_hybrid_cycle_resource_config config;
        config.kv_tokens = int64_t(kv_tokens);
        config.max_inflight = 2;
        config.cuda_weight_budget_bytes =
            size_t(budget_mib) * 1024 * 1024;
        config.cuda_safety_reserve_bytes = 512ULL * 1024 * 1024;
        config.attention_graph_cache = true;
        config.attention_scratch_pool = true;
        std::string error;
        ok = resource_pool.prepare(
            *model, layout, cpu, phone, cuda, config, error);
        if (!ok) {
            std::fprintf(stderr,
                "[CYCLE_FULL_RESOURCE_PROBE] status=FAIL "
                "phase=PREPARE reason=%s\n", error.c_str());
        } else {
            const auto & selected = resource_pool.execution_layout();
            ok = selected.pc_cuda_layers.size() == size_t(n_layers) &&
                resource_pool.prepared_layers() == n_layers;
            for (int l = 0; l < n_layers && ok; ++l) {
                const auto * item = resource_pool.layer(l);
                const auto kind = selected.kinds[size_t(l)];
                const bool pc_full =
                    kind == llama_hybrid_cycle_layer_kind::PC_CPU;
                const bool phone_owned =
                    kind == llama_hybrid_cycle_layer_kind::PHONE_GPU ||
                    kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C;
                const bool cuda_tail =
                    kind == llama_hybrid_cycle_layer_kind::PC_CUDA_TAIL;
                const auto expected_backend = phone_owned ? phone :
                    (pc_full || cuda_tail ? cuda : cpu);
                ok = item && item->physical_layer == l &&
                    item->kv && item->kv->buffer &&
                    item->kv->keys && item->kv->values &&
                    item->kv->max_tokens == int64_t(kv_tokens) &&
                    item->kv->committed_tokens == 0 &&
                    item->graph_cache && item->scratch_pool &&
                    item->attn_backend == expected_backend &&
                    (!pc_full || selected.pc_cuda_layers[size_t(l)] != 0) &&
                    item->cuda_weights == nullptr;
                if (ok) {
                    std::printf(
                        "[CYCLE_FULL_RESOURCE_LAYER] physical=%d kind=%s "
                        "attn=%s kv_tokens=%" PRId64
                        " cuda_weight_resident=0 graph_cache=1 scratch_pool=1 "
                        "status=READY\n",
                        l,
                        llama_hybrid_cycle_kind_name(kind),
                        ggml_backend_name(item->attn_backend),
                        item->kv->max_tokens);
                } else {
                    std::fprintf(stderr,
                        "[CYCLE_FULL_RESOURCE_LAYER] physical=%d "
                        "status=FAIL reason=RESOURCE_OR_BACKEND_MISMATCH\n",
                        l);
                }
            }
            if (ok) {
                // Lazy residency: PC Full is already routed to CUDA before
                // any mirror is loaded. Fault one layer in explicitly.
                int first_pc_full = -1;
                for (int l = 0; l < n_layers; ++l) {
                    if (selected.kinds[size_t(l)] ==
                        llama_hybrid_cycle_layer_kind::PC_CPU) {
                        first_pc_full = l;
                        break;
                    }
                }
                std::string residency_error;
                ok = first_pc_full >= 0 &&
                    resource_pool.ensure_pc_full_cuda_resident(
                        first_pc_full, residency_error) &&
                    resource_pool.layer(first_pc_full)->cuda_weights != nullptr &&
                    resource_pool.cuda_resident_layers() == 1 &&
                    resource_pool.cuda_residency_loads() == 1;
                if (!ok) {
                    std::fprintf(stderr,
                        "[CYCLE_FULL_RESOURCE_PROBE] status=FAIL "
                        "reason=CUDA_RESIDENCY detail=%s\n",
                        residency_error.c_str());
                }
            }
            if (ok) {
                // Validate the six independent REAL physical Phone regions:
                // each C2P, Phone Full and P2C stage must borrow its own
                // absolute-layer K/V, graph metadata and scratch owner.
                // No RPC graph is executed by this resource-only probe.
                llama_hybrid_cycle_phone_regions phone_regions;
                const int chunk_size = 64;
                const int chunks = int((kv_tokens + chunk_size - 1) /
                                       chunk_size);
                const int inflight = std::min(2, chunks);
                ok = phone_regions.prepare(
                    selected, int64_t(kv_tokens), chunk_size,
                    inflight, error) &&
                    phone_regions.validate_resources(resource_pool, error) &&
                    phone_regions.region_count() == 6;
                if (!ok) {
                    std::fprintf(stderr,
                        "[CYCLE_FULL_PHONE_REGION] status=FAIL reason=%s\n",
                        error.c_str());
                }
                for (int cycle = 0;
                     cycle < phone_regions.region_count() && ok; ++cycle) {
                    const auto * region = phone_regions.by_index(cycle);
                    if (!region) {
                        ok = false;
                        break;
                    }
                    std::printf(
                        "[CYCLE_FULL_PHONE_REGION] cycle=%d c2p=%d "
                        "phone=[%d,%d) p2c=%d chunks=%d "
                        "max_inflight=%d resource_owner=PHYSICAL_LAYER "
                        "status=READY\n",
                        cycle, region->location.c2p_layer,
                        region->location.phone_begin,
                        region->location.phone_end,
                        region->location.p2c_layer,
                        region->n_chunks, region->max_inflight);
                }
            }
            if (ok) {
                // State descriptors are independent of graph/KV ownership.
                // The same 48-layer layout feeds the ONE global task ring.
                llama_hybrid_cycle_execution_session_config desc;
                desc.layout = selected;
                desc.physical_layer_count = n_layers;
                desc.tokens = int64_t(kv_tokens);
                desc.chunk_tokens = 64;
                llama_hybrid_cycle_execution_session<> session(desc);
                for (int l = 0; l < n_layers && ok; ++l) {
                    ok = session.borrow_kv(
                        l, resource_pool.layer(l)->kv.get());
                }
                ok = ok && session.layer_count() == n_layers &&
                    session.at(7, 0).tokens == session.at(8, 0).tokens &&
                    session.at(7, 0).start == session.at(8, 0).start;
            }
        }
        // All resources are freed BEFORE model/Backends go away.
        // No GPU/RPC task was ever submitted by this resource-only test.
        resource_pool.clear();
    }
    model.reset();
    if (cuda) ggml_backend_free(cuda);
    ggml_backend_free(phone);
    ggml_backend_free(cpu);
    std::printf("[CYCLE_FULL_RESOURCE_PROBE] layers=%d "
                "actual_forward=NOT_RUN logits=NOT_RUN "
                "status=%s\n", n_layers, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
