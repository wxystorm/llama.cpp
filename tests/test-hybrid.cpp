#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "llama-hybrid.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: test-hybrid <rpc-endpoint> <model.gguf> [probe-layer]\n");
        return 1;
    }

    const char * endpoint   = argv[1];
    const char * model_path = argv[2];

    //
    // 1. 初始化 PC CPU + Phone RPC backend
    //
    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    if (cpu == nullptr) {
        fprintf(stderr, "failed to initialize CPU backend\n");
        return 1;
    }

    ggml_backend_t phone = ggml_backend_rpc_init(endpoint, 0);

    if (phone == nullptr) {
        fprintf(stderr, "failed to initialize RPC backend: %s\n", endpoint);
        ggml_backend_free(cpu);
        return 1;
    }

    ggml_backend_t     gpu     = nullptr;
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev != nullptr) {
        gpu = ggml_backend_dev_init(gpu_dev, nullptr);
        if (gpu != nullptr) {
            printf("GPU backend: %s\n", ggml_backend_name(gpu));
        }
    }

    try {
        //
        // 2. 只打开 GGUF metadata
        //
        std::vector<std::string> splits;

        llama_model_loader ml(nullptr,     // metadata
                              nullptr,     // set_tensor_data
                              nullptr,     // set_tensor_data_ud
                              model_path,  // GGUF path
                              splits,
                              nullptr,     // FILE *
                              false,       // use_mmap
                              false,       // use_direct_io
                              false,       // check_tensors
                              true,        // no_alloc
                              nullptr,     // kv overrides
                              nullptr      // tensor buffer overrides
        );

        printf("=== MODEL ===\n");
        printf("path: %s\n", model_path);
        printf("arch: %s\n", ml.get_arch_name().c_str());

        //
        // 3. 扫描 dense FFN 层数量
        //
        int n_layer = 0;

        for (int i = 0; i < 1024; ++i) {
            llama_hybrid_ffn_desc tmp;

            if (!ml.get_hybrid_ffn_desc(i, tmp)) {
                break;
            }

            n_layer++;
        }

        if (n_layer <= 0) {
            fprintf(stderr,
                    "failed to find dense FFN layers in model\n"
                    "this test currently expects FFN_GATE/FFN_UP/FFN_DOWN\n");

            if (gpu != nullptr) {
                ggml_backend_free(gpu);
            }
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }

        //
        // 默认取中间层，也可以 argv[3] 手动指定
        //
        int probe_layer = n_layer / 2;

        if (argc >= 4) {
            probe_layer = std::atoi(argv[3]);
        }

        if (probe_layer < 0 || probe_layer >= n_layer) {
            fprintf(stderr, "invalid probe layer %d, model has %d dense FFN layers\n", probe_layer, n_layer);

            if (gpu != nullptr) {
                ggml_backend_free(gpu);
            }
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }

        //
        // 4. 从真实 GGUF metadata 中读取这一层 FFN
        //
        llama_hybrid_ffn_desc desc;

        if (!ml.get_hybrid_ffn_desc(probe_layer, desc)) {
            fprintf(stderr, "failed to get FFN descriptor for layer %d\n", probe_layer);

            if (gpu != nullptr) {
                ggml_backend_free(gpu);
            }
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }

        llama_hybrid_attn_desc attn_desc;
        if (!ml.get_hybrid_attn_desc(probe_layer, attn_desc)) {
            fprintf(stderr, "failed to get attention descriptor for layer %d\n", probe_layer);

            if (gpu != nullptr) {
                ggml_backend_free(gpu);
            }
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }

        printf("layers: %d\n", n_layer);
        printf("probe_layer: %d\n", probe_layer);

        printf(
            "FFN: n_embd=%lld n_ff=%lld "
            "norm=%s gate=%s up=%s down=%s weight_bytes=%zu\n",
            (long long) desc.n_embd, (long long) desc.n_ff, ggml_type_name(desc.ffn_norm_type),
            ggml_type_name(desc.gate_type), ggml_type_name(desc.up_type), ggml_type_name(desc.down_type),
            desc.weight_bytes);

        printf(
            "ATTN: n_embd=%lld q=%lld k=%lld v=%lld o_in=%lld heads=%d kv_heads=%d "
            "q_type=%s k_type=%s v_type=%s o_type=%s q_norm=%d k_norm=%d\n",
            (long long) attn_desc.n_embd, (long long) attn_desc.q_dim, (long long) attn_desc.k_dim,
            (long long) attn_desc.v_dim, (long long) attn_desc.o_in_dim, attn_desc.n_head, attn_desc.n_head_kv,
            ggml_type_name(attn_desc.q_type), ggml_type_name(attn_desc.k_type), ggml_type_name(attn_desc.v_type),
            ggml_type_name(attn_desc.o_type), attn_desc.has_q_norm ? 1 : 0, attn_desc.has_k_norm ? 1 : 0);

        printf("ATTN RoPE: type=%d n_rot=%d n_ctx_orig=%d freq_base=%g freq_scale=%g\n", (int) attn_desc.rope_type,
               attn_desc.n_rot, attn_desc.n_ctx_orig, attn_desc.freq_base, attn_desc.freq_scale);

        //
        // 5. 初始化真实尺寸的 hybrid profile
        //
        llama_hybrid_profile profile;

        profile.n_layer = n_layer;
        profile.n_embd  = desc.n_embd;
        profile.n_ff    = desc.n_ff;

        // 对应 K = 1 / 2 / 4
        profile.probe_tokens           = 16;
        profile.probe_chunk_min_tokens = 4;

        if (!ml.get_hybrid_weight_bytes(n_layer, profile.model_weight_bytes, profile.non_layer_weight_bytes,
                                        profile.layer_weight_bytes, profile.layer_attn_forced_bytes,
                                        profile.layer_ffn_split_bytes, profile.layer_mirrored_bytes)) {
            fprintf(stderr, "failed to collect model weight bytes\n");

            if (gpu != nullptr) {
                ggml_backend_free(gpu);
            }
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }

        for (int il = 0; il < n_layer; ++il) {
            printf("MEM layer=%d total=%zu attn_forced=%zu ffn_split=%zu mirrored=%zu\n", il,
                   profile.layer_weight_bytes[il], profile.layer_attn_forced_bytes[il],
                   profile.layer_ffn_split_bytes[il], profile.layer_mirrored_bytes[il]);
        }

        printf("\n=== MEMORY PROFILE ===\n");

        if (!llama_hybrid_profile_memory(profile, cpu, phone, gpu)) {
            fprintf(stderr, "memory profile failed\n");

            if (gpu != nullptr) {
                ggml_backend_free(gpu);
            }
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }

        //
        // 6. Phase 2:
        //    使用真实 n_embd 测试 PC <-> Phone RPC
        //
        printf("\n=== RPC PROFILE ===\n");

        if (!llama_hybrid_profile_rpc(profile, cpu, phone)) {
            fprintf(stderr, "RPC profile failed\n");

            if (gpu != nullptr) {
                ggml_backend_free(gpu);
            }
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }

        //
        // 7. Phase 3A:
        //    使用真实 shape + quant type
        //    CPU 跑一遍，手机 RPC 跑一遍
        //
        printf("\n=== FFN PROFILE ===\n");

        if (!llama_hybrid_profile_ffn(profile, desc, cpu, phone)) {
            fprintf(stderr, "FFN profile failed\n");

            if (gpu != nullptr) {
                ggml_backend_free(gpu);
            }
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }

        printf("\n=== ATTENTION PROFILE ===\n");

        if (!llama_hybrid_profile_attention(profile, attn_desc, cpu, phone, gpu)) {
            fprintf(stderr, "attention profile failed\n");

            if (gpu != nullptr) {
                ggml_backend_free(gpu);
            }
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }

        printf("\n=== FULL LAYER PROFILE ===\n");

        if (!llama_hybrid_profile_full_layer(profile, attn_desc, desc, cpu, phone, gpu)) {
            fprintf(stderr, "full layer profile failed\n");

            if (gpu != nullptr) {
                ggml_backend_free(gpu);
            }
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }

        printf("\n=== PHONE BLOCK PROFILE ===\n");

        if (!llama_hybrid_profile_phone_blocks(profile, attn_desc, desc, phone)) {
            fprintf(stderr, "phone block profile failed\n");

            if (gpu != nullptr) {
                ggml_backend_free(gpu);
            }
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }

        //
        // 8. 打印最终 profile
        //
        printf("\n=== FINAL PROFILE ===\n");

        llama_hybrid_profile_print(profile);

        printf("\n=== PHASE 4A PLAN ENUM ===\n");

        llama_hybrid_constraints constraints;
        constraints.target_ctx      = 512;
        constraints.score_kv_tokens = 512;

        const std::vector<llama_hybrid_plan> feasible = llama_hybrid_enumerate_feasible_plans(profile, constraints);
        printf("feasible plans: %zu\n", feasible.size());

        if (feasible.empty()) {
            fprintf(stderr, "no feasible hybrid plans\n");

            if (gpu != nullptr) {
                ggml_backend_free(gpu);
            }
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }

        printf("\n=== PHASE 4B PLAN SCORE ===\n");

        std::vector<llama_hybrid_plan> ranked;
        ranked.reserve(feasible.size());
        for (auto plan : feasible) {
            if (llama_hybrid_score_plan(profile, constraints, plan)) {
                ranked.push_back(plan);
            }
        }

        std::sort(ranked.begin(), ranked.end(), [](const llama_hybrid_plan & a, const llama_hybrid_plan & b) {
            if (a.predicted_ms != b.predicted_ms) {
                return a.predicted_ms < b.predicted_ms;
            }
            return a.gpu_memory < b.gpu_memory;
        });

        if (ranked.empty()) {
            fprintf(stderr, "no scoreable hybrid plans\n");

            if (gpu != nullptr) {
                ggml_backend_free(gpu);
            }
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }

        printf("ranked plans: %zu\n", ranked.size());
        for (size_t i = 0; i < std::min<size_t>(ranked.size(), 10); ++i) {
            printf("\n--- rank %zu ---\n", i + 1);
            llama_hybrid_plan_print(ranked[i]);
        }

        printf("\n=== BEST PLAN ===\n");
        const llama_hybrid_plan & best = ranked.front();
        llama_hybrid_plan_print(best);

        if (!llama_hybrid_runtime_plan_set(best)) {
            throw std::runtime_error("failed to publish the best hybrid plan");
        }

        llama_hybrid_plan verified;
        if (!llama_hybrid_runtime_plan_get(verified)) {
            throw std::runtime_error("failed to read back the best hybrid plan");
        }
        printf("[HYBRID_RUNTIME_VERIFY] T=%d P=%d C=%d R=%.3f G=%d K=%d\n", verified.tensor_layers,
               verified.phone_layers, verified.pc_layers, verified.tensor_pc_ratio, verified.gpu_pc_layers,
               verified.tensor_chunks_per_ubatch);

    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());

        if (gpu != nullptr) {
            ggml_backend_free(gpu);
        }
        ggml_backend_free(phone);
        ggml_backend_free(cpu);
        return 1;
    }

    if (gpu != nullptr) {
        ggml_backend_free(gpu);
    }
    ggml_backend_free(phone);
    ggml_backend_free(cpu);

    return 0;
}
