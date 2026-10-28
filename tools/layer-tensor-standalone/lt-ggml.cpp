#include "lt-ggml.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-hybrid.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <utility>

namespace lt {
namespace {

void init_backends_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        ggml_backend_load_all();
        llama_backend_init();
    });
}

void set_process_env(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

void unset_process_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

std::string lower_ascii(std::string value) {
    for (char & c : value) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return value;
}

ggml_backend_dev_t select_device(
        const std::string & requested,
        side local_side,
        std::string & error) {
    const std::string lowered = lower_ascii(requested);

    if (lowered.empty() || lowered == "auto") {
        if (local_side == side::pc) {
            ggml_backend_dev_t cpu =
                ggml_backend_dev_by_type(
                    GGML_BACKEND_DEVICE_TYPE_CPU);
            if (cpu == nullptr) {
                error = "CPU backend is not available";
            }
            return cpu;
        }

        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const auto type =
                ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                return dev;
            }
        }

        // Some mobile backends advertise themselves as ACCEL rather than
        // GPU/IGPU. Prefer a true GPU classification, but accept ACCEL as a
        // Phone-local fallback before failing.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) ==
                GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                return dev;
            }
        }

        error =
            "auto Phone device did not find a GPU/IGPU/ACCEL backend";
        return nullptr;
    }

    if (lowered == "cpu") {
        ggml_backend_dev_t cpu =
            ggml_backend_dev_by_type(
                GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu == nullptr) {
            error = "CPU backend is not available";
        }
        return cpu;
    }

    ggml_backend_dev_t exact =
        ggml_backend_dev_by_name(requested.c_str());
    if (exact == nullptr) {
        std::ostringstream oss;
        oss << "GGML device not found: " << requested << ". available=";
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            if (i != 0) {
                oss << ",";
            }
            oss << ggml_backend_dev_name(
                ggml_backend_dev_get(i));
        }
        error = oss.str();
    }
    return exact;
}

struct graph_hold {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ggml_tensor * primary = nullptr;
    ggml_tensor * residual = nullptr;
    ggml_tensor * hidden = nullptr;
    ggml_tensor * topk = nullptr;
    ggml_tensor * weights = nullptr;

    ~graph_hold() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::shared_ptr<graph_hold> new_graph_hold() {
    constexpr size_t meta_bytes = 16U * 1024U * 1024U;

    ggml_init_params params {};
    params.mem_size = meta_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return nullptr;
    }

    auto result = std::make_shared<graph_hold>();
    result->ctx = ctx;
    return result;
}

size_t quantized_dim0_offset(
        const ggml_tensor * tensor,
        int64_t element_begin) {
    const int64_t block =
        ggml_blck_size(tensor->type);
    GGML_ASSERT(block > 0);
    GGML_ASSERT(element_begin % block == 0);

    return static_cast<size_t>(
        element_begin / block) * tensor->nb[0];
}


uint32_t hp_n_layer(
        const llama_hparams & hp) {
    return hp.n_layer_all -
        hp.n_layer_nextn;
}

bool hp_is_swa(
        const llama_hparams & hp,
        uint32_t il) {
    GGML_ASSERT(il < hp.n_layer_all);
    return hp.is_swa_impl[il] != 0;
}

uint32_t hp_n_head(
        const llama_hparams & hp,
        uint32_t il) {
    GGML_ASSERT(il < hp.n_layer_all);
    return hp.n_head_arr[il];
}

uint32_t hp_n_head_kv(
        const llama_hparams & hp,
        uint32_t il) {
    GGML_ASSERT(il < hp.n_layer_all);
    return hp.n_head_kv_arr[il];
}

uint32_t hp_n_embd_head_k(
        const llama_hparams & hp,
        uint32_t il) {
    GGML_ASSERT(il < hp.n_layer_all);
    return hp_is_swa(hp, il) ?
        hp.n_embd_head_k_swa :
        hp.n_embd_head_k_full;
}

uint32_t hp_n_embd_head_v(
        const llama_hparams & hp,
        uint32_t il) {
    GGML_ASSERT(il < hp.n_layer_all);
    return hp_is_swa(hp, il) ?
        hp.n_embd_head_v_swa :
        hp.n_embd_head_v_full;
}

uint32_t hp_n_rot(
        const llama_hparams & hp,
        uint32_t il) {
    GGML_ASSERT(il < hp.n_layer_all);
    return hp_is_swa(hp, il) ?
        hp.n_rot_swa :
        hp.n_rot_full;
}

} // namespace

struct ggml_compute::impl {
    struct prefix_nodes {
        ggml_tensor * pos = nullptr;
        ggml_tensor * mask = nullptr;

        ggml_tensor * k_write = nullptr;
        ggml_tensor * v_write = nullptr;

        ggml_tensor * residual = nullptr;
        ggml_tensor * hidden = nullptr;
        ggml_tensor * topk = nullptr;
        ggml_tensor * weights = nullptr;
    };

    ggml_compute_config config;

    llama_model * model = nullptr;
    ggml_backend_dev_t device = nullptr;
    ggml_backend_t backend = nullptr;
    std::string backend_device_name;

    ggml_context * kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buffer = nullptr;
    std::vector<ggml_tensor *> kv_k;
    std::vector<ggml_tensor *> kv_v;

    // One persistent activation arena for the whole prompt. Every chunk owns
    // a fixed token range inside this long-lived backend buffer and overwrites
    // that range after each completed local layer.
    ggml_context * activation_ctx = nullptr;
    ggml_backend_buffer_t activation_buffer = nullptr;
    ggml_tensor * activation_root = nullptr;
    std::vector<ggml_tensor *> activation_chunks;

    struct materialized_ffn_shard {
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        ggml_tensor * down = nullptr;
        int64_t begin = 0;
        int64_t width = 0;
    };

    // OpenCL cannot safely execute Q4_K/Q6_K MoE sub-views because its
    // quantized weights are backend-repacked. Materialize the Phone shard as
    // independent backend tensors so the repack is built for the shard shape.
    ggml_context * phone_shard_ctx = nullptr;
    ggml_backend_buffer_t phone_shard_buffer = nullptr;
    std::vector<materialized_ffn_shard> phone_shards;

    int32_t total_tokens = 0;
    int32_t n_embd = 0;
    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    int32_t n_ff_exp = 0;

    std::mutex backend_mutex;
    std::mutex cache_mutex;

    std::unordered_map<key, std::shared_ptr<graph_hold>, key_hash>
        owner_cache;
    std::unordered_map<key, std::shared_ptr<graph_hold>, key_hash>
        remote_cache;

    ~impl() {
        owner_cache.clear();
        remote_cache.clear();

        if (phone_shard_buffer != nullptr) {
            ggml_backend_buffer_free(phone_shard_buffer);
            phone_shard_buffer = nullptr;
        }
        if (phone_shard_ctx != nullptr) {
            ggml_free(phone_shard_ctx);
            phone_shard_ctx = nullptr;
        }

        if (activation_buffer != nullptr) {
            ggml_backend_buffer_free(activation_buffer);
            activation_buffer = nullptr;
        }
        if (activation_ctx != nullptr) {
            ggml_free(activation_ctx);
            activation_ctx = nullptr;
        }

        if (kv_buffer != nullptr) {
            ggml_backend_buffer_free(kv_buffer);
            kv_buffer = nullptr;
        }
        if (kv_ctx != nullptr) {
            ggml_free(kv_ctx);
            kv_ctx = nullptr;
        }

        if (backend != nullptr) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
        if (model != nullptr) {
            llama_model_free(model);
            model = nullptr;
        }
    }


    bool read_gate_up_shard(
            llama_model_loader & loader,
            const ggml_tensor * source,
            ggml_tensor * dest,
            int64_t begin,
            std::string & error) {
        if (source == nullptr ||
            dest == nullptr ||
            source->type != dest->type ||
            source->ne[0] != dest->ne[0] ||
            source->ne[2] != dest->ne[2] ||
            begin < 0 ||
            begin + dest->ne[1] > source->ne[1]) {
            error = "invalid gate/up shard materialization shape";
            return false;
        }

        const size_t dst_bytes =
            ggml_nbytes(dest);
        std::vector<uint8_t> packed(dst_bytes);

        const size_t copy_bytes =
            static_cast<size_t>(dest->ne[1]) *
            source->nb[1];

        for (int64_t expert = 0;
             expert < source->ne[2];
             ++expert) {
            const size_t src_off =
                static_cast<size_t>(expert) *
                    source->nb[2] +
                static_cast<size_t>(begin) *
                    source->nb[1];
            const size_t dst_off =
                static_cast<size_t>(expert) *
                    dest->nb[2];

            if (dst_off > packed.size() ||
                copy_bytes > packed.size() - dst_off ||
                !loader.read_local_tensor(
                    source->name,
                    packed.data() + dst_off,
                    src_off,
                    copy_bytes)) {
                error =
                    std::string("failed to read gate/up shard from GGUF: ") +
                    source->name;
                return false;
            }
        }

        ggml_backend_tensor_set(
            dest,
            packed.data(),
            0,
            packed.size());
        return true;
    }

    bool read_down_shard(
            llama_model_loader & loader,
            const ggml_tensor * source,
            ggml_tensor * dest,
            int64_t begin,
            std::string & error) {
        if (source == nullptr ||
            dest == nullptr ||
            source->type != dest->type ||
            source->ne[1] != dest->ne[1] ||
            source->ne[2] != dest->ne[2] ||
            begin < 0 ||
            begin + dest->ne[0] > source->ne[0]) {
            error = "invalid down shard materialization shape";
            return false;
        }

        const int64_t block =
            ggml_blck_size(source->type);
        if (block <= 0 ||
            begin % block != 0 ||
            dest->ne[0] % block != 0) {
            error = "down shard is not quantization-block aligned";
            return false;
        }

        const size_t src_dim0_off =
            quantized_dim0_offset(
                source,
                begin);
        const size_t row_copy_bytes =
            dest->nb[1];

        std::vector<uint8_t> packed(
            ggml_nbytes(dest));
        std::vector<uint8_t> expert_src(
            source->nb[2]);

        for (int64_t expert = 0;
             expert < source->ne[2];
             ++expert) {
            const size_t expert_src_off =
                static_cast<size_t>(expert) *
                source->nb[2];

            if (!loader.read_local_tensor(
                    source->name,
                    expert_src.data(),
                    expert_src_off,
                    expert_src.size())) {
                error =
                    std::string("failed to read down expert from GGUF: ") +
                    source->name;
                return false;
            }

            for (int64_t row = 0;
                 row < source->ne[1];
                 ++row) {
                const size_t src_off =
                    static_cast<size_t>(row) *
                        source->nb[1] +
                    src_dim0_off;
                const size_t dst_off =
                    static_cast<size_t>(expert) *
                        dest->nb[2] +
                    static_cast<size_t>(row) *
                        dest->nb[1];

                if (src_off > expert_src.size() ||
                    row_copy_bytes >
                        expert_src.size() - src_off ||
                    dst_off > packed.size() ||
                    row_copy_bytes >
                        packed.size() - dst_off) {
                    error = "down shard packing range is invalid";
                    return false;
                }

                std::memcpy(
                    packed.data() + dst_off,
                    expert_src.data() + src_off,
                    row_copy_bytes);
            }
        }

        ggml_backend_tensor_set(
            dest,
            packed.data(),
            0,
            packed.size());
        return true;
    }

    bool initialize_phone_shards(
            std::string & error) {
        const auto device_type =
            ggml_backend_dev_type(device);

        if (config.local_side != side::phone ||
            config.runtime_plan.execution_mode != mode::tensor ||
            device_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            return true;
        }

        const size_t layer_count =
            static_cast<size_t>(
                config.runtime_plan.layers);

        ggml_init_params shard_params {};
        shard_params.mem_size =
            (layer_count * 3 + 32) *
            ggml_tensor_overhead();
        shard_params.mem_buffer = nullptr;
        shard_params.no_alloc = true;

        phone_shard_ctx =
            ggml_init(shard_params);
        if (phone_shard_ctx == nullptr) {
            error =
                "failed to create Phone FFN shard context";
            return false;
        }

        phone_shards.resize(layer_count);

        for (int il = 0;
             il < config.runtime_plan.layers;
             ++il) {
            const key task { il, 0 };
            int64_t begin = 0;
            int64_t width = 0;

            if (!shard_for_side(
                    task,
                    side::phone,
                    begin,
                    width,
                    error)) {
                return false;
            }

            const llama_layer & layer =
                model->layers[
                    static_cast<size_t>(il)];

            materialized_ffn_shard & shard =
                phone_shards[
                    static_cast<size_t>(il)];
            shard.begin = begin;
            shard.width = width;

            shard.gate =
                ggml_new_tensor_3d(
                    phone_shard_ctx,
                    layer.ffn_gate_exps->type,
                    layer.ffn_gate_exps->ne[0],
                    width,
                    layer.ffn_gate_exps->ne[2]);
            shard.up =
                ggml_new_tensor_3d(
                    phone_shard_ctx,
                    layer.ffn_up_exps->type,
                    layer.ffn_up_exps->ne[0],
                    width,
                    layer.ffn_up_exps->ne[2]);
            shard.down =
                ggml_new_tensor_3d(
                    phone_shard_ctx,
                    layer.ffn_down_exps->type,
                    width,
                    layer.ffn_down_exps->ne[1],
                    layer.ffn_down_exps->ne[2]);

            if (shard.gate == nullptr ||
                shard.up == nullptr ||
                shard.down == nullptr) {
                error =
                    "failed to create Phone FFN shard tensors";
                return false;
            }

            char name[GGML_MAX_NAME];
            std::snprintf(
                name,
                sizeof(name),
                "lt.phone.blk.%d.ffn_gate_exps.shard",
                il);
            ggml_set_name(shard.gate, name);
            std::snprintf(
                name,
                sizeof(name),
                "lt.phone.blk.%d.ffn_up_exps.shard",
                il);
            ggml_set_name(shard.up, name);
            std::snprintf(
                name,
                sizeof(name),
                "lt.phone.blk.%d.ffn_down_exps.shard",
                il);
            ggml_set_name(shard.down, name);
        }

        phone_shard_buffer =
            ggml_backend_alloc_ctx_tensors(
                phone_shard_ctx,
                backend);
        if (phone_shard_buffer == nullptr) {
            error =
                "failed to allocate Phone FFN shard buffer";
            return false;
        }

        try {
            std::vector<std::string> splits;
            llama_model_loader loader(
                nullptr,
                nullptr,
                nullptr,
                config.model_path,
                splits,
                nullptr,
                false,
                false,
                false,
                true,
                nullptr,
                nullptr);

            for (int il = 0;
                 il < config.runtime_plan.layers;
                 ++il) {
                const llama_layer & layer =
                    model->layers[
                        static_cast<size_t>(il)];
                materialized_ffn_shard & shard =
                    phone_shards[
                        static_cast<size_t>(il)];

                if (!read_gate_up_shard(
                        loader,
                        layer.ffn_gate_exps,
                        shard.gate,
                        shard.begin,
                        error) ||
                    !read_gate_up_shard(
                        loader,
                        layer.ffn_up_exps,
                        shard.up,
                        shard.begin,
                        error) ||
                    !read_down_shard(
                        loader,
                        layer.ffn_down_exps,
                        shard.down,
                        shard.begin,
                        error)) {
                    return false;
                }

                std::fprintf(
                    stderr,
                    "[LT_GGML_PHONE_SHARD] layer=%d begin=%lld width=%lld "
                    "gate=%.3fMiB up=%.3fMiB down=%.3fMiB "
                    "buffer=%s status=READY\n",
                    il,
                    static_cast<long long>(shard.begin),
                    static_cast<long long>(shard.width),
                    static_cast<double>(
                        ggml_nbytes(shard.gate)) /
                        (1024.0 * 1024.0),
                    static_cast<double>(
                        ggml_nbytes(shard.up)) /
                        (1024.0 * 1024.0),
                    static_cast<double>(
                        ggml_nbytes(shard.down)) /
                        (1024.0 * 1024.0),
                    ggml_backend_buffer_name(
                        phone_shard_buffer));
            }
        } catch (const std::exception & ex) {
            error =
                std::string("Phone FFN shard GGUF load failed: ") +
                ex.what();
            return false;
        }

        ggml_backend_synchronize(backend);
        return true;
    }

    bool initialize(std::string & error) {
        init_backends_once();

        device = select_device(
            config.device,
            config.local_side,
            error);
        if (device == nullptr) {
            return false;
        }

        const auto device_type =
            ggml_backend_dev_type(device);

        if (config.local_side == side::pc &&
            device_type != GGML_BACKEND_DEVICE_TYPE_CPU) {
            error =
                "real standalone v1 keeps PC on CPU; use --device cpu";
            return false;
        }

        backend_device_name = ggml_backend_dev_name(device);

        std::vector<ggml_backend_dev_t> model_devices;
        llama_model_params params =
            llama_model_default_params();

        params.use_mmap = false;
        params.split_mode = LLAMA_SPLIT_MODE_NONE;
        params.no_alloc = false;

        if (device_type == GGML_BACKEND_DEVICE_TYPE_GPU ||
            device_type == GGML_BACKEND_DEVICE_TYPE_IGPU ||
            device_type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            model_devices.push_back(device);
            model_devices.push_back(nullptr);
            params.devices = model_devices.data();
            params.n_gpu_layers = -1;
        } else {
            params.n_gpu_layers = 0;
        }

        set_process_env(
            "LLAMA_HYBRID_EXPERIMENT_N_LAYERS",
            std::to_string(config.runtime_plan.layers));

        model = llama_model_load_from_file(
            config.model_path.c_str(),
            params);

        unset_process_env(
            "LLAMA_HYBRID_EXPERIMENT_N_LAYERS");

        if (model == nullptr) {
            error = "failed to load truncated model: " +
                config.model_path;
            return false;
        }

        if (model->arch != LLM_ARCH_QWEN3MOE) {
            error =
                "standalone real adapter currently supports Qwen3-MoE only";
            return false;
        }

        if (static_cast<int32_t>(
                hp_n_layer(model->hparams)) !=
            config.runtime_plan.layers) {
            std::ostringstream oss;
            oss << "loaded layer count mismatch: model="
                << hp_n_layer(model->hparams)
                << " plan=" << config.runtime_plan.layers;
            error = oss.str();
            return false;
        }

        n_embd = static_cast<int32_t>(
            model->hparams.n_embd);
        n_expert = static_cast<int32_t>(
            model->hparams.n_expert);
        n_expert_used = static_cast<int32_t>(
            model->hparams.n_expert_used);
        n_ff_exp = static_cast<int32_t>(
            model->hparams.n_ff_exp);

        if (n_embd <= 0 ||
            n_expert <= 0 ||
            n_expert_used <= 0 ||
            n_ff_exp <= 0) {
            error = "invalid Qwen3-MoE dimensions";
            return false;
        }

        for (int il = 0;
             il < config.runtime_plan.layers;
             ++il) {
            const llama_layer & layer =
                model->layers[static_cast<size_t>(il)];

            if (layer.attn_norm == nullptr ||
                layer.wq == nullptr ||
                layer.wk == nullptr ||
                layer.wv == nullptr ||
                layer.wo == nullptr ||
                layer.attn_q_norm == nullptr ||
                layer.attn_k_norm == nullptr ||
                layer.ffn_norm == nullptr ||
                layer.ffn_gate_inp == nullptr ||
                layer.ffn_gate_exps == nullptr ||
                layer.ffn_up_exps == nullptr ||
                layer.ffn_down_exps == nullptr) {
                error =
                    "Qwen3-MoE layer is missing a required tensor";
                return false;
            }

            if (layer.wqkv != nullptr ||
                layer.ffn_gate_up_exps != nullptr ||
                layer.ffn_gate_exps_s != nullptr ||
                layer.ffn_up_exps_s != nullptr ||
                layer.ffn_down_exps_s != nullptr) {
                error =
                    "standalone real adapter v1 does not support fused/scaled "
                    "Qwen3-MoE expert tensors";
                return false;
            }
        }

        if (config.runtime_plan.execution_mode == mode::tensor) {
            for (int il = 0;
                 il < config.runtime_plan.layers;
                 ++il) {
                const llama_layer & layer =
                    model->layers[static_cast<size_t>(il)];
                const float requested =
                    config.runtime_plan.pc_ratio_for_layer(il);

                const int64_t pc_width =
                    llama_hybrid_ffn_shard_size(
                        n_ff_exp,
                        layer.ffn_down_exps->type,
                        requested,
                        0);
                const int64_t phone_width =
                    llama_hybrid_ffn_shard_size(
                        n_ff_exp,
                        layer.ffn_down_exps->type,
                        requested,
                        1);

                if (pc_width <= 0 ||
                    phone_width <= 0 ||
                    pc_width + phone_width != n_ff_exp) {
                    std::ostringstream oss;
                    oss << "invalid Tensor shard at layer "
                        << il
                        << ": requested_pc_ratio="
                        << requested
                        << " pc_width=" << pc_width
                        << " phone_width=" << phone_width
                        << " n_ff=" << n_ff_exp;
                    error = oss.str();
                    return false;
                }

                if (config.log_events) {
                    std::fprintf(
                        stderr,
                        "[LT_GGML_SHARD] side=%s layer=%d owner=%s "
                        "requested_pc_ratio=%.6f "
                        "pc_begin=%lld pc_width=%lld "
                        "phone_begin=0 phone_width=%lld "
                        "effective_pc_ratio=%.6f\n",
                        side_name(config.local_side),
                        il,
                        side_name(
                            config.runtime_plan.owner_for_layer(il)),
                        requested,
                        static_cast<long long>(phone_width),
                        static_cast<long long>(pc_width),
                        static_cast<long long>(phone_width),
                        static_cast<double>(pc_width) /
                            static_cast<double>(n_ff_exp));
                }
            }
        }

        if (!initialize_phone_shards(error)) {
            return false;
        }

        if (device_type != GGML_BACKEND_DEVICE_TYPE_CPU) {
            auto check_gpu_tensor =
                [&](int il,
                    const char * name,
                    const ggml_tensor * tensor) -> bool {
                    if (tensor == nullptr) {
                        error =
                            std::string("missing tensor: ") +
                            name;
                        return false;
                    }

                    const char * buffer_name =
                        tensor->buffer != nullptr ?
                            ggml_backend_buffer_name(
                                tensor->buffer) :
                            "NULL";

                    std::fprintf(
                        stderr,
                        "[LT_GGML_WEIGHT] side=%s layer=%d "
                        "tensor=%s buffer=%s extra=%p\n",
                        side_name(config.local_side),
                        il,
                        name,
                        buffer_name,
                        tensor->extra);

                    if (tensor->buffer == nullptr ||
                        tensor->extra == nullptr) {
                        std::ostringstream oss;
                        oss << "GPU tensor is not backend-resident: layer="
                            << il
                            << " tensor=" << name
                            << " buffer=" << buffer_name
                            << " extra=" << tensor->extra;
                        error = oss.str();
                        return false;
                    }

                    return true;
                };

            for (int il = 0;
                 il < config.runtime_plan.layers;
                 ++il) {
                const llama_layer & layer =
                    model->layers[
                        static_cast<size_t>(il)];

                if (!check_gpu_tensor(
                        il, "attn_norm",
                        layer.attn_norm) ||
                    !check_gpu_tensor(
                        il, "wq",
                        layer.wq) ||
                    !check_gpu_tensor(
                        il, "wk",
                        layer.wk) ||
                    !check_gpu_tensor(
                        il, "wv",
                        layer.wv) ||
                    !check_gpu_tensor(
                        il, "wo",
                        layer.wo) ||
                    !check_gpu_tensor(
                        il, "attn_q_norm",
                        layer.attn_q_norm) ||
                    !check_gpu_tensor(
                        il, "attn_k_norm",
                        layer.attn_k_norm) ||
                    !check_gpu_tensor(
                        il, "ffn_norm",
                        layer.ffn_norm) ||
                    !check_gpu_tensor(
                        il, "ffn_gate_inp",
                        layer.ffn_gate_inp) ||
                    !check_gpu_tensor(
                        il, "ffn_gate_exps",
                        layer.ffn_gate_exps) ||
                    !check_gpu_tensor(
                        il, "ffn_up_exps",
                        layer.ffn_up_exps) ||
                    !check_gpu_tensor(
                        il, "ffn_down_exps",
                        layer.ffn_down_exps)) {
                    return false;
                }
            }
        }

        backend = ggml_backend_dev_init(
            device,
            nullptr);
        if (backend == nullptr) {
            error = "failed to initialize execution backend: " +
                backend_device_name;
            return false;
        }

        if (config.threads <= 0) {
            error = "backend thread count must be > 0";
            return false;
        }

        ggml_backend_reg_t backend_reg =
            ggml_backend_dev_backend_reg(device);
        if (backend_reg != nullptr) {
            auto set_threads =
                reinterpret_cast<ggml_backend_set_n_threads_t>(
                    ggml_backend_reg_get_proc_address(
                        backend_reg,
                        "ggml_backend_set_n_threads"));
            if (set_threads != nullptr) {
                set_threads(
                    backend,
                    config.threads);
            }
        }

        total_tokens = std::accumulate(
            config.runtime_plan.chunk_tokens.begin(),
            config.runtime_plan.chunk_tokens.end(),
            int32_t(0));

        if (total_tokens <= 0) {
            error = "invalid total token count";
            return false;
        }

        if (!initialize_kv(error)) {
            return false;
        }
        if (!initialize_activation_buffer(error)) {
            return false;
        }

        const llama_layer & layer0 = model->layers[0];

        std::fprintf(stderr,
            "[LT_GGML_MODEL] side=%s device=%s description=%s "
            "layers=%d tokens=%d n_embd=%d experts=%d topk=%d "
            "ff=%d q_type=%s down_type=%s "
            "threads=%d weight_buffer=%s status=READY\n",
            side_name(config.local_side),
            backend_device_name.c_str(),
            ggml_backend_dev_description(device),
            config.runtime_plan.layers,
            total_tokens,
            n_embd,
            n_expert,
            n_expert_used,
            n_ff_exp,
            ggml_type_name(layer0.wq->type),
            ggml_type_name(layer0.ffn_down_exps->type),
            config.threads,
            layer0.wq->buffer != nullptr ?
                ggml_backend_buffer_name(layer0.wq->buffer) :
                "NULL");

        return true;
    }

    bool initialize_kv(std::string & error) {
        ggml_init_params params {};
        params.mem_size =
            static_cast<size_t>(
                config.runtime_plan.layers * 4 + 32) *
                ggml_tensor_overhead();
        params.mem_buffer = nullptr;
        params.no_alloc = true;

        kv_ctx = ggml_init(params);
        if (kv_ctx == nullptr) {
            error = "failed to create KV metadata context";
            return false;
        }

        kv_k.resize(
            static_cast<size_t>(config.runtime_plan.layers),
            nullptr);
        kv_v.resize(
            static_cast<size_t>(config.runtime_plan.layers),
            nullptr);

        for (int il = 0;
             il < config.runtime_plan.layers;
             ++il) {
            const llama_hparams & hp = model->hparams;
            const int64_t head_k = hp_n_embd_head_k(hp, il);
            const int64_t head_v = hp_n_embd_head_v(hp, il);
            const int64_t heads_kv = hp_n_head_kv(hp, il);

            kv_k[static_cast<size_t>(il)] =
                ggml_new_tensor_3d(
                    kv_ctx,
                    GGML_TYPE_F16,
                    head_k,
                    total_tokens,
                    heads_kv);
            kv_v[static_cast<size_t>(il)] =
                ggml_new_tensor_3d(
                    kv_ctx,
                    GGML_TYPE_F16,
                    head_v,
                    total_tokens,
                    heads_kv);
        }

        kv_buffer =
            ggml_backend_alloc_ctx_tensors(
                kv_ctx,
                backend);
        if (kv_buffer == nullptr) {
            error = "failed to allocate standalone KV buffer";
            return false;
        }

        ggml_backend_buffer_clear(kv_buffer, 0);
        return true;
    }

    bool initialize_activation_buffer(std::string & error) {
        const size_t chunk_count =
            config.runtime_plan.chunk_tokens.size();

        ggml_init_params params {};
        params.mem_size =
            (chunk_count + 40) *
            ggml_tensor_overhead();
        params.mem_buffer = nullptr;
        params.no_alloc = true;

        activation_ctx = ggml_init(params);
        if (activation_ctx == nullptr) {
            error =
                "failed to create activation arena metadata context";
            return false;
        }

        activation_root =
            ggml_new_tensor_2d(
                activation_ctx,
                GGML_TYPE_F32,
                n_embd,
                total_tokens);

        if (activation_root == nullptr) {
            error =
                "failed to create persistent activation arena";
            return false;
        }

        activation_chunks.resize(
            chunk_count,
            nullptr);

        int32_t token_begin = 0;
        for (size_t chunk = 0;
             chunk < chunk_count;
             ++chunk) {
            const int32_t tokens =
                config.runtime_plan.chunk_tokens[chunk];

            activation_chunks[chunk] =
                ggml_view_2d(
                    activation_ctx,
                    activation_root,
                    n_embd,
                    tokens,
                    activation_root->nb[1],
                    static_cast<size_t>(token_begin) *
                        activation_root->nb[1]);

            if (activation_chunks[chunk] == nullptr) {
                error =
                    "failed to create persistent activation chunk view";
                return false;
            }

            token_begin += tokens;
        }

        if (token_begin != total_tokens) {
            error =
                "activation arena token partition mismatch";
            return false;
        }

        activation_buffer =
            ggml_backend_alloc_ctx_tensors(
                activation_ctx,
                backend);
        if (activation_buffer == nullptr) {
            error =
                "failed to allocate persistent activation arena";
            return false;
        }

        ggml_backend_buffer_clear(
            activation_buffer,
            0);

        if (activation_root->buffer == nullptr ||
            (ggml_backend_dev_type(device) !=
                 GGML_BACKEND_DEVICE_TYPE_CPU &&
             activation_root->extra == nullptr)) {
            error =
                "persistent activation root is not backend-ready";
            return false;
        }

        for (size_t chunk = 0;
             chunk < chunk_count;
             ++chunk) {
            ggml_tensor * view =
                activation_chunks[chunk];

            if (view->buffer == nullptr ||
                (ggml_backend_dev_type(device) !=
                     GGML_BACKEND_DEVICE_TYPE_CPU &&
                 view->extra == nullptr)) {
                std::ostringstream oss;
                oss << "persistent activation view is not backend-ready"
                    << ": chunk=" << chunk
                    << " buffer="
                    << (view->buffer != nullptr ?
                        ggml_backend_buffer_name(view->buffer) :
                        "NULL")
                    << " extra=" << view->extra;
                error = oss.str();
                return false;
            }
        }

        std::fprintf(
            stderr,
            "[LT_GGML_ACTIVATION_ARENA] side=%s tokens=%d chunks=%zu "
            "n_embd=%d bytes=%.3f MiB buffer=%s status=READY\n",
            side_name(config.local_side),
            total_tokens,
            chunk_count,
            n_embd,
            static_cast<double>(
                ggml_backend_buffer_get_size(
                    activation_buffer)) /
                (1024.0 * 1024.0),
            ggml_backend_buffer_name(
                activation_buffer));

        return true;
    }

    int32_t chunk_tokens(const key & task) const {
        if (task.chunk < 0 ||
            static_cast<size_t>(task.chunk) >=
                config.runtime_plan.chunk_tokens.size()) {
            return -1;
        }
        return config.runtime_plan.chunk_tokens[
            static_cast<size_t>(task.chunk)];
    }

    int32_t chunk_begin(const key & task) const {
        int32_t begin = 0;
        for (int32_t i = 0; i < task.chunk; ++i) {
            begin += config.runtime_plan.chunk_tokens[
                static_cast<size_t>(i)];
        }
        return begin;
    }


    ggml_tensor * activation_chunk_view(
            const key & task) const {
        if (task.chunk < 0 ||
            static_cast<size_t>(task.chunk) >=
                activation_chunks.size()) {
            return nullptr;
        }

        return activation_chunks[
            static_cast<size_t>(task.chunk)];
    }

    std::shared_ptr<graph_hold> take_cache(
            std::unordered_map<
                key,
                std::shared_ptr<graph_hold>,
                key_hash> & cache,
            const key & task) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        auto it = cache.find(task);
        if (it == cache.end()) {
            return nullptr;
        }

        auto result = it->second;
        cache.erase(it);
        return result;
    }

    void put_cache(
            std::unordered_map<
                key,
                std::shared_ptr<graph_hold>,
                key_hash> & cache,
            const key & task,
            std::shared_ptr<graph_hold> state) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[task] = std::move(state);
    }

    bool validate_hidden_size(
            const key & task,
            const std::vector<float> & values,
            std::string & error) const {
        const int32_t tokens = chunk_tokens(task);
        if (tokens <= 0) {
            error = "invalid task chunk";
            return false;
        }

        const size_t expected =
            static_cast<size_t>(n_embd) *
            static_cast<size_t>(tokens);

        if (values.size() != expected) {
            std::ostringstream oss;
            oss << "hidden size mismatch for L"
                << task.layer << "C" << task.chunk
                << ": got=" << values.size()
                << " expected=" << expected;
            error = oss.str();
            return false;
        }
        return true;
    }

    bool validate_ready(
            const key & task,
            const ready_payload & ready,
            std::string & error) const {
        if (!validate_hidden_size(
                task,
                ready.residual,
                error) ||
            !validate_hidden_size(
                task,
                ready.hidden,
                error)) {
            return false;
        }

        const size_t route_count =
            static_cast<size_t>(n_expert_used) *
            static_cast<size_t>(chunk_tokens(task));

        if (ready.topk.size() != route_count ||
            ready.weights.size() != route_count) {
            std::ostringstream oss;
            oss << "route size mismatch for L"
                << task.layer << "C" << task.chunk
                << ": topk=" << ready.topk.size()
                << " weights=" << ready.weights.size()
                << " expected=" << route_count;
            error = oss.str();
            return false;
        }
        return true;
    }

    bool make_input(
            const key & task,
            const std::vector<float> & host_input,
            const std::shared_ptr<graph_hold> & graph,
            ggml_tensor *& input,
            ggml_tensor *& host_input_tensor,
            std::shared_ptr<graph_hold> & cached_input,
            std::string & error) {
        (void) cached_input;

        const int32_t tokens =
            chunk_tokens(task);
        if (tokens <= 0) {
            error = "invalid task chunk";
            return false;
        }

        if (!host_input.empty()) {
            if (!validate_hidden_size(
                    task,
                    host_input,
                    error)) {
                return false;
            }

            host_input_tensor =
                ggml_new_tensor_2d(
                    graph->ctx,
                    GGML_TYPE_F32,
                    n_embd,
                    tokens);
            input = host_input_tensor;
            return true;
        }

        // Empty host input is the explicit device-local continuation signal.
        // Unlike the old temporary-graph pointer reuse, this tensor is owned
        // by a persistent backend buffer for the entire benchmark lifetime.
        input =
            activation_chunk_view(task);
        host_input_tensor = nullptr;

        if (input == nullptr ||
            input->buffer == nullptr ||
            (ggml_backend_dev_type(device) !=
                 GGML_BACKEND_DEVICE_TYPE_CPU &&
             input->extra == nullptr)) {
            std::ostringstream oss;
            oss << "persistent activation chunk view is not backend-ready"
                << ": layer=" << task.layer
                << " chunk=" << task.chunk
                << " tensor=" << input;
            error = oss.str();
            return false;
        }

        return true;
    }

    bool build_prefix(
            const key & task,
            ggml_context * ctx,
            ggml_tensor * input,
            prefix_nodes & out,
            std::vector<int32_t> & pos_data,
            std::vector<ggml_fp16_t> & mask_data,
            std::string & error) {
        if (task.layer < 0 ||
            task.layer >= config.runtime_plan.layers) {
            error = "prefix layer is out of range";
            return false;
        }

        const int32_t tokens = chunk_tokens(task);
        const int32_t begin = chunk_begin(task);
        const int32_t end = begin + tokens;

        const llama_hparams & hp = model->hparams;
        llama_layer & layer =
            model->layers[
                static_cast<size_t>(task.layer)];

        const int64_t n_head = hp_n_head(hp, task.layer);
        const int64_t n_head_kv =
            hp_n_head_kv(hp, task.layer);
        const int64_t head_k =
            hp_n_embd_head_k(hp, task.layer);
        const int64_t head_v =
            hp_n_embd_head_v(hp, task.layer);
        const int64_t n_rot =
            hp_n_rot(hp, task.layer);

        if (head_k <= 0 ||
            head_v <= 0 ||
            n_head <= 0 ||
            n_head_kv <= 0 ||
            layer.wq->ne[1] !=
                head_k * n_head ||
            layer.wk->ne[1] !=
                head_k * n_head_kv ||
            layer.wv->ne[1] !=
                head_v * n_head_kv ||
            layer.wo->ne[0] !=
                head_v * n_head) {
            error = "unsupported Qwen3-MoE attention shape";
            return false;
        }

        ggml_tensor * attn_norm =
            ggml_rms_norm(
                ctx,
                input,
                hp.f_norm_rms_eps);
        attn_norm =
            ggml_mul(
                ctx,
                attn_norm,
                layer.attn_norm);

        ggml_tensor * q =
            ggml_mul_mat(ctx, layer.wq, attn_norm);
        ggml_tensor * k =
            ggml_mul_mat(ctx, layer.wk, attn_norm);
        ggml_tensor * v =
            ggml_mul_mat(ctx, layer.wv, attn_norm);

        q = ggml_reshape_3d(
            ctx,
            q,
            head_k,
            n_head,
            tokens);
        k = ggml_reshape_3d(
            ctx,
            k,
            head_k,
            n_head_kv,
            tokens);
        v = ggml_reshape_3d(
            ctx,
            v,
            head_v,
            n_head_kv,
            tokens);

        q = ggml_rms_norm(
            ctx,
            q,
            hp.f_norm_rms_eps);
        q = ggml_mul(
            ctx,
            q,
            layer.attn_q_norm);

        k = ggml_rms_norm(
            ctx,
            k,
            hp.f_norm_rms_eps);
        k = ggml_mul(
            ctx,
            k,
            layer.attn_k_norm);

        out.pos =
            ggml_new_tensor_1d(
                ctx,
                GGML_TYPE_I32,
                tokens);

        const int32_t n_ctx_orig =
            hp.n_ctx_orig_yarn > 0 ?
                static_cast<int32_t>(
                    hp.n_ctx_orig_yarn) :
                static_cast<int32_t>(
                    hp.n_ctx_train);

        q = ggml_rope_ext(
            ctx,
            q,
            out.pos,
            nullptr,
            static_cast<int>(n_rot),
            static_cast<int>(
                LLAMA_ROPE_TYPE_NEOX),
            n_ctx_orig,
            hp.rope_freq_base_train,
            hp.rope_freq_scale_train,
            0.0f,
            1.0f,
            32.0f,
            1.0f);

        k = ggml_rope_ext(
            ctx,
            k,
            out.pos,
            nullptr,
            static_cast<int>(n_rot),
            static_cast<int>(
                LLAMA_ROPE_TYPE_NEOX),
            n_ctx_orig,
            hp.rope_freq_base_train,
            hp.rope_freq_scale_train,
            0.0f,
            1.0f,
            32.0f,
            1.0f);

        q = ggml_permute(
            ctx, q, 0, 2, 1, 3);
        k = ggml_permute(
            ctx, k, 0, 2, 1, 3);
        v = ggml_permute(
            ctx, v, 0, 2, 1, 3);

        k = ggml_cast(
            ctx, k, GGML_TYPE_F16);
        v = ggml_cast(
            ctx, v, GGML_TYPE_F16);

        ggml_tensor * k_cache =
            kv_k[static_cast<size_t>(task.layer)];
        ggml_tensor * v_cache =
            kv_v[static_cast<size_t>(task.layer)];

        ggml_tensor * k_slot =
            ggml_view_3d(
                ctx,
                k_cache,
                head_k,
                tokens,
                n_head_kv,
                k_cache->nb[1],
                k_cache->nb[2],
                static_cast<size_t>(begin) *
                    k_cache->nb[1]);
        ggml_tensor * v_slot =
            ggml_view_3d(
                ctx,
                v_cache,
                head_v,
                tokens,
                n_head_kv,
                v_cache->nb[1],
                v_cache->nb[2],
                static_cast<size_t>(begin) *
                    v_cache->nb[1]);

        out.k_write =
            ggml_cpy(ctx, k, k_slot);
        out.v_write =
            ggml_cpy(ctx, v, v_slot);

        ggml_tensor * k_all =
            ggml_view_3d(
                ctx,
                k_cache,
                head_k,
                end,
                n_head_kv,
                k_cache->nb[1],
                k_cache->nb[2],
                0);
        ggml_tensor * v_all =
            ggml_view_3d(
                ctx,
                v_cache,
                head_v,
                end,
                n_head_kv,
                v_cache->nb[1],
                v_cache->nb[2],
                0);

        out.mask =
            ggml_new_tensor_2d(
                ctx,
                GGML_TYPE_F16,
                end,
                tokens);

        ggml_tensor * attn =
            ggml_flash_attn_ext(
                ctx,
                q,
                k_all,
                v_all,
                out.mask,
                1.0f /
                    std::sqrt(
                        static_cast<float>(head_k)),
                0.0f,
                0.0f);
        ggml_flash_attn_ext_set_prec(
            attn,
            GGML_PREC_F32);

        attn =
            ggml_reshape_2d(
                ctx,
                attn,
                head_v * n_head,
                tokens);

        ggml_tensor * attn_out =
            ggml_mul_mat(
                ctx,
                layer.wo,
                attn);

        out.residual =
            ggml_add(
                ctx,
                attn_out,
                input);

        out.hidden =
            ggml_rms_norm(
                ctx,
                out.residual,
                hp.f_norm_rms_eps);
        out.hidden =
            ggml_mul(
                ctx,
                out.hidden,
                layer.ffn_norm);

        ggml_tensor * logits =
            ggml_mul_mat(
                ctx,
                layer.ffn_gate_inp,
                out.hidden);
        ggml_tensor * probs =
            ggml_soft_max(
                ctx,
                logits);

        out.topk =
            ggml_argsort_top_k(
                ctx,
                probs,
                n_expert_used);

        probs =
            ggml_reshape_3d(
                ctx,
                probs,
                1,
                n_expert,
                tokens);

        out.weights =
            ggml_get_rows(
                ctx,
                probs,
                out.topk);
        out.weights =
            ggml_reshape_2d(
                ctx,
                out.weights,
                n_expert_used,
                tokens);

        ggml_tensor * weights_sum =
            ggml_sum_rows(
                ctx,
                out.weights);
        weights_sum =
            ggml_clamp(
                ctx,
                weights_sum,
                6.103515625e-5f,
                std::numeric_limits<float>::infinity());
        out.weights =
            ggml_div(
                ctx,
                out.weights,
                weights_sum);
        out.weights =
            ggml_reshape_3d(
                ctx,
                out.weights,
                1,
                n_expert_used,
                tokens);

        if (hp.expert_weights_scale != 0.0f &&
            hp.expert_weights_scale != 1.0f) {
            out.weights =
                ggml_scale(
                    ctx,
                    out.weights,
                    hp.expert_weights_scale);
        }

        pos_data.resize(
            static_cast<size_t>(tokens));
        for (int32_t i = 0; i < tokens; ++i) {
            pos_data[static_cast<size_t>(i)] =
                begin + i;
        }

        mask_data.resize(
            static_cast<size_t>(end) *
            static_cast<size_t>(tokens));

        const ggml_fp16_t neg_inf =
            ggml_fp32_to_fp16(
                -std::numeric_limits<float>::infinity());
        const ggml_fp16_t zero =
            ggml_fp32_to_fp16(0.0f);

        for (int32_t iq = 0; iq < tokens; ++iq) {
            for (int32_t ik = 0; ik < end; ++ik) {
                mask_data[
                    static_cast<size_t>(iq) *
                        static_cast<size_t>(end) +
                    static_cast<size_t>(ik)] =
                    ik <= begin + iq ?
                        zero : neg_inf;
            }
        }

        return true;
    }

    ggml_tensor * weight_view_gate_up(
            ggml_context * ctx,
            ggml_tensor * tensor,
            int64_t begin,
            int64_t width) {
        if (begin == 0 &&
            width == tensor->ne[1]) {
            return tensor;
        }

        return ggml_view_3d(
            ctx,
            tensor,
            tensor->ne[0],
            width,
            tensor->ne[2],
            tensor->nb[1],
            tensor->nb[2],
            static_cast<size_t>(begin) *
                tensor->nb[1]);
    }

    ggml_tensor * weight_view_down(
            ggml_context * ctx,
            ggml_tensor * tensor,
            int64_t begin,
            int64_t width) {
        if (begin == 0 &&
            width == tensor->ne[0]) {
            return tensor;
        }

        return ggml_view_3d(
            ctx,
            tensor,
            width,
            tensor->ne[1],
            tensor->ne[2],
            tensor->nb[1],
            tensor->nb[2],
            quantized_dim0_offset(
                tensor,
                begin));
    }

    ggml_tensor * build_moe_partial(
            ggml_context * ctx,
            const key & task,
            ggml_tensor * hidden,
            ggml_tensor * ids,
            ggml_tensor * mix_weights,
            int64_t shard_begin,
            int64_t shard_width,
            std::string & error) {
        llama_layer & layer =
            model->layers[
                static_cast<size_t>(task.layer)];

        if (shard_width <= 0 ||
            shard_begin < 0 ||
            shard_begin + shard_width >
                layer.ffn_down_exps->ne[0]) {
            error = "invalid expert shard";
            return nullptr;
        }

        const int64_t down_block =
            ggml_blck_size(
                layer.ffn_down_exps->type);
        if (shard_begin % down_block != 0 ||
            shard_width % down_block != 0) {
            error =
                "expert shard is not quantization-block aligned";
            return nullptr;
        }

        ggml_tensor * gate_w =
            weight_view_gate_up(
                ctx,
                layer.ffn_gate_exps,
                shard_begin,
                shard_width);
        ggml_tensor * up_w =
            weight_view_gate_up(
                ctx,
                layer.ffn_up_exps,
                shard_begin,
                shard_width);
        ggml_tensor * down_w =
            weight_view_down(
                ctx,
                layer.ffn_down_exps,
                shard_begin,
                shard_width);

        const int32_t tokens =
            chunk_tokens(task);

        ggml_tensor * input3 =
            ggml_reshape_3d(
                ctx,
                hidden,
                n_embd,
                1,
                tokens);

        ggml_tensor * gate =
            ggml_mul_mat_id(
                ctx,
                gate_w,
                input3,
                ids);
        ggml_tensor * up =
            ggml_mul_mat_id(
                ctx,
                up_w,
                input3,
                ids);

        ggml_tensor * activated =
            ggml_swiglu_split(
                ctx,
                gate,
                up);

        ggml_tensor * experts =
            ggml_mul_mat_id(
                ctx,
                down_w,
                activated,
                ids);

        experts =
            ggml_mul(
                ctx,
                experts,
                mix_weights);

        std::vector<ggml_tensor *> expert_views;
        expert_views.reserve(
            static_cast<size_t>(n_expert_used));

        for (int ie = 0;
             ie < n_expert_used;
             ++ie) {
            expert_views.push_back(
                ggml_view_2d(
                    ctx,
                    experts,
                    n_embd,
                    tokens,
                    experts->nb[2],
                    static_cast<size_t>(ie) *
                        experts->nb[1]));
        }

        ggml_tensor * output =
            expert_views.front();

        for (int ie = 1;
             ie < n_expert_used;
             ++ie) {
            output =
                ggml_add(
                    ctx,
                    output,
                    expert_views[
                        static_cast<size_t>(ie)]);
        }

        if (n_expert_used == 1) {
            output =
                ggml_cont(
                    ctx,
                    output);
        }

        return output;
    }

    bool shard_for_side(
            const key & task,
            side shard_side,
            int64_t & begin,
            int64_t & width,
            std::string & error) {
        llama_layer & layer =
            model->layers[
                static_cast<size_t>(task.layer)];

        const float pc_ratio =
            config.runtime_plan.pc_ratio_for_layer(
                task.layer);

        const int64_t pc_width =
            llama_hybrid_ffn_shard_size(
                n_ff_exp,
                layer.ffn_down_exps->type,
                pc_ratio,
                0);
        const int64_t phone_width =
            llama_hybrid_ffn_shard_size(
                n_ff_exp,
                layer.ffn_down_exps->type,
                pc_ratio,
                1);

        if (pc_width <= 0 ||
            phone_width <= 0 ||
            pc_width + phone_width != n_ff_exp) {
            error =
                "tensor ratio produced an empty/invalid expert shard";
            return false;
        }

        if (shard_side == side::pc) {
            begin = phone_width;
            width = pc_width;
        } else {
            begin = 0;
            width = phone_width;
        }

        return true;
    }

    bool shard_for_task(
            const key & task,
            int64_t & begin,
            int64_t & width,
            std::string & error) {
        return shard_for_side(
            task,
            config.local_side,
            begin,
            width,
            error);
    }

    bool allocate_state(
            const std::shared_ptr<graph_hold> & state,
            std::string & error) {
        state->buffer =
            ggml_backend_alloc_ctx_tensors(
                state->ctx,
                backend);
        if (state->buffer == nullptr) {
            error =
                "failed to allocate GGML graph tensors";
            return false;
        }
        return true;
    }

    bool run_owner_prefix(
            const key & task,
            const std::vector<float> & host_input,
            ready_payload & ready,
            std::string & error) {
        auto state = new_graph_hold();
        if (state == nullptr) {
            error =
                "failed to create owner graph context";
            return false;
        }

        ggml_tensor * input = nullptr;
        ggml_tensor * host_input_tensor = nullptr;
        std::shared_ptr<graph_hold> cached_input;

        if (!make_input(
                task,
                host_input,
                state,
                input,
                host_input_tensor,
                cached_input,
                error)) {
            return false;
        }

        prefix_nodes prefix;
        std::vector<int32_t> pos_data;
        std::vector<ggml_fp16_t> mask_data;

        if (!build_prefix(
                task,
                state->ctx,
                input,
                prefix,
                pos_data,
                mask_data,
                error)) {
            return false;
        }

        ggml_cgraph * graph =
            ggml_new_graph_custom(
                state->ctx,
                2048,
                false);

        // KV writes are aliasing side effects, so force them before the
        // attention consumer graph just like the main Qwen3-MoE builder.
        ggml_build_forward_expand(
            graph,
            prefix.k_write);
        ggml_build_forward_expand(
            graph,
            prefix.v_write);
        ggml_build_forward_expand(
            graph,
            prefix.weights);

        if (!allocate_state(
                state,
                error)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(
                backend_mutex);

            if (host_input_tensor != nullptr) {
                ggml_backend_tensor_set(
                    host_input_tensor,
                    host_input.data(),
                    0,
                    host_input.size() *
                        sizeof(float));
            }

            ggml_backend_tensor_set(
                prefix.pos,
                pos_data.data(),
                0,
                pos_data.size() *
                    sizeof(int32_t));

            ggml_backend_tensor_set(
                prefix.mask,
                mask_data.data(),
                0,
                mask_data.size() *
                    sizeof(ggml_fp16_t));

            const ggml_status status =
                ggml_backend_graph_compute(
                    backend,
                    graph);
            ggml_backend_synchronize(backend);

            if (status != GGML_STATUS_SUCCESS) {
                error =
                    "OWNER prefix GGML compute failed";
                return false;
            }

            const size_t hidden_count =
                static_cast<size_t>(n_embd) *
                static_cast<size_t>(
                    chunk_tokens(task));
            const size_t route_count =
                static_cast<size_t>(n_expert_used) *
                static_cast<size_t>(
                    chunk_tokens(task));

            ready.residual.resize(hidden_count);
            ready.hidden.resize(hidden_count);
            ready.topk.resize(route_count);
            ready.weights.resize(route_count);

            ggml_backend_tensor_get(
                prefix.residual,
                ready.residual.data(),
                0,
                hidden_count * sizeof(float));
            ggml_backend_tensor_get(
                prefix.hidden,
                ready.hidden.data(),
                0,
                hidden_count * sizeof(float));
            ggml_backend_tensor_get(
                prefix.topk,
                ready.topk.data(),
                0,
                route_count * sizeof(int32_t));
            ggml_backend_tensor_get(
                prefix.weights,
                ready.weights.data(),
                0,
                route_count * sizeof(float));
        }

        state->residual = prefix.residual;
        state->hidden = prefix.hidden;
        state->topk = prefix.topk;
        state->weights = prefix.weights;

        put_cache(
            owner_cache,
            task,
            state);

        return true;
    }

    bool run_owner_local_ffn(
            const key & task,
            std::vector<float> & partial,
            std::string & error) {
        auto prefix =
            take_cache(
                owner_cache,
                task);
        if (prefix == nullptr ||
            prefix->hidden == nullptr ||
            prefix->topk == nullptr ||
            prefix->weights == nullptr) {
            error =
                "OWNER local FFN prefix state missing";
            return false;
        }

        int64_t shard_begin = 0;
        int64_t shard_width = 0;
        if (!shard_for_task(
                task,
                shard_begin,
                shard_width,
                error)) {
            return false;
        }

        auto state = new_graph_hold();
        if (state == nullptr) {
            error =
                "failed to create OWNER FFN context";
            return false;
        }

        ggml_tensor * output =
            build_moe_partial(
                state->ctx,
                task,
                prefix->hidden,
                prefix->topk,
                prefix->weights,
                shard_begin,
                shard_width,
                error);
        if (output == nullptr) {
            return false;
        }

        ggml_cgraph * graph =
            ggml_new_graph_custom(
                state->ctx,
                1024,
                false);
        ggml_build_forward_expand(
            graph,
            output);

        if (!allocate_state(
                state,
                error)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(
                backend_mutex);

            const ggml_status status =
                ggml_backend_graph_compute(
                    backend,
                    graph);
            ggml_backend_synchronize(backend);

            if (status != GGML_STATUS_SUCCESS) {
                error =
                    "OWNER local FFN GGML compute failed";
                return false;
            }

            const size_t count =
                static_cast<size_t>(n_embd) *
                static_cast<size_t>(
                    chunk_tokens(task));
            partial.resize(count);

            ggml_backend_tensor_get(
                output,
                partial.data(),
                0,
                count * sizeof(float));
        }

        return true;
    }

    bool run_ffn_shard_reference(
            const key & task,
            const ready_payload & ready,
            side shard_side,
            std::vector<float> & partial,
            std::string & error) {
        if (!validate_ready(
                task,
                ready,
                error)) {
            return false;
        }

        int64_t shard_begin = 0;
        int64_t shard_width = 0;
        if (!shard_for_side(
                task,
                shard_side,
                shard_begin,
                shard_width,
                error)) {
            return false;
        }

        const int32_t tokens =
            chunk_tokens(task);

        auto state = new_graph_hold();
        if (state == nullptr) {
            error =
                "failed to create FFN shard reference context";
            return false;
        }

        ggml_tensor * hidden =
            ggml_new_tensor_2d(
                state->ctx,
                GGML_TYPE_F32,
                n_embd,
                tokens);
        ggml_tensor * ids =
            ggml_new_tensor_2d(
                state->ctx,
                GGML_TYPE_I32,
                n_expert_used,
                tokens);
        ggml_tensor * weights =
            ggml_new_tensor_3d(
                state->ctx,
                GGML_TYPE_F32,
                1,
                n_expert_used,
                tokens);

        ggml_tensor * output =
            build_moe_partial(
                state->ctx,
                task,
                hidden,
                ids,
                weights,
                shard_begin,
                shard_width,
                error);
        if (output == nullptr) {
            return false;
        }

        ggml_cgraph * graph =
            ggml_new_graph_custom(
                state->ctx,
                1024,
                false);
        ggml_build_forward_expand(
            graph,
            output);

        if (!allocate_state(
                state,
                error)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(
                backend_mutex);

            ggml_backend_tensor_set(
                hidden,
                ready.hidden.data(),
                0,
                ready.hidden.size() *
                    sizeof(float));
            ggml_backend_tensor_set(
                ids,
                ready.topk.data(),
                0,
                ready.topk.size() *
                    sizeof(int32_t));
            ggml_backend_tensor_set(
                weights,
                ready.weights.data(),
                0,
                ready.weights.size() *
                    sizeof(float));

            const ggml_status status =
                ggml_backend_graph_compute(
                    backend,
                    graph);
            ggml_backend_synchronize(
                backend);

            if (status != GGML_STATUS_SUCCESS) {
                error =
                    "FFN shard reference GGML compute failed";
                return false;
            }

            const size_t count =
                static_cast<size_t>(n_embd) *
                static_cast<size_t>(tokens);
            partial.resize(count);
            ggml_backend_tensor_get(
                output,
                partial.data(),
                0,
                count * sizeof(float));
        }

        return true;
    }

    bool run_remote_ffn(
            const key & task,
            const ready_payload & ready,
            std::vector<float> & partial,
            std::string & error) {
        if (!validate_ready(
                task,
                ready,
                error)) {
            return false;
        }

        int64_t shard_begin = 0;
        int64_t shard_width = 0;
        if (!shard_for_task(
                task,
                shard_begin,
                shard_width,
                error)) {
            return false;
        }

        const int32_t tokens =
            chunk_tokens(task);

        auto state = new_graph_hold();
        if (state == nullptr) {
            error =
                "failed to create REMOTE_FFN context";
            return false;
        }

        ggml_tensor * residual =
            ggml_new_tensor_2d(
                state->ctx,
                GGML_TYPE_F32,
                n_embd,
                tokens);
        ggml_tensor * hidden =
            ggml_new_tensor_2d(
                state->ctx,
                GGML_TYPE_F32,
                n_embd,
                tokens);
        ggml_tensor * ids =
            ggml_new_tensor_2d(
                state->ctx,
                GGML_TYPE_I32,
                n_expert_used,
                tokens);
        ggml_tensor * weights =
            ggml_new_tensor_3d(
                state->ctx,
                GGML_TYPE_F32,
                1,
                n_expert_used,
                tokens);

        ggml_tensor * output =
            build_moe_partial(
                state->ctx,
                task,
                hidden,
                ids,
                weights,
                shard_begin,
                shard_width,
                error);
        if (output == nullptr) {
            return false;
        }

        ggml_cgraph * graph =
            ggml_new_graph_custom(
                state->ctx,
                1024,
                false);
        ggml_build_forward_expand(
            graph,
            output);

        if (!allocate_state(
                state,
                error)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(
                backend_mutex);

            ggml_backend_tensor_set(
                residual,
                ready.residual.data(),
                0,
                ready.residual.size() *
                    sizeof(float));
            ggml_backend_tensor_set(
                hidden,
                ready.hidden.data(),
                0,
                ready.hidden.size() *
                    sizeof(float));
            ggml_backend_tensor_set(
                ids,
                ready.topk.data(),
                0,
                ready.topk.size() *
                    sizeof(int32_t));
            ggml_backend_tensor_set(
                weights,
                ready.weights.data(),
                0,
                ready.weights.size() *
                    sizeof(float));

            const ggml_status status =
                ggml_backend_graph_compute(
                    backend,
                    graph);
            ggml_backend_synchronize(backend);

            if (status != GGML_STATUS_SUCCESS) {
                error =
                    "REMOTE_FFN GGML compute failed";
                return false;
            }
        }

        state->primary = output;
        state->residual = residual;

        if (config.runtime_plan.debug_phone_partial &&
            config.local_side == side::phone) {
            const size_t count =
                static_cast<size_t>(n_embd) *
                static_cast<size_t>(tokens);
            partial.resize(count);

            std::lock_guard<std::mutex> lock(
                backend_mutex);
            ggml_backend_tensor_get(
                output,
                partial.data(),
                0,
                count * sizeof(float));
        } else {
            // Normal performance path: keep the local shard device-resident
            // until REDUCE and avoid a host readback.
            partial.clear();
        }

        put_cache(
            remote_cache,
            task,
            state);

        return true;
    }

    bool run_reduce(
            const key & task,
            const std::vector<float> & remote_partial,
            std::vector<float> & output,
            std::string & error) {
        auto local =
            take_cache(
                remote_cache,
                task);
        if (local == nullptr ||
            local->primary == nullptr ||
            local->residual == nullptr) {
            error =
                "REDUCE local partial state missing";
            return false;
        }

        if (!validate_hidden_size(
                task,
                remote_partial,
                error)) {
            return false;
        }

        const int32_t tokens =
            chunk_tokens(task);

        auto state = new_graph_hold();
        if (state == nullptr) {
            error =
                "failed to create REDUCE context";
            return false;
        }

        ggml_tensor * peer_partial =
            ggml_new_tensor_2d(
                state->ctx,
                GGML_TYPE_F32,
                n_embd,
                tokens);

        ggml_tensor * sum =
            ggml_add(
                state->ctx,
                local->primary,
                peer_partial);
        ggml_tensor * reduced =
            ggml_add(
                state->ctx,
                local->residual,
                sum);

        ggml_cgraph * graph =
            ggml_new_graph_custom(
                state->ctx,
                64,
                false);
        ggml_build_forward_expand(
            graph,
            reduced);

        if (!allocate_state(
                state,
                error)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(
                backend_mutex);

            ggml_backend_tensor_set(
                peer_partial,
                remote_partial.data(),
                0,
                remote_partial.size() *
                    sizeof(float));

            const ggml_status status =
                ggml_backend_graph_compute(
                    backend,
                    graph);
            ggml_backend_synchronize(backend);

            if (status != GGML_STATUS_SUCCESS) {
                error =
                    "REDUCE GGML compute failed";
                return false;
            }

            const bool final_layer =
                task.layer + 1 ==
                    config.runtime_plan.layers;

            if (final_layer) {
                const size_t count =
                    static_cast<size_t>(n_embd) *
                    static_cast<size_t>(tokens);
                output.resize(count);

                ggml_backend_tensor_get(
                    reduced,
                    output.data(),
                    0,
                    count * sizeof(float));
            } else {
                ggml_tensor * arena_view =
                    activation_chunk_view(task);

                if (arena_view == nullptr ||
                    arena_view->buffer == nullptr ||
                    (ggml_backend_dev_type(device) !=
                         GGML_BACKEND_DEVICE_TYPE_CPU &&
                     arena_view->extra == nullptr)) {
                    error =
                        "REDUCE activation chunk view is not backend-ready";
                    return false;
                }

                ggml_backend_tensor_copy_async(
                    backend,
                    backend,
                    reduced,
                    arena_view);
                ggml_backend_synchronize(
                    backend);

                // Empty output tells runtime that the next OWNER input is
                // already resident in its persistent activation arena.
                output.clear();

                if (config.log_events) {
                    std::fprintf(
                        stderr,
                        "[LT_GGML_LOCAL_CONT] side=%s "
                        "from_layer=%d to_layer=%d chunk=%d "
                        "arena=FIXED_RANGE mode=REDUCE status=READY\n",
                        side_name(config.local_side),
                        task.layer,
                        task.layer + 1,
                        task.chunk);
                }
            }
        }

        return true;
    }

    bool run_full_layer(
            const key & task,
            const std::vector<float> & host_input,
            std::vector<float> & output,
            std::string & error) {
        auto state = new_graph_hold();
        if (state == nullptr) {
            error =
                "failed to create FULL_LAYER context";
            return false;
        }

        ggml_tensor * input = nullptr;
        ggml_tensor * host_input_tensor = nullptr;
        std::shared_ptr<graph_hold> cached_input;

        if (!make_input(
                task,
                host_input,
                state,
                input,
                host_input_tensor,
                cached_input,
                error)) {
            return false;
        }

        prefix_nodes prefix;
        std::vector<int32_t> pos_data;
        std::vector<ggml_fp16_t> mask_data;

        if (!build_prefix(
                task,
                state->ctx,
                input,
                prefix,
                pos_data,
                mask_data,
                error)) {
            return false;
        }

        ggml_tensor * ffn =
            build_moe_partial(
                state->ctx,
                task,
                prefix.hidden,
                prefix.topk,
                prefix.weights,
                0,
                n_ff_exp,
                error);
        if (ffn == nullptr) {
            return false;
        }

        ggml_tensor * layer_out =
            ggml_add(
                state->ctx,
                prefix.residual,
                ffn);

        ggml_cgraph * graph =
            ggml_new_graph_custom(
                state->ctx,
                2048,
                false);
        ggml_build_forward_expand(
            graph,
            prefix.k_write);
        ggml_build_forward_expand(
            graph,
            prefix.v_write);
        ggml_build_forward_expand(
            graph,
            layer_out);

        if (!allocate_state(
                state,
                error)) {
            return false;
        }

        const bool final_layer =
            task.layer + 1 ==
                config.runtime_plan.layers;
        const bool next_local =
            !final_layer &&
            config.runtime_plan.owner_for_layer(
                task.layer + 1) ==
                config.local_side;

        {
            std::lock_guard<std::mutex> lock(
                backend_mutex);

            if (host_input_tensor != nullptr) {
                ggml_backend_tensor_set(
                    host_input_tensor,
                    host_input.data(),
                    0,
                    host_input.size() *
                        sizeof(float));
            }

            ggml_backend_tensor_set(
                prefix.pos,
                pos_data.data(),
                0,
                pos_data.size() *
                    sizeof(int32_t));
            ggml_backend_tensor_set(
                prefix.mask,
                mask_data.data(),
                0,
                mask_data.size() *
                    sizeof(ggml_fp16_t));

            const ggml_status status =
                ggml_backend_graph_compute(
                    backend,
                    graph);
            ggml_backend_synchronize(backend);

            if (status != GGML_STATUS_SUCCESS) {
                error =
                    "FULL_LAYER GGML compute failed";
                return false;
            }

            if (next_local) {
                ggml_tensor * arena_view =
                    activation_chunk_view(task);

                if (arena_view == nullptr ||
                    arena_view->buffer == nullptr ||
                    (ggml_backend_dev_type(device) !=
                         GGML_BACKEND_DEVICE_TYPE_CPU &&
                     arena_view->extra == nullptr)) {
                    error =
                        "FULL_LAYER activation chunk view is not backend-ready";
                    return false;
                }

                ggml_backend_tensor_copy_async(
                    backend,
                    backend,
                    layer_out,
                    arena_view);
                ggml_backend_synchronize(
                    backend);
                output.clear();

                if (config.log_events) {
                    std::fprintf(
                        stderr,
                        "[LT_GGML_LOCAL_CONT] side=%s "
                        "from_layer=%d to_layer=%d chunk=%d "
                        "arena=FIXED_RANGE mode=FULL status=READY\n",
                        side_name(config.local_side),
                        task.layer,
                        task.layer + 1,
                        task.chunk);
                }
            } else {
                const size_t count =
                    static_cast<size_t>(n_embd) *
                    static_cast<size_t>(
                        chunk_tokens(task));
                output.resize(count);

                ggml_backend_tensor_get(
                    layer_out,
                    output.data(),
                    0,
                    count * sizeof(float));
            }
        }

        return true;
    }

    bool run_ffn_split_selfcheck(
            const key & task,
            const std::vector<float> & host_input,
            float pc_ratio,
            std::string & report,
            std::string & error) {
        if (config.local_side != side::pc ||
            ggml_backend_dev_type(device) !=
                GGML_BACKEND_DEVICE_TYPE_CPU) {
            error = "FFN split selfcheck requires PC CPU backend";
            return false;
        }

        if (!validate_hidden_size(
                task,
                host_input,
                error)) {
            return false;
        }

        llama_layer & layer =
            model->layers[
                static_cast<size_t>(task.layer)];

        const int64_t pc_width =
            llama_hybrid_ffn_shard_size(
                n_ff_exp,
                layer.ffn_down_exps->type,
                pc_ratio,
                0);
        const int64_t phone_width =
            llama_hybrid_ffn_shard_size(
                n_ff_exp,
                layer.ffn_down_exps->type,
                pc_ratio,
                1);

        if (pc_width <= 0 ||
            phone_width <= 0 ||
            pc_width + phone_width != n_ff_exp) {
            error = "FFN split selfcheck ratio produces invalid shard";
            return false;
        }

        auto state = new_graph_hold();
        if (state == nullptr) {
            error = "failed to create FFN split selfcheck context";
            return false;
        }

        ggml_tensor * input = nullptr;
        ggml_tensor * host_input_tensor = nullptr;
        std::shared_ptr<graph_hold> cached_input;

        if (!make_input(
                task,
                host_input,
                state,
                input,
                host_input_tensor,
                cached_input,
                error)) {
            return false;
        }

        prefix_nodes prefix;
        std::vector<int32_t> pos_data;
        std::vector<ggml_fp16_t> mask_data;

        if (!build_prefix(
                task,
                state->ctx,
                input,
                prefix,
                pos_data,
                mask_data,
                error)) {
            return false;
        }

        // Mirror the current physical mapping exactly:
        //   Phone/head = [0, phone_width)
        //   PC/tail    = [phone_width, n_ff_exp)
        ggml_tensor * full =
            build_moe_partial(
                state->ctx,
                task,
                prefix.hidden,
                prefix.topk,
                prefix.weights,
                0,
                n_ff_exp,
                error);
        if (full == nullptr) {
            return false;
        }

        ggml_tensor * head =
            build_moe_partial(
                state->ctx,
                task,
                prefix.hidden,
                prefix.topk,
                prefix.weights,
                0,
                phone_width,
                error);
        if (head == nullptr) {
            return false;
        }

        ggml_tensor * tail =
            build_moe_partial(
                state->ctx,
                task,
                prefix.hidden,
                prefix.topk,
                prefix.weights,
                phone_width,
                pc_width,
                error);
        if (tail == nullptr) {
            return false;
        }

        ggml_tensor * split =
            ggml_add(
                state->ctx,
                head,
                tail);

        ggml_cgraph * graph =
            ggml_new_graph_custom(
                state->ctx,
                4096,
                false);
        ggml_build_forward_expand(
            graph,
            prefix.k_write);
        ggml_build_forward_expand(
            graph,
            prefix.v_write);
        ggml_build_forward_expand(
            graph,
            full);
        ggml_build_forward_expand(
            graph,
            head);
        ggml_build_forward_expand(
            graph,
            tail);
        ggml_build_forward_expand(
            graph,
            split);

        if (!allocate_state(
                state,
                error)) {
            return false;
        }

        const size_t count =
            static_cast<size_t>(n_embd) *
            static_cast<size_t>(
                chunk_tokens(task));

        std::vector<float> full_host(count);
        std::vector<float> head_host(count);
        std::vector<float> tail_host(count);
        std::vector<float> split_host(count);

        {
            std::lock_guard<std::mutex> lock(
                backend_mutex);

            GGML_ASSERT(
                host_input_tensor != nullptr);
            ggml_backend_tensor_set(
                host_input_tensor,
                host_input.data(),
                0,
                host_input.size() *
                    sizeof(float));
            ggml_backend_tensor_set(
                prefix.pos,
                pos_data.data(),
                0,
                pos_data.size() *
                    sizeof(int32_t));
            ggml_backend_tensor_set(
                prefix.mask,
                mask_data.data(),
                0,
                mask_data.size() *
                    sizeof(ggml_fp16_t));

            const ggml_status status =
                ggml_backend_graph_compute(
                    backend,
                    graph);
            ggml_backend_synchronize(
                backend);

            if (status != GGML_STATUS_SUCCESS) {
                error = "FFN split selfcheck GGML compute failed";
                return false;
            }

            ggml_backend_tensor_get(
                full,
                full_host.data(),
                0,
                count * sizeof(float));
            ggml_backend_tensor_get(
                head,
                head_host.data(),
                0,
                count * sizeof(float));
            ggml_backend_tensor_get(
                tail,
                tail_host.data(),
                0,
                count * sizeof(float));
            ggml_backend_tensor_get(
                split,
                split_host.data(),
                0,
                count * sizeof(float));
        }

        double diff_sq = 0.0;
        double full_sq = 0.0;
        double max_abs = 0.0;
        double max_rel = 0.0;

        for (size_t i = 0; i < count; ++i) {
            const double a =
                static_cast<double>(
                    full_host[i]);
            const double b =
                static_cast<double>(
                    split_host[i]);
            const double d =
                std::fabs(a - b);

            diff_sq += d * d;
            full_sq += a * a;
            max_abs =
                std::max(
                    max_abs,
                    d);
            max_rel =
                std::max(
                    max_rel,
                    d /
                        std::max(
                            1e-12,
                            std::fabs(a)));
        }

        const double rms =
            std::sqrt(
                diff_sq /
                std::max<size_t>(1, count));
        const double full_rms =
            std::sqrt(
                full_sq /
                std::max<size_t>(1, count));
        const double rel_rms =
            rms /
            std::max(
                1e-12,
                full_rms);

        std::ostringstream oss;
        oss << std::setprecision(12)
            << "layer=" << task.layer
            << " chunk=" << task.chunk
            << " pc_ratio=" << pc_ratio
            << " phone_begin=0"
            << " phone_width=" << phone_width
            << " pc_begin=" << phone_width
            << " pc_width=" << pc_width
            << " full_checksum="
            << checksum(full_host)
            << " head_checksum="
            << checksum(head_host)
            << " tail_checksum="
            << checksum(tail_host)
            << " split_checksum="
            << checksum(split_host)
            << " max_abs=" << max_abs
            << " rms=" << rms
            << " rel_rms=" << rel_rms
            << " max_rel=" << max_rel;

        report = oss.str();
        return true;
    }

};

ggml_compute::ggml_compute(
        std::unique_ptr<impl> impl_value)
    : impl_(std::move(impl_value)) {
}

ggml_compute::~ggml_compute() = default;

std::shared_ptr<ggml_compute> ggml_compute::create(
        ggml_compute_config config,
        std::string & error) {
    auto implementation =
        std::make_unique<impl>();
    implementation->config =
        std::move(config);

    if (!implementation->initialize(error)) {
        return nullptr;
    }

    return std::shared_ptr<ggml_compute>(
        new ggml_compute(
            std::move(implementation)));
}

int32_t ggml_compute::n_embd() const {
    return impl_->n_embd;
}

int32_t ggml_compute::n_layers() const {
    return impl_->config.runtime_plan.layers;
}

const char * ggml_compute::device_name() const {
    return impl_->backend_device_name.c_str();
}

bool ggml_compute::embed_tokens(
        const std::vector<int32_t> & tokens,
        std::vector<float> & output,
        std::string & error) {
    if (impl_->config.local_side != side::pc) {
        error =
            "token embedding is a PC prelude";
        return false;
    }
    if (tokens.empty()) {
        error = "cannot embed an empty token list";
        return false;
    }

    auto state = new_graph_hold();
    if (state == nullptr) {
        error =
            "failed to create embedding context";
        return false;
    }

    ggml_tensor * ids =
        ggml_new_tensor_1d(
            state->ctx,
            GGML_TYPE_I32,
            static_cast<int64_t>(tokens.size()));

    ggml_tensor * embedded =
        ggml_get_rows(
            state->ctx,
            impl_->model->tok_embd,
            ids);

    ggml_cgraph * graph =
        ggml_new_graph_custom(
            state->ctx,
            64,
            false);
    ggml_build_forward_expand(
        graph,
        embedded);

    if (!impl_->allocate_state(
            state,
            error)) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(
            impl_->backend_mutex);

        ggml_backend_tensor_set(
            ids,
            tokens.data(),
            0,
            tokens.size() *
                sizeof(int32_t));

        const ggml_status status =
            ggml_backend_graph_compute(
                impl_->backend,
                graph);
        ggml_backend_synchronize(
            impl_->backend);

        if (status != GGML_STATUS_SUCCESS) {
            error =
                "token embedding GGML compute failed";
            return false;
        }

        output.resize(
            static_cast<size_t>(impl_->n_embd) *
            tokens.size());

        ggml_backend_tensor_get(
            embedded,
            output.data(),
            0,
            output.size() *
                sizeof(float));
    }

    return true;
}

bool ggml_compute::ffn_split_selfcheck(
        const key & task,
        const std::vector<float> & input,
        float pc_ratio,
        std::string & report,
        std::string & error) {
    return impl_->run_ffn_split_selfcheck(
        task,
        input,
        pc_ratio,
        report,
        error);
}

bool ggml_compute::ffn_shard_reference(
        const key & task,
        const ready_payload & ready,
        side shard_side,
        std::vector<float> & partial,
        std::string & error) {
    return impl_->run_ffn_shard_reference(
        task,
        ready,
        shard_side,
        partial,
        error);
}

bool ggml_compute::owner_prefix(
        const key & task,
        const std::vector<float> & input,
        ready_payload & ready) {
    std::string error;
    const bool ok =
        impl_->run_owner_prefix(
            task,
            input,
            ready,
            error);

    if (!ok) {
        std::fprintf(stderr,
            "[LT_GGML_ERROR] side=%s phase=OWNER_PREFIX "
            "layer=%d chunk=%d reason=%s\n",
            side_name(impl_->config.local_side),
            task.layer,
            task.chunk,
            error.c_str());
    }
    return ok;
}

bool ggml_compute::local_ffn(
        const key & task,
        const ready_payload &,
        std::vector<float> & partial) {
    std::string error;
    const bool ok =
        impl_->run_owner_local_ffn(
            task,
            partial,
            error);

    if (!ok) {
        std::fprintf(stderr,
            "[LT_GGML_ERROR] side=%s phase=LOCAL_FFN "
            "layer=%d chunk=%d reason=%s\n",
            side_name(impl_->config.local_side),
            task.layer,
            task.chunk,
            error.c_str());
    }
    return ok;
}

bool ggml_compute::remote_ffn(
        const key & task,
        const ready_payload & ready,
        std::vector<float> & partial) {
    std::string error;
    const bool ok =
        impl_->run_remote_ffn(
            task,
            ready,
            partial,
            error);

    if (!ok) {
        std::fprintf(stderr,
            "[LT_GGML_ERROR] side=%s phase=REMOTE_FFN "
            "layer=%d chunk=%d reason=%s\n",
            side_name(impl_->config.local_side),
            task.layer,
            task.chunk,
            error.c_str());
    }
    return ok;
}

bool ggml_compute::reduce(
        const key & task,
        const ready_payload &,
        const std::vector<float> &,
        const std::vector<float> & remote_partial,
        std::vector<float> & output) {
    std::string error;
    const bool ok =
        impl_->run_reduce(
            task,
            remote_partial,
            output,
            error);

    if (!ok) {
        std::fprintf(stderr,
            "[LT_GGML_ERROR] side=%s phase=REDUCE "
            "layer=%d chunk=%d reason=%s\n",
            side_name(impl_->config.local_side),
            task.layer,
            task.chunk,
            error.c_str());
    }
    return ok;
}

bool ggml_compute::full_layer(
        const key & task,
        const std::vector<float> & input,
        std::vector<float> & output) {
    std::string error;
    const bool ok =
        impl_->run_full_layer(
            task,
            input,
            output,
            error);

    if (!ok) {
        std::fprintf(stderr,
            "[LT_GGML_ERROR] side=%s phase=FULL_LAYER "
            "layer=%d chunk=%d reason=%s\n",
            side_name(impl_->config.local_side),
            task.layer,
            task.chunk,
            error.c_str());
    }
    return ok;
}

bool read_text_file(
        const std::string & path,
        std::string & text,
        std::string & error) {
    std::ifstream stream(
        path,
        std::ios::binary);
    if (!stream) {
        error =
            "failed to open prompt file: " + path;
        return false;
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    text = buffer.str();

    if (text.empty()) {
        error = "prompt file is empty";
        return false;
    }
    return true;
}

bool tokenize_prompt(
        const std::string & model_path,
        const std::string & prompt,
        std::vector<int32_t> & tokens,
        std::string & error) {
    init_backends_once();

    llama_model_params params =
        llama_model_default_params();
    params.vocab_only = true;
    params.n_gpu_layers = 0;
    params.use_mmap = true;
    params.split_mode = LLAMA_SPLIT_MODE_NONE;

    llama_model * model =
        llama_model_load_from_file(
            model_path.c_str(),
            params);
    if (model == nullptr) {
        error =
            "failed to load model vocabulary";
        return false;
    }

    const llama_vocab * vocab =
        llama_model_get_vocab(model);

    std::vector<llama_token> tmp(
        prompt.size() + 32);

    int32_t count =
        llama_tokenize(
            vocab,
            prompt.data(),
            static_cast<int32_t>(prompt.size()),
            tmp.data(),
            static_cast<int32_t>(tmp.size()),
            true,
            true);

    if (count < 0) {
        const int32_t needed = -count;
        tmp.resize(
            static_cast<size_t>(needed));

        count =
            llama_tokenize(
                vocab,
                prompt.data(),
                static_cast<int32_t>(prompt.size()),
                tmp.data(),
                needed,
                true,
                true);
    }

    if (count <= 0) {
        llama_model_free(model);
        error = "prompt tokenization failed";
        return false;
    }

    tokens.assign(
        tmp.begin(),
        tmp.begin() + count);

    llama_model_free(model);
    return true;
}

std::vector<int32_t> split_prompt_chunks(
        int32_t token_count,
        int32_t chunk_tokens) {
    std::vector<int32_t> result;

    if (token_count <= 0 ||
        chunk_tokens <= 0) {
        return result;
    }

    int32_t remaining = token_count;
    while (remaining > 0) {
        const int32_t take =
            std::min(
                remaining,
                chunk_tokens);
        result.push_back(take);
        remaining -= take;
    }

    return result;
}

} // namespace lt
