// Scoped persistent CPU workers for model-load Hybrid/Region Ring profiling.
// The actual prefill threadpool is not available until llama_context exists,
// so the profile owns an equivalent pool for its entire measurement lifetime.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

class llama_hybrid_profile_cpu_threadpool {
public:
    llama_hybrid_profile_cpu_threadpool() = default;
    llama_hybrid_profile_cpu_threadpool(const llama_hybrid_profile_cpu_threadpool &) = delete;
    llama_hybrid_profile_cpu_threadpool & operator=(const llama_hybrid_profile_cpu_threadpool &) = delete;

    ~llama_hybrid_profile_cpu_threadpool() {
        reset();
    }

    bool attach(ggml_backend_t cpu_backend, int n_threads) {
        if (pool_ != nullptr || cpu_backend == nullptr ||
                n_threads <= 0 || n_threads > GGML_MAX_N_THREADS) {
            return false;
        }

        ggml_backend_dev_t device = ggml_backend_get_device(cpu_backend);
        ggml_backend_reg_t reg =
            device != nullptr ? ggml_backend_dev_backend_reg(device) : nullptr;
        if (reg == nullptr) {
            return false;
        }

        using new_fn = ggml_threadpool_t (*)(ggml_threadpool_params *);
        using free_fn = void (*)(ggml_threadpool_t);
        using bind_fn = void (*)(ggml_backend_t, ggml_threadpool_t);

        const auto create = reinterpret_cast<new_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
        const auto destroy = reinterpret_cast<free_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free"));
        const auto bind = reinterpret_cast<bind_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
        if (create == nullptr || destroy == nullptr || bind == nullptr) {
            return false;
        }

        auto threadpool_params = ggml_threadpool_params_default(n_threads);
        ggml_threadpool_t pool = create(&threadpool_params);
        if (pool == nullptr) {
            return false;
        }

        backend_ = cpu_backend;
        pool_ = pool;
        destroy_ = destroy;
        bind_ = bind;
        bind_(backend_, pool_);
        return true;
    }

    // The owner must outlive this guard. Unbind first (which pauses workers),
    // then free the pool; every early return from autoplan follows this path.
    void reset() {
        if (pool_ != nullptr) {
            bind_(backend_, nullptr);
            destroy_(pool_);
            pool_ = nullptr;
            backend_ = nullptr;
            destroy_ = nullptr;
            bind_ = nullptr;
        }
    }

    bool attached() const {
        return pool_ != nullptr;
    }

    ggml_threadpool_t get() const {
        return pool_;
    }

private:
    ggml_backend_t backend_ = nullptr;
    ggml_threadpool_t pool_ = nullptr;
    void (*destroy_)(ggml_threadpool_t) = nullptr;
    void (*bind_)(ggml_backend_t, ggml_threadpool_t) = nullptr;
};
