#pragma once

#include "lt-runtime.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lt {

struct ggml_compute_config {
    side local_side = side::pc;
    plan runtime_plan;

    std::string model_path;

    // "cpu" selects the CPU backend. "auto" selects the first GPU/IGPU and
    // falls back to CPU. Any other value is interpreted as an exact GGML
    // device name.
    std::string device = "cpu";
    int32_t threads = 8;

    bool log_events = false;
};

class ggml_compute final : public compute {
public:
    static std::shared_ptr<ggml_compute> create(
        ggml_compute_config config,
        std::string & error);

    ~ggml_compute() override;

    ggml_compute(const ggml_compute &) = delete;
    ggml_compute & operator=(const ggml_compute &) = delete;

    int32_t n_embd() const;
    int32_t n_layers() const;
    const char * device_name() const;

    // PC-only prelude used before the task runtime starts. Token embeddings are
    // real model embeddings; the result is F32 [n_embd, n_tokens].
    bool embed_tokens(
        const std::vector<int32_t> & tokens,
        std::vector<float> & output,
        std::string & error);

    bool owner_prefix(
        const key & task,
        const std::vector<float> & input,
        ready_payload & ready) override;

    bool local_ffn(
        const key & task,
        const ready_payload & ready,
        std::vector<float> & partial) override;

    bool remote_ffn(
        const key & task,
        const ready_payload & ready,
        std::vector<float> & partial) override;

    bool reduce(
        const key & task,
        const ready_payload & ready,
        const std::vector<float> & local_partial,
        const std::vector<float> & remote_partial,
        std::vector<float> & output) override;

    bool full_layer(
        const key & task,
        const std::vector<float> & input,
        std::vector<float> & output) override;

private:
    struct impl;

    explicit ggml_compute(std::unique_ptr<impl> impl_value);

    std::unique_ptr<impl> impl_;
};

// Tokenize on PC using the model vocabulary only. This happens before the
// truncated compute model is loaded and therefore does not materialize model
// weights.
bool tokenize_prompt(
    const std::string & model_path,
    const std::string & prompt,
    std::vector<int32_t> & tokens,
    std::string & error);

bool read_text_file(
    const std::string & path,
    std::string & text,
    std::string & error);

std::vector<int32_t> split_prompt_chunks(
    int32_t token_count,
    int32_t chunk_tokens);

} // namespace lt
