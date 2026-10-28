#include "lt-ggml.h"
#include "lt-runtime.h"
#include "lt-tcp.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct options {
    bool self_test = false;
    bool ffn_split_selfcheck = false;
    bool trace = false;

    std::string connect_endpoint;
    std::string model_path;
    std::string prompt_file;
    std::string device = "cpu";

    lt::mode execution_mode = lt::mode::tensor;

    int layers = 4;

    // Reference/self-test only. Real runs derive the number of chunks from
    // the tokenized prompt and --chunk-tokens.
    int chunks = 3;
    int chunk_tokens = 64;
    int hidden = 256;

    int owner_burst = 1;
    int max_owner_ahead = 1;
    int threads = 8;
    lt::side first_owner = lt::side::pc;

    float c2p_pc_ratio = 0.5f;
    float p2c_pc_ratio = 0.5f;

    std::vector<lt::side> layer_owners;
};

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s --self-test [--trace]\n"
        "  %s --ffn-split-selfcheck --model MODEL.gguf --prompt-file prompt.txt "
        "[--chunk-tokens N] [--pc-ratio R]\n"
        "  %s --connect HOST:PORT --model MODEL.gguf --prompt-file prompt.txt "
        "--mode tensor|layer --layers N --chunk-tokens N [options]\n\n"
        "real-model options:\n"
        "  --model PATH             local PC GGUF path\n"
        "  --prompt-file PATH       prompt text to tokenize on PC\n"
        "  --device NAME            PC GGML device; v1 requires cpu\n"
        "  --pc-ratio R             set both C2P and P2C physical PC FFN share\n"
        "  --c2p-ratio R            C2P physical PC FFN share\n"
        "  --p2c-ratio R            P2C physical PC FFN share\n"
        "  --chunk-tokens N         Tensor/Layer chunk size (tail is automatic)\n"
        "  --layer-owners CSV       Layer mode owners, e.g. pc,pc,phone,phone,pc,pc\n"
        "  --first-owner pc|phone   Tensor layer-0 Attention owner (default pc)\n"
        "  --owner-burst N          OWNER burst before waiting REMOTE_FFN (default 1)\n"
        "  --max-owner-ahead N      Tensor frontier lead over downstream READY (default 1)\n"
        "  --threads N              PC CPU GGML threads (default 8)\n"
        "  --trace                  print task transitions\n\n"
        "reference self-test options:\n"
        "  --chunks N               synthetic chunk count\n"
        "  --hidden N               synthetic hidden width\n",
        argv0, argv0, argv0);
}

bool parse_side(
        const std::string & text,
        lt::side & value) {
    if (text == "pc" || text == "PC") {
        value = lt::side::pc;
        return true;
    }
    if (text == "phone" || text == "PHONE") {
        value = lt::side::phone;
        return true;
    }
    return false;
}

bool parse_layer_owners(
        const std::string & text,
        std::vector<lt::side> & values) {
    values.clear();

    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t comma =
            text.find(',', begin);

        const std::string item =
            text.substr(
                begin,
                comma == std::string::npos ?
                    std::string::npos :
                    comma - begin);

        lt::side owner;
        if (!parse_side(item, owner)) {
            return false;
        }

        values.push_back(owner);

        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }

    return !values.empty();
}

bool parse_int(
        const char * text,
        int & value) {
    char * end = nullptr;
    const long parsed =
        std::strtol(text, &end, 10);

    if (end == text ||
        *end != '\0' ||
        parsed <= 0 ||
        parsed > 1000000) {
        return false;
    }

    value = static_cast<int>(parsed);
    return true;
}

bool parse_ratio(
        const char * text,
        float & value) {
    char * end = nullptr;
    const float parsed =
        std::strtof(text, &end);

    if (end == text ||
        *end != '\0' ||
        !(parsed > 0.0f && parsed < 1.0f)) {
        return false;
    }

    value = parsed;
    return true;
}

bool parse_options(
        int argc,
        char ** argv,
        options & opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto need_value =
            [&](const char * name) -> const char * {
                if (i + 1 >= argc) {
                    std::fprintf(
                        stderr,
                        "%s requires a value\n",
                        name);
                    return nullptr;
                }
                return argv[++i];
            };

        if (arg == "--self-test") {
            opts.self_test = true;
        } else if (arg == "--ffn-split-selfcheck") {
            opts.ffn_split_selfcheck = true;
        } else if (arg == "--trace") {
            opts.trace = true;
        } else if (arg == "--connect") {
            const char * value =
                need_value("--connect");
            if (value == nullptr) return false;
            opts.connect_endpoint = value;
        } else if (arg == "--model") {
            const char * value =
                need_value("--model");
            if (value == nullptr) return false;
            opts.model_path = value;
        } else if (arg == "--prompt-file") {
            const char * value =
                need_value("--prompt-file");
            if (value == nullptr) return false;
            opts.prompt_file = value;
        } else if (arg == "--device") {
            const char * value =
                need_value("--device");
            if (value == nullptr) return false;
            opts.device = value;
        } else if (arg == "--mode") {
            const char * value =
                need_value("--mode");
            if (value == nullptr) return false;

            const std::string text = value;
            if (text == "tensor") {
                opts.execution_mode =
                    lt::mode::tensor;
            } else if (text == "layer") {
                opts.execution_mode =
                    lt::mode::layer;
            } else {
                return false;
            }
        } else if (arg == "--layers") {
            const char * value =
                need_value("--layers");
            if (value == nullptr ||
                !parse_int(
                    value,
                    opts.layers)) {
                return false;
            }
        } else if (arg == "--chunks") {
            const char * value =
                need_value("--chunks");
            if (value == nullptr ||
                !parse_int(
                    value,
                    opts.chunks)) {
                return false;
            }
        } else if (arg == "--chunk-tokens") {
            const char * value =
                need_value("--chunk-tokens");
            if (value == nullptr ||
                !parse_int(
                    value,
                    opts.chunk_tokens)) {
                return false;
            }
        } else if (arg == "--hidden") {
            const char * value =
                need_value("--hidden");
            if (value == nullptr ||
                !parse_int(
                    value,
                    opts.hidden)) {
                return false;
            }
        } else if (arg == "--max-owner-ahead") {
            const char * value =
                need_value("--max-owner-ahead");
            if (value == nullptr) {
                return false;
            }
            char * end = nullptr;
            const long parsed =
                std::strtol(value, &end, 10);
            if (end == value ||
                *end != '\0' ||
                parsed < 0 ||
                parsed > 1000000) {
                return false;
            }
            opts.max_owner_ahead =
                static_cast<int>(parsed);
        } else if (arg == "--threads") {
            const char * value =
                need_value("--threads");
            if (value == nullptr ||
                !parse_int(
                    value,
                    opts.threads)) {
                return false;
            }
        } else if (arg == "--owner-burst") {
            const char * value =
                need_value("--owner-burst");
            if (value == nullptr ||
                !parse_int(
                    value,
                    opts.owner_burst)) {
                return false;
            }
        } else if (arg == "--first-owner") {
            const char * value =
                need_value("--first-owner");
            if (value == nullptr ||
                !parse_side(
                    value,
                    opts.first_owner)) {
                return false;
            }
        } else if (arg == "--pc-ratio") {
            const char * value =
                need_value("--pc-ratio");
            float ratio = 0.0f;
            if (value == nullptr ||
                !parse_ratio(
                    value,
                    ratio)) {
                return false;
            }
            opts.c2p_pc_ratio = ratio;
            opts.p2c_pc_ratio = ratio;
        } else if (arg == "--c2p-ratio") {
            const char * value =
                need_value("--c2p-ratio");
            if (value == nullptr ||
                !parse_ratio(
                    value,
                    opts.c2p_pc_ratio)) {
                return false;
            }
        } else if (arg == "--p2c-ratio") {
            const char * value =
                need_value("--p2c-ratio");
            if (value == nullptr ||
                !parse_ratio(
                    value,
                    opts.p2c_pc_ratio)) {
                return false;
            }
        } else if (arg == "--layer-owners") {
            const char * value =
                need_value("--layer-owners");
            if (value == nullptr ||
                !parse_layer_owners(
                    value,
                    opts.layer_owners)) {
                return false;
            }
        } else if (arg == "--help" ||
                   arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(
                stderr,
                "unknown option: %s\n",
                arg.c_str());
            return false;
        }
    }

    return true;
}

lt::plan make_plan(
        const options & opts,
        const std::vector<int32_t> * real_chunks = nullptr) {
    lt::plan runtime_plan;

    runtime_plan.execution_mode =
        opts.execution_mode;
    runtime_plan.layers =
        opts.layers;

    if (real_chunks != nullptr) {
        runtime_plan.chunk_tokens =
            *real_chunks;
    } else {
        runtime_plan.chunk_tokens.assign(
            static_cast<size_t>(opts.chunks),
            opts.chunk_tokens);
    }

    runtime_plan.first_tensor_owner =
        opts.first_owner;
    runtime_plan.owner_burst_limit =
        opts.owner_burst;
    runtime_plan.max_owner_ahead =
        opts.max_owner_ahead;
    runtime_plan.c2p_pc_ratio =
        opts.c2p_pc_ratio;
    runtime_plan.p2c_pc_ratio =
        opts.p2c_pc_ratio;

    if (opts.execution_mode == lt::mode::layer) {
        if (!opts.layer_owners.empty()) {
            runtime_plan.layer_owners =
                opts.layer_owners;
        } else {
            // Neutral default for ad-hoc runs. The future grid runner will
            // enumerate the Layer candidates explicitly.
            runtime_plan.layer_owners.assign(
                static_cast<size_t>(opts.layers),
                lt::side::pc);

            const int first_phone =
                opts.layers / 3;
            const int last_phone =
                opts.layers -
                opts.layers / 3;

            for (int layer = first_phone;
                 layer < last_phone;
                 ++layer) {
                runtime_plan.layer_owners[
                    static_cast<size_t>(layer)] =
                    lt::side::phone;
            }
        }
    }

    return runtime_plan;
}

std::vector<float> make_input(
        int hidden,
        int chunk,
        int tokens) {
    const size_t count =
        static_cast<size_t>(hidden) *
        static_cast<size_t>(tokens);

    std::vector<float> values(count);

    for (size_t i = 0;
         i < count;
         ++i) {
        values[i] =
            0.001f *
            static_cast<float>(
                1 +
                chunk * 13 +
                static_cast<int>(i % 97));
    }

    return values;
}

class loopback_transport final : public lt::transport {
public:
    void bind(lt::runtime * peer) {
        peer_ = peer;
    }

    bool send(
            const lt::message & value) override {
        return peer_ != nullptr &&
            peer_->on_message(value);
    }

private:
    lt::runtime * peer_ = nullptr;
};

bool run_self_test_case(
        lt::mode execution_mode,
        bool trace) {
    options opts;
    opts.execution_mode =
        execution_mode;
    opts.layers = 6;
    opts.chunks = 4;
    opts.chunk_tokens = 8;
    opts.hidden = 16;
    opts.owner_burst = 1;
    opts.first_owner =
        lt::side::pc;

    if (execution_mode == lt::mode::layer) {
        opts.layer_owners = {
            lt::side::pc,
            lt::side::pc,
            lt::side::phone,
            lt::side::phone,
            lt::side::pc,
            lt::side::pc,
        };
    }

    const lt::plan runtime_plan =
        make_plan(opts);

    auto pc_transport =
        std::make_shared<loopback_transport>();
    auto phone_transport =
        std::make_shared<loopback_transport>();

    auto pc_compute =
        std::make_shared<lt::reference_compute>(
            0.0f);
    auto phone_compute =
        std::make_shared<lt::reference_compute>(
            0.02f);

    lt::runtime pc(
        lt::side::pc,
        runtime_plan,
        pc_compute,
        pc_transport,
        trace);

    lt::runtime phone(
        lt::side::phone,
        runtime_plan,
        phone_compute,
        phone_transport,
        trace);

    pc_transport->bind(&phone);
    phone_transport->bind(&pc);

    if (!pc.start() ||
        !phone.start()) {
        return false;
    }

    for (int chunk = 0;
         chunk < opts.chunks;
         ++chunk) {
        std::vector<float> input =
            make_input(
                opts.hidden,
                chunk,
                opts.chunk_tokens);

        if (runtime_plan.owner_for_layer(0) ==
            lt::side::pc) {
            if (!pc.submit_initial(
                    chunk,
                    std::move(input))) {
                return false;
            }
        } else {
            if (!phone.submit_initial(
                    chunk,
                    std::move(input))) {
                return false;
            }
        }
    }

    const bool ok =
        pc.wait_until_complete(
            static_cast<uint64_t>(
                opts.chunks));

    const lt::stats pc_stats =
        pc.snapshot_stats();
    const lt::stats phone_stats =
        phone.snapshot_stats();

    pc.stop();
    phone.stop();

    if (!ok ||
        pc.failed() ||
        phone.failed()) {
        std::fprintf(
            stderr,
            "[LT_SELF_TEST] mode=%s status=FAIL "
            "pc_error=%s phone_error=%s\n",
            lt::mode_name(execution_mode),
            pc.failure_reason().c_str(),
            phone.failure_reason().c_str());
        return false;
    }

    std::fprintf(
        stderr,
        "[LT_SELF_TEST] mode=%s status=PASS "
        "pc_owner=%llu pc_remote=%llu pc_reduce=%llu "
        "phone_owner=%llu phone_remote=%llu phone_reduce=%llu\n",
        lt::mode_name(execution_mode),
        static_cast<unsigned long long>(
            pc_stats.owner_tasks),
        static_cast<unsigned long long>(
            pc_stats.remote_ffn_tasks),
        static_cast<unsigned long long>(
            pc_stats.reduce_tasks),
        static_cast<unsigned long long>(
            phone_stats.owner_tasks),
        static_cast<unsigned long long>(
            phone_stats.remote_ffn_tasks),
        static_cast<unsigned long long>(
            phone_stats.reduce_tasks));

    return true;
}

int run_ffn_split_selfcheck(
        const options & opts) {
    if (opts.model_path.empty() ||
        opts.prompt_file.empty()) {
        std::fprintf(
            stderr,
            "--model and --prompt-file are required "
            "for --ffn-split-selfcheck\n");
        return 2;
    }

    options diag = opts;
    diag.execution_mode = lt::mode::tensor;
    diag.layers = 1;
    diag.first_owner = lt::side::pc;

    std::string error;
    std::string prompt;
    if (!lt::read_text_file(
            diag.prompt_file,
            prompt,
            error)) {
        std::fprintf(
            stderr,
            "[LT_FFN_SPLIT_SELF] phase=PROMPT "
            "status=FAIL reason=%s\n",
            error.c_str());
        return 2;
    }

    std::vector<int32_t> tokens;
    if (!lt::tokenize_prompt(
            diag.model_path,
            prompt,
            tokens,
            error)) {
        std::fprintf(
            stderr,
            "[LT_FFN_SPLIT_SELF] phase=TOKENIZE "
            "status=FAIL reason=%s\n",
            error.c_str());
        return 2;
    }

    const std::vector<int32_t> chunk_sizes =
        lt::split_prompt_chunks(
            static_cast<int32_t>(
                tokens.size()),
            diag.chunk_tokens);
    if (chunk_sizes.empty()) {
        std::fprintf(
            stderr,
            "[LT_FFN_SPLIT_SELF] phase=CHUNK status=FAIL\n");
        return 2;
    }

    const lt::plan runtime_plan =
        make_plan(
            diag,
            &chunk_sizes);

    std::string reason;
    if (!runtime_plan.valid(&reason)) {
        std::fprintf(
            stderr,
            "[LT_FFN_SPLIT_SELF] phase=PLAN "
            "status=FAIL reason=%s\n",
            reason.c_str());
        return 2;
    }

    lt::ggml_compute_config compute_config;
    compute_config.local_side =
        lt::side::pc;
    compute_config.runtime_plan =
        runtime_plan;
    compute_config.model_path =
        diag.model_path;
    compute_config.device = "cpu";
    compute_config.threads =
        diag.threads;
    compute_config.log_events =
        diag.trace;

    auto compute =
        lt::ggml_compute::create(
            std::move(compute_config),
            error);
    if (compute == nullptr) {
        std::fprintf(
            stderr,
            "[LT_FFN_SPLIT_SELF] phase=MODEL "
            "status=FAIL reason=%s\n",
            error.c_str());
        return 4;
    }

    size_t token_offset = 0;
    for (size_t chunk = 0;
         chunk < chunk_sizes.size();
         ++chunk) {
        const size_t count =
            static_cast<size_t>(
                chunk_sizes[chunk]);

        std::vector<int32_t> chunk_token_ids(
            tokens.begin() +
                static_cast<std::ptrdiff_t>(
                    token_offset),
            tokens.begin() +
                static_cast<std::ptrdiff_t>(
                    token_offset + count));

        std::vector<float> embedded;
        if (!compute->embed_tokens(
                chunk_token_ids,
                embedded,
                error)) {
            std::fprintf(
                stderr,
                "[LT_FFN_SPLIT_SELF] phase=EMBED chunk=%zu "
                "status=FAIL reason=%s\n",
                chunk,
                error.c_str());
            return 4;
        }

        std::string report;
        const lt::key task {
            0,
            static_cast<int32_t>(chunk)
        };

        if (!compute->ffn_split_selfcheck(
                task,
                embedded,
                diag.c2p_pc_ratio,
                report,
                error)) {
            std::fprintf(
                stderr,
                "[LT_FFN_SPLIT_SELF] phase=CHECK chunk=%zu "
                "status=FAIL reason=%s\n",
                chunk,
                error.c_str());
            return 4;
        }

        std::printf(
            "[LT_FFN_SPLIT_SELF] %s status=OBSERVED\n",
            report.c_str());

        token_offset += count;
    }

    std::printf(
        "[LT_FFN_SPLIT_SELF] tokens=%zu chunks=%zu "
        "XT=%d ratio=%.6f status=PASS\n",
        tokens.size(),
        chunk_sizes.size(),
        diag.chunk_tokens,
        diag.c2p_pc_ratio);

    return 0;
}

int run_real(
        const options & opts) {
    if (opts.connect_endpoint.empty() ||
        opts.model_path.empty() ||
        opts.prompt_file.empty()) {
        std::fprintf(
            stderr,
            "--connect, --model and --prompt-file are required "
            "for real-model runs\n");
        return 2;
    }

    std::string error;
    std::string prompt;

    if (!lt::read_text_file(
            opts.prompt_file,
            prompt,
            error)) {
        std::fprintf(
            stderr,
            "[LT_PC] phase=PROMPT status=FAIL reason=%s\n",
            error.c_str());
        return 2;
    }

    std::vector<int32_t> tokens;
    if (!lt::tokenize_prompt(
            opts.model_path,
            prompt,
            tokens,
            error)) {
        std::fprintf(
            stderr,
            "[LT_PC] phase=TOKENIZE status=FAIL reason=%s\n",
            error.c_str());
        return 2;
    }

    const std::vector<int32_t> chunk_sizes =
        lt::split_prompt_chunks(
            static_cast<int32_t>(
                tokens.size()),
            opts.chunk_tokens);

    if (chunk_sizes.empty()) {
        std::fprintf(
            stderr,
            "[LT_PC] phase=CHUNK status=FAIL\n");
        return 2;
    }

    const lt::plan runtime_plan =
        make_plan(
            opts,
            &chunk_sizes);

    std::string reason;
    if (!runtime_plan.valid(&reason)) {
        std::fprintf(
            stderr,
            "invalid plan: %s\n",
            reason.c_str());
        return 2;
    }

    auto channel =
        std::make_shared<lt::tcp_channel>();

    if (!channel->connect_to(
            opts.connect_endpoint,
            error)) {
        std::fprintf(
            stderr,
            "connect failed: %s\n",
            error.c_str());
        return 3;
    }

    if (!channel->send_plan(
            runtime_plan)) {
        std::fprintf(
            stderr,
            "failed to send plan\n");
        return 3;
    }

    const auto prelude_begin =
        std::chrono::steady_clock::now();

    lt::ggml_compute_config compute_config;
    compute_config.local_side =
        lt::side::pc;
    compute_config.runtime_plan =
        runtime_plan;
    compute_config.model_path =
        opts.model_path;
    compute_config.device =
        opts.device;
    compute_config.threads =
        opts.threads;
    compute_config.log_events =
        opts.trace;

    auto compute =
        lt::ggml_compute::create(
            std::move(compute_config),
            error);

    if (compute == nullptr) {
        std::fprintf(
            stderr,
            "[LT_PC] phase=MODEL status=FAIL reason=%s\n",
            error.c_str());
        return 4;
    }

    // Real token embeddings are common work for every Layer/Tensor candidate.
    // Materialize them before the measured layer interval.
    std::vector<std::vector<float>> initial_chunks;
    initial_chunks.reserve(
        chunk_sizes.size());

    size_t token_offset = 0;
    for (size_t chunk = 0;
         chunk < chunk_sizes.size();
         ++chunk) {
        const size_t count =
            static_cast<size_t>(
                chunk_sizes[chunk]);

        std::vector<int32_t> chunk_token_ids(
            tokens.begin() +
                static_cast<std::ptrdiff_t>(
                    token_offset),
            tokens.begin() +
                static_cast<std::ptrdiff_t>(
                    token_offset + count));

        std::vector<float> embedded;
        if (!compute->embed_tokens(
                chunk_token_ids,
                embedded,
                error)) {
            std::fprintf(
                stderr,
                "[LT_PC] phase=EMBED chunk=%zu "
                "status=FAIL reason=%s\n",
                chunk,
                error.c_str());
            return 4;
        }

        initial_chunks.push_back(
            std::move(embedded));
        token_offset += count;
    }

    if (token_offset != tokens.size()) {
        std::fprintf(
            stderr,
            "[LT_PC] phase=EMBED status=FAIL "
            "reason=TOKEN_PARTITION_MISMATCH\n");
        return 4;
    }

    // Worker sends this only after its own local GGUF has loaded and the
    // OpenCL/other backend is ready. This makes model load symmetric and
    // excludes it from wall_ms.
    lt::message worker_ready;
    if (!channel->recv_message(
            worker_ready) ||
        worker_ready.kind !=
            lt::message_kind::model_ready) {
        std::fprintf(
            stderr,
            "[LT_PC] phase=PHONE_MODEL_READY status=FAIL\n");
        return 4;
    }

    const auto prelude_end =
        std::chrono::steady_clock::now();

    auto transport =
        std::make_shared<lt::socket_transport>(
            channel);

    lt::runtime pc(
        lt::side::pc,
        runtime_plan,
        compute,
        transport,
        opts.trace);

    if (!pc.start()) {
        return 5;
    }

    std::atomic<bool> rx_ok { true };
    std::atomic<bool> shutting_down { false };

    std::thread rx([&]() {
        lt::message incoming;

        while (channel->recv_message(
                incoming)) {
            if (incoming.kind ==
                lt::message_kind::model_ready) {
                std::fprintf(
                    stderr,
                    "[LT_PC] phase=RX status=FAIL "
                    "reason=DUPLICATE_MODEL_READY\n");
                rx_ok.store(false);
                return;
            }

            if (!pc.on_message(
                    std::move(incoming))) {
                rx_ok.store(false);
                return;
            }

            incoming = {};
        }

        if (!shutting_down.load() &&
            !pc.failed()) {
            rx_ok.store(false);

            // Wake wait_until_complete() on an unexpected peer disconnect.
            lt::message stop_event;
            stop_event.kind =
                lt::message_kind::stop;
            (void) pc.on_message(
                std::move(stop_event));
        }
    });

    std::fprintf(
        stderr,
        "[LT_PC] phase=RUN mode=%s layers=%d "
        "tokens=%zu chunks=%zu XT=%d "
        "c2p_pc_ratio=%.6f p2c_pc_ratio=%.6f "
        "max_owner_ahead=%d pc_device=%s status=BEGIN\n",
        lt::mode_name(
            runtime_plan.execution_mode),
        runtime_plan.layers,
        tokens.size(),
        chunk_sizes.size(),
        opts.chunk_tokens,
        runtime_plan.c2p_pc_ratio,
        runtime_plan.p2c_pc_ratio,
        runtime_plan.max_owner_ahead,
        compute->device_name());

    const auto begin =
        std::chrono::steady_clock::now();

    for (size_t chunk = 0;
         chunk < initial_chunks.size();
         ++chunk) {
        if (runtime_plan.owner_for_layer(0) ==
            lt::side::pc) {
            if (!pc.submit_initial(
                    static_cast<int32_t>(chunk),
                    std::move(
                        initial_chunks[chunk]))) {
                rx_ok.store(false);
                break;
            }
        } else {
            lt::message initial;
            initial.kind =
                lt::message_kind::initial;
            initial.task = {
                0,
                static_cast<int32_t>(chunk)
            };
            initial.hidden =
                std::move(
                    initial_chunks[chunk]);

            if (!channel->send_message(
                    initial)) {
                rx_ok.store(false);
                break;
            }
        }
    }

    bool completed = false;
    if (rx_ok.load()) {
        completed =
            pc.wait_until_complete(
                static_cast<uint64_t>(
                    chunk_sizes.size()));
    }

    const auto end =
        std::chrono::steady_clock::now();

    shutting_down.store(true);

    lt::message stop;
    stop.kind =
        lt::message_kind::stop;
    (void) channel->send_message(stop);

    channel->close();

    if (rx.joinable()) {
        rx.join();
    }

    pc.stop();

    if (!completed ||
        pc.failed() ||
        !rx_ok.load()) {
        std::fprintf(
            stderr,
            "[LT_RUN] mode=%s status=FAIL reason=%s\n",
            lt::mode_name(
                runtime_plan.execution_mode),
            pc.failure_reason().empty() ?
                "RX_OR_PROTOCOL_FAILURE" :
                pc.failure_reason().c_str());
        return 6;
    }

    const double wall_ms =
        std::chrono::duration<double, std::milli>(
            end - begin).count();

    const double prelude_ms =
        std::chrono::duration<double, std::milli>(
            prelude_end -
            prelude_begin).count();

    const lt::stats stats =
        pc.snapshot_stats();

    std::printf(
        "[LT_RUN] mode=%s layers=%d tokens=%zu chunks=%zu "
        "XT=%d c2p_pc_ratio=%.6f p2c_pc_ratio=%.6f "
        "prelude_ms=%.3f wall_ms=%.3f tok_per_s=%.3f "
        "owner=%llu remote=%llu reduce=%llu full=%llu "
        "final_chunks=%llu checksum=%.9f "
        "status=PASS engine=GGML_REAL\n",
        lt::mode_name(
            runtime_plan.execution_mode),
        runtime_plan.layers,
        tokens.size(),
        chunk_sizes.size(),
        opts.chunk_tokens,
        runtime_plan.c2p_pc_ratio,
        runtime_plan.p2c_pc_ratio,
        prelude_ms,
        wall_ms,
        wall_ms > 0.0 ?
            1000.0 *
                static_cast<double>(
                    tokens.size()) /
                wall_ms :
            0.0,
        static_cast<unsigned long long>(
            stats.owner_tasks),
        static_cast<unsigned long long>(
            stats.remote_ffn_tasks),
        static_cast<unsigned long long>(
            stats.reduce_tasks),
        static_cast<unsigned long long>(
            stats.full_layer_tasks),
        static_cast<unsigned long long>(
            stats.final_chunks),
        stats.final_checksum_sum);

    return 0;
}

} // namespace

int main(
        int argc,
        char ** argv) {
    options opts;

    if (!parse_options(
            argc,
            argv,
            opts)) {
        usage(argv[0]);
        return 2;
    }

    if (opts.ffn_split_selfcheck) {
        return run_ffn_split_selfcheck(opts);
    }

    if (opts.self_test) {
        const bool tensor_ok =
            run_self_test_case(
                lt::mode::tensor,
                opts.trace);

        const bool layer_ok =
            run_self_test_case(
                lt::mode::layer,
                opts.trace);

        return tensor_ok &&
            layer_ok ?
            0 : 1;
    }

    return run_real(opts);
}
