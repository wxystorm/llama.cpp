#include "lt-ggml.h"
#include "lt-runtime.h"
#include "lt-tcp.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

namespace {

struct options {
    std::string listen_endpoint =
        "0.0.0.0:50061";
    std::string model_path;
    std::string device = "auto";
    int threads = 8;
    bool trace = false;
};

void usage(const char * argv0) {
    std::fprintf(
        stderr,
        "usage: %s --model MODEL.gguf "
        "[--listen HOST:PORT] [--device NAME] [--trace]\n\n"
        "  --model PATH       Phone-local GGUF path\n"
        "  --listen HOST:PORT default 0.0.0.0:50061\n"
        "  --device NAME      GGML device; default auto selects first GPU/IGPU\n"
        "  --threads N        backend threads when supported (default 8)\n"
        "  --trace            print task transitions\n",
        argv0);
}

bool parse_options(
        int argc,
        char ** argv,
        options & opts) {
    for (int i = 1;
         i < argc;
         ++i) {
        const std::string arg = argv[i];

        if (arg == "--trace") {
            opts.trace = true;
        } else if (arg == "--listen") {
            if (i + 1 >= argc) {
                return false;
            }
            opts.listen_endpoint =
                argv[++i];
        } else if (arg == "--model") {
            if (i + 1 >= argc) {
                return false;
            }
            opts.model_path =
                argv[++i];
        } else if (arg == "--device") {
            if (i + 1 >= argc) {
                return false;
            }
            opts.device =
                argv[++i];
        } else if (arg == "--threads") {
            if (i + 1 >= argc) {
                return false;
            }
            char * end = nullptr;
            const long value =
                std::strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' ||
                value <= 0 || value > 1024) {
                return false;
            }
            opts.threads =
                static_cast<int>(value);
        } else if (arg == "--help" ||
                   arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            return false;
        }
    }

    return !opts.model_path.empty();
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

    auto channel =
        std::make_shared<lt::tcp_channel>();

    std::string error;

    std::fprintf(
        stderr,
        "[LT_WORKER] event=LISTEN endpoint=%s\n",
        opts.listen_endpoint.c_str());

    if (!channel->listen_and_accept(
            opts.listen_endpoint,
            error)) {
        std::fprintf(
            stderr,
            "[LT_WORKER] event=LISTEN status=FAIL reason=%s\n",
            error.c_str());
        return 3;
    }

    lt::plan runtime_plan;

    if (!channel->recv_plan(
            runtime_plan)) {
        std::fprintf(
            stderr,
            "[LT_WORKER] event=PLAN status=FAIL\n");
        return 4;
    }

    std::fprintf(
        stderr,
        "[LT_WORKER] event=PLAN status=READY "
        "mode=%s layers=%d chunks=%zu first_owner=%s "
        "c2p_pc_ratio=%.6f p2c_pc_ratio=%.6f\n",
        lt::mode_name(
            runtime_plan.execution_mode),
        runtime_plan.layers,
        runtime_plan.chunk_tokens.size(),
        lt::side_name(
            runtime_plan.owner_for_layer(0)),
        runtime_plan.c2p_pc_ratio,
        runtime_plan.p2c_pc_ratio);

    lt::ggml_compute_config compute_config;
    compute_config.local_side =
        lt::side::phone;
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
            "[LT_WORKER] event=MODEL status=FAIL reason=%s\n",
            error.c_str());
        return 5;
    }

    auto transport =
        std::make_shared<lt::socket_transport>(
            channel);

    lt::runtime phone(
        lt::side::phone,
        runtime_plan,
        compute,
        transport,
        opts.trace);

    if (!phone.start()) {
        std::fprintf(
            stderr,
            "[LT_WORKER] event=RUNTIME status=FAIL reason=%s\n",
            phone.failure_reason().c_str());
        return 6;
    }

    lt::message model_ready;
    model_ready.kind =
        lt::message_kind::model_ready;

    if (!channel->send_message(
            model_ready)) {
        std::fprintf(
            stderr,
            "[LT_WORKER] event=MODEL_READY status=FAIL\n");
        phone.stop();
        return 6;
    }

    std::fprintf(
        stderr,
        "[LT_WORKER] event=MODEL_READY "
        "device=%s n_embd=%d layers=%d status=READY\n",
        compute->device_name(),
        compute->n_embd(),
        compute->n_layers());

    bool received_stop = false;
    lt::message incoming;

    while (channel->recv_message(
            incoming)) {
        if (incoming.kind ==
            lt::message_kind::stop) {
            received_stop = true;
            (void) phone.on_message(
                std::move(incoming));
            break;
        }

        if (incoming.kind ==
            lt::message_kind::model_ready) {
            std::fprintf(
                stderr,
                "[LT_WORKER] status=FAIL "
                "reason=UNEXPECTED_MODEL_READY\n");
            break;
        }

        if (!phone.on_message(
                std::move(incoming))) {
            break;
        }

        incoming = {};
    }

    phone.stop();

    const lt::stats stats =
        phone.snapshot_stats();

    if (phone.failed()) {
        std::fprintf(
            stderr,
            "[LT_WORKER] status=FAIL reason=%s\n",
            phone.failure_reason().c_str());
        return 7;
    }

    std::fprintf(
        stderr,
        "[LT_WORKER] status=%s "
        "owner=%llu remote=%llu reduce=%llu full=%llu "
        "engine=GGML_REAL\n",
        received_stop ?
            "PASS" :
            "DISCONNECTED",
        static_cast<unsigned long long>(
            stats.owner_tasks),
        static_cast<unsigned long long>(
            stats.remote_ffn_tasks),
        static_cast<unsigned long long>(
            stats.reduce_tasks),
        static_cast<unsigned long long>(
            stats.full_layer_tasks));

    return received_stop ? 0 : 8;
}
