#include "arg.h"
#include "common.h"
#include "log.h"

#include "cli-context.h"
#include "ggml-backend.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <signal.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void signal_handler(int) {
    if (cli_context::interrupted().load()) {
        // second Ctrl+C - exit immediately
        // make sure to clear colors before exiting (not using LOG or console.cpp here to avoid deadlock)
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    cli_context::interrupted().store(true);
}
#endif

// Opt-in CPU/RPC checkpoint trace (slower: the scheduler synchronizes nodes).
static bool cli_rpc_trace_tensor(ggml_tensor * t, bool ask, void *) {
    if (!t) return false;
    const char * name = ggml_get_name(t);
    if (!name || !*name) return false;

    const bool output = strcmp(name, "result_norm") == 0 || strcmp(name, "result_output") == 0;
    const char * dash = strrchr(name, '-');
    const int layer = dash && dash[1] >= '0' && dash[1] <= '9' ? atoi(dash + 1) : -1;
    const bool chosen_layer = layer == 0 || (layer >= 0 && layer % 4 == 3);
    const bool chosen_name =
        strncmp(name, "attn_norm-", 10) == 0 ||
        strncmp(name, "Qcur-", 5) == 0 ||
        strncmp(name, "Kcur-", 5) == 0 ||
        strncmp(name, "attn_out-", 9) == 0 ||
        strncmp(name, "ffn_inp-", 8) == 0 ||
        strncmp(name, "ffn_norm-", 9) == 0 ||
        strncmp(name, "ffn_up-", 7) == 0 ||
        strncmp(name, "ffn_out-", 8) == 0 ||
        strncmp(name, "l_out-", 6) == 0;

    const bool selected = (output || (chosen_layer && chosen_name)) &&
        t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
        t->ne[0] > 0 && t->ne[1] > 0 && t->ne[2] == 1 && t->ne[3] == 1 &&
        ggml_nbytes(t) <= 8*1024*1024;
    if (ask) return selected;
    if (!selected) return true;
    if (!t->buffer && !(t->view_src && t->view_src->buffer)) return true;

    const size_t nrows = (size_t) t->ne[0];
    const size_t tok = (size_t) t->ne[1] - 1;
    std::vector<float> v((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    const float * last = v.data() + nrows*tok;
    double sum = 0, sumsq = 0, maxabs = 0;
    size_t bad = 0;
    for (size_t i = 0; i < nrows; ++i) {
        const double val = last[i];
        if (!std::isfinite(val)) { ++bad; continue; }
        sum += val;
        sumsq += val*val;
        maxabs = (std::max)(maxabs, std::fabs(val));
    }
    auto sample = [&](size_t i) { return i < nrows ? (double) last[i] : 0.0; };
    fprintf(stderr,
        "[CLI_TENSOR_TRACE] name=%s op=%s ne0=%zu ne1=%" PRId64 " token=%zu mean=%.9g rms=%.9g maxabs=%.9g nonfinite=%zu v0=%.9g v31=%.9g v32=%.9g v255=%.9g vlast=%.9g\n",
        name, ggml_op_name(t->op), nrows, t->ne[1], tok,
        sum/nrows, std::sqrt(sumsq/nrows), maxabs, bad,
        sample(0), sample(31), sample(32), sample(255), sample(nrows-1));

    if (strcmp(name, "result_output") == 0) {
        const ggml_tensor * weight = t->src[0];
        const ggml_tensor * input  = t->src[1];
        const ggml_backend_buffer_t wbuf = weight ? (weight->view_src ? weight->view_src->buffer : weight->buffer) : nullptr;
        fprintf(stderr,
            "[CLI_LMHEAD_INFO] weight=%s type=%s ne0=%" PRId64 " ne1=%" PRId64
            " wbytes=%zu buffer=%s usage=%d input=%s input_type=%s input_ne1=%" PRId64 "\n",
            weight ? ggml_get_name(weight) : "(null)",
            weight ? ggml_type_name(weight->type) : "(null)",
            weight ? weight->ne[0] : 0, weight ? weight->ne[1] : 0,
            weight ? ggml_nbytes(weight) : 0,
            wbuf ? ggml_backend_buffer_name(wbuf) : "(none)",
            wbuf ? (int) ggml_backend_buffer_get_usage(wbuf) : -1,
            input ? ggml_get_name(input) : "(null)",
            input ? ggml_type_name(input->type) : "(null)",
            input ? input->ne[1] : 0);

        size_t first_bad = nrows, last_bad = nrows, first_huge = nrows, last_huge = nrows;
        size_t num_nan = 0, num_inf = 0, num_huge = 0, first_bad_sample = 0;
        std::array<size_t, 8> bad_indices = {};
        for (size_t i = 0; i < nrows; ++i) {
            const float x = last[i];
            if (!std::isfinite(x)) {
                if (first_bad == nrows) first_bad = i;
                last_bad = i;
                if (first_bad_sample < bad_indices.size()) bad_indices[first_bad_sample++] = i;
                if (std::isnan(x)) ++num_nan; else ++num_inf;
            } else if (std::fabs((double) x) > 1000.0) {
                ++num_huge;
                if (first_huge == nrows) first_huge = i;
                last_huge = i;
            }
        }
        fprintf(stderr,
            "[CLI_LOGITS_DIAG] ne0=%zu nan=%zu inf=%zu huge_gt_1000=%zu first_bad=%zu last_bad=%zu first_huge=%zu last_huge=%zu bad_samples=",
            nrows, num_nan, num_inf, num_huge,
            first_bad, last_bad, first_huge, last_huge);
        for (size_t i = 0; i < first_bad_sample; ++i) {
            fprintf(stderr, "%s%zu", i ? "," : "", bad_indices[i]);
        }
        fprintf(stderr, "\n");
        std::array<double, 5> top;
        std::array<size_t, 5> ids = {};
        top.fill(-std::numeric_limits<double>::infinity());
        for (size_t i = 0; i < nrows; ++i) {
            const double x = last[i];
            if (!std::isfinite(x) || x <= top[4]) continue;
            int pos = 4;
            while (pos > 0 && x > top[(size_t) pos-1]) {
                top[(size_t) pos] = top[(size_t) pos-1];
                ids[(size_t) pos] = ids[(size_t) pos-1];
                --pos;
            }
            top[(size_t) pos] = x;
            ids[(size_t) pos] = i;
        }
        fprintf(stderr,
            "[CLI_LOGITS_TOP] token=%zu top0=%zu:%.9g top1=%zu:%.9g top2=%zu:%.9g top3=%zu:%.9g top4=%zu:%.9g\n",
            tok, ids[0], top[0], ids[1], top[1], ids[2], top[2],
            ids[3], top[3], ids[4], top[4]);
    }
    return true;
}

// satisfies -Wmissing-declarations
int llama_cli(int argc, char ** argv);

int llama_cli(int argc, char ** argv) {
    common_params params;

    params.verbosity = LOG_LEVEL_ERROR; // by default, less verbose logs

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }

    if (std::getenv("GGML_RPC_TENSOR_TRACE") != nullptr) {
        params.cb_eval = cli_rpc_trace_tensor;
        params.cb_eval_user_data = nullptr;
        fprintf(stderr, "[CLI_TENSOR_TRACE_ENABLED] CPU/RPC checkpoints active\n");
    }

    llama_backend_init();
    llama_numa_init(params.numa);

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    cli_context ctx_cli(params);

    if (!ctx_cli.init()) {
        return 1;
    }

    return ctx_cli.run();
}
