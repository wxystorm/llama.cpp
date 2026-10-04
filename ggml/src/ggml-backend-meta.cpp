#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include <cinttypes>

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

static void meta_debug_tensor_data(const ggml_tensor * tensor, const char * tag) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        return;
    }

    const size_t n = ggml_nelements(tensor);
    std::vector<float> data(n);
    ggml_backend_tensor_get(tensor, data.data(), 0, n * sizeof(float));

    double sum = 0.0;
    double l2 = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double x = data[i];
        sum += x;
        l2 += x * x;
        max_abs = std::max(max_abs, std::abs(x));
    }

    printf(
        "[NUMDBG] %s tensor=%s n=%zu "
        "sum=%.9f l2=%.9f max=%.9f "
        "v0=%.9f v1=%.9f v2=%.9f v3=%.9f "
        "v4=%.9f v5=%.9f v6=%.9f v7=%.9f\n",
        tag,
        tensor->name,
        n,
        sum,
        std::sqrt(l2),
        max_abs,
        n > 0 ? data[0] : 0.0f,
        n > 1 ? data[1] : 0.0f,
        n > 2 ? data[2] : 0.0f,
        n > 3 ? data[3] : 0.0f,
        n > 4 ? data[4] : 0.0f,
        n > 5 ? data[5] : 0.0f,
        n > 6 ? data[6] : 0.0f,
        n > 7 ? data[7] : 0.0f);
}

static void meta_debug_tensor(
        ggml_backend_t backend,
        const ggml_tensor * tensor,
        const char * tag) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        return;
    }

    ggml_backend_synchronize(backend);
    meta_debug_tensor_data(tensor, tag);
}

static bool ggml_backend_meta_parse_decode_ffn_chunk(
        const char * name,
        int & chunk,
        int & layer) {
    if (name == nullptr) {
        return false;
    }

    int n = 0;
    if (std::sscanf(name, "ffn_down_chunk_%d-%d%n", &chunk, &layer, &n) != 2) {
        return false;
    }
    return name[n] == '\0';
}

static bool ggml_backend_meta_parse_prefill_norm_chunk(
        const char * name,
        int & chunk,
        int & layer) {
    if (name == nullptr) {
        return false;
    }

    int n = 0;
    if (std::sscanf(
            name,
            "prefill_ffn_norm_chunk_%d-%d%n",
            &chunk,
            &layer,
            &n) != 2) {
        return false;
    }

    return name[n] == '\0';
}

static bool ggml_backend_meta_parse_prefill_down_chunk(
        const char * name,
        int & chunk,
        int & layer) {
    if (name == nullptr) {
        return false;
    }

    int n = 0;
    if (std::sscanf(name, "prefill_ffn_down_chunk_%d-%d%n", &chunk, &layer, &n) != 2) {
        return false;
    }
    return name[n] == '\0';
}

static bool ggml_backend_meta_parse_phone_route_weights(
        const char * name,
        int & chunk,
        int & layer) {
    if (name == nullptr) {
        return false;
    }

    int n = 0;
    if (std::sscanf(name, "phone_prefill_route_weights_chunk_%d-%d%n",
                    &chunk, &layer, &n) == 2 && name[n] == '\0') {
        return true;
    }

    chunk = -1;
    n = 0;
    return std::sscanf(name, "phone_moe_route_weights-%d%n",
                       &layer, &n) == 1 && name[n] == '\0';
}

static bool ggml_backend_meta_parse_phone_route_topk(
        const char * name,
        int & chunk,
        int & layer) {
    if (name == nullptr) {
        return false;
    }

    int n = 0;
    if (std::sscanf(name, "phone_prefill_route_topk_chunk_%d-%d%n",
                    &chunk, &layer, &n) == 2 && name[n] == '\0') {
        return true;
    }

    chunk = -1;
    n = 0;
    return std::sscanf(name, "phone_moe_route_topk-%d%n",
                       &layer, &n) == 1 && name[n] == '\0';
}

static bool ggml_backend_meta_parse_prefill_wave_ffn_inp_chunk(
        const char * name,
        int & chunk,
        int & layer) {
    if (name == nullptr) {
        return false;
    }

    int n = 0;
    if (std::sscanf(name, "prefill_wave_ffn_inp_chunk_%d-%d%n", &chunk, &layer, &n) != 2) {
        return false;
    }
    return name[n] == '\0';
}

static bool ggml_backend_meta_parse_prefill_wave_attn_out_chunk(
        const char * name,
        int & chunk,
        int & layer) {
    if (name == nullptr) {
        return false;
    }

    int n = 0;
    if (std::sscanf(name, "prefill_wave_attn_out_chunk_%d-%d%n", &chunk, &layer, &n) != 2) {
        return false;
    }
    return name[n] == '\0';
}

static bool ggml_backend_meta_parse_prefill_wave_attn_out_group(
        const char * name,
        int & chunk_begin,
        int & chunk_count,
        int & layer) {
    if (name == nullptr) {
        return false;
    }

    int n = 0;
    if (std::sscanf(
            name, "prefill_wave_attn_out_group_%d_%d-%d%n",
            &chunk_begin, &chunk_count, &layer, &n) != 3) {
        return false;
    }
    return name[n] == '\0';
}

static bool ggml_backend_meta_parse_prefill_wave_l_out_chunk(
        const char * name,
        int & chunk,
        int & layer) {
    if (name == nullptr) {
        return false;
    }

    int n = 0;
    if (std::sscanf(name, "prefill_wave_l_out_chunk_%d-%d%n", &chunk, &layer, &n) != 2) {
        return false;
    }
    return name[n] == '\0';
}

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
struct ggml_backend_meta_simple_tensor_container {
    std::vector<ggml_context_ptr> ctxs;
    std::map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;   //这个是一个映射表，key是原始张量，value是切分后的张量列表

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params, const int n_simple) {
        ctxs.reserve(n_simple);
        for (int i = 0; i < n_simple; i++) {
            ctxs.emplace_back(ggml_init(params));
        }
    }
    ggml_backend_meta_simple_tensor_container() {}
};

struct ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    static constexpr int STC_COMPUTE_COUNT = 3;

    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[STC_COMPUTE_COUNT];
    int stc_compute_index      = 0;
    int stc_compute_index_next = 0;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;

    int debug;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            ggml_backend_meta_simple_tensor_container & stc_compute_2,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)),
              stc_compute{std::move(stc_compute_0), std::move(stc_compute_1), std::move(stc_compute_2)} {
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        return stc_compute[stc_compute_index];
    }

    ggml_backend_meta_simple_tensor_container * find_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return &stc_static;
        }

        auto & current = stc_compute[stc_compute_index];
        if (current.simple_tensors.find(tensor) != current.simple_tensors.end()) {
            return &current;
        }

        for (int i = 0; i < STC_COMPUTE_COUNT; ++i) {
            if (i == stc_compute_index) {
                continue;
            }

            auto & stc = stc_compute[i];
            if (stc.simple_tensors.find(tensor) != stc.simple_tensors.end()) {
                return &stc;
            }
        }

        return nullptr;
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(
        const struct ggml_tensor * tensor, size_t index) {

    if (tensor == nullptr ||
        tensor->buffer == nullptr ||
        !ggml_backend_buffer_is_meta(tensor->buffer)) {

        fprintf(stderr,
            "[META_SIMPLE_NONMETA] "
            "tensor=%p name=%s op=%s buffer=%p "
            "view_src=%p view_name=%s view_buffer=%p "
            "view_offs=%zu index=%zu "
            "ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
            (const void *) tensor,
            tensor != nullptr ? tensor->name : "(null)",
            tensor != nullptr ? ggml_op_name(tensor->op) : "(null)",
            tensor != nullptr ? (void *) tensor->buffer : nullptr,

            tensor != nullptr ? (void *) tensor->view_src : nullptr,
            tensor != nullptr && tensor->view_src != nullptr ?
                tensor->view_src->name : "(null)",
            tensor != nullptr && tensor->view_src != nullptr ?
                (void *) tensor->view_src->buffer : nullptr,

            tensor != nullptr ? tensor->view_offs : 0,
            index,

            tensor != nullptr ? tensor->ne[0] : 0,
            tensor != nullptr ? tensor->ne[1] : 0,
            tensor != nullptr ? tensor->ne[2] : 0,
            tensor != nullptr ? tensor->ne[3] : 0);
    }

    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));

    ggml_backend_meta_buffer_context * buf_ctx =
        (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    GGML_ASSERT(index < buf_ctx->bufs.size());

    auto * stc = buf_ctx->find_simple_tensor_container(tensor);
    if (stc == nullptr) {
        return nullptr;
    }

    auto it = stc->simple_tensors.find(tensor);
    if (it == stc->simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}
// 判断一个张量是否是静态的，即是否在静态简单张量容器中存在
static bool ggml_backend_meta_tensor_is_static(
        const ggml_tensor * tensor) {

    if (tensor == nullptr ||
        tensor->buffer == nullptr ||
        !ggml_backend_buffer_is_meta(tensor->buffer)) {
        return false;
    }

    auto * buf_ctx =
        static_cast<ggml_backend_meta_buffer_context *>(
            tensor->buffer->context);

    return buf_ctx->stc_static.simple_tensors.find(tensor) !=
           buf_ctx->stc_static.simple_tensors.end();
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const bool hybrid_split_detail =
        std::getenv("GGML_META_ALLOC_TRACE") != nullptr &&
        std::strncmp(tensor->name, "ffn_moe_weights-", 16) == 0;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                if (std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_CPU_CHUNK_STAGE") != nullptr) {
                    std::string lhs_ne;
                    std::string rhs_ne;
                    for (size_t j = 0; j < n_bufs; ++j) {
                        if (j > 0) {
                            lhs_ne += ",";
                            rhs_ne += ",";
                        }
                        lhs_ne += std::to_string(ret.ne[j]);
                        rhs_ne += std::to_string(src_ss[i].ne[j]);
                    }
                    GGML_LOG_ERROR(
                        "[HYBRID_META_SPLIT_MISMATCH] tensor=%s op=%s src_index=%zu "
                        "lhs_axis=%s lhs_ne={%s} lhs_nr=%u lhs_segments=%u "
                        "rhs=%s rhs_axis=%s rhs_ne={%s} rhs_nr=%u rhs_segments=%u\n",
                        tensor->name,
                        ggml_op_name(tensor->op),
                        i,
                        ggml_backend_meta_split_axis_name(ret.axis),
                        lhs_ne.c_str(),
                        ret.nr[0],
                        ret.n_segments,
                        tensor->src[i] != nullptr ? tensor->src[i]->name : "(null)",
                        ggml_backend_meta_split_axis_name(src_ss[i].axis),
                        rhs_ne.c_str(),
                        src_ss[i].nr[0],
                        src_ss[i].n_segments);
                }
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        GGML_ABORT("fatal error");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1);
                if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                    return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                }
                int64_t base_ne_in = tensor->src[0]->ne[0];
                for (int dim = 1; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                base_ne_in /= src_ss[0].nr[0];
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    const int64_t base_ne_out_next = base_ne_out *= tensor->ne[dim];
                    if (base_ne_out_next % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out_next/base_ne_in)}, 1};
                    }
                    if (base_ne_out_next > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                    base_ne_out = base_ne_out_next;
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (hybrid_split_detail) {
            GGML_LOG_ERROR(
                "[HYBRID_META_GET_ROWS] tensor=%s point=RULE "
                "src0_axis=%s src0_segments=%u src0_nr0=%u "
                "src1_axis=%s src1_segments=%u src1_nr0=%u\n",
                tensor->name,
                ggml_backend_meta_split_axis_name(src_ss[0].axis),
                src_ss[0].n_segments,
                src_ss[0].nr[0],
                ggml_backend_meta_split_axis_name(src_ss[1].axis),
                src_ss[1].n_segments,
                src_ss[1].nr[0]);
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(                             src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis <= GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            }
            return ret;
        }

        int wave_l_out_chunk = -1;
        int wave_l_out_layer = -1;
        const bool is_prefill_wave_l_out =
            tensor->op == GGML_OP_ADD &&
            ggml_backend_meta_parse_prefill_wave_l_out_chunk(
                tensor->name, wave_l_out_chunk, wave_l_out_layer);

        if (is_prefill_wave_l_out) {
            // The split FFN produces PARTIAL tensors on PC/Phone, but the
            // explicit wavefront l_out node is committed only after Phone has
            // been reduced into the PC down tensor. Treat the committed node
            // as MIRRORED so downstream Attention sees a full hidden state.
            //
            // This result is independent of the source split states. Return
            // before recursively walking the residual/source chain so staged
            // graphs do not build unnecessarily deep split-state recursion.
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                continue;
            }
            if (hybrid_split_detail) {
                const ggml_tensor * src = tensor->src[i];
                GGML_LOG_ERROR(
                    "[HYBRID_META_GET_ROWS] tensor=%s point=SRC_STATE_BEGIN src_index=%zu "
                    "src=%s src_op=%s view=%s view_src=%s view_offs=%zu "
                    "ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] "
                    "nb=[%zu,%zu,%zu,%zu]\n",
                    tensor->name,
                    i,
                    src->name,
                    ggml_op_name(src->op),
                    src->view_src != nullptr ? "yes" : "no",
                    src->view_src != nullptr ? src->view_src->name : "(null)",
                    src->view_offs,
                    src->ne[0], src->ne[1], src->ne[2], src->ne[3],
                    src->nb[0], src->nb[1], src->nb[2], src->nb[3]);
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            if (hybrid_split_detail) {
                GGML_LOG_ERROR(
                    "[HYBRID_META_GET_ROWS] tensor=%s point=SRC_STATE_END src_index=%zu "
                    "axis=%s segments=%u nr0=%u\n",
                    tensor->name,
                    i,
                    ggml_backend_meta_split_axis_name(src_ss[i].axis),
                    src_ss[i].n_segments,
                    src_ss[i].nr[0]);
            }
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        // Compute tensor storage is reused across staged graph generations.
        // The whole cache is invalidated once when the STC generation switches
        // in ggml_backend_meta_buffer_init_tensor(). Do not clear the entire
        // cache from inside recursive split-state evaluation: doing so destroys
        // memoized ancestor states and can turn a linear walk into unbounded
        // recomputation deep enough to overflow the host stack.
        //
        // A mismatch inside one generation should be local in normal operation.
        // Drop this stale entry and recompute it while preserving unrelated
        // split states. If this starts firing repeatedly, GGML_META_STC_DEBUG
        // can be used to diagnose an unexpected in-generation tensor mutation.
        buf_ctx->split_state_cache.erase(it);
        it = buf_ctx->split_state_cache.end();
    }

    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor->src[i], true);
                GGML_ASSERT(split_state.n_segments == 1);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]) + "x" + std::to_string(split_state.nr[0]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                const ggml_backend_meta_split_state & ss = buf_ctx->split_state_cache[key].first;
                ne_info += std::to_string(ss.ne[j]) + "x" + std::to_string(ss.nr[0]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < ret.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += ret.ne[s*n_bufs + j] * ret.nr[s];
            }
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    auto * stc = buf_ctx->find_simple_tensor_container(tensor);
    if (stc == nullptr) {
        stc = &buf_ctx->get_simple_tensor_container(tensor);
    }
    return ggml_backend_meta_get_split_state(*stc, tensor, assume_sync);
}

static bool ggml_backend_meta_tensor_is_mirrored(
        const ggml_tensor * tensor, bool assume_sync) {
    if (tensor == nullptr || tensor->buffer == nullptr ||
        !ggml_backend_buffer_is_meta(tensor->buffer)) {
        return false;
    }
    return ggml_backend_meta_get_split_state(tensor, assume_sync).axis ==
        GGML_BACKEND_SPLIT_AXIS_MIRRORED;
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    const bool hybrid_init_detail =
        std::getenv("GGML_META_ALLOC_TRACE") != nullptr &&
        (std::strncmp(tensor->name, "ffn_moe_probs-", 14) == 0 ||
         std::strncmp(tensor->name, "ffn_moe_weights-", 16) == 0);

    if (hybrid_init_detail) {
        GGML_LOG_ERROR(
            "[HYBRID_META_INIT_DETAIL] tensor=%s op=%s point=SPLIT_BEGIN "
            "src0=%s src1=%s simple_bufs=%zu\n",
            tensor->name,
            ggml_op_name(tensor->op),
            tensor->src[0] != nullptr ? tensor->src[0]->name : "(null)",
            tensor->src[1] != nullptr ? tensor->src[1]->name : "(null)",
            n_simple_bufs);
    }

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    if (hybrid_init_detail) {
        GGML_LOG_ERROR(
            "[HYBRID_META_INIT_DETAIL] tensor=%s point=SPLIT_END "
            "axis=%s segments=%u nr0=%u\n",
            tensor->name,
            ggml_backend_meta_split_axis_name(split_state.axis),
            split_state.n_segments,
            split_state.nr[0]);
    }
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctxs[j].get(); //第j个张量后端目录，初始里面没有张量
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if ((simple_buf != nullptr) && ggml_backend_buffer_is_multi_buffer(simple_buf)) {
            // see https://github.com/ggml-org/llama.cpp/issues/22197
            GGML_ABORT("multi buffers are not supported by the meta backend");
        }

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        if (hybrid_init_detail) {
            GGML_LOG_ERROR(
                "[HYBRID_META_INIT_DETAIL] tensor=%s point=SIMPLE_NEW_BEGIN "
                "backend=%zu ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] "
                "simple_buf=%p\n",
                tensor->name,
                j,
                ne[0], ne[1], ne[2], ne[3],
                (void *) simple_buf);
        }
        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        if (hybrid_init_detail) {
            GGML_LOG_ERROR(
                "[HYBRID_META_INIT_DETAIL] tensor=%s point=SIMPLE_NEW_END "
                "backend=%zu simple_tensor=%p\n",
                tensor->name,
                j,
                (void *) t_ij);
        }
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }
        t_ij->extra = tensor->extra;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                if (hybrid_init_detail) {
                    GGML_LOG_ERROR(
                        "[HYBRID_META_INIT_DETAIL] tensor=%s point=SRC_MAP_BEGIN "
                        "backend=%zu src_index=%d src=%s src_buffer=%p\n",
                        tensor->name,
                        j,
                        i,
                        tensor->src[i]->name,
                        (void *) tensor->src[i]->buffer);
                }
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
                if (hybrid_init_detail) {
                    GGML_LOG_ERROR(
                        "[HYBRID_META_INIT_DETAIL] tensor=%s point=SRC_MAP_END "
                        "backend=%zu src_index=%d mapped=%p\n",
                        tensor->name,
                        j,
                        i,
                        (void *) t_ij->src[i]);
                }
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                if (std::getenv("GGML_META_TP_FFN_FLAG_TRACE") != nullptr &&
                        (std::strstr(tensor->name, "ffn_moe") != nullptr ||
                         std::strstr(tensor->name, "ffn_down") != nullptr)) {
                    int64_t src_ne_0 = 0;
                    int64_t src_ne_1 = 0;
                    for (size_t s = 0; s < split_state_src.n_segments; ++s) {
                        if (n_simple_bufs > 0) {
                            src_ne_0 += split_state_src.ne[s*n_simple_bufs + 0] *
                                        split_state_src.nr[s];
                        }
                        if (n_simple_bufs > 1) {
                            src_ne_1 += split_state_src.ne[s*n_simple_bufs + 1] *
                                        split_state_src.nr[s];
                        }
                    }
                    printf(
                        "[TP_FFN_FLAG_DISABLE] tensor=%s op=%s backend=%zu "
                        "src=%s src_axis=%s src_ne={%" PRId64 ",%" PRId64 "}\n",
                        tensor->name,
                        ggml_op_name(tensor->op),
                        j,
                        tensor->src[i]->name,
                        ggml_backend_meta_split_axis_name(split_state_src.axis),
                        src_ne_0,
                        src_ne_1);
                }
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    if (hybrid_init_detail) {
        GGML_LOG_ERROR(
            "[HYBRID_META_INIT_DETAIL] tensor=%s point=MAP_INSERT_BEGIN entries=%zu\n",
            tensor->name,
            stc.simple_tensors.size());
    }
    stc.simple_tensors[tensor] = simple_tensors;
    if (hybrid_init_detail) {
        GGML_LOG_ERROR(
            "[HYBRID_META_INIT_DETAIL] tensor=%s point=MAP_INSERT_END entries=%zu\n",
            tensor->name,
            stc.simple_tensors.size());
    }

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;

    if (buf_ctx->stc_compute_index != buf_ctx->stc_compute_index_next) {
        if (std::getenv("GGML_META_STC_DEBUG") != nullptr) {
            GGML_LOG_DEBUG(
                "[META_SPLIT_CACHE_ROTATE] cur=%d next=%d entries=%zu tensor=%s\n",
                buf_ctx->stc_compute_index,
                buf_ctx->stc_compute_index_next,
                buf_ctx->split_state_cache.size(),
                tensor->name);
        }

        // The allocator reuses ggml_tensor addresses across staged graphs.
        // Invalidate split states once at the generation boundary, before any
        // tensor from the new STC is initialized. Keeping this out of recursive
        // split-state lookup preserves memoization for the entire graph and
        // prevents stale-address recovery from recursively clearing the cache.
        buf_ctx->split_state_cache.clear();
        buf_ctx->stc_compute_index = buf_ctx->stc_compute_index_next;
    }

    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

static enum ggml_backend_local_file_result
ggml_backend_meta_try_set_local_file_2d(
        ggml_tensor * tensor,
        uint64_t src_offset,
        uint64_t dst_offset,
        uint64_t copy_size,
        uint64_t n_copies,
        uint64_t src_stride,
        uint64_t dst_stride) {

    if (tensor == nullptr || tensor->buffer == nullptr) {
        return GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
    }

    ggml_backend_buffer_type_t buft =
        ggml_backend_buffer_get_type(tensor->buffer);

    if (buft == nullptr) {
        return GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
    }

    ggml_backend_dev_t dev =
        ggml_backend_buft_get_device(buft);

    if (dev == nullptr) {
        return GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
    }

    ggml_backend_reg_t reg =
        ggml_backend_dev_backend_reg(dev);

    if (reg == nullptr) {
        return GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
    }

    void * proc =
        ggml_backend_reg_get_proc_address(
            reg,
            GGML_BACKEND_RPC_SET_LOCAL_2D_PROC);

    if (proc == nullptr) {
        return GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
    }

    auto fn =
        reinterpret_cast<ggml_backend_rpc_set_local_2d_t>(
            proc);

    return fn(
        tensor,
        src_offset,
        dst_offset,
        copy_size,
        n_copies,
        src_stride,
        dst_stride);
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }
    const bool is_static_tensor =
                    ggml_backend_meta_tensor_is_static(tensor);

                /*
                * 再增加几项防御性限制：
                * 本地文件快速路径只处理原始、非 view 的静态张量。
                */
                const bool can_use_local_file =
                    is_static_tensor &&
                    tensor->op == GGML_OP_NONE &&
                    tensor->view_src == nullptr &&
                    tensor->name[0] != '\0';

    switch (split_state.axis) {
    case GGML_BACKEND_SPLIT_AXIS_0:
    case GGML_BACKEND_SPLIT_AXIS_1:
    case GGML_BACKEND_SPLIT_AXIS_2: {
        // 完整 Meta tensor 中，一个完整 chunk 的字节数
        const size_t chunk_size_full =
            tensor->nb[split_state.axis + 1];

        GGML_ASSERT(chunk_size_full != 0);
        GGML_ASSERT(offset % chunk_size_full == 0);
        GGML_ASSERT(size   % chunk_size_full == 0);

        const int64_t i_start =
            offset / chunk_size_full;

        const int64_t i_stop =
            (offset + size) / chunk_size_full;

        GGML_ASSERT(i_start >= 0);
        GGML_ASSERT(i_stop >= i_start);

        size_t offset_j = 0;

        for (size_t j = 0; j < n_bufs; ++j) {
            ggml_tensor * simple_tensor =
                ggml_backend_meta_buffer_simple_tensor(
                    tensor,
                    j);

            GGML_ASSERT(simple_tensor != nullptr);

            const size_t chunk_size_j =
                simple_tensor->nb[split_state.axis + 1];

            if (chunk_size_j == 0) {
                continue;
            }

            /*
             * 本次写入在局部目标张量中的起始偏移。
             */
            const size_t simple_offset =
                static_cast<size_t>(i_start) *
                chunk_size_j;

            /*
             * 本地文件源偏移必须是相对于“完整源张量起点”的偏移。
             *
             * offset：
             *   当前写入区域在完整 Meta tensor 中的起始偏移。
             *
             * offset_j：
             *   当前设备分片在每个完整 chunk 内的偏移。
             */
            const uint64_t src_offset =
                static_cast<uint64_t>(offset) +
                static_cast<uint64_t>(offset_j);

            const uint64_t dst_offset =
                static_cast<uint64_t>(simple_offset);

            const uint64_t copy_size =
                static_cast<uint64_t>(chunk_size_j);

            const uint64_t n_copies =
                static_cast<uint64_t>(i_stop - i_start);

            const uint64_t src_stride =
                static_cast<uint64_t>(chunk_size_full);

            const uint64_t dst_stride =
                static_cast<uint64_t>(chunk_size_j);

            ggml_backend_local_file_result result =
    GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
            // 只有是静态才走
            if (can_use_local_file) {
                result =
                    ggml_backend_meta_try_set_local_file_2d(
                        simple_tensor,
                        src_offset,
                        dst_offset,
                        copy_size,
                        n_copies,
                        src_stride,
                        dst_stride);
            }

            switch (result) {
                case GGML_BACKEND_LOCAL_FILE_HANDLED: {
                    /*
                     * RPC 服务端已经从本地 GGUF 读取并写入，
                     * 不再发送 data 中的权重。
                     */
                } break;

                case GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED: {
                    /*
                     * CPU、CUDA、旧版 RPC，或者服务端没有本地模型，
                     * 使用 llama.cpp 原来的内存复制路径。
                     */
                    ggml_backend_tensor_set_2d(
                        simple_tensor,
                        static_cast<const char *>(data) +
                            offset_j,
                        simple_offset,
                        chunk_size_j,
                        static_cast<size_t>(
                            i_stop - i_start),
                        chunk_size_j,
                        chunk_size_full);
                } break;

                case GGML_BACKEND_LOCAL_FILE_ERROR: {
                    GGML_ABORT(
                        "failed to load tensor '%s' "
                        "from RPC local file: "
                        "src_offset=%" PRIu64
                        ", dst_offset=%" PRIu64
                        ", copy_size=%" PRIu64
                        ", n_copies=%" PRIu64,
                        simple_tensor->name,
                        src_offset,
                        dst_offset,
                        copy_size,
                        n_copies);
                } break;

                default: {
                    GGML_ABORT(
                        "invalid RPC local-file result");
                }
            }

            /*
             * 注意：无论本地加载还是普通复制，
             * 都要继续累计下一个设备的源分片偏移。
             */
            offset_j += chunk_size_j;
        }

        /*
         * 检查所有设备的局部分片是否刚好组成完整 chunk。
         */
        GGML_ASSERT(offset_j == chunk_size_full);
    } break;

    case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
    for (size_t j = 0; j < n_bufs; ++j) {
        ggml_tensor * simple_tensor =
            ggml_backend_meta_buffer_simple_tensor(
                tensor,
                j);

        GGML_ASSERT(simple_tensor != nullptr);

        ggml_backend_local_file_result result =
            GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;

        /*
         * 只有静态模型权重才尝试 RPC 本地 GGUF。
         *
         * 运行时 embd、tokens、mask 等直接保持
         * NOT_SUPPORTED，然后进入普通 set_tensor。
         */
        if (can_use_local_file) {
            result =
                ggml_backend_meta_try_set_local_file_2d(
                    simple_tensor,
                    /* src_offset = */ static_cast<uint64_t>(offset),
                    /* dst_offset = */ static_cast<uint64_t>(offset),
                    /* copy_size  = */ static_cast<uint64_t>(size),
                    /* n_copies   = */ 1,
                    /* src_stride = */ static_cast<uint64_t>(size),
                    /* dst_stride = */ static_cast<uint64_t>(size));
        }

        switch (result) {
            case GGML_BACKEND_LOCAL_FILE_HANDLED: {
                // RPC服务端已经从本地GGUF加载完成
            } break;

            case GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED: {
                /*
                 * 包括：
                 * 1. CPU/CUDA后端；
                 * 2. 运行时计算张量；
                 * 3. RPC不支持本地文件扩展；
                 * 4. 非静态张量。
                 */
                ggml_backend_tensor_set(
                    simple_tensor,
                    data,
                    offset,
                    size);
            } break;

            case GGML_BACKEND_LOCAL_FILE_ERROR: {
                GGML_ABORT(
                    "failed to load mirrored tensor '%s' "
                    "from RPC local file",
                    simple_tensor->name);
            } break;

            default: {
                GGML_ABORT(
                    "invalid RPC local-file result");
            }
        }
    }
} break;

    case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
        /*
         * 这里不能直接从本地 GGUF 加载。
         *
         * 写入的数据不是文件中的原始数据，
         * 而是原始 F32 数据除以 n_bufs 后产生的新数据。
         */
        GGML_ASSERT(tensor->type == GGML_TYPE_F32);

        const int64_t ne =
            ggml_nelements(tensor);

        std::vector<float> tmp;
        tmp.reserve(ne);

        for (int64_t i = 0; i < ne; ++i) {
            tmp.push_back(
                static_cast<const float *>(data)[i] /
                n_bufs);
        }

        for (size_t j = 0; j < n_bufs; ++j) {
            ggml_tensor * simple_tensor =
                ggml_backend_meta_buffer_simple_tensor(
                    tensor,
                    j);

            ggml_backend_tensor_set(
                simple_tensor,
                tmp.data(),
                offset,
                size);
        }
    } break;

    default: {
        GGML_ABORT("fatal error");
    }
}
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (std::getenv("GGML_META_PIPELINE_DEBUG") != nullptr &&
            std::strcmp(tensor->name, "l_out-23") == 0) {
        const ggml_tensor * simple0 = n_bufs > 0 ?
            ggml_backend_meta_buffer_simple_tensor(tensor, 0) : nullptr;
        const ggml_tensor * simple1 = n_bufs > 1 ?
            ggml_backend_meta_buffer_simple_tensor(tensor, 1) : nullptr;
        printf(
            "[META_EXPORT] tensor=%s ptr=%p axis=%s simple0=%p simple1=%p "
            "s0=%s s1=%s\n",
            tensor->name, (const void *) tensor,
            ggml_backend_meta_split_axis_name(split_state.axis),
            (const void *) simple0, (const void *) simple1,
            simple0 != nullptr ? simple0->name : "(null)",
            simple1 != nullptr ? simple1->name : "(null)");
        meta_debug_tensor_data(simple0, "EXPORT simple0");
        meta_debug_tensor_data(simple1, "EXPORT simple1");
    }

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr, // TODO implement
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const int64_t total_begin_us = ggml_time_us();
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_2(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    std::vector<int64_t> simple_alloc_us(n_simple_bufts, 0);
    std::vector<size_t> simple_alloc_size(n_simple_bufts, 0);
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);
        const int64_t simple_begin_us = ggml_time_us();
        bufs.push_back(ggml_backend_buft_alloc_buffer(simple_buft, size));
        simple_alloc_us[i] = ggml_time_us() - simple_begin_us;
        if (bufs.back() == nullptr) {
            GGML_LOG_ERROR(
                "[META_BUFFER_ALLOC_FAIL] index=%zu buft=%s requested_mib=%.3f "
                "alloc_ms=%.3f simple_backends=%zu\n",
                i,
                ggml_backend_buft_name(simple_buft),
                size / 1048576.0,
                simple_alloc_us[i] / 1000.0,
                n_simple_bufts);
            for (size_t j = 0; j + 1 < bufs.size(); ++j) {
                ggml_backend_buffer_free(bufs[j]);
            }
            return nullptr;
        }
        simple_alloc_size[i] = ggml_backend_buffer_get_size(bufs.back());
        max_size = std::max(max_size, simple_alloc_size[i]);
    }
    ggml_backend_meta_buffer_context * buf_ctx =
        new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, stc_compute_2, bufs);

    const int64_t total_us = ggml_time_us() - total_begin_us;
    const bool timing_debug = std::getenv("GGML_ALLOC_TIMING_DEBUG") != nullptr;
    if (timing_debug || total_us >= 10000) {
        GGML_LOG_DEBUG(
            "[META_BUFFER_ALLOC] requested_mib=%.3f simple_backends=%zu total_ms=%.3f max_mib=%.3f\n",
            size / 1048576.0,
            n_simple_bufts,
            total_us / 1000.0,
            max_size / 1048576.0);
        for (size_t i = 0; i < n_simple_bufts; ++i) {
            ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);
            GGML_LOG_DEBUG(
                "[META_BUFFER_ALLOC_SIMPLE] index=%zu buft=%s requested_mib=%.3f actual_mib=%.3f alloc_ms=%.3f\n",
                i,
                ggml_backend_buft_name(simple_buft),
                size / 1048576.0,
                simple_alloc_size[i] / 1048576.0,
                simple_alloc_us[i] / 1000.0);
        }
    }

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    constexpr size_t tensor_chunk_headroom = 16;
    constexpr size_t auxiliary_headroom    = 4;
    constexpr size_t compute_headroom      = tensor_chunk_headroom + auxiliary_headroom;
    const ggml_init_params params_static = {
        /*.mem_size   =*/ ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom*ggml_get_mem_size(ctx),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_2(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx =
        new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, stc_compute_2, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {    //这里是张量切分主要逻辑
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t); //后端切分张量
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctxs[i].get();
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        if (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        GGML_ASSERT(meta_buf_ctx->bufs[i]);
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;    //里面的context指向的就是meta_buf_ctx
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_compute_workers;
struct ggml_backend_meta_transfer_worker;

struct ggml_backend_meta_context {
    static constexpr size_t PREFILL_RETURN_LANES = 2;
    static constexpr size_t PREFILL_ROUTE_LANES = 4;

    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;
        std::array<ggml_backend_buffer_ptr, PREFILL_RETURN_LANES> prefill_reduce_bufs;
        std::vector<std::array<ggml_backend_buffer_ptr, 3>> prefill_route_stage_bufs;
        // Private Phone-side storage used by Router-first deferred FFN.
        // The full normalized hidden is produced directly into one layer-local
        // buffer, while top-k/weights use per-chunk buffers.
        ggml_backend_buffer_ptr prefill_phone_hidden_stage_buf;
        std::vector<std::array<ggml_backend_buffer_ptr, 3>> prefill_phone_ffn_stage_bufs;
        // Private Phone-side PC-partial staging. One buffer per Tensor chunk
        // lets network return overlap the PC worker without overwriting an
        // earlier partial before the layer-tail ADD.
        std::vector<ggml_backend_buffer_ptr> prefill_phone_return_stage_bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_cgraph *>  cgraphs_early;
    std::vector<ggml_cgraph *>  cgraphs_phone_fused;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;
    uint64_t                    next_snapshot_seq = 1;
    uint64_t                    next_phone_prefill_route_seq = 1;
    uint64_t                    next_phone_prefill_ffn_seq = 1;
    int                         tensor_phone_first_layer = -1;
    int                         tensor_phone_last_layer  = -1;

    ggml_backend_meta_tensor_profile tensor_profile {};
    std::mutex                       tensor_profile_mutex;
    std::map<int, int64_t>           tensor_profile_wave_layer_start_us;
    std::map<int, int64_t>           tensor_profile_wave_layer_compute_wall_us;
    std::map<int, int64_t>           tensor_profile_wave_layer_barrier_us;
    int64_t                          tensor_profile_wave_span_begin_us = 0;
    int64_t                          tensor_profile_wave_span_end_us   = 0;

    ggml_backend_meta_compute_workers * compute_workers = nullptr;
    ggml_backend_meta_transfer_worker * transfer_worker = nullptr;
    ggml_backend_meta_transfer_worker * prefill_input_worker = nullptr;
    ggml_backend_meta_transfer_worker * prefill_pc_worker = nullptr;
    std::array<ggml_backend_meta_transfer_worker *, PREFILL_RETURN_LANES>
        prefill_return_workers { nullptr, nullptr };
    std::vector<std::shared_ptr<std::vector<uint8_t>>>
        prefill_pc_return_host_payloads;
    std::array<ggml_backend_meta_transfer_worker *, PREFILL_RETURN_LANES> prefill_reduce_workers { nullptr,
                                                                                                  nullptr };
    std::array<ggml_backend_meta_transfer_worker *, PREFILL_ROUTE_LANES> prefill_route_workers { nullptr,
                                                                                                nullptr };

    // Dedicated async Meta graph state used only by the hybrid Phone scheduler.
    // The worker executes the ordinary synchronous Meta graph implementation;
    // only the caller thread is released early after rebuild/preparation.
    bool                                 async_graph_compute_enabled = false;
    bool                                 async_graph_active = false;
    bool                                 async_graph_prepared = false;
    ggml_status                          async_graph_status = GGML_STATUS_SUCCESS;
    std::mutex                           async_graph_mutex;
    std::condition_variable              async_graph_cv;
    std::thread                          async_graph_thread;

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~ggml_backend_meta_context();
};

struct ggml_backend_meta_compute_workers {
    ggml_backend_meta_context * backend_ctx;
    size_t n_backends;

    std::vector<std::thread> workers;
    std::vector<ggml_status> statuses;
    std::vector<int64_t> backend_time_us;
    std::vector<int64_t> backend_call_count;
    std::vector<int64_t> completed_time_us;

    std::mutex mutex;
    std::condition_variable task_cv;
    std::condition_variable done_cv;
    std::vector<size_t> generation;
    std::vector<size_t> completed_generation;
    std::vector<ggml_cgraph *> task_graph;
    bool stop = false;

    ggml_backend_meta_compute_workers(ggml_backend_meta_context * backend_ctx, size_t n_backends) :
        backend_ctx(backend_ctx),
        n_backends(n_backends),
        statuses(n_backends, GGML_STATUS_SUCCESS),
        backend_time_us(n_backends, 0),
        backend_call_count(n_backends, 0),
        completed_time_us(n_backends, 0),
        generation(n_backends, 0),
        completed_generation(n_backends, 0),
        task_graph(n_backends, nullptr) {
        if (n_backends <= 1) {
            return;
        }

        workers.reserve(n_backends);
        for (size_t j = 0; j < n_backends; ++j) {
            workers.emplace_back([this, j]() {
                size_t seen_generation = 0;
                std::unique_lock<std::mutex> lock(mutex);
                while (true) {
                    task_cv.wait(lock, [&]() { return stop || generation[j] != seen_generation; });
                    if (stop) {
                        return;
                    }

                    seen_generation = generation[j];
                    ggml_cgraph * graph = task_graph[j];
                    lock.unlock();

                    auto & bcj = this->backend_ctx->backend_configs[j];
                    const int64_t start_us = ggml_time_us();
                    const ggml_status status = ggml_backend_graph_compute_async(
                            bcj.backend, graph);
                    const int64_t elapsed_us = ggml_time_us() - start_us;
                    const int64_t completed_us = ggml_time_us();

                    lock.lock();
                    statuses[j] = status;
                    backend_time_us[j] += elapsed_us;
                    backend_call_count[j] += 1;
                    completed_time_us[j] = completed_us;
                    completed_generation[j] = seen_generation;
                    done_cv.notify_all();
                }
            });
        }
    }

    ~ggml_backend_meta_compute_workers() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        task_cv.notify_all();
        for (auto & worker : workers) {
            worker.join();
        }
    }

    ggml_status compute(size_t i) {
        if (n_backends == 1) {
            auto & bcj = backend_ctx->backend_configs[0];
            const int64_t start_us = ggml_time_us();
            const ggml_status status = ggml_backend_graph_compute_async(
                    bcj.backend, bcj.cgraphs[i].cgraph_main);
            backend_time_us[0] += ggml_time_us() - start_us;
            backend_call_count[0] += 1;
            return status;
        }

        for (size_t j = 0; j < n_backends; ++j) {
            start(j, i);
        }

        for (size_t j = 0; j < n_backends; ++j) {
            const ggml_status status = wait(j);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    void start(size_t j, size_t i) {
        start_graph(j, backend_ctx->backend_configs[j].cgraphs[i].cgraph_main);
    }

    void start_graph(size_t j, ggml_cgraph * graph) {
        std::lock_guard<std::mutex> lock(mutex);
        GGML_ASSERT(completed_generation[j] == generation[j]);
        GGML_ASSERT(graph != nullptr);
        task_graph[j] = graph;
        ++generation[j];
        task_cv.notify_all();
    }

    ggml_status wait(size_t j) {
        std::unique_lock<std::mutex> lock(mutex);
        const size_t target_generation = generation[j];
        done_cv.wait(lock, [&]() { return completed_generation[j] == target_generation; });
        return statuses[j];
    }

    int64_t completed_at(size_t j) {
        std::lock_guard<std::mutex> lock(mutex);
        return completed_time_us[j];
    }

    void reset_timings() {
        std::fill(backend_time_us.begin(), backend_time_us.end(), 0);
        std::fill(backend_call_count.begin(), backend_call_count.end(), 0);
    }
};

struct ggml_backend_meta_transfer_worker {
    struct task {
        uint64_t id;
        std::function<ggml_status(uint64_t)> run;
    };

    std::thread worker;
    std::mutex mutex;
    std::condition_variable task_cv;
    std::condition_variable done_cv;
    std::condition_variable stage_cv;
    std::deque<task> tasks;
    uint64_t submitted = 0;
    uint64_t completed = 0;
    uint64_t stage_ready = 0;
    ggml_status status = GGML_STATUS_SUCCESS;
    bool stop = false;

    ggml_backend_meta_transfer_worker() : worker([this]() {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            task_cv.wait(lock, [&]() { return stop || !tasks.empty(); });
            if (stop && tasks.empty()) {
                return;
            }

            task current = std::move(tasks.front());
            tasks.pop_front();
            lock.unlock();
            const ggml_status task_status = current.run(current.id);
            lock.lock();
            if (status == GGML_STATUS_SUCCESS) {
                status = task_status;
            }
            completed = current.id;
            done_cv.notify_all();
            stage_cv.notify_all();
        }
    }) {}

    ~ggml_backend_meta_transfer_worker() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        task_cv.notify_one();
        worker.join();
    }

    uint64_t enqueue(std::function<ggml_status(uint64_t)> run) {
        std::lock_guard<std::mutex> lock(mutex);
        const uint64_t id = ++submitted;
        tasks.push_back({ id, std::move(run) });
        task_cv.notify_one();
        return id;
    }

    ggml_status wait(uint64_t id) {
        std::unique_lock<std::mutex> lock(mutex);
        done_cv.wait(lock, [&]() { return completed >= id; });
        return status;
    }

    bool is_completed(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex);
        return completed >= id;
    }

    void mark_stage_ready(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex);
        stage_ready = std::max(stage_ready, id);
        stage_cv.notify_all();
    }

    ggml_status wait_stage_ready(uint64_t id) {
        std::unique_lock<std::mutex> lock(mutex);
        stage_cv.wait(lock, [&]() { return stage_ready >= id || completed >= id; });
        return status;
    }
};

struct ggml_backend_meta_stage_ready_context {
    ggml_backend_meta_transfer_worker * worker;
    uint64_t task_id;
};

static void ggml_backend_meta_stage_ready(void * user_data) {
    auto * context = static_cast<ggml_backend_meta_stage_ready_context *>(user_data);
    context->worker->mark_stage_ready(context->task_id);
}

static ggml_backend_rpc_set_stage_ready_t ggml_backend_meta_get_stage_ready_setter(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_set_stage_ready_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_RPC_SET_STAGE_READY_PROC));
}

static ggml_backend_rpc_set_route_transfer_lane_t
ggml_backend_meta_get_route_transfer_lane_setter(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_set_route_transfer_lane_t>(
        ggml_backend_reg_get_proc_address(
            reg,
            GGML_BACKEND_RPC_SET_ROUTE_TRANSFER_LANE_PROC));
}

static ggml_backend_rpc_set_route_wait_seq_t
ggml_backend_meta_get_route_wait_seq_setter(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_set_route_wait_seq_t>(
        ggml_backend_reg_get_proc_address(
            reg,
            GGML_BACKEND_RPC_SET_ROUTE_WAIT_SEQ_PROC));
}

static ggml_backend_rpc_route_mark_ready_t
ggml_backend_meta_get_route_mark_ready(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_route_mark_ready_t>(
        ggml_backend_reg_get_proc_address(
            reg,
            GGML_BACKEND_RPC_ROUTE_MARK_READY_PROC));
}

static ggml_backend_rpc_route_snapshot_ready_t
ggml_backend_meta_get_route_snapshot_ready(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_route_snapshot_ready_t>(
        ggml_backend_reg_get_proc_address(
            reg,
            GGML_BACKEND_RPC_ROUTE_SNAPSHOT_READY_PROC));
}

static ggml_backend_rpc_get_route_snapshot_t
ggml_backend_meta_get_route_snapshot(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_get_route_snapshot_t>(
        ggml_backend_reg_get_proc_address(
            reg,
            GGML_BACKEND_RPC_GET_ROUTE_SNAPSHOT_PROC));
}

static ggml_backend_rpc_fence_t ggml_backend_meta_get_rpc_fence(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_fence_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_RPC_FENCE_PROC));
}

static ggml_backend_rpc_get_tensor_batch3_t ggml_backend_meta_get_tensor_batch3(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_get_tensor_batch3_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_RPC_GET_TENSOR_BATCH3_PROC));
}

static ggml_backend_rpc_set_tensor_graph_t ggml_backend_meta_get_set_tensor_graph(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_set_tensor_graph_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_RPC_SET_TENSOR_GRAPH_PROC));
}

static ggml_backend_rpc_set_tensor_async_return_t
ggml_backend_meta_get_set_tensor_async_return(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_set_tensor_async_return_t>(
        ggml_backend_reg_get_proc_address(
            reg,
            GGML_BACKEND_RPC_SET_TENSOR_ASYNC_RETURN_PROC));
}

static ggml_backend_rpc_phone_ffn_mark_ready_t
ggml_backend_meta_get_phone_ffn_mark_ready(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_phone_ffn_mark_ready_t>(
        ggml_backend_reg_get_proc_address(
            reg,
            GGML_BACKEND_RPC_PHONE_FFN_MARK_READY_PROC));
}

static ggml_backend_rpc_set_tensor_async_return_wait_t
ggml_backend_meta_get_set_tensor_async_return_wait(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<
        ggml_backend_rpc_set_tensor_async_return_wait_t>(
            ggml_backend_reg_get_proc_address(
                reg,
                GGML_BACKEND_RPC_SET_TENSOR_ASYNC_RETURN_WAIT_PROC));
}

static ggml_backend_rpc_snapshot_arm_t ggml_backend_meta_get_snapshot_arm(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_snapshot_arm_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_RPC_SNAPSHOT_ARM_PROC));
}

static ggml_backend_rpc_prepare_graph_snapshot_t ggml_backend_meta_get_graph_snapshot_preparer(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_prepare_graph_snapshot_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_RPC_PREPARE_GRAPH_SNAPSHOT_PROC));
}

static ggml_backend_rpc_prepare_fused_ffn_input_t ggml_backend_meta_get_fused_ffn_input_preparer(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_prepare_fused_ffn_input_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_RPC_PREPARE_FUSED_FFN_INPUT_PROC));
}

static ggml_backend_rpc_set_snapshot_read_t ggml_backend_meta_get_snapshot_read_setter(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_set_snapshot_read_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_RPC_SET_SNAPSHOT_READ_PROC));
}

static ggml_backend_rpc_get_snapshot_stats_t ggml_backend_meta_get_snapshot_stats_getter(
        ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_get_snapshot_stats_t>(
        ggml_backend_reg_get_proc_address(
            reg, GGML_BACKEND_RPC_GET_SNAPSHOT_STATS_PROC));
}

static ggml_backend_rpc_wait_snapshot_ready_t ggml_backend_meta_get_snapshot_ready_waiter(ggml_backend_t backend) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }

    return reinterpret_cast<ggml_backend_rpc_wait_snapshot_ready_t>(
        ggml_backend_reg_get_proc_address(reg, GGML_BACKEND_RPC_WAIT_SNAPSHOT_READY_PROC));
}

ggml_backend_meta_context::~ggml_backend_meta_context() {
    if (async_graph_thread.joinable()) {
        async_graph_thread.join();
    }
    delete compute_workers;
    delete transfer_worker;
    delete prefill_input_worker;
    delete prefill_pc_worker;
    for (auto * worker : prefill_return_workers) {
        delete worker;
    }
    for (auto * worker : prefill_reduce_workers) {
        delete worker;
    }
    for (auto * worker : prefill_route_workers) {
        delete worker;
    }
    if (comm_ctx != nullptr) {
        ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
        GGML_ASSERT(comm_free != nullptr);
        comm_free(comm_ctx);
    }
    for (auto & bc : backend_configs) {
        ggml_backend_free(bc.backend);
    }
}

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);
    GGML_ASSERT(split_state.nr[0]      == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static ggml_status ggml_backend_meta_wait_async_graph_impl(
        ggml_backend_meta_context * backend_ctx) {
    if (backend_ctx->async_graph_thread.joinable()) {
        backend_ctx->async_graph_thread.join();
    }

    std::lock_guard<std::mutex> lock(backend_ctx->async_graph_mutex);
    backend_ctx->async_graph_active = false;
    backend_ctx->async_graph_prepared = false;
    return backend_ctx->async_graph_status;
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    ggml_backend_meta_context * backend_ctx =
        (ggml_backend_meta_context *) backend->context;
    (void) ggml_backend_meta_wait_async_graph_impl(backend_ctx);

    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

static enum ggml_status ggml_backend_meta_graph_compute_impl(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    const int64_t meta_graph_start_us = ggml_time_us();
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    ggml_backend_rpc_snapshot_stats rpc_snapshot_stats_begin {};
    ggml_backend_rpc_get_snapshot_stats_t rpc_snapshot_stats_getter = nullptr;
    bool have_rpc_snapshot_stats = false;
    if (n_backends > 1) {
        rpc_snapshot_stats_getter =
            ggml_backend_meta_get_snapshot_stats_getter(
                backend_ctx->backend_configs[1].backend);
        if (rpc_snapshot_stats_getter != nullptr) {
            have_rpc_snapshot_stats =
                rpc_snapshot_stats_getter(
                    backend_ctx->backend_configs[1].backend,
                    &rpc_snapshot_stats_begin);
        }
    }

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);
    const int64_t meta_rebuild_begin_us = needs_rebuild ? ggml_time_us() : 0;
    int64_t meta_rebuild_us = 0;

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            buf_ctx->stc_compute_index_next =
                (buf_ctx->stc_compute_index + 1) % ggml_backend_meta_buffer_context::STC_COMPUTE_COUNT;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->stc_compute[buf_ctx->stc_compute_index_next];

            if (std::getenv("GGML_META_STC_DEBUG") != nullptr) {
                printf("[META_STC_ROTATE] uid=%" PRIu64 " buf=%p cur=%d next=%d clear_entries=%zu first=%s last=%s\n",
                       cgraph->uid, (void *) buf, buf_ctx->stc_compute_index, buf_ctx->stc_compute_index_next,
                       stc.simple_tensors.size(), cgraph->n_nodes > 0 ? cgraph->nodes[0]->name : "(none)",
                       cgraph->n_nodes > 0 ? cgraph->nodes[cgraph->n_nodes - 1]->name : "(none)");
            }

            for (ggml_context_ptr & ctx : stc.ctxs) {
                ggml_reset(ctx.get());
            }
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                    bcj.nodes[i] = node;
                    continue;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                if (!bcj.nodes[i]) {
                    fprintf(stderr,
                            "[META_MISSING_SIMPLE] j=%zu i=%d name=%s op=%s tensor=%p buffer=%p view_src=%p "
                            "view_offs=%zu uid=%" PRIu64 "\n",
                            j, i, node->name, ggml_op_name(node->op), (void *) node, (void *) node->buffer,
                            (void *) node->view_src, node->view_offs, cgraph->uid);
                }
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        if (n_backends == 2 &&
                backend_ctx->tensor_phone_first_layer >= 0) {
            bool in_phone_router = false;
            int phone_router_layer = -1;

            for (int i = 0; i < cgraph->n_nodes; ++i) {
                ggml_tensor * node = cgraph->nodes[i];

                int parsed_layer = -1;
                int parsed_chars = 0;
                const bool router_begin =
                    std::sscanf(node->name, "ffn_moe_logits-%d%n",
                                &parsed_layer, &parsed_chars) == 1 &&
                    node->name[parsed_chars] == '\0' &&
                    parsed_layer >= backend_ctx->tensor_phone_first_layer &&
                    parsed_layer < backend_ctx->tensor_phone_last_layer;

                if (router_begin) {
                    in_phone_router = true;
                    phone_router_layer = parsed_layer;
                }

                if (in_phone_router) {
                    backend_ctx->backend_configs[0].nodes[i]->flags &=
                        ~GGML_TENSOR_FLAG_COMPUTE;
                    backend_ctx->backend_configs[1].nodes[i]->flags |=
                        GGML_TENSOR_FLAG_COMPUTE;
                }

                int route_chunk = -1;
                int route_layer = -1;
                if (in_phone_router &&
                        ggml_backend_meta_parse_phone_route_weights(
                            node->name, route_chunk, route_layer) &&
                        route_layer == phone_router_layer) {
                    if (std::getenv("GGML_META_PIPELINE_DEBUG") != nullptr) {
                        printf(
                            "[TENSOR_PHONE_ROUTER_OWNER] layer=%d chunk=%d "
                            "begin=ffn_moe_logits end=%s owner=PHONE\n",
                            route_layer, route_chunk, node->name);
                    }
                    in_phone_router = false;
                    phone_router_layer = -1;
                }
            }
        }

        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_tensor * node = cgraph->nodes[i];
            const bool is_decode_result_norm =
                node->ne[1] == 1 &&
                std::strncmp(node->name, "result_norm", 11) == 0;
            if (!is_decode_result_norm) {
                continue;
            }
            for (size_t j = 1; j < n_backends; ++j) {
                backend_ctx->backend_configs[j].nodes[i]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }

        // TENSOR_PHONE_PRIMARY keeps the activation/control path on Phone:
        // Attention -> residual (ffn_inp) -> FFN norm -> Router.  PC joins only
        // after routing is complete, where it receives the normalized hidden
        // state plus the tiny top-k/weight tensors used by the expert FFN.
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (n_backends != 2 ||
                    backend_ctx->tensor_phone_first_layer < 0) {
                break;
            }

            int layer = -1;
            int parsed = 0;
            const char * name = cgraph->nodes[i]->name;
            const bool is_ffn_inp =
                std::sscanf(name, "ffn_inp-%d%n", &layer, &parsed) == 1 &&
                name[parsed] == '\0';

            if (!is_ffn_inp) {
                layer = -1;
                parsed = 0;
                const bool is_ffn_norm =
                    std::sscanf(name, "ffn_norm-%d%n", &layer, &parsed) == 1 &&
                    name[parsed] == '\0';
                if (!is_ffn_norm) {
                    continue;
                }
            }

            if (layer < backend_ctx->tensor_phone_first_layer ||
                    layer >= backend_ctx->tensor_phone_last_layer) {
                continue;
            }

            backend_ctx->backend_configs[0].nodes[i]->flags &=
                ~GGML_TENSOR_FLAG_COMPUTE;
            backend_ctx->backend_configs[1].nodes[i]->flags |=
                GGML_TENSOR_FLAG_COMPUTE;

            if (std::getenv("GGML_META_PIPELINE_DEBUG") != nullptr) {
                printf(
                    "[TENSOR_PHONE_CONTROL_OWNER] layer=%d node=%s owner=PHONE\n",
                    layer, name);
            }
        }

        // Phone-primary single-owner mode keeps every Tensor-layer l_out on
        // Phone, including the terminal Tensor layer.  If the following region
        // also runs on Phone, the activation stays local.  If the following
        // region needs PC, the ordinary single-active-backend boundary copy
        // below transfers the completed terminal l_out exactly once.
        const bool phone_primary_single_owner =
            std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER") != nullptr;
        if (n_backends == 2 &&
                backend_ctx->tensor_phone_first_layer >= 0) {
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                int layer = -1;
                int parsed = 0;
                const char * name = cgraph->nodes[i]->name;
                if (std::sscanf(name, "l_out-%d%n", &layer, &parsed) != 1 ||
                        name[parsed] != '\0' ||
                        layer < backend_ctx->tensor_phone_first_layer ||
                        layer >= backend_ctx->tensor_phone_last_layer) {
                    continue;
                }

                if (phone_primary_single_owner) {
                    backend_ctx->backend_configs[0].nodes[i]->flags &=
                        ~GGML_TENSOR_FLAG_COMPUTE;
                } else {
                    backend_ctx->backend_configs[0].nodes[i]->flags |=
                        GGML_TENSOR_FLAG_COMPUTE;
                }
                backend_ctx->backend_configs[1].nodes[i]->flags |=
                    GGML_TENSOR_FLAG_COMPUTE;

                if (std::getenv("GGML_META_PIPELINE_DEBUG") != nullptr) {
                    if (phone_primary_single_owner) {
                        printf(
                            "[TENSOR_PHONE_V23_L_OUT_OWNER] layer=%d node=%s "
                            "owner=PHONE scope=ALL\n",
                            layer, name);
                    } else {
                        printf(
                            "[TENSOR_PHONE_V2_L_OUT_MIRROR] layer=%d node=%s "
                            "compute={PC,PHONE} scope=BASELINE\n",
                            layer, name);
                    }
                }
            }
        }

        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_tensor * node = cgraph->nodes[i];
            const bool is_decode_result_output =
                node->ne[1] == 1 &&
                std::strncmp(node->name, "result_output", 13) == 0;
            if (!is_decode_result_output) {
                continue;
            }

            const ggml_backend_meta_split_state split_state =
                ggml_backend_meta_get_split_state(node, /* assume_sync = */ false);
            if (split_state.axis < GGML_BACKEND_SPLIT_AXIS_0 ||
                    split_state.axis > GGML_BACKEND_SPLIT_AXIS_3) {
                continue;
            }

            for (size_t j = 0; j < n_backends; ++j) {
                ggml_tensor * simple = backend_ctx->backend_configs[j].nodes[i];
                if (simple != nullptr && ggml_nelements(simple) > 0) {
                    simple->flags |= GGML_TENSOR_FLAG_COMPUTE;
                }
            }
        }

        for (int i = 0; i < cgraph->n_nodes; ++i) {
            int chunk = -1;
            int layer = -1;
            if (!ggml_backend_meta_parse_prefill_norm_chunk(
                    cgraph->nodes[i]->name, chunk, layer)) {
                continue;
            }

            const bool phone_primary_norm =
                n_backends == 2 &&
                backend_ctx->tensor_phone_first_layer >= 0 &&
                layer >= backend_ctx->tensor_phone_first_layer &&
                layer < backend_ctx->tensor_phone_last_layer;

            if (phone_primary_norm) {
                backend_ctx->backend_configs[0].nodes[i]->flags &=
                    ~GGML_TENSOR_FLAG_COMPUTE;
                backend_ctx->backend_configs[1].nodes[i]->flags |=
                    GGML_TENSOR_FLAG_COMPUTE;
            } else {
                for (size_t j = 1; j < n_backends; ++j) {
                    backend_ctx->backend_configs[j].nodes[i]->flags &=
                        ~GGML_TENSOR_FLAG_COMPUTE;
                }
            }
        }

        // Wavefront l_out is a real PC graph node.  Its static position is now
        // immediately before Attention(L+1,C), so executing it in the normal
        // PC graph preserves allocator lifetime and ordering.  Secondary
        // backends keep the mirrored placeholder but never compute this ADD.
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            int chunk = -1;
            int layer = -1;
            if (!ggml_backend_meta_parse_prefill_wave_l_out_chunk(
                    cgraph->nodes[i]->name, chunk, layer)) {
                continue;
            }
            for (size_t j = 1; j < n_backends; ++j) {
                backend_ctx->backend_configs[j].nodes[i]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                size_t attn_active_count = 0;
size_t attn_active_backend = SIZE_MAX;

for (size_t j = 0; j < n_backends; ++j) {
    if (backend_ctx->backend_configs[j].nodes[id]->flags &
            GGML_TENSOR_FLAG_COMPUTE) {
        ++attn_active_count;
        attn_active_backend = j;
    }
}

const bool pc_only_attn =
    std::strncmp(node->name, "attn_out-", 9) == 0 &&
    attn_active_count == 1 &&
    attn_active_backend == 0;

const bool decode_pc_only_attn = pc_only_attn && node->ne[1] == 1;
const bool prefill_pc_only_attn = pc_only_attn && node->ne[1] > 1;

if (decode_pc_only_attn || prefill_pc_only_attn) {
    std::set<const ggml_tensor *> delayed_nodes = { node };

    for (int ii = id + 1; ii < cgraph->n_nodes; ++ii) {
        ggml_tensor * next = cgraph->nodes[ii];
        bool depends_on_delayed = false;

        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (next->src[s] == nullptr) {
                continue;
            }

            if (delayed_nodes.count(next->src[s]) != 0) {
                depends_on_delayed = true;
                continue;
            }

            if (!ggml_backend_meta_tensor_is_mirrored(next->src[s], false)) {
                return i;
            }
        }

        if (depends_on_delayed) {
            delayed_nodes.insert(next);
        }

        if (decode_pc_only_attn && std::strncmp(next->name, "ffn_norm-", 9) == 0) {
            return depends_on_delayed ? ii : i;
        }

        int chunk = -1;
        int layer = -1;
        if (prefill_pc_only_attn &&
                ggml_backend_meta_parse_prefill_norm_chunk(next->name, chunk, layer) &&
                chunk == 0) {
            return depends_on_delayed ? ii : i;
        }

        if (next->buffer == nullptr || !ggml_backend_buffer_is_meta(next->buffer)) {
            return i;
        }
        if (ggml_backend_meta_get_split_state(next, false).axis ==
                GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return i;
        }
    }

    return i;
}

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        int next_prefill_chunk = -1;
                        int next_prefill_layer = -1;
                        if (ggml_backend_meta_parse_prefill_norm_chunk(
                                next->name, next_prefill_chunk, next_prefill_layer)) {
                            break;
                        }
                        if (!ggml_backend_meta_tensor_is_mirrored(next, false)) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (!ggml_backend_meta_tensor_is_mirrored(next->src[s], false)) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            next->src[1] != nullptr && next->src[1]->buffer != nullptr &&
                            ggml_backend_buffer_is_meta(next->src[1]->buffer) &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_tensor_is_mirrored(next->src[2], false)) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_tensor_is_mirrored(next->src[1], false)) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            auto is_passthrough_host_view = [](const ggml_tensor * node) -> bool {
                return node->view_src != nullptr &&
                       node->view_src->op == GGML_OP_NONE &&
                       node->view_src->buffer != nullptr &&
                       ggml_backend_buffer_is_host(node->view_src->buffer);
            };

            // DAG prefill creates token-range views of host inputs (for example
            // position slices). Such VIEW nodes are no-op/pass-through nodes for
            // the Meta split planner and are intentionally skipped below. The
            // old end-of-graph test used the literal last cgraph node, however,
            // so a trailing host VIEW could prevent the final subgraph from ever
            // being closed and leave i_start < cgraph->n_nodes.
            int last_meta_node = cgraph->n_nodes - 1;
            while (last_meta_node >= 0 &&
                   is_passthrough_host_view(cgraph->nodes[last_meta_node])) {
                --last_meta_node;
            }
            GGML_ASSERT(last_meta_node >= 0);

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_passthrough_host_view(node)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }
                const bool is_decode_result_norm =
                    node->ne[1] == 1 &&
                    std::strncmp(node->name, "result_norm", 11) == 0;
                int prefill_chunk = -1;
                int prefill_layer = -1;
                const bool is_prefill_norm_chunk =
                    ggml_backend_meta_parse_prefill_norm_chunk(
                        node->name, prefill_chunk, prefill_layer);

                int phone_route_chunk = -1;
                int phone_route_layer = -1;
                const bool is_phone_route_boundary =
                    ggml_backend_meta_parse_phone_route_weights(
                        node->name, phone_route_chunk, phone_route_layer);

                int phone_primary_l_out_layer = -1;
                int phone_primary_l_out_parsed = 0;
                const bool is_phone_primary_l_out =
                    n_backends == 2 &&
                    backend_ctx->tensor_phone_first_layer >= 0 &&
                    std::sscanf(
                        node->name, "l_out-%d%n",
                        &phone_primary_l_out_layer,
                        &phone_primary_l_out_parsed) == 1 &&
                    node->name[phone_primary_l_out_parsed] == '\0' &&
                    phone_primary_l_out_layer >=
                        backend_ctx->tensor_phone_first_layer &&
                    phone_primary_l_out_layer <
                        backend_ctx->tensor_phone_last_layer;

                const bool new_subgraph =
                    i == last_meta_node ||
                    split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL ||
                    is_decode_result_norm ||
                    is_prefill_norm_chunk ||
                    is_phone_route_boundary ||
                    is_phone_primary_l_out;
                if (!new_subgraph) {
                    continue;
                }

                const int i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }

            // The last simple-backend subgraph already extends to
            // cgraph->n_nodes when it is materialized below, so any trailing
            // pass-through host VIEWs are naturally kept in that final graph.
            // They must be the only nodes left after the final real Meta node.
            for (int i = i_start; i < cgraph->n_nodes; ++i) {
                GGML_ASSERT(is_passthrough_host_view(cgraph->nodes[i]));
            }
            i_start = cgraph->n_nodes;
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 5 * backend_ctx->n_reduce_steps; // tmp + reduce + residual + layer + output view
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_graphs_early = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, false);
            const size_t mem_graphs_phone_fused = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, false);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux +
                                                mem_per_device_graphs_early + mem_per_device_nodes_aux) +
                                      mem_graphs_phone_fused,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < n_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), cgraph->n_nodes, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->cgraphs_early.resize(n_backends*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_early.size(); k++) {
                backend_ctx->cgraphs_early[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, false);
            }
            backend_ctx->cgraphs_phone_fused.resize(backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_phone_fused.size(); ++k) {
                backend_ctx->cgraphs_phone_fused[k] =
                    ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, false);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_node_stop - i_node_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                if (cgraph->uid != 0) {
                    uint64_t uid = cgraph->uid;
                    uid ^= (uint64_t) (j + 1)       * 0x9e3779b97f4a7c15ULL;
                    uid ^= (uint64_t) (i_graph + 1) * 0xbf58476d1ce4e5b9ULL;
                    uid ^= uid >> 30;
                    uid *= 0xbf58476d1ce4e5b9ULL;
                    uid ^= uid >> 27;
                    uid *= 0x94d049bb133111ebULL;
                    uid ^= uid >> 31;
                    cgraph_ij->uid = uid != 0 ? uid : ggml_graph_next_uid();
                } else {
                    cgraph_ij->uid = ggml_graph_next_uid();
                }
            }
        }
    }
    if (needs_rebuild) {
        meta_rebuild_us = ggml_time_us() - meta_rebuild_begin_us;
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        GGML_ASSERT(ina < backend_ctx->nodes_aux.size());
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    auto set_prefill_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t lane) {
        GGML_ASSERT(lane < ggml_backend_meta_context::PREFILL_RETURN_LANES);
        auto & bcj = backend_ctx->backend_configs[j];
        auto & buf_ptr = bcj.prefill_reduce_bufs[lane];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        GGML_ASSERT(iga < backend_ctx->cgraphs_aux.size());
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    int64_t reduce_copy_wait_us = 0;
    int64_t reduce_add_us       = 0;
    int64_t reduce_zero_us      = 0;
    int64_t reduce_comm_us      = 0;
    size_t  reduce_count        = 0;
    size_t  reduce_comm_count   = 0;
    size_t  reduce_fallback_count = 0;
    size_t  direct_copy_count   = 0;
    size_t  reduce_to_primary_count = 0;
    size_t  reduce_zero_copy_skips = 0;
    int64_t reduce_max_us       = 0;
    struct reduce_copy_stats {
        size_t   count    = 0;
        uint64_t bytes    = 0;
        int64_t  total_us = 0;
        int64_t  max_us   = 0;
    };
    std::vector<reduce_copy_stats> reduce_copy_by_direction(n_backends*n_backends);
    const bool pipeline_debug = std::getenv("GGML_META_PIPELINE_DEBUG") != nullptr;
    const bool phone_exit_profile =
        std::getenv("GGML_META_PHONE_EXIT_PROFILE") != nullptr;
    const bool meta_sg_timing =
        std::getenv("GGML_META_SG_TIMING") != nullptr;
    const bool meta_timing_debug =
        pipeline_debug || std::getenv("GGML_META_TIMING_DEBUG") != nullptr;
    const bool return_path_debug = std::getenv("GGML_RETURN_PATH_DEBUG") != nullptr;
    size_t reduce_copy_detail_count = 0;
    int64_t pipeline_submit_sum_us = 0;
    int64_t pipeline_gap_sum_us    = 0;
    int64_t pipeline_gap_max_us    = 0;
    size_t  pipeline_gap_count     = 0;
    int64_t tensor_attn_us         = 0;
    int64_t tensor_pc_ffn_us       = 0;
    int64_t tensor_phone_us        = 0;
    int64_t tensor_wait_us         = 0;
    int64_t lane_reuse_wait_count  = 0;
    int64_t lane_reuse_wait_us     = 0;
    int64_t lane_reuse_wait_max_us = 0;
    std::array<int64_t, ggml_backend_meta_context::PREFILL_RETURN_LANES>
        lane_reuse_wait_count_by_lane {};
    std::array<int64_t, ggml_backend_meta_context::PREFILL_RETURN_LANES>
        lane_reuse_wait_us_by_lane {};
    int64_t layer_barrier_wait_count  = 0;
    int64_t layer_barrier_wait_us     = 0;
    int64_t layer_barrier_wait_max_us = 0;
    int64_t layer_barrier_wait_max_layer = -1;
    int64_t layer_barrier_wait_max_pending = 0;
    int64_t layer_barrier_wait_max_last_lane = -1;
    struct pipeline_gap_timing {
        bool valid = false;

        size_t subgraph = SIZE_MAX;

        int64_t pc_start_us  = 0;
        int64_t pc_submit_us = 0;
    };
    pipeline_gap_timing pipeline_gap;
    std::mutex meta_copy_stats_mutex;

    auto record_copy_wait = [&](int64_t copy_us) {
        std::lock_guard<std::mutex> lock(meta_copy_stats_mutex);
        reduce_copy_wait_us += copy_us;
    };

    auto record_reduce_add = [&](int64_t add_us, bool to_primary) {
        std::lock_guard<std::mutex> lock(meta_copy_stats_mutex);
        reduce_add_us += add_us;
        if (to_primary) {
            ++reduce_to_primary_count;
        }
    };

    auto record_direct_copy = [&]() {
        std::lock_guard<std::mutex> lock(meta_copy_stats_mutex);
        ++direct_copy_count;
    };

    auto has_copy_detail_budget = [&]() {
        std::lock_guard<std::mutex> lock(meta_copy_stats_mutex);
        return reduce_copy_detail_count < 4;
    };

    auto record_meta_copy = [&](size_t i, size_t j_src, size_t j_dst, const ggml_tensor * node_src, int64_t copy_us) {
        std::lock_guard<std::mutex> lock(meta_copy_stats_mutex);
        const size_t copy_bytes = ggml_nbytes(node_src);
        auto & stats = reduce_copy_by_direction[j_src*n_backends + j_dst];
        ++stats.count;
        stats.bytes += copy_bytes;
        stats.total_us += copy_us;
        stats.max_us = std::max(stats.max_us, copy_us);

        const bool is_result_norm =
            std::strncmp(node_src->name, "result_norm", 11) == 0;
        if (pipeline_debug && (reduce_copy_detail_count < 4 || is_result_norm)) {
            printf("[META_COPY] sg=%zu %zu->%zu tensor=%s bytes=%zu "
                   "ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] time=%.3f ms\n",
                   i, j_src, j_dst, node_src->name, copy_bytes,
                   node_src->ne[0], node_src->ne[1], node_src->ne[2], node_src->ne[3],
                   copy_us / 1000.0);
            if (!is_result_norm) {
                ++reduce_copy_detail_count;
            }
        }
    };

    struct meta_snapshot_prepare {
        bool prepared = false;
        uint32_t slot = 0;
        uint64_t seq = 0;
    };
    std::vector<meta_snapshot_prepare> snapshot_prepares(backend_ctx->n_subgraphs);

    auto get_ffn_down_boundary_node = [&](size_t j, size_t sg) -> ggml_tensor * {
        if (j >= n_backends || sg >= backend_ctx->n_subgraphs) {
            return nullptr;
        }

        ggml_cgraph * graph = backend_ctx->backend_configs[j].cgraphs[sg].cgraph_main;
        if (graph == nullptr || graph->n_nodes == 0) {
            return nullptr;
        }

        ggml_tensor * node = graph->nodes[graph->n_nodes - 1];
        int chunk = -1;
        int layer = -1;
        if (!ggml_backend_meta_parse_decode_ffn_chunk(node->name, chunk, layer) ||
                !(node->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            return nullptr;
        }

        return node;
    };

    auto get_prefill_down_boundary_node = [&](size_t j, size_t sg) -> ggml_tensor * {
        if (j >= n_backends || sg >= backend_ctx->n_subgraphs) {
            return nullptr;
        }

        ggml_cgraph * graph = backend_ctx->backend_configs[j].cgraphs[sg].cgraph_main;
        if (graph == nullptr || graph->n_nodes == 0) {
            return nullptr;
        }

        ggml_tensor * node = graph->nodes[graph->n_nodes - 1];
        int chunk = -1;
        int layer = -1;
        if (!ggml_backend_meta_parse_prefill_down_chunk(node->name, chunk, layer) ||
                !(node->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            return nullptr;
        }

        return node;
    };

    auto get_early_graph = [&](size_t j, size_t sg) -> ggml_cgraph * {
        GGML_ASSERT(j < n_backends);
        GGML_ASSERT(sg < backend_ctx->max_subgraphs);
        return backend_ctx->cgraphs_early[j*backend_ctx->max_subgraphs + sg];
    };

    auto prepare_graph_snapshot = [&](size_t sg, ggml_tensor * tensor) {
        if (sg >= snapshot_prepares.size() || snapshot_prepares[sg].prepared) {
            return;
        }

        auto & bc_phone = backend_ctx->backend_configs[1];
        const auto prepare = ggml_backend_meta_get_graph_snapshot_preparer(bc_phone.backend);
        const auto snapshot_arm = ggml_backend_meta_get_snapshot_arm(bc_phone.backend);
        const auto set_snapshot_read = ggml_backend_meta_get_snapshot_read_setter(bc_phone.backend);
        if (prepare == nullptr || snapshot_arm == nullptr || set_snapshot_read == nullptr) {
            return;
        }

        const uint64_t seq = backend_ctx->next_snapshot_seq++;
        const uint32_t slot = seq & 1;
        const bool prepared = prepare(
            bc_phone.backend,
            tensor,
            0,
            ggml_nbytes(tensor),
            slot,
            seq);
        GGML_ASSERT(prepared);

        snapshot_prepares[sg].prepared = true;
        snapshot_prepares[sg].slot = slot;
        snapshot_prepares[sg].seq = seq;
    };

    uint64_t pending_prefill_input_task = 0;
    int pending_prefill_input_layer = -1;
    int pending_prefill_input_chunk = -1;

    struct phone_prefill_route_task {
        size_t lane = 0;
        uint64_t task = 0;
        uint64_t producer_seq = 0;
        ggml_backend_meta_transfer_worker * worker = nullptr;

        // Original Phone graph inputs.
        ggml_tensor * src_hidden = nullptr;
        ggml_tensor * src_topk = nullptr;
        ggml_tensor * src_weights = nullptr;

        // Optional private Phone copies.  When present, route transfer reads
        // these stable buffers and the local Phone FFN is restored/submitted
        // later at the layer barrier.
        ggml_tensor * phone_stage_hidden = nullptr;
        ggml_tensor * phone_stage_topk = nullptr;
        ggml_tensor * phone_stage_weights = nullptr;

        // Private PC-side route staging.
        ggml_tensor * stage_hidden = nullptr;
        ggml_tensor * stage_topk = nullptr;
        ggml_tensor * stage_weights = nullptr;
        ggml_tensor * dst_hidden = nullptr;
        ggml_tensor * dst_topk = nullptr;
        ggml_tensor * dst_weights = nullptr;
    };
    std::map<std::pair<int, int>, phone_prefill_route_task>
        pending_phone_prefill_routes;
    std::array<uint64_t, ggml_backend_meta_context::PREFILL_ROUTE_LANES>
        pending_phone_prefill_route_lane_task { 0, 0 };

    struct phone_prefill_lane1_return_gate {
        std::mutex mutex;
        std::condition_variable cv;
        bool open = false;
        bool cancelled = false;
    };
    std::map<int, std::shared_ptr<phone_prefill_lane1_return_gate>>
        phone_prefill_lane1_return_gates;

    // If graph execution exits early while a lane-1 return is waiting for the
    // last route handoff, wake it so worker destruction cannot deadlock.
    struct phone_prefill_lane1_return_gate_guard {
        std::map<int, std::shared_ptr<phone_prefill_lane1_return_gate>> & gates;

        ~phone_prefill_lane1_return_gate_guard() {
            for (auto & entry : gates) {
                const auto & gate = entry.second;
                if (gate == nullptr) {
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lock(gate->mutex);
                    gate->cancelled = true;
                    gate->open = true;
                }
                gate->cv.notify_all();
            }
        }
    } phone_prefill_lane1_return_gate_guard_instance {
        phone_prefill_lane1_return_gates
    };

    struct phone_prefill_pc_branch {
        int layer = -1;
        int chunk = -1;
        size_t sg = 0;
        uint64_t pc_task = 0;
        uint64_t return_task = 0;
        size_t return_lane = 0;
        uint64_t phone_ffn_seq = 0;
        ggml_tensor * return_stage = nullptr;
    };
    std::deque<phone_prefill_pc_branch> pending_phone_prefill_pc_branches;

    struct phone_prefill_deferred_phone_branch {
        int layer = -1;
        int chunk = -1;
        size_t sg = 0;
        ggml_tensor * src_hidden = nullptr;
        ggml_tensor * src_topk = nullptr;
        ggml_tensor * src_weights = nullptr;
        ggml_tensor * stage_hidden = nullptr;
        ggml_tensor * stage_topk = nullptr;
        ggml_tensor * stage_weights = nullptr;
    };
    std::deque<phone_prefill_deferred_phone_branch>
        pending_phone_prefill_phone_branches;

    std::map<size_t, int> deferred_phone_prefill_return_sgs;
    std::array<uint64_t, ggml_backend_meta_context::PREFILL_RETURN_LANES> pending_prefill_reduce_task { 0, 0 };
    std::array<int, ggml_backend_meta_context::PREFILL_RETURN_LANES> pending_prefill_reduce_layer { -1, -1 };
    std::array<int, ggml_backend_meta_context::PREFILL_RETURN_LANES> pending_prefill_reduce_chunk { -1, -1 };
    std::array<ggml_backend_meta_transfer_worker *, ggml_backend_meta_context::PREFILL_RETURN_LANES>
        pending_prefill_reduce_worker { nullptr, nullptr };
    auto has_pending_prefill_reduce = [&]() {
        return std::any_of(pending_prefill_reduce_task.begin(), pending_prefill_reduce_task.end(),
                           [](uint64_t task) { return task != 0; });
    };
    auto has_pending_prefill_reduce_for_layer = [&](int layer) {
        for (size_t lane = 0; lane < ggml_backend_meta_context::PREFILL_RETURN_LANES; ++lane) {
            if (pending_prefill_reduce_task[lane] != 0 && pending_prefill_reduce_layer[lane] == layer) {
                return true;
            }
        }
        return false;
    };
    auto wait_prefill_reduce_lane = [&](size_t lane) {
        if (pending_prefill_reduce_task[lane] == 0) {
            return GGML_STATUS_SUCCESS;
        }
        GGML_ASSERT(pending_prefill_reduce_worker[lane] != nullptr);
        const int64_t wait_start_us = ggml_time_us();
        const ggml_status status =
            pending_prefill_reduce_worker[lane]->wait(pending_prefill_reduce_task[lane]);
        tensor_wait_us += ggml_time_us() - wait_start_us;
        pending_prefill_reduce_task[lane]   = 0;
        pending_prefill_reduce_layer[lane]  = -1;
        pending_prefill_reduce_chunk[lane]  = -1;
        pending_prefill_reduce_worker[lane] = nullptr;
        return status;
    };

    auto wait_prefill_reduce_dependency = [&](int layer, int chunk, bool & waited) {
        waited = false;
        for (size_t lane = 0; lane < ggml_backend_meta_context::PREFILL_RETURN_LANES; ++lane) {
            if (pending_prefill_reduce_task[lane] == 0 ||
                pending_prefill_reduce_layer[lane] != layer ||
                pending_prefill_reduce_chunk[lane] != chunk) {
                continue;
            }
            waited = true;
            return wait_prefill_reduce_lane(lane);
        }
        return GGML_STATUS_SUCCESS;
    };
    auto wait_prefill_reduces_before = [&](int layer_exclusive, bool & waited, int64_t & wait_us) {
        waited = false;
        wait_us = 0;
        ggml_status result = GGML_STATUS_SUCCESS;
        for (size_t lane = 0; lane < ggml_backend_meta_context::PREFILL_RETURN_LANES; ++lane) {
            if (pending_prefill_reduce_task[lane] == 0 ||
                pending_prefill_reduce_layer[lane] >= layer_exclusive) {
                continue;
            }
            waited = true;
            const int64_t begin_us = ggml_time_us();
            const ggml_status status = wait_prefill_reduce_lane(lane);
            wait_us += ggml_time_us() - begin_us;
            if (result == GGML_STATUS_SUCCESS && status != GGML_STATUS_SUCCESS) {
                result = status;
            }
        }
        return result;
    };
    auto wait_all_prefill_reduces = [&](int * last_lane) {
        ggml_status result = GGML_STATUS_SUCCESS;
        int64_t max_lane_wait_us = -1;
        if (last_lane != nullptr) {
            *last_lane = -1;
        }
        for (size_t lane = 0; lane < ggml_backend_meta_context::PREFILL_RETURN_LANES; ++lane) {
            const bool measure_lane = last_lane != nullptr && pending_prefill_reduce_task[lane] != 0;
            const int64_t lane_wait_start_us = measure_lane ? ggml_time_us() : 0;
            const ggml_status status = wait_prefill_reduce_lane(lane);
            if (measure_lane) {
                const int64_t lane_wait_us = ggml_time_us() - lane_wait_start_us;
                if (lane_wait_us > max_lane_wait_us) {
                    max_lane_wait_us = lane_wait_us;
                    *last_lane = (int) lane;
                }
            }
            if (result == GGML_STATUS_SUCCESS && status != GGML_STATUS_SUCCESS) {
                result = status;
            }
        }
        return result;
    };
    int deferred_phone_exit_layer = -1;
    ggml_tensor * debug_terminal_handoff_dst = nullptr;
    auto debug_handoff_watch = [&](const char * tag) {
        if (!pipeline_debug || debug_terminal_handoff_dst == nullptr) {
            return;
        }

        printf(
            "[HANDOFF_WATCH] %s tensor=%p data=%p buffer=%p "
            "view_src=%p view_offs=%zu\n",
            tag, (void *) debug_terminal_handoff_dst,
            debug_terminal_handoff_dst->data,
            (void *) debug_terminal_handoff_dst->buffer,
            (void *) debug_terminal_handoff_dst->view_src,
            debug_terminal_handoff_dst->view_offs);
        meta_debug_tensor(
            backend_ctx->backend_configs[0].backend,
            debug_terminal_handoff_dst, tag);
    };
    auto subgraph_is_prefill_pc_only_attn = [&](size_t sg) -> bool {
    if (n_backends != 2 || sg >= backend_ctx->n_subgraphs) {
        return false;
    }

    auto * pc_graph =
        backend_ctx->backend_configs[0].cgraphs[sg].cgraph_main;
    auto * phone_graph =
        backend_ctx->backend_configs[1].cgraphs[sg].cgraph_main;

    if (pc_graph == nullptr || phone_graph == nullptr ||
            pc_graph->n_nodes == 0 || phone_graph->n_nodes == 0) {
        return false;
    }

    ggml_tensor * pc_last =
        pc_graph->nodes[pc_graph->n_nodes - 1];

    ggml_tensor * phone_last =
        phone_graph->nodes[phone_graph->n_nodes - 1];

    return
        pc_last->ne[1] > 1 &&
        std::strncmp(pc_last->name, "attn_out-", 9) == 0 &&
        (pc_last->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 &&
        (phone_last->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
};
    auto subgraph_is_prefill_pc_only_ffn = [&](size_t sg) -> bool {
    if (sg == 0 || n_backends != 2 ||
            sg >= backend_ctx->n_subgraphs) {
        return false;
    }

    if (!subgraph_is_prefill_pc_only_attn(sg - 1)) {
        return false;
    }

    auto * pc_graph =
        backend_ctx->backend_configs[0].cgraphs[sg].cgraph_main;
    auto * phone_graph =
        backend_ctx->backend_configs[1].cgraphs[sg].cgraph_main;

    if (pc_graph == nullptr || phone_graph == nullptr ||
            pc_graph->n_nodes == 0 || phone_graph->n_nodes == 0) {
        return false;
    }

    ggml_tensor * pc_last =
        pc_graph->nodes[pc_graph->n_nodes - 1];

    ggml_tensor * phone_last =
        phone_graph->nodes[phone_graph->n_nodes - 1];

    return
        (pc_last->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 &&
        (phone_last->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
};

    auto subgraph_has_compute = [&](size_t backend, size_t sg) -> bool {
        if (backend >= n_backends || sg >= backend_ctx->n_subgraphs) {
            return false;
        }

        ggml_cgraph * graph = backend_ctx->backend_configs[backend].cgraphs[sg].cgraph_main;
        if (graph == nullptr) {
            return false;
        }

        for (int k = 0; k < graph->n_nodes; ++k) {
            if (graph->nodes[k]->flags & GGML_TENSOR_FLAG_COMPUTE) {
                return true;
            }
        }
        return false;
    };

    auto subgraph_is_prefill_norm_pc_only = [&](size_t sg) -> bool {
        if (n_backends != 2 || sg >= backend_ctx->n_subgraphs) {
            return false;
        }

        ggml_cgraph * pc_graph = backend_ctx->backend_configs[0].cgraphs[sg].cgraph_main;
        ggml_cgraph * phone_graph = backend_ctx->backend_configs[1].cgraphs[sg].cgraph_main;
        if (pc_graph == nullptr || phone_graph == nullptr ||
                pc_graph->n_nodes == 0 || phone_graph->n_nodes == 0) {
            return false;
        }

        ggml_tensor * pc_last = pc_graph->nodes[pc_graph->n_nodes - 1];
        ggml_tensor * phone_last = phone_graph->nodes[phone_graph->n_nodes - 1];
        int pc_chunk = -1;
        int pc_layer = -1;
        int phone_chunk = -1;
        int phone_layer = -1;
        return ggml_backend_meta_parse_prefill_norm_chunk(
                   pc_last->name, pc_chunk, pc_layer) &&
               ggml_backend_meta_parse_prefill_norm_chunk(
                   phone_last->name, phone_chunk, phone_layer) &&
               pc_chunk == phone_chunk && pc_layer == phone_layer;
    };

    auto subgraph_is_decode_pc_only_norm = [&](size_t sg) -> bool {
        if (n_backends != 2 || sg >= backend_ctx->n_subgraphs) {
            return false;
        }

        ggml_cgraph * pc_graph = backend_ctx->backend_configs[0].cgraphs[sg].cgraph_main;
        ggml_cgraph * phone_graph = backend_ctx->backend_configs[1].cgraphs[sg].cgraph_main;
        if (pc_graph == nullptr || phone_graph == nullptr ||
                pc_graph->n_nodes == 0 || phone_graph->n_nodes == 0) {
            return false;
        }

        ggml_tensor * pc_last = pc_graph->nodes[pc_graph->n_nodes - 1];
        ggml_tensor * phone_last = phone_graph->nodes[phone_graph->n_nodes - 1];
        if (pc_last->ne[1] != 1) {
            return false;
        }

        const bool pc_is_norm =
            std::strncmp(pc_last->name, "ffn_norm-", 9) == 0 ||
            std::strncmp(pc_last->name, "result_norm", 11) == 0;
        const bool phone_is_norm =
            std::strncmp(phone_last->name, "ffn_norm-", 9) == 0 ||
            std::strncmp(phone_last->name, "result_norm", 11) == 0;
        return pc_is_norm && phone_is_norm &&
               (pc_last->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 &&
               (phone_last->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
    };
    auto subgraph_is_decode_pc_only_ffn = [&](size_t sg) -> bool {
        if (sg == 0 || n_backends != 2 || sg >= backend_ctx->n_subgraphs) {
            return false;
        }

        if (!subgraph_is_decode_pc_only_norm(sg - 1)) {
            return false;
        }

        ggml_cgraph * pc_graph = backend_ctx->backend_configs[0].cgraphs[sg].cgraph_main;
        ggml_cgraph * phone_graph = backend_ctx->backend_configs[1].cgraphs[sg].cgraph_main;
        if (pc_graph == nullptr || phone_graph == nullptr ||
                pc_graph->n_nodes == 0 || phone_graph->n_nodes == 0) {
            return false;
        }

        ggml_tensor * pc_last = pc_graph->nodes[pc_graph->n_nodes - 1];
        ggml_tensor * phone_last = phone_graph->nodes[phone_graph->n_nodes - 1];
        return pc_last->ne[1] == 1 &&
               (pc_last->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 &&
               (phone_last->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
    };
    auto subgraph_is_prefill_pc_only = [&](size_t sg) -> bool {
    return
        subgraph_is_prefill_pc_only_attn(sg) ||
        subgraph_is_prefill_pc_only_ffn(sg);
};
    auto subgraph_will_execute_phone = [&](size_t sg) -> bool {
    if (subgraph_is_prefill_pc_only(sg) ||
            subgraph_is_prefill_norm_pc_only(sg) ||
            subgraph_is_decode_pc_only_norm(sg) ||
            subgraph_is_decode_pc_only_ffn(sg)) {
        return false;
    }

    return subgraph_has_compute(1, sg);
};
    auto subgraph_last_has_compute = [&](size_t backend, size_t sg) -> bool {
        if (backend >= n_backends || sg >= backend_ctx->n_subgraphs) {
            return false;
        }

        ggml_cgraph * graph =
            backend_ctx->backend_configs[backend].cgraphs[sg].cgraph_main;
        if (graph == nullptr || graph->n_nodes == 0) {
            return false;
        }

        ggml_tensor * last = graph->nodes[graph->n_nodes - 1];
        return (last->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
    };
    auto subgraph_is_phone_owned = [&](size_t sg) -> bool {
        return n_backends == 2 &&
               !subgraph_last_has_compute(0, sg) &&
               subgraph_last_has_compute(1, sg);
    };
    auto subgraph_will_execute_pc = [&](size_t sg) -> bool {
        if (subgraph_is_phone_owned(sg)) {
            return false;
        }
        return subgraph_has_compute(0, sg);
    };
    auto subgraph_is_phone_only = [&](size_t sg) -> bool {
        return subgraph_is_phone_owned(sg);
    };
    std::map<int, bool> layer_phone_primary_cache;
    auto layer_attention_phone_owned = [&](int layer) -> bool {
        if (n_backends != 2 || layer < 0) {
            return false;
        }
        if (backend_ctx->tensor_phone_first_layer >= 0 &&
            layer >= backend_ctx->tensor_phone_first_layer &&
            layer < backend_ctx->tensor_phone_last_layer) {
            return true;
        }

        const auto cached = layer_phone_primary_cache.find(layer);
        if (cached != layer_phone_primary_cache.end()) {
            return cached->second;
        }

        char attn_name[64];
        char residual_name[64];
        std::snprintf(attn_name, sizeof(attn_name), "attn_out-%d", layer);
        std::snprintf(residual_name, sizeof(residual_name), "ffn_inp-%d", layer);

        bool marker_seen = false;
        bool pc_compute = false;
        bool phone_compute = false;

        for (size_t sg = 0; sg < backend_ctx->n_subgraphs; ++sg) {
            for (size_t backend = 0; backend < 2; ++backend) {
                ggml_cgraph * graph =
                    backend_ctx->backend_configs[backend].cgraphs[sg].cgraph_main;
                if (graph == nullptr) {
                    continue;
                }
                for (int k = 0; k < graph->n_nodes; ++k) {
                    ggml_tensor * node = graph->nodes[k];
                    if (std::strcmp(node->name, attn_name) != 0 &&
                        std::strcmp(node->name, residual_name) != 0) {
                        continue;
                    }

                    marker_seen = true;
                    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                        continue;
                    }
                    if (backend == 0) {
                        pc_compute = true;
                    } else {
                        phone_compute = true;
                    }
                }
            }
        }

        const bool phone_primary =
            marker_seen && !pc_compute && phone_compute;
        layer_phone_primary_cache[layer] = phone_primary;
        return phone_primary;
    };

    std::map<int, bool> layer_tensor_phone_primary_cache;
    auto layer_is_tensor_phone_primary = [&](int layer) -> bool {
        if (backend_ctx->tensor_phone_first_layer >= 0 &&
            layer >= backend_ctx->tensor_phone_first_layer &&
            layer < backend_ctx->tensor_phone_last_layer) {
            return true;
        }
        if (n_backends != 2 || layer < 0 ||
            !layer_attention_phone_owned(layer)) {
            return false;
        }

        const auto cached =
            layer_tensor_phone_primary_cache.find(layer);
        if (cached != layer_tensor_phone_primary_cache.end()) {
            return cached->second;
        }

        std::set<std::pair<bool, int>> pc_chunks;
        std::set<std::pair<bool, int>> phone_chunks;
        for (size_t sg = 0; sg < backend_ctx->n_subgraphs; ++sg) {
            for (size_t backend = 0; backend < 2; ++backend) {
                ggml_cgraph * graph =
                    backend_ctx->backend_configs[backend].cgraphs[sg].cgraph_main;
                if (graph == nullptr) {
                    continue;
                }
                for (int k = 0; k < graph->n_nodes; ++k) {
                    ggml_tensor * node = graph->nodes[k];
                    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                        continue;
                    }
                    int chunk = -1;
                    int parsed_layer = -1;
                    const bool prefill = ggml_backend_meta_parse_prefill_down_chunk(
                        node->name, chunk, parsed_layer);
                    if (!prefill && !ggml_backend_meta_parse_decode_ffn_chunk(
                            node->name, chunk, parsed_layer)) {
                        continue;
                    }
                    if (parsed_layer == layer) {
                        (backend == 0 ? pc_chunks : phone_chunks).insert({prefill, chunk});
                    }
                }
            }
        }

        bool split_ffn_seen = false;
        for (const auto & chunk : pc_chunks) {
            if (phone_chunks.count(chunk) != 0) {
                split_ffn_seen = true;
                break;
            }
        }

        layer_tensor_phone_primary_cache[layer] = split_ffn_seen;
        return split_ffn_seen;
    };

    auto subgraph_tensor_phone_primary_layer =
        [&](size_t sg, int & layer) -> bool {
            layer = -1;
            if (sg >= backend_ctx->n_subgraphs) {
                return false;
            }

            for (size_t backend = 0;
                 backend < n_backends;
                 ++backend) {
                ggml_cgraph * graph =
                    backend_ctx->backend_configs[backend]
                        .cgraphs[sg].cgraph_main;
                if (graph == nullptr) {
                    continue;
                }

                for (int k = 0; k < graph->n_nodes; ++k) {
                    ggml_tensor * node = graph->nodes[k];
                    int parsed_layer = -1;
                    int chunk = -1;

                    const bool parsed =
                        std::sscanf(
                            node->name, "attn_out-%d",
                            &parsed_layer) == 1 ||
                        std::sscanf(
                            node->name, "ffn_inp-%d",
                            &parsed_layer) == 1 ||
                        std::sscanf(
                            node->name, "ffn_norm-%d",
                            &parsed_layer) == 1 ||
                        ggml_backend_meta_parse_prefill_norm_chunk(
                            node->name, chunk, parsed_layer) ||
                        ggml_backend_meta_parse_prefill_down_chunk(
                            node->name, chunk, parsed_layer) ||
                        ggml_backend_meta_parse_decode_ffn_chunk(
                            node->name, chunk, parsed_layer);

                    if (parsed &&
                        layer_is_tensor_phone_primary(parsed_layer)) {
                        layer = parsed_layer;
                        return true;
                    }
                }
            }

            return false;
        };

    auto prefill_layer_hands_off_to_phone = [&](size_t sg, int layer) -> bool {
        if (layer_is_tensor_phone_primary(layer)) {
            return false;
        }
        if (layer_attention_phone_owned(layer)) {
            return true;
        }
        for (size_t next = sg + 1; next < backend_ctx->n_subgraphs; ++next) {
            ggml_cgraph * graph = backend_ctx->backend_configs[0].cgraphs[next].cgraph_main;
            if (graph == nullptr || graph->n_nodes == 0) {
                continue;
            }

            ggml_tensor * last = graph->nodes[graph->n_nodes - 1];
            int next_chunk = -1;
            int next_layer = -1;
            const bool is_prefill_chunk =
                ggml_backend_meta_parse_prefill_norm_chunk(last->name, next_chunk, next_layer) ||
                ggml_backend_meta_parse_prefill_down_chunk(last->name, next_chunk, next_layer);
            if (is_prefill_chunk) {
                if (next_layer > layer) {
                    return false;
                }
                continue;
            }

            if (subgraph_is_phone_only(next)) {
                return true;
            }
            if (subgraph_is_prefill_pc_only(next)) {
                return false;
            }
        }
        return false;
    };
    auto prefill_is_last_down_chunk = [&](size_t sg, int layer) -> bool {
        for (size_t next = sg + 1; next < backend_ctx->n_subgraphs; ++next) {
            ggml_cgraph * graph = backend_ctx->backend_configs[0].cgraphs[next].cgraph_main;
            if (graph == nullptr || graph->n_nodes == 0) {
                continue;
            }

            ggml_tensor * last = graph->nodes[graph->n_nodes - 1];
            int next_chunk = -1;
            int next_layer = -1;
            if (ggml_backend_meta_parse_prefill_down_chunk(
                    last->name, next_chunk, next_layer)) {
                return next_layer != layer;
            }
            if (subgraph_is_phone_only(next) || subgraph_is_prefill_pc_only(next)) {
                return true;
            }
        }
        return true;
    };
    auto decode_layer_hands_off_to_phone = [&](size_t sg, int layer) -> bool {
        if (layer_is_tensor_phone_primary(layer)) {
            return false;
        }
        if (layer_attention_phone_owned(layer)) {
            return true;
        }
        for (size_t next = sg + 1; next < backend_ctx->n_subgraphs; ++next) {
            ggml_cgraph * graph = backend_ctx->backend_configs[0].cgraphs[next].cgraph_main;
            if (graph == nullptr || graph->n_nodes == 0) {
                continue;
            }

            ggml_tensor * last = graph->nodes[graph->n_nodes - 1];
            int next_chunk = -1;
            int next_layer = -1;
            if (ggml_backend_meta_parse_decode_ffn_chunk(
                    last->name, next_chunk, next_layer)) {
                if (next_layer > layer) {
                    return false;
                }
                continue;
            }
            if (subgraph_is_phone_only(next)) {
                return true;
            }
            if (subgraph_is_decode_pc_only_norm(next)) {
                return false;
            }
        }
        return false;
    };
    auto decode_is_last_down_chunk = [&](size_t sg, int layer) -> bool {
        for (size_t next = sg + 1; next < backend_ctx->n_subgraphs; ++next) {
            ggml_cgraph * graph = backend_ctx->backend_configs[0].cgraphs[next].cgraph_main;
            if (graph == nullptr || graph->n_nodes == 0) {
                continue;
            }

            int next_chunk = -1;
            int next_layer = -1;
            ggml_tensor * last = graph->nodes[graph->n_nodes - 1];
            if (ggml_backend_meta_parse_decode_ffn_chunk(
                    last->name, next_chunk, next_layer)) {
                return next_layer != layer;
            }
            if (subgraph_is_phone_only(next) || subgraph_is_decode_pc_only_norm(next)) {
                return true;
            }
        }
        return true;
    };
    auto find_recent_ffn_inp = [&](size_t backend, size_t sg, int layer) -> ggml_tensor * {
        if (backend >= n_backends || sg >= backend_ctx->n_subgraphs) {
            return nullptr;
        }

        char expected[64];
        std::snprintf(expected, sizeof(expected), "ffn_inp-%d", layer);

        for (int64_t s = (int64_t) sg; s >= 0; --s) {
            ggml_cgraph * graph =
                backend_ctx->backend_configs[backend].cgraphs[(size_t) s].cgraph_main;
            if (graph == nullptr) {
                continue;
            }
            for (int k = graph->n_nodes - 1; k >= 0; --k) {
                ggml_tensor * tensor = graph->nodes[k];
                if (std::strcmp(tensor->name, expected) == 0) {
                    return tensor;
                }
            }
        }
        return nullptr;
    };

    auto find_recent_ffn_inp_layer = [&](size_t backend, size_t sg, int & layer) -> ggml_tensor * {
        if (backend >= n_backends || sg >= backend_ctx->n_subgraphs) {
            return nullptr;
        }

        for (int64_t s = (int64_t) sg; s >= 0; --s) {
            ggml_cgraph * graph =
                backend_ctx->backend_configs[backend].cgraphs[(size_t) s].cgraph_main;
            if (graph == nullptr) {
                continue;
            }
            for (int k = graph->n_nodes - 1; k >= 0; --k) {
                ggml_tensor * tensor = graph->nodes[k];
                int parsed_layer = -1;
                if (std::sscanf(tensor->name, "ffn_inp-%d", &parsed_layer) == 1) {
                    layer = parsed_layer;
                    return tensor;
                }
            }
        }
        return nullptr;
    };
    auto find_layer_output = [&](size_t backend, int layer) -> ggml_tensor * {
        if (backend >= n_backends) {
            return nullptr;
        }
        for (size_t sg = 0; sg < backend_ctx->n_subgraphs; ++sg) {
            ggml_cgraph * graph = backend_ctx->backend_configs[backend].cgraphs[sg].cgraph_main;
            if (graph == nullptr) {
                continue;
            }
            for (int k = 0; k < graph->n_nodes; ++k) {
                int parsed_layer = -1;
                if (std::sscanf(graph->nodes[k]->name, "l_out-%d", &parsed_layer) == 1 &&
                        parsed_layer == layer) {
                    return graph->nodes[k];
                }
            }
        }
        return nullptr;
    };
    auto find_exact_named_tensor = [&](size_t backend, const char * expected) -> ggml_tensor * {
        if (backend >= n_backends) {
            return nullptr;
        }

        for (int64_t sg = (int64_t) backend_ctx->n_subgraphs - 1; sg >= 0; --sg) {
            ggml_cgraph * graph =
                backend_ctx->backend_configs[backend].cgraphs[(size_t) sg].cgraph_main;
            if (graph == nullptr) {
                continue;
            }
            for (int k = graph->n_nodes - 1; k >= 0; --k) {
                ggml_tensor * tensor = graph->nodes[k];
                if (tensor != nullptr && std::strcmp(tensor->name, expected) == 0) {
                    return tensor;
                }
            }
            for (int k = graph->n_leafs - 1; k >= 0; --k) {
                ggml_tensor * tensor = graph->leafs[k];
                if (tensor != nullptr && std::strcmp(tensor->name, expected) == 0) {
                    return tensor;
                }
            }
        }
        return nullptr;
    };
    auto debug_layer_input = [&](size_t backend, size_t sg, int layer, bool after_compute) {
        if (!pipeline_debug || backend >= n_backends || sg >= backend_ctx->n_subgraphs) {
            return;
        }

        ggml_cgraph * graph = backend_ctx->backend_configs[backend].cgraphs[sg].cgraph_main;
        if (graph == nullptr) {
            return;
        }

        char expected[64];
        std::snprintf(expected, sizeof(expected), "norm-%d", layer);
        for (int k = 0; k < graph->n_nodes; ++k) {
            ggml_tensor * norm = graph->nodes[k];
            if (std::strcmp(norm->name, expected) != 0 ||
                    !(norm->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                continue;
            }

            ggml_tensor * src0 = norm->src[0];
            if (src0 == nullptr) {
                continue;
            }

            // At the first Phone-primary Tensor layer, prove that the tensor
            // consumed by Phone is exactly the preceding PC layer output.
            // Unlike the ordinary per-token input trace, this also runs for
            // prefill matrices (ne[1] > 1).
            const bool trace_tp_entry =
                std::getenv("GGML_META_TP_FFN_NUMERIC_TRACE") != nullptr &&
                !after_compute &&
                backend == 1 &&
                layer == backend_ctx->tensor_phone_first_layer &&
                src0->type == GGML_TYPE_F32;
            if (trace_tp_entry) {
                char prev_name[64];
                std::snprintf(
                    prev_name, sizeof(prev_name),
                    "l_out-%d", layer - 1);
                ggml_tensor * pc_prev =
                    find_exact_named_tensor(0, prev_name);

                if (pc_prev != nullptr &&
                        pc_prev->type == GGML_TYPE_F32 &&
                        ggml_nelements(pc_prev) == ggml_nelements(src0)) {
                    auto fence_for_trace = [&](ggml_backend_t b) {
                        const ggml_backend_rpc_fence_t rpc_fence =
                            ggml_backend_meta_get_rpc_fence(b);
                        if (rpc_fence != nullptr) {
                            rpc_fence(b);
                        } else {
                            ggml_backend_synchronize(b);
                        }
                    };

                    auto & pc_cfg = backend_ctx->backend_configs[0];
                    auto & ph_cfg = backend_ctx->backend_configs[1];
                    fence_for_trace(pc_cfg.backend);
                    fence_for_trace(ph_cfg.backend);

                    const size_t n = ggml_nelements(src0);
                    std::vector<float> pc_values(n);
                    std::vector<float> phone_values(n);
                    ggml_backend_tensor_get(
                        pc_prev, pc_values.data(), 0, n * sizeof(float));
                    ggml_backend_tensor_get(
                        src0, phone_values.data(), 0, n * sizeof(float));

                    double pc_sum = 0.0;
                    double phone_sum = 0.0;
                    double pc_l2 = 0.0;
                    double phone_l2 = 0.0;
                    double diff_l2 = 0.0;
                    double max_abs_diff = 0.0;
                    size_t max_diff_index = 0;
                    for (size_t q = 0; q < n; ++q) {
                        const double a = pc_values[q];
                        const double b = phone_values[q];
                        const double d = a - b;
                        pc_sum += a;
                        phone_sum += b;
                        pc_l2 += a * a;
                        phone_l2 += b * b;
                        diff_l2 += d * d;
                        if (std::abs(d) > max_abs_diff) {
                            max_abs_diff = std::abs(d);
                            max_diff_index = q;
                        }
                    }

                    printf(
                        "[TP_ENTRY_NUMERIC] layer=%d pc=%s phone=%s n=%zu "
                        "pc_sum=%.9f phone_sum=%.9f pc_l2=%.9f phone_l2=%.9f "
                        "max_abs_diff=%.9g rms_diff=%.9g max_diff_index=%zu "
                        "pc_v0=%.9f phone_v0=%.9f "
                        "pc_v1=%.9f phone_v1=%.9f\n",
                        layer, pc_prev->name, src0->name, n,
                        pc_sum, phone_sum,
                        std::sqrt(pc_l2), std::sqrt(phone_l2),
                        max_abs_diff,
                        n > 0 ? std::sqrt(diff_l2 / (double) n) : 0.0,
                        max_diff_index,
                        n > 0 ? pc_values[0] : 0.0f,
                        n > 0 ? phone_values[0] : 0.0f,
                        n > 1 ? pc_values[1] : 0.0f,
                        n > 1 ? phone_values[1] : 0.0f);
                } else {
                    printf(
                        "[TP_ENTRY_NUMERIC] layer=%d status=SKIP "
                        "pc_prev=%s phone_src=%s\n",
                        layer,
                        pc_prev != nullptr ? pc_prev->name : "(null)",
                        src0->name);
                }
            }

            if (src0->ne[1] != 1) {
                continue;
            }

            int producer_idx = -1;
            for (int p = 0; p < graph->n_nodes; ++p) {
                if (graph->nodes[p] == src0) {
                    producer_idx = p;
                    break;
                }
            }
            const bool producer_in_sg =
                producer_idx >= 0 &&
                (src0->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;

            printf(
                "[META_INPUT%s] layer=%d norm=%s src0=%s "
                "src0_ptr=%p src0_buf=%p bytes=%zu op=%s "
                "producer_in_sg=%d producer_idx=%d\n",
                after_compute ? "_POST" : "",
                layer, norm->name, src0->name,
                (void *) src0, (void *) src0->buffer, ggml_nbytes(src0),
                ggml_op_name(src0->op), (int) producer_in_sg, producer_idx);

            // A source produced inside this same subgraph has not run yet when
            // the PRE trace executes. Reading it here reports stale/zero arena
            // contents and can also perturb OpenCL/RPC timing. Defer only those
            // sources until immediately after this subgraph completes.
            if (!after_compute && producer_in_sg) {
                printf(
                    "[NUMDBG_DEFER] layer=%d backend=%zu tensor=%s "
                    "reason=producer_in_current_sg producer_idx=%d\n",
                    layer, backend, src0->name, producer_idx);
                return;
            }
            if (after_compute != producer_in_sg) {
                return;
            }

            char tag[96];
            if (after_compute) {
                std::snprintf(
                    tag, sizeof(tag),
                    "TP layer%d norm input POST backend%zu",
                    layer, backend);
            } else {
                std::snprintf(
                    tag, sizeof(tag),
                    "TP layer%d norm input backend%zu",
                    layer, backend);
            }
            meta_debug_tensor(backend_ctx->backend_configs[backend].backend, src0, tag);
            return;
        }
    };
    auto disable_layer_output_producers = [&](size_t backend, size_t start_sg, int layer) -> ggml_tensor * {
        if (backend >= n_backends) {
            return nullptr;
        }

        char expected[64];
        std::snprintf(expected, sizeof(expected), "l_out-%d", layer);
        for (size_t sg = start_sg; sg < backend_ctx->n_subgraphs; ++sg) {
            ggml_cgraph * graph = backend_ctx->backend_configs[backend].cgraphs[sg].cgraph_main;
            if (graph == nullptr) {
                continue;
            }
            for (int k = 0; k < graph->n_nodes; ++k) {
                ggml_tensor * tensor = graph->nodes[k];
                if (std::strcmp(tensor->name, expected) != 0) {
                    continue;
                }
                for (int disabled = 0; disabled <= k; ++disabled) {
                    graph->nodes[disabled]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                }
                if (pipeline_debug) {
                    printf(
                        "[DISABLE_L_OUT_PRODUCER] sg=%zu layer=%d "
                        "through=%d tensor=%s\n",
                        sg, layer, k, tensor->name);
                }
                return tensor;
            }
        }
        return nullptr;
    };
    auto has_layer_output_producer = [&](size_t backend, size_t start_sg, int layer) -> bool {
        if (backend >= n_backends || layer < 0) {
            return false;
        }
        char expected[64];
        std::snprintf(expected, sizeof(expected), "l_out-%d", layer);
        for (size_t sg = start_sg; sg < backend_ctx->n_subgraphs; ++sg) {
            ggml_cgraph * graph = backend_ctx->backend_configs[backend].cgraphs[sg].cgraph_main;
            if (graph == nullptr) {
                continue;
            }
            for (int k = 0; k < graph->n_nodes; ++k) {
                if (std::strcmp(graph->nodes[k]->name, expected) == 0) {
                    return true;
                }
            }
        }
        return false;
    };
    auto subgraph_is_exact_l_out_bridge = [&](size_t sg, int layer) -> bool {
        if (n_backends != 2 || sg >= backend_ctx->n_subgraphs) {
            return false;
        }

        char expected[64];
        std::snprintf(expected, sizeof(expected), "l_out-%d", layer);
        for (size_t backend = 0; backend < n_backends; ++backend) {
            ggml_cgraph * graph =
                backend_ctx->backend_configs[backend].cgraphs[sg].cgraph_main;
            if (graph == nullptr || graph->n_nodes != 1 ||
                    std::strcmp(graph->nodes[0]->name, expected) != 0) {
                return false;
            }
        }
        return true;
    };
    auto find_down_chunk = [&](size_t backend, bool prefill, int layer, int chunk) -> ggml_tensor * {
        if (backend >= n_backends) {
            return nullptr;
        }
        for (size_t sg = 0; sg < backend_ctx->n_subgraphs; ++sg) {
            ggml_cgraph * graph = backend_ctx->backend_configs[backend].cgraphs[sg].cgraph_main;
            if (graph == nullptr) {
                continue;
            }
            for (int k = 0; k < graph->n_nodes; ++k) {
                int parsed_chunk = -1;
                int parsed_layer = -1;
                const bool parsed = prefill ?
                    ggml_backend_meta_parse_prefill_down_chunk(
                        graph->nodes[k]->name, parsed_chunk, parsed_layer) :
                    ggml_backend_meta_parse_decode_ffn_chunk(
                        graph->nodes[k]->name, parsed_chunk, parsed_layer);
                if (parsed && parsed_layer == layer && parsed_chunk == chunk) {
                    return graph->nodes[k];
                }
            }
        }
        return nullptr;
    };
    auto down_chunk_offset = [&](size_t backend, bool prefill, int layer, int chunk) -> size_t {
        size_t offset = 0;
        for (int current = 0; current < chunk; ++current) {
            ggml_tensor * tensor = find_down_chunk(backend, prefill, layer, current);
            GGML_ASSERT(tensor != nullptr);
            offset += ggml_nbytes(tensor);
        }
        return offset;
    };

    bool return_wavefront_graph = false;
    int  return_wavefront_first_layer = std::numeric_limits<int>::max();
    if (n_backends == 2) {
        for (size_t sg = 0; sg < backend_ctx->n_subgraphs; ++sg) {
            ggml_cgraph * graph = backend_ctx->backend_configs[0].cgraphs[sg].cgraph_main;
            if (graph == nullptr) {
                continue;
            }
            for (int node_id = 0; node_id < graph->n_nodes; ++node_id) {
                int chunk = -1;
                int layer = -1;
                if (ggml_backend_meta_parse_prefill_wave_ffn_inp_chunk(
                        graph->nodes[node_id]->name, chunk, layer)) {
                    return_wavefront_graph = true;
                    return_wavefront_first_layer = std::min(return_wavefront_first_layer, layer);
                }
            }
        }
    }

    int64_t return_wave_dependency_wait_count  = 0;
    int64_t return_wave_dependency_wait_us     = 0;
    int64_t return_wave_dependency_wait_max_us = 0;
    int64_t return_wave_overlap_boundaries     = 0;
    int64_t return_wave_ahead_attn_chunks      = 0;
    int64_t return_wave_ahead_phone_submits    = 0;
    int64_t return_wave_phone_credit_wait_count = 0;
    int64_t return_wave_phone_credit_wait_us    = 0;
    int64_t return_wave_phone_credit_wait_max_us = 0;
    std::deque<uint64_t> return_wave_phone_credit_seqs;
    ggml_backend_rpc_wait_snapshot_ready_t return_wave_snapshot_ready_waiter =
        n_backends > 1 ?
        ggml_backend_meta_get_snapshot_ready_waiter(backend_ctx->backend_configs[1].backend) :
        nullptr;
    int64_t return_wave_old_layer_wait_count   = 0;
    int64_t return_wave_old_layer_wait_us      = 0;
    std::map<int, int64_t> return_wave_layer_start_us;
    std::map<int, int64_t> return_wave_layer_compute_wall_us;
    std::map<int, int64_t> return_wave_layer_barrier_us;

    // Pure-observation Phone-primary stage profiler.  This intentionally does
    // not add synchronization, graph submissions, tensor reads, or ownership
    // changes.  It only consumes timing values that the normal path already
    // produces, plus ggml_time_us() wall-clock stamps around existing calls.
    const bool tensor_phone_stage_profile =
        std::getenv("GGML_META_TENSOR_PHONE_STAGE_PROFILE") != nullptr;
    const bool tensor_expert_load_profile =
        std::getenv("GGML_META_TENSOR_EXPERT_LOAD_PROFILE") != nullptr;
    const bool phone_prefill_chunk_pipeline =
        std::getenv("GGML_META_PHONE_PREFILL_CHUNK_PIPELINE") != nullptr;
    const bool phone_prefill_defer_phone_ffn =
        std::getenv("GGML_META_PHONE_PREFILL_DEFER_PHONE_FFN") != nullptr;
    const bool phone_prefill_async_return =
        std::getenv("GGML_META_PHONE_PREFILL_ASYNC_RETURN") != nullptr;
    const bool phone_prefill_producer_route =
        std::getenv("GGML_META_PHONE_PREFILL_PRODUCER_ROUTE") != nullptr;
    const bool phone_prefill_ordered_return =
        std::getenv("GGML_META_PHONE_PREFILL_ORDERED_RETURN") != nullptr;
    const bool phone_prefill_chunk_join =
        std::getenv("GGML_META_PHONE_PREFILL_CHUNK_JOIN") != nullptr;
    const bool phone_prefill_route_lane_swap =
        std::getenv("GGML_RPC_ROUTE_LANE_SWAP") != nullptr;

    auto phone_prefill_route_lane_for_chunk =
        [&](int chunk) -> size_t {
            GGML_ASSERT(chunk >= 0);
            size_t lane =
                static_cast<size_t>(chunk) %
                ggml_backend_meta_context::PREFILL_ROUTE_LANES;
            if (phone_prefill_route_lane_swap && lane < 2) {
                lane ^= size_t(1);
            }
            return lane;
        };

    if ((pipeline_debug || tensor_phone_stage_profile) &&
            phone_prefill_route_lane_swap) {
        printf(
            "[PHONE_PREFILL_ROUTE_LANE_MAP] swap=1 "
            "chunk_even=1 chunk_odd=0\n");
    }

    // Fine-grained Phone-primary correctness fences for A/B isolation.
    // The legacy STRICT_FENCE remains an umbrella and preserves its older
    // conservative behavior. The split switches only add their own fence.
    const bool phone_primary_strict_all =
        std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_STRICT_FENCE") != nullptr;
    const bool phone_primary_strict_route =
        phone_primary_strict_all ||
        std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_STRICT_ROUTE") != nullptr;
    const bool phone_primary_strict_ffn_handoff =
        phone_primary_strict_all ||
        std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_STRICT_FFN_HANDOFF") != nullptr;
    const bool phone_primary_strict_reduce =
        phone_primary_strict_all ||
        std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_STRICT_REDUCE") != nullptr;
    const ggml_backend_rpc_set_tensor_async_return_t
        phone_prefill_async_return_set =
            n_backends > 1 ?
            ggml_backend_meta_get_set_tensor_async_return(
                backend_ctx->backend_configs[1].backend) :
            nullptr;
    const bool phone_prefill_async_return_active =
        phone_prefill_async_return &&
        phone_prefill_async_return_set != nullptr &&
        phone_prefill_async_return_set(
            backend_ctx->backend_configs[1].backend,
            nullptr,
            nullptr,
            0);

    const bool phone_prefill_ordered_return_active =
        phone_prefill_ordered_return &&
        phone_prefill_async_return_active &&
        !phone_primary_strict_reduce;

    const ggml_backend_rpc_phone_ffn_mark_ready_t
        phone_prefill_ffn_mark_ready =
            n_backends > 1 ?
            ggml_backend_meta_get_phone_ffn_mark_ready(
                backend_ctx->backend_configs[1].backend) :
            nullptr;
    const ggml_backend_rpc_set_tensor_async_return_wait_t
        phone_prefill_async_return_wait =
            n_backends > 1 ?
            ggml_backend_meta_get_set_tensor_async_return_wait(
                backend_ctx->backend_configs[1].backend) :
            nullptr;

    const bool phone_prefill_chunk_join_active =
        phone_prefill_chunk_join &&
        phone_prefill_producer_route &&
        phone_prefill_ordered_return_active &&
        phone_prefill_ffn_mark_ready != nullptr &&
        phone_prefill_async_return_wait != nullptr &&
        phone_prefill_ffn_mark_ready(
            backend_ctx->backend_configs[1].backend,
            0) &&
        phone_prefill_async_return_wait(
            backend_ctx->backend_configs[1].backend,
            nullptr,
            nullptr,
            0,
            0,
            0);

    if (pipeline_debug && phone_prefill_async_return) {
        printf(
            "[PHONE_PREFILL_ASYNC_RETURN_CAP] "
            "requested=1 active=%d ordered=%d chunk_join=%d\n",
            phone_prefill_async_return_active ? 1 : 0,
            phone_prefill_ordered_return_active ? 1 : 0,
            phone_prefill_chunk_join_active ? 1 : 0);
    }

    struct phone_prefill_binding_backup {
        ggml_tensor * tensor = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        void * data = nullptr;
    };
    std::map<int, std::vector<phone_prefill_binding_backup>>
        phone_prefill_direct_bindings;
    std::map<std::pair<int, int>, uint64_t>
        phone_prefill_route_producer_seq;

    auto backup_phone_prefill_binding =
        [&](int layer, ggml_tensor * tensor) {
            GGML_ASSERT(tensor != nullptr);
            auto & bindings = phone_prefill_direct_bindings[layer];
            for (const auto & binding : bindings) {
                if (binding.tensor == tensor) {
                    return;
                }
            }
            bindings.push_back({
                tensor,
                tensor->buffer,
                tensor->data,
            });
        };

    auto rebind_phone_prefill_view =
        [&](int layer, ggml_tensor * view) {
            GGML_ASSERT(view != nullptr);
            GGML_ASSERT(view->view_src != nullptr);
            GGML_ASSERT(view->view_src->buffer != nullptr);
            GGML_ASSERT(view->view_src->data != nullptr);

            // This view is already initialized by the scheduler.  Calling
            // ggml_backend_view_init() again would assert because buffer/data
            // are non-null.  Preserve its original binding and rewrite only
            // the descriptor used for RPC serialization.
            backup_phone_prefill_binding(layer, view);
            view->buffer = view->view_src->buffer;
            view->data =
                static_cast<char *>(view->view_src->data) +
                view->view_offs;
        };

    auto bind_phone_prefill_storage =
        [&](int layer,
            ggml_tensor * tensor,
            ggml_backend_buffer_ptr & buf) {
            GGML_ASSERT(tensor != nullptr);
            ggml_tensor * storage =
                tensor->view_src != nullptr ?
                    tensor->view_src : tensor;

            backup_phone_prefill_binding(layer, storage);

            const size_t need = ggml_nbytes(storage);
            if (!buf ||
                    ggml_backend_buffer_get_size(buf.get()) < need) {
                buf.reset(
                    ggml_backend_alloc_buffer(
                        backend_ctx->backend_configs[1].backend,
                        need));
            }
            GGML_ASSERT(buf != nullptr);

            storage->buffer = buf.get();
            storage->data = ggml_backend_buffer_get_base(buf.get());

            if (tensor->view_src != nullptr) {
                rebind_phone_prefill_view(layer, tensor);
            }
        };

    auto restore_phone_prefill_direct_bindings =
        [&](int layer) {
            auto binding_it =
                phone_prefill_direct_bindings.find(layer);
            if (binding_it == phone_prefill_direct_bindings.end()) {
                return;
            }

            for (auto it = binding_it->second.rbegin();
                 it != binding_it->second.rend();
                 ++it) {
                it->tensor->buffer = it->buffer;
                it->tensor->data = it->data;
            }

            phone_prefill_direct_bindings.erase(binding_it);
        };

    auto phone_prefill_direct_defer_eligible =
        [&](int layer) {
            if (!phone_prefill_chunk_pipeline ||
                    !phone_prefill_defer_phone_ffn ||
                    phone_prefill_producer_route ||
                    n_backends != 2 ||
                    !layer_is_tensor_phone_primary(layer) ||
                    std::getenv(
                        "LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER") ==
                        nullptr ||
                    std::getenv(
                        "LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE") ==
                        nullptr ||
                    phone_primary_strict_all ||
                    std::getenv(
                        "GGML_META_TP_CONTROL_TRACE_LAYER") !=
                        nullptr) {
                return false;
            }

            if (find_down_chunk(0, true, layer, 1) == nullptr ||
                    find_down_chunk(1, true, layer, 1) == nullptr) {
                return false;
            }

            ggml_backend_t phone_backend =
                backend_ctx->backend_configs[1].backend;
            const ggml_backend_rpc_route_mark_ready_t mark_ready =
                ggml_backend_meta_get_route_mark_ready(phone_backend);
            return
                ggml_backend_meta_get_tensor_batch3(phone_backend) !=
                    nullptr &&
                ggml_backend_meta_get_stage_ready_setter(
                    phone_backend) != nullptr &&
                ggml_backend_meta_get_route_transfer_lane_setter(
                    phone_backend) != nullptr &&
                ggml_backend_meta_get_route_wait_seq_setter(
                    phone_backend) != nullptr &&
                mark_ready != nullptr &&
                mark_ready(phone_backend, 0);
        };

    auto bind_phone_prefill_hidden_direct =
        [&](int layer) -> bool {
            if (!phone_prefill_direct_defer_eligible(layer)) {
                return false;
            }

            char hidden_name[96];
            std::snprintf(
                hidden_name,
                sizeof(hidden_name),
                "prefill_ffn_norm_chunk_0-%d",
                layer);
            ggml_tensor * hidden0 =
                find_exact_named_tensor(1, hidden_name);
            if (hidden0 == nullptr) {
                return false;
            }

            auto & hidden_buf =
                backend_ctx->backend_configs[1].
                    prefill_phone_hidden_stage_buf;
            bind_phone_prefill_storage(
                layer,
                hidden0,
                hidden_buf);

            ggml_tensor * hidden_storage =
                hidden0->view_src != nullptr ?
                    hidden0->view_src : hidden0;
            for (int chunk = 0; chunk < 64; ++chunk) {
                std::snprintf(
                    hidden_name,
                    sizeof(hidden_name),
                    "prefill_ffn_norm_chunk_%d-%d",
                    chunk,
                    layer);
                ggml_tensor * hidden =
                    find_exact_named_tensor(1, hidden_name);
                if (hidden == nullptr) {
                    if (chunk > 0) {
                        break;
                    }
                    continue;
                }
                if (hidden->view_src == hidden_storage) {
                    rebind_phone_prefill_view(layer, hidden);
                }
            }

            return true;
        };

    auto bind_phone_prefill_route_outputs_direct =
        [&](int layer, int chunk) -> bool {
            if (!phone_prefill_direct_defer_eligible(layer) ||
                    chunk < 0) {
                return false;
            }

            char topk_name[96];
            char weights_name[96];
            std::snprintf(
                topk_name,
                sizeof(topk_name),
                "phone_prefill_route_topk_chunk_%d-%d",
                chunk,
                layer);
            std::snprintf(
                weights_name,
                sizeof(weights_name),
                "phone_prefill_route_weights_chunk_%d-%d",
                chunk,
                layer);

            ggml_tensor * topk =
                find_exact_named_tensor(1, topk_name);
            ggml_tensor * weights =
                find_exact_named_tensor(1, weights_name);
            if (topk == nullptr || weights == nullptr) {
                return false;
            }

            auto & slots =
                backend_ctx->backend_configs[1].
                    prefill_phone_ffn_stage_bufs;
            if (slots.size() <= static_cast<size_t>(chunk)) {
                slots.resize(static_cast<size_t>(chunk) + 1);
            }

            bind_phone_prefill_storage(
                layer,
                topk,
                slots[static_cast<size_t>(chunk)][1]);
            bind_phone_prefill_storage(
                layer,
                weights,
                slots[static_cast<size_t>(chunk)][2]);
            return true;
        };


    // Producer-driven Phone->PC route path.
    //
    // The PC route worker posts GET_ROUTE_SNAPSHOT(seq,lane) before the Phone
    // Router producer graph runs. The dedicated route socket waits on an
    // immutable server-side lane slot. When the Phone graph completes, Meta
    // synchronously captures hidden/top-k/weights into that slot and only then
    // publishes READY. The consumer never reads a live producer tensor.
    auto prearm_phone_prefill_route =
        [&](size_t sg, int layer, int chunk) -> bool {
            if (!phone_prefill_producer_route ||
                    !phone_prefill_chunk_pipeline ||
                    phone_primary_strict_route ||
                    chunk < 0 ||
                    n_backends != 2 ||
                    std::getenv(
                        "LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER") ==
                        nullptr ||
                    std::getenv(
                        "LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE") ==
                        nullptr) {
                return false;
            }

            if (pending_phone_prefill_routes.find(
                    { layer, chunk }) !=
                    pending_phone_prefill_routes.end()) {
                return true;
            }

            auto & bcj_src = backend_ctx->backend_configs[1];
            auto & bcj_dst = backend_ctx->backend_configs[0];

            const ggml_backend_rpc_get_route_snapshot_t
                route_get_snapshot =
                    ggml_backend_meta_get_route_snapshot(
                        bcj_src.backend);
            const ggml_backend_rpc_route_snapshot_ready_t
                route_snapshot_ready =
                    ggml_backend_meta_get_route_snapshot_ready(
                        bcj_src.backend);
            const ggml_backend_rpc_set_stage_ready_t route_set_stage_ready =
                ggml_backend_meta_get_stage_ready_setter(
                    bcj_src.backend);

            if (route_get_snapshot == nullptr ||
                    route_snapshot_ready == nullptr ||
                    route_set_stage_ready == nullptr ||
                    !route_snapshot_ready(
                        bcj_src.backend,
                        0,
                        0,
                        nullptr,
                        nullptr,
                        nullptr)) {
                return false;
            }

            char hidden_name[96];
            char topk_name[96];
            char weights_name[96];
            std::snprintf(
                hidden_name,
                sizeof(hidden_name),
                "prefill_ffn_norm_chunk_%d-%d",
                chunk,
                layer);
            std::snprintf(
                topk_name,
                sizeof(topk_name),
                "phone_prefill_route_topk_chunk_%d-%d",
                chunk,
                layer);
            std::snprintf(
                weights_name,
                sizeof(weights_name),
                "phone_prefill_route_weights_chunk_%d-%d",
                chunk,
                layer);

            ggml_tensor * src_hidden =
                find_exact_named_tensor(1, hidden_name);
            ggml_tensor * dst_hidden =
                find_exact_named_tensor(0, hidden_name);
            ggml_tensor * src_topk =
                find_exact_named_tensor(1, topk_name);
            ggml_tensor * dst_topk =
                find_exact_named_tensor(0, topk_name);
            ggml_tensor * src_weights =
                find_exact_named_tensor(1, weights_name);
            ggml_tensor * dst_weights =
                find_exact_named_tensor(0, weights_name);

            if (src_hidden == nullptr || dst_hidden == nullptr ||
                    src_topk == nullptr || dst_topk == nullptr ||
                    src_weights == nullptr || dst_weights == nullptr ||
                    !ggml_are_same_layout(src_hidden, dst_hidden) ||
                    !ggml_are_same_layout(src_topk, dst_topk) ||
                    !ggml_are_same_layout(src_weights, dst_weights)) {
                return false;
            }

            const size_t lane =
                phone_prefill_route_lane_for_chunk(chunk);
            auto & route_worker =
                backend_ctx->prefill_route_workers[lane];
            if (route_worker == nullptr) {
                route_worker =
                    new ggml_backend_meta_transfer_worker();
            }

            auto & stage_slots =
                backend_ctx->backend_configs[0].
                    prefill_route_stage_bufs;
            if (stage_slots.size() <=
                    static_cast<size_t>(chunk)) {
                stage_slots.resize(
                    static_cast<size_t>(chunk) + 1);
            }

            ggml_tensor * stage_hidden =
                get_node_aux(dst_hidden);
            ggml_tensor * stage_topk =
                get_node_aux(dst_topk);
            ggml_tensor * stage_weights =
                get_node_aux(dst_weights);

            ggml_tensor * stage_tensors[3] = {
                stage_hidden,
                stage_topk,
                stage_weights,
            };
            ggml_tensor * dst_tensors[3] = {
                dst_hidden,
                dst_topk,
                dst_weights,
            };

            auto & stage_bufs =
                stage_slots[static_cast<size_t>(chunk)];
            for (size_t stage_index = 0;
                    stage_index < 3;
                    ++stage_index) {
                const size_t need =
                    ggml_nbytes(dst_tensors[stage_index]);
                auto & buf = stage_bufs[stage_index];
                if (!buf ||
                        ggml_backend_buffer_get_size(buf.get()) <
                            need) {
                    buf.reset(
                        ggml_backend_alloc_buffer(
                            bcj_dst.backend,
                            need));
                }
                GGML_ASSERT(buf != nullptr);
                stage_tensors[stage_index]->buffer = buf.get();
                stage_tensors[stage_index]->data =
                    ggml_backend_buffer_get_base(buf.get());
            }

            const uint64_t producer_seq =
                backend_ctx->next_phone_prefill_route_seq++;
            ggml_backend_t route_src_backend = bcj_src.backend;
            ggml_backend_t route_dst_backend = bcj_dst.backend;

            const uint64_t route_task =
                route_worker->enqueue(
                    [&, route_worker, route_get_snapshot,
                        route_set_stage_ready, producer_seq,
                        src_hidden, stage_hidden,
                        src_topk, stage_topk,
                        src_weights, stage_weights,
                        dst_hidden, dst_topk, dst_weights,
                        route_src_backend, route_dst_backend,
                        lane, layer, chunk, sg]
                    (uint64_t task_id) -> ggml_status {
                        ggml_backend_meta_stage_ready_context
                            stage_context {
                                route_worker,
                                task_id,
                            };

                        route_set_stage_ready(
                            ggml_backend_meta_stage_ready,
                            &stage_context);

                        const int64_t route_begin_us =
                            ggml_time_us();
                        const bool used = route_get_snapshot(
                            route_src_backend,
                            route_dst_backend,
                            producer_seq,
                            static_cast<uint32_t>(lane),
                            stage_hidden,
                            stage_topk,
                            stage_weights);
                        const int64_t route_us =
                            ggml_time_us() -
                            route_begin_us;

                        route_set_stage_ready(nullptr, nullptr);

                        if (!used) {
                            return GGML_STATUS_FAILED;
                        }

                        record_copy_wait(route_us);
                        record_meta_copy(
                            sg, 1, 0, src_hidden, route_us);
                        record_meta_copy(
                            sg, 1, 0, src_topk, 0);
                        record_meta_copy(
                            sg, 1, 0, src_weights, 0);

                        if (pipeline_debug ||
                                tensor_phone_stage_profile) {
                            printf(
                                "[PHONE_PREFILL_ROUTE_MAILBOX_READY] "
                                "layer=%d chunk=%d lane=%zu "
                                "seq=%" PRIu64 " task=%" PRIu64
                                " total_ms=%.3f\n",
                                layer,
                                chunk,
                                lane,
                                producer_seq,
                                task_id,
                                route_us / 1000.0);
                        }
                        return GGML_STATUS_SUCCESS;
                    });

            pending_phone_prefill_route_lane_task[lane] =
                route_task;
            phone_prefill_route_producer_seq[
                { layer, chunk }] = producer_seq;
            pending_phone_prefill_routes[
                { layer, chunk }] = {
                    lane,
                    route_task,
                    producer_seq,
                    route_worker,
                    src_hidden,
                    src_topk,
                    src_weights,
                    nullptr,
                    nullptr,
                    nullptr,
                    stage_hidden,
                    stage_topk,
                    stage_weights,
                    dst_hidden,
                    dst_topk,
                    dst_weights,
                };

            if (pipeline_debug ||
                    tensor_phone_stage_profile) {
                printf(
                    "[PHONE_PREFILL_ROUTE_MAILBOX_ARM] "
                    "layer=%d chunk=%d lane=%zu seq=%" PRIu64
                    " task=%" PRIu64 " bytes=%zu\n",
                    layer,
                    chunk,
                    lane,
                    producer_seq,
                    route_task,
                    ggml_nbytes(src_hidden) +
                        ggml_nbytes(src_topk) +
                        ggml_nbytes(src_weights));
            }

            return true;
        };

    struct tensor_expert_load_accum {
        uint64_t assignments = 0;
        int64_t read_us = 0;
        std::vector<uint64_t> counts;
    };

    std::mutex tensor_expert_load_mutex;
    std::map<int, tensor_expert_load_accum> tensor_expert_load_by_layer;
    std::map<int, std::pair<int64_t, int64_t>> tensor_expert_shard_by_layer;

    auto tensor_expert_top_string =
        [](const std::vector<uint64_t> & counts, size_t limit) {
            std::vector<std::pair<uint64_t, size_t>> ranked;
            ranked.reserve(counts.size());
            for (size_t expert = 0; expert < counts.size(); ++expert) {
                if (counts[expert] != 0) {
                    ranked.push_back({ counts[expert], expert });
                }
            }
            std::sort(
                ranked.begin(), ranked.end(),
                [](const auto & a, const auto & b) {
                    if (a.first != b.first) {
                        return a.first > b.first;
                    }
                    return a.second < b.second;
                });
            if (ranked.size() > limit) {
                ranked.resize(limit);
            }

            std::string out;
            for (size_t i = 0; i < ranked.size(); ++i) {
                if (i != 0) {
                    out += ",";
                }
                out += std::to_string(ranked[i].second);
                out += ":";
                out += std::to_string(ranked[i].first);
            }
            return out;
        };

    auto tensor_moe_ffn_shard_width =
        [](const ggml_cgraph * graph) -> int64_t {
            if (graph == nullptr) {
                return -1;
            }
            for (int node_index = 0;
                 node_index < graph->n_nodes;
                 ++node_index) {
                const ggml_tensor * node = graph->nodes[node_index];
                if (node == nullptr ||
                        node->op != GGML_OP_MUL_MAT_ID ||
                        node->src[0] == nullptr) {
                    continue;
                }

                const ggml_tensor * weight = node->src[0];
                const char * name = weight->name;
                if (name == nullptr) {
                    continue;
                }
                if (std::strstr(name, "ffn_gate_exps.weight") != nullptr ||
                        std::strstr(name, "ffn_up_exps.weight") != nullptr) {
                    return weight->ne[1];
                }
                if (std::strstr(name, "ffn_down_exps.weight") != nullptr) {
                    return weight->ne[0];
                }
            }
            return -1;
        };

    struct tensor_phone_stage_timing {
        bool mode_set = false;
        bool decode = false;

        int64_t pre_route_wall_us = 0;
        int64_t pre_route_pc_worker_us = 0;
        int64_t pre_route_phone_worker_us = 0;

        int64_t route_handoff_us = 0;

        int64_t ffn_wall_us = 0;
        int64_t pc_ffn_worker_us = 0;
        int64_t phone_ffn_worker_us = 0;

        int64_t return_client_wall_us = 0;

        int route_compute_count = 0;
        int route_handoff_count = 0;
        int ffn_compute_count = 0;
        int return_count = 0;
    };

    std::map<std::pair<int, int>, tensor_phone_stage_timing>
        tensor_phone_stage_timings;

    auto tensor_phone_stage_entry =
        [&](int layer, int chunk, bool decode)
            -> tensor_phone_stage_timing & {
            const int chunk_key = chunk >= 0 ? chunk : 0;
            auto & timing =
                tensor_phone_stage_timings[{ layer, chunk_key }];
            timing.mode_set = true;
            timing.decode = decode;
            return timing;
        };

    auto perform_phone_prefill_oneway_return =
        [&](size_t sg, int layer, int chunk) -> ggml_status {
            constexpr size_t j_src = 0;
            constexpr size_t j_dst = 1;
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_cgraph * pc_graph = bcj_src.cgraphs[sg].cgraph_main;
            ggml_cgraph * phone_graph = bcj_dst.cgraphs[sg].cgraph_main;
            GGML_ASSERT(pc_graph != nullptr && phone_graph != nullptr);
            GGML_ASSERT(pc_graph->n_nodes > 0 && phone_graph->n_nodes > 0);

            ggml_tensor * node_src = pc_graph->nodes[pc_graph->n_nodes - 1];
            ggml_tensor * node_dst = phone_graph->nodes[phone_graph->n_nodes - 1];

            int pc_chunk = -1;
            int pc_layer = -1;
            int phone_chunk = -1;
            int phone_layer = -1;
            GGML_ASSERT(ggml_backend_meta_parse_prefill_down_chunk(
                node_src->name, pc_chunk, pc_layer));
            GGML_ASSERT(ggml_backend_meta_parse_prefill_down_chunk(
                node_dst->name, phone_chunk, phone_layer));
            GGML_ASSERT(pc_chunk == chunk && phone_chunk == chunk);
            GGML_ASSERT(pc_layer == layer && phone_layer == layer);
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));
            GGML_ASSERT(ggml_nbytes(node_src) == ggml_nbytes(node_dst));

            const int64_t return_begin_us =
                tensor_phone_stage_profile ? ggml_time_us() : 0;

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, 0);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src =
                node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;

            const bool strict_phone_primary_fence =
                phone_primary_strict_reduce;
            const auto fence_backend = [&](ggml_backend_t backend) {
                const ggml_backend_rpc_fence_t rpc_fence =
                    ggml_backend_meta_get_rpc_fence(backend);
                if (rpc_fence != nullptr) {
                    rpc_fence(backend);
                } else {
                    ggml_backend_synchronize(backend);
                }
            };

            int64_t src_fence_us = 0;
            if (strict_phone_primary_fence) {
                const int64_t begin_us = ggml_time_us();
                fence_backend(bcj_src.backend);
                src_fence_us = ggml_time_us() - begin_us;
            }

            const ggml_backend_rpc_set_tensor_graph_t set_tensor_graph =
                ggml_backend_meta_get_set_tensor_graph(bcj_dst.backend);
            const bool allow_fused_set_add =
                !strict_phone_primary_fence &&
                set_tensor_graph != nullptr &&
                std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_DISABLE_FUSED_SET_ADD") == nullptr;

            bool fused_set_add_used = false;
            int64_t fused_set_add_us = 0;
            int64_t copy_us = 0;
            int64_t dst_fence_us = 0;
            int64_t add_submit_us = 0;

            if (allow_fused_set_add) {
                const int64_t begin_us = ggml_time_us();
                fused_set_add_used = set_tensor_graph(
                    bcj_src.backend,
                    bcj_dst.backend,
                    node_src,
                    node_tmp,
                    cgraph_aux);
                fused_set_add_us = ggml_time_us() - begin_us;
                if (fused_set_add_used) {
                    copy_us = fused_set_add_us;
                    record_copy_wait(fused_set_add_us);
                    record_meta_copy(
                        sg, j_src, j_dst, node_src, fused_set_add_us);
                }
            }

            if (!fused_set_add_used) {
                const int64_t copy_begin_us = ggml_time_us();
                ggml_backend_tensor_copy_async(
                    bcj_src.backend, bcj_dst.backend, node_src, node_tmp);
                copy_us = ggml_time_us() - copy_begin_us;
                record_copy_wait(copy_us);
                record_meta_copy(sg, j_src, j_dst, node_src, copy_us);

                if (strict_phone_primary_fence) {
                    const int64_t begin_us = ggml_time_us();
                    fence_backend(bcj_dst.backend);
                    dst_fence_us = ggml_time_us() - begin_us;
                }

                const int64_t add_begin_us = ggml_time_us();
                const ggml_status status =
                    ggml_backend_graph_compute_async(
                        bcj_dst.backend, cgraph_aux);
                add_submit_us = ggml_time_us() - add_begin_us;
                record_reduce_add(add_submit_us, true);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }

            int64_t add_fence_us = 0;
            if (strict_phone_primary_fence) {
                const int64_t begin_us = ggml_time_us();
                fence_backend(bcj_dst.backend);
                add_fence_us = ggml_time_us() - begin_us;
            }

            if (tensor_phone_stage_profile) {
                const int64_t wall_us = ggml_time_us() - return_begin_us;
                auto & stage = tensor_phone_stage_entry(layer, chunk, false);
                stage.return_client_wall_us += wall_us;
                stage.return_count += 1;
                printf(
                    "[TENSOR_PHONE_STAGE_COMM] stage=return_set_add "
                    "mode=prefill layer=%d chunk=%d sg=%zu wall_ms=%.3f "
                    "fused=%d fused_ms=%.3f copy_ms=%.3f "
                    "add_submit_ms=%.3f pipeline=1\n",
                    layer,
                    chunk,
                    sg,
                    wall_us / 1000.0,
                    fused_set_add_used ? 1 : 0,
                    fused_set_add_us / 1000.0,
                    copy_us / 1000.0,
                    add_submit_us / 1000.0);
            }

            if (pipeline_debug) {
                printf(
                    "[PHONE_PREFILL_PIPE_RETURN] layer=%d chunk=%d sg=%zu "
                    "bytes=%zu fused=%d src_fence_ms=%.3f copy_ms=%.3f "
                    "dst_fence_ms=%.3f add_ms=%.3f add_fence_ms=%.3f\n",
                    layer,
                    chunk,
                    sg,
                    ggml_nbytes(node_src),
                    fused_set_add_used ? 1 : 0,
                    src_fence_us / 1000.0,
                    copy_us / 1000.0,
                    dst_fence_us / 1000.0,
                    add_submit_us / 1000.0,
                    add_fence_us / 1000.0);
            }

            return GGML_STATUS_SUCCESS;
        };

    auto submit_phone_prefill_staged_add =
        [&](size_t sg,
            int layer,
            int chunk,
            ggml_tensor * return_stage) -> ggml_status {
            constexpr size_t j_dst = 1;
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];
            ggml_cgraph * phone_graph =
                bcj_dst.cgraphs[sg].cgraph_main;
            GGML_ASSERT(phone_graph != nullptr);
            GGML_ASSERT(phone_graph->n_nodes > 0);
            GGML_ASSERT(return_stage != nullptr);

            ggml_tensor * node_dst =
                phone_graph->nodes[phone_graph->n_nodes - 1];
            int phone_chunk = -1;
            int phone_layer = -1;
            GGML_ASSERT(ggml_backend_meta_parse_prefill_down_chunk(
                node_dst->name, phone_chunk, phone_layer));
            GGML_ASSERT(phone_chunk == chunk);
            GGML_ASSERT(phone_layer == layer);
            GGML_ASSERT(ggml_are_same_layout(node_dst, return_stage));

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src =
                node_dst->view_src == nullptr ?
                    node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = return_stage;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * add_graph = get_cgraph_aux();
            add_graph->nodes[0] = node_red;
            add_graph->n_nodes = 1;

            const int64_t submit_begin_us = ggml_time_us();
            const ggml_status status =
                ggml_backend_graph_compute_async(
                    bcj_dst.backend,
                    add_graph);
            const int64_t submit_us =
                ggml_time_us() - submit_begin_us;
            record_reduce_add(submit_us, true);

            if (tensor_phone_stage_profile) {
                auto & stage =
                    tensor_phone_stage_entry(layer, chunk, false);
                stage.return_client_wall_us += submit_us;
                stage.return_count += 1;
            }

            if (pipeline_debug || tensor_phone_stage_profile) {
                printf(
                    "[PHONE_PREFILL_ASYNC_ADD_SUBMIT] "
                    "layer=%d chunk=%d sg=%zu bytes=%zu submit_ms=%.3f "
                    "status=%d\n",
                    layer,
                    chunk,
                    sg,
                    ggml_nbytes(return_stage),
                    submit_us / 1000.0,
                    (int) status);
            }

            return status;
        };

    auto join_phone_prefill_pc_chunk =
        [&](int layer, int chunk, bool wait_for_return) -> ggml_status {
            if (!phone_prefill_chunk_join_active) {
                return GGML_STATUS_SUCCESS;
            }

            auto branch_it =
                std::find_if(
                    pending_phone_prefill_pc_branches.begin(),
                    pending_phone_prefill_pc_branches.end(),
                    [&](const phone_prefill_pc_branch & branch) {
                        return branch.layer == layer &&
                            branch.chunk == chunk;
                    });

            if (branch_it ==
                    pending_phone_prefill_pc_branches.end()) {
                return GGML_STATUS_FAILED;
            }

            GGML_ASSERT(branch_it->return_task != 0);
            GGML_ASSERT(branch_it->phone_ffn_seq != 0);
            GGML_ASSERT(branch_it->return_stage != nullptr);
            GGML_ASSERT(
                branch_it->return_lane <
                ggml_backend_meta_context::PREFILL_RETURN_LANES);
            ggml_backend_meta_transfer_worker * return_worker =
                backend_ctx->prefill_return_workers[
                    branch_it->return_lane];
            GGML_ASSERT(return_worker != nullptr);

            const bool pc_ready_before =
                backend_ctx->prefill_pc_worker != nullptr &&
                backend_ctx->prefill_pc_worker->is_completed(
                    branch_it->pc_task);
            const bool return_ready_before =
                return_worker->is_completed(
                    branch_it->return_task);

            // Normal chunk boundaries are opportunistic only.  If the return
            // has not completed, leave this branch pending and let the next
            // Phone chunk run instead of turning the per-chunk dependency into
            // a host-side barrier.  The last chunk of the layer calls this
            // helper with wait_for_return=true and drains every residual
            // branch before the next layer is allowed to start.
            if (!wait_for_return && !return_ready_before) {
                if (pipeline_debug ||
                        tensor_phone_stage_profile) {
                    printf(
                        "[PHONE_PREFILL_CHUNK_JOIN_DEFER] "
                        "layer=%d chunk=%d sg=%zu seq=%" PRIu64
                        " pc_ready=%d return_ready=0\n",
                        branch_it->layer,
                        branch_it->chunk,
                        branch_it->sg,
                        branch_it->phone_ffn_seq,
                        pc_ready_before ? 1 : 0);
                }
                return GGML_STATUS_SUCCESS;
            }

            const int64_t join_wait_begin_us = ggml_time_us();
            const ggml_status return_status =
                return_worker->wait(
                    branch_it->return_task);
            const int64_t join_wait_us =
                ggml_time_us() - join_wait_begin_us;
            if (return_status != GGML_STATUS_SUCCESS) {
                return return_status;
            }

            // The return ACK is emitted only after the Phone server has seen
            // this chunk's FFN-ready seq and enqueued the PC-partial write.
            // Submit ADD immediately for this chunk.  It goes to the same
            // ordered Phone compute socket/queue, so write -> ADD is ordered
            // without a host fence.
            const int64_t add_begin_us = ggml_time_us();
            const ggml_status add_status =
                submit_phone_prefill_staged_add(
                    branch_it->sg,
                    branch_it->layer,
                    branch_it->chunk,
                    branch_it->return_stage);
            const int64_t add_us =
                ggml_time_us() - add_begin_us;
            if (add_status != GGML_STATUS_SUCCESS) {
                return add_status;
            }

            if (pipeline_debug ||
                    tensor_phone_stage_profile) {
                printf(
                    "[PHONE_PREFILL_CHUNK_JOIN] "
                    "layer=%d chunk=%d sg=%zu seq=%" PRIu64
                    " mode=%s return_lane=%zu pc_ready_before=%d "
                    "return_ready_before=%d wait_ms=%.3f "
                    "add_submit_ms=%.3f\n",
                    branch_it->layer,
                    branch_it->chunk,
                    branch_it->sg,
                    branch_it->phone_ffn_seq,
                    wait_for_return ?
                        "LAYER_BARRIER" : "EAGER",
                    branch_it->return_lane,
                    pc_ready_before ? 1 : 0,
                    return_ready_before ? 1 : 0,
                    join_wait_us / 1000.0,
                    add_us / 1000.0);
            }

            const int joined_layer = branch_it->layer;
            const int joined_chunk = branch_it->chunk;
            const size_t joined_sg = branch_it->sg;
            pending_phone_prefill_routes.erase(
                { joined_layer, joined_chunk });
            deferred_phone_prefill_return_sgs.erase(joined_sg);
            pending_phone_prefill_pc_branches.erase(branch_it);

            const bool layer_pending =
                std::any_of(
                    pending_phone_prefill_pc_branches.begin(),
                    pending_phone_prefill_pc_branches.end(),
                    [&](const phone_prefill_pc_branch & branch) {
                        return branch.layer == joined_layer;
                    });
            if (!layer_pending) {
                phone_prefill_lane1_return_gates.erase(joined_layer);
                if (pipeline_debug ||
                        tensor_phone_stage_profile) {
                    printf(
                        "[PHONE_PREFILL_CHUNK_JOIN_LAYER_READY] "
                        "layer=%d ready_after_chunk=%d mode=PER_CHUNK\n",
                        joined_layer,
                        joined_chunk);
                }
            }

            return GGML_STATUS_SUCCESS;
        };

    auto has_pending_phone_prefill_pc_for_layer =
        [&](int layer) -> bool {
            return std::any_of(
                pending_phone_prefill_pc_branches.begin(),
                pending_phone_prefill_pc_branches.end(),
                [&](const phone_prefill_pc_branch & branch) {
                    return branch.layer == layer;
                });
        };

    auto wait_phone_prefill_pc_dependency =
        [&](int layer, int chunk, bool & waited, int64_t & wait_us)
            -> ggml_status {
            waited = false;
            wait_us = 0;

            const auto branch_it =
                std::find_if(
                    pending_phone_prefill_pc_branches.begin(),
                    pending_phone_prefill_pc_branches.end(),
                    [&](const phone_prefill_pc_branch & branch) {
                        return branch.layer == layer &&
                            branch.chunk == chunk;
                    });
            if (branch_it ==
                    pending_phone_prefill_pc_branches.end()) {
                // Already reaped opportunistically.
                return GGML_STATUS_SUCCESS;
            }

            waited = true;
            const int64_t begin_us = ggml_time_us();
            const ggml_status status =
                join_phone_prefill_pc_chunk(layer, chunk, true);
            wait_us = ggml_time_us() - begin_us;
            return status;
        };

    auto reap_phone_prefill_pc_layer =
        [&](int layer, bool wait_for_all) -> ggml_status {
            if (!phone_prefill_chunk_join_active) {
                return GGML_STATUS_SUCCESS;
            }

            // Two return lanes run independently. Reap whichever branch has
            // completed first. At the layer boundary, only block when neither
            // lane has a completed branch; this avoids delaying a ready ADD
            // behind an unrelated return on the other lane.
            while (true) {
                auto branch_it =
                    pending_phone_prefill_pc_branches.end();
                auto wait_candidate =
                    pending_phone_prefill_pc_branches.end();

                for (auto it =
                         pending_phone_prefill_pc_branches.begin();
                     it != pending_phone_prefill_pc_branches.end();
                     ++it) {
                    if (it->layer != layer) {
                        continue;
                    }

                    if (wait_candidate ==
                            pending_phone_prefill_pc_branches.end()) {
                        wait_candidate = it;
                    }

                    GGML_ASSERT(
                        it->return_lane <
                        ggml_backend_meta_context::PREFILL_RETURN_LANES);
                    ggml_backend_meta_transfer_worker * return_worker =
                        backend_ctx->prefill_return_workers[
                            it->return_lane];
                    GGML_ASSERT(return_worker != nullptr);

                    if (return_worker->is_completed(
                            it->return_task)) {
                        branch_it = it;
                        break;
                    }
                }

                if (branch_it ==
                        pending_phone_prefill_pc_branches.end()) {
                    if (!wait_for_all ||
                            wait_candidate ==
                                pending_phone_prefill_pc_branches.end()) {
                        return GGML_STATUS_SUCCESS;
                    }
                    branch_it = wait_candidate;
                }

                const int chunk = branch_it->chunk;
                const ggml_status status =
                    join_phone_prefill_pc_chunk(
                        layer,
                        chunk,
                        wait_for_all);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        };

    auto reap_phone_prefill_pc_layers_before =
        [&](int layer_exclusive, bool & waited, int64_t & wait_us)
            -> ggml_status {
            waited = false;
            wait_us = 0;

            while (true) {
                int old_layer = std::numeric_limits<int>::max();
                for (const auto & branch :
                        pending_phone_prefill_pc_branches) {
                    if (branch.layer < layer_exclusive) {
                        old_layer = std::min(
                            old_layer, branch.layer);
                    }
                }
                if (old_layer ==
                        std::numeric_limits<int>::max()) {
                    return GGML_STATUS_SUCCESS;
                }

                waited = true;
                const int64_t begin_us = ggml_time_us();
                const ggml_status status =
                    reap_phone_prefill_pc_layer(
                        old_layer, true);
                wait_us += ggml_time_us() - begin_us;
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        };

    auto drain_phone_prefill_pc_layer =
        [&](int layer) -> ggml_status {
            if (backend_ctx->prefill_pc_worker == nullptr) {
                return GGML_STATUS_SUCCESS;
            }

            const int64_t tail_begin_us = ggml_time_us();
            size_t layer_chunk_count = 0;
            size_t pc_ready_at_entry = 0;
            for (const auto & branch : pending_phone_prefill_pc_branches) {
                if (branch.layer != layer) {
                    continue;
                }
                ++layer_chunk_count;
                if (backend_ctx->prefill_pc_worker->is_completed(
                        branch.pc_task)) {
                    ++pc_ready_at_entry;
                }
            }

            ggml_backend_t phone_backend =
                backend_ctx->backend_configs[1].backend;

            // Optional Router-first scheduling.  Each deferred local Phone FFN
            // has private copies of hidden/topk/weights, so allocator reuse by
            // later chunks cannot corrupt its inputs.  Do not restore those
            // bytes into allocator-managed tensors at the layer tail: their
            // storage may already have been reused.  Instead, temporarily
            // rebind the graph's three external tensor descriptors to private
            // staging while RPC serializes each Phone FFN graph.
            size_t phone_deferred_submitted = 0;
            int64_t phone_restore_submit_us = 0;
            int64_t phone_ffn_submit_us = 0;
            for (auto it = pending_phone_prefill_phone_branches.begin();
                 it != pending_phone_prefill_phone_branches.end();) {
                if (it->layer != layer) {
                    ++it;
                    continue;
                }

                GGML_ASSERT(backend_ctx->compute_workers != nullptr);
                GGML_ASSERT(it->src_hidden != nullptr);
                GGML_ASSERT(it->src_topk != nullptr);
                GGML_ASSERT(it->src_weights != nullptr);
                GGML_ASSERT(it->stage_hidden != nullptr);
                GGML_ASSERT(it->stage_topk != nullptr);
                GGML_ASSERT(it->stage_weights != nullptr);

                ggml_cgraph * phone_graph =
                    backend_ctx->backend_configs[1].
                        cgraphs[it->sg].cgraph_main;
                GGML_ASSERT(phone_graph != nullptr);

                // Inputs are already directly bound to their private
                // producer buffers.  Force serialization so the remote graph
                // records those bindings; no local rebind or restore is
                // needed around this submission.
                const int64_t bind_begin_us = ggml_time_us();
                const uint64_t saved_uid = phone_graph->uid;
                phone_graph->uid = 0;
                const int64_t bind_us = ggml_time_us() - bind_begin_us;
                phone_restore_submit_us += bind_us;

                const int64_t submit_begin_us = ggml_time_us();
                backend_ctx->compute_workers->start_graph(
                    1, phone_graph);
                const ggml_status phone_status =
                    backend_ctx->compute_workers->wait(1);
                const int64_t submit_us =
                    ggml_time_us() - submit_begin_us;
                phone_ffn_submit_us += submit_us;

                phone_graph->uid = saved_uid;

                if (phone_status != GGML_STATUS_SUCCESS) {
                    return phone_status;
                }

                if (pipeline_debug) {
                    printf(
                        "[PHONE_PREFILL_PHONE_DEFER_SUBMIT] "
                        "layer=%d chunk=%d sg=%zu "
                        "bind_ms=%.3f ffn_submit_ms=%.3f\n",
                        it->layer,
                        it->chunk,
                        it->sg,
                        bind_us / 1000.0,
                        submit_us / 1000.0);
                }

                ++phone_deferred_submitted;
                it = pending_phone_prefill_phone_branches.erase(it);
            }

            // All local Phone FFNs are now submitted fire-and-forget on the
            // compute socket.  Before any PC partial is returned and added
            // into Phone state, fence once at the layer barrier so the final
            // local partial is guaranteed complete.  PC worker tasks continue
            // to make progress while this fence waits.
            const ggml_backend_rpc_fence_t phone_fence =
                ggml_backend_meta_get_rpc_fence(phone_backend);
            GGML_ASSERT(phone_fence != nullptr);

            const int64_t phone_fence_begin_us = ggml_time_us();
            phone_fence(phone_backend);
            const int64_t phone_fence_us =
                ggml_time_us() - phone_fence_begin_us;

            size_t pc_ready_after_phone_fence = 0;
            for (const auto & branch : pending_phone_prefill_pc_branches) {
                if (branch.layer == layer &&
                        backend_ctx->prefill_pc_worker->is_completed(
                            branch.pc_task)) {
                    ++pc_ready_after_phone_fence;
                }
            }

            if (pipeline_debug || tensor_phone_stage_profile) {
                printf(
                    "[PHONE_PREFILL_LAYER_FENCE] "
                    "layer=%d wait_ms=%.3f\n",
                    layer,
                    phone_fence_us / 1000.0);
            }

            size_t async_return_chunks = 0;
            size_t return_ready_after_phone_fence = 0;
            for (const auto & branch : pending_phone_prefill_pc_branches) {
                if (branch.layer != layer || branch.return_task == 0) {
                    continue;
                }
                ++async_return_chunks;
                GGML_ASSERT(
                    branch.return_lane <
                    ggml_backend_meta_context::PREFILL_RETURN_LANES);
                ggml_backend_meta_transfer_worker * return_worker =
                    backend_ctx->prefill_return_workers[
                        branch.return_lane];
                GGML_ASSERT(return_worker != nullptr);
                if (return_worker->is_completed(
                        branch.return_task)) {
                    ++return_ready_after_phone_fence;
                }
            }
            GGML_ASSERT(
                async_return_chunks == 0 ||
                async_return_chunks == layer_chunk_count);
            const bool use_async_return =
                async_return_chunks != 0;

            int64_t pc_wait_total_us = 0;
            int64_t return_wait_total_us = 0;
            int64_t return_fence_us = 0;
            int64_t add_submit_total_us = 0;
            int64_t add_fence_us = 0;
            int64_t return_total_us = 0;
            size_t drained_chunks = 0;

            if (use_async_return) {
                // Network transfer has already been running concurrently with
                // the PC worker and local Phone FFNs.  At the tail, wait only
                // for any residual PC/return work that failed to hide.
                for (auto & branch : pending_phone_prefill_pc_branches) {
                    if (branch.layer != layer) {
                        continue;
                    }
                    GGML_ASSERT(branch.return_task != 0);
                    GGML_ASSERT(branch.return_stage != nullptr);

                    const int64_t pc_wait_begin_us = ggml_time_us();
                    const ggml_status pc_status =
                        backend_ctx->prefill_pc_worker->wait(
                            branch.pc_task);
                    const int64_t pc_wait_us =
                        ggml_time_us() - pc_wait_begin_us;
                    pc_wait_total_us += pc_wait_us;
                    if (pc_status != GGML_STATUS_SUCCESS) {
                        return pc_status;
                    }

                    GGML_ASSERT(
                        branch.return_lane <
                        ggml_backend_meta_context::PREFILL_RETURN_LANES);
                    ggml_backend_meta_transfer_worker * return_worker =
                        backend_ctx->prefill_return_workers[
                            branch.return_lane];
                    GGML_ASSERT(return_worker != nullptr);

                    const int64_t return_wait_begin_us =
                        ggml_time_us();
                    const ggml_status return_status =
                        return_worker->wait(
                            branch.return_task);
                    const int64_t return_wait_us =
                        ggml_time_us() - return_wait_begin_us;
                    return_wait_total_us += return_wait_us;
                    if (return_status != GGML_STATUS_SUCCESS) {
                        return return_status;
                    }

                    if (pipeline_debug ||
                            tensor_phone_stage_profile) {
                        printf(
                            "[PHONE_PREFILL_ASYNC_DRAIN] "
                            "layer=%d chunk=%d sg=%zu "
                            "pc_wait_ms=%.3f return_wait_ms=%.3f\n",
                            branch.layer,
                            branch.chunk,
                            branch.sg,
                            pc_wait_us / 1000.0,
                            return_wait_us / 1000.0);
                    }
                }

                // Return ACK means the server has already enqueued the
                // OpenCL write. In ordered-return mode the server owns the
                // CL_FALSE source payload until a later device synchronize,
                // so ADD can rely on in-order queue sequencing instead of a
                // host-side return fence.
                if (!phone_prefill_ordered_return_active) {
                    const int64_t return_fence_begin_us =
                        ggml_time_us();
                    phone_fence(phone_backend);
                    return_fence_us =
                        ggml_time_us() - return_fence_begin_us;
                }

                // Queue every ADD on the Phone compute socket. Later Phone
                // graph commands use the same ordered RPC compute socket and
                // OpenCL queue, so ordered-return mode does not need a host
                // completion fence after the ADD submissions either.
                for (auto & branch : pending_phone_prefill_pc_branches) {
                    if (branch.layer != layer) {
                        continue;
                    }
                    const int64_t add_begin_us = ggml_time_us();
                    const ggml_status add_status =
                        submit_phone_prefill_staged_add(
                            branch.sg,
                            branch.layer,
                            branch.chunk,
                            branch.return_stage);
                    add_submit_total_us +=
                        ggml_time_us() - add_begin_us;
                    if (add_status != GGML_STATUS_SUCCESS) {
                        return add_status;
                    }
                }

                if (!phone_prefill_ordered_return_active) {
                    const int64_t add_fence_begin_us =
                        ggml_time_us();
                    phone_fence(phone_backend);
                    add_fence_us =
                        ggml_time_us() - add_fence_begin_us;
                }

                return_total_us =
                    return_wait_total_us +
                    return_fence_us +
                    add_submit_total_us +
                    add_fence_us;

                if (pipeline_debug ||
                        tensor_phone_stage_profile) {
                    printf(
                        "[PHONE_PREFILL_RETURN_OVERLAP] "
                        "layer=%d chunks=%zu ready_after_phone_fence=%zu "
                        "return_wait_ms=%.3f return_fence_ms=%.3f "
                        "add_submit_ms=%.3f add_fence_ms=%.3f "
                        "residual_ms=%.3f\n",
                        layer,
                        async_return_chunks,
                        return_ready_after_phone_fence,
                        return_wait_total_us / 1000.0,
                        return_fence_us / 1000.0,
                        add_submit_total_us / 1000.0,
                        add_fence_us / 1000.0,
                        return_total_us / 1000.0);
                }

                for (auto it =
                         pending_phone_prefill_pc_branches.begin();
                     it != pending_phone_prefill_pc_branches.end();) {
                    if (it->layer != layer) {
                        ++it;
                        continue;
                    }
                    ++drained_chunks;
                    pending_phone_prefill_routes.erase(
                        { it->layer, it->chunk });
                    deferred_phone_prefill_return_sgs.erase(
                        it->sg);
                    it =
                        pending_phone_prefill_pc_branches.erase(it);
                }
            } else {
                // Patch-8 / feature-off fallback: preserve the previous
                // correct fused SET+ADD tail path unchanged.
                for (auto it =
                         pending_phone_prefill_pc_branches.begin();
                     it != pending_phone_prefill_pc_branches.end();) {
                    if (it->layer != layer) {
                        ++it;
                        continue;
                    }

                    const int64_t wait_begin_us =
                        ggml_time_us();
                    const ggml_status pc_status =
                        backend_ctx->prefill_pc_worker->wait(
                            it->pc_task);
                    const int64_t wait_us =
                        ggml_time_us() - wait_begin_us;
                    pc_wait_total_us += wait_us;
                    if (pc_status != GGML_STATUS_SUCCESS) {
                        return pc_status;
                    }

                    const int64_t return_begin_us =
                        ggml_time_us();
                    const ggml_status return_status =
                        perform_phone_prefill_oneway_return(
                            it->sg,
                            it->layer,
                            it->chunk);
                    const int64_t return_us =
                        ggml_time_us() - return_begin_us;
                    return_total_us += return_us;
                    if (return_status != GGML_STATUS_SUCCESS) {
                        return return_status;
                    }

                    if (pipeline_debug) {
                        printf(
                            "[PHONE_PREFILL_PIPE_DRAIN] "
                            "layer=%d chunk=%d sg=%zu "
                            "pc_wait_ms=%.3f return_ms=%.3f\n",
                            it->layer,
                            it->chunk,
                            it->sg,
                            wait_us / 1000.0,
                            return_us / 1000.0);
                    }

                    ++drained_chunks;
                    pending_phone_prefill_routes.erase(
                        { it->layer, it->chunk });
                    deferred_phone_prefill_return_sgs.erase(
                        it->sg);
                    it =
                        pending_phone_prefill_pc_branches.erase(it);
                }
            }

            if (pipeline_debug || tensor_phone_stage_profile) {
                const int64_t tail_wall_us =
                    ggml_time_us() - tail_begin_us;
                printf(
                    "[PHONE_PREFILL_PC_DRAIN] "
                    "layer=%d chunks=%zu ready_at_entry=%zu "
                    "ready_after_phone_fence=%zu "
                    "return_ready_after_phone_fence=%zu "
                    "phone_deferred=%zu async_return=%d ordered_return=%d "
                    "bind_ms=%.3f phone_ffn_submit_ms=%.3f "
                    "phone_fence_ms=%.3f pc_wait_ms=%.3f "
                    "return_wait_ms=%.3f return_fence_ms=%.3f "
                    "add_submit_ms=%.3f add_fence_ms=%.3f "
                    "return_ms=%.3f tail_ms=%.3f\n",
                    layer,
                    drained_chunks,
                    pc_ready_at_entry,
                    pc_ready_after_phone_fence,
                    return_ready_after_phone_fence,
                    phone_deferred_submitted,
                    use_async_return ? 1 : 0,
                    phone_prefill_ordered_return_active ? 1 : 0,
                    phone_restore_submit_us / 1000.0,
                    phone_ffn_submit_us / 1000.0,
                    phone_fence_us / 1000.0,
                    pc_wait_total_us / 1000.0,
                    return_wait_total_us / 1000.0,
                    return_fence_us / 1000.0,
                    add_submit_total_us / 1000.0,
                    add_fence_us / 1000.0,
                    return_total_us / 1000.0,
                    tail_wall_us / 1000.0);
            }

            if (tensor_expert_load_profile) {
                tensor_expert_load_accum layer_load;
                {
                    std::lock_guard<std::mutex> lock(
                        tensor_expert_load_mutex);
                    const auto load_it =
                        tensor_expert_load_by_layer.find(layer);
                    if (load_it !=
                            tensor_expert_load_by_layer.end()) {
                        layer_load = load_it->second;
                        tensor_expert_load_by_layer.erase(load_it);
                    }
                }

                size_t unique_experts = 0;
                for (uint64_t count : layer_load.counts) {
                    unique_experts += count != 0 ? 1 : 0;
                }
                const std::string top =
                    tensor_expert_top_string(
                        layer_load.counts, 12);

                int64_t pc_ff = -1;
                int64_t phone_ff = -1;
                const auto shard_it =
                    tensor_expert_shard_by_layer.find(layer);
                if (shard_it !=
                        tensor_expert_shard_by_layer.end()) {
                    pc_ff = shard_it->second.first;
                    phone_ff = shard_it->second.second;
                    tensor_expert_shard_by_layer.erase(shard_it);
                }
                const int64_t total_ff =
                    pc_ff >= 0 && phone_ff >= 0 ?
                        pc_ff + phone_ff : -1;

                printf(
                    "[TENSOR_EXPERT_LOAD_LAYER] "
                    "layer=%d parity=%s chunks=%zu "
                    "assignments=%" PRIu64 " unique=%zu "
                    "top=%s read_ms=%.3f "
                    "pc_ff=%" PRId64 " phone_ff=%" PRId64
                    " pc_pct=%.3f phone_pct=%.3f\n",
                    layer,
                    (layer & 1) != 0 ? "odd" : "even",
                    drained_chunks,
                    layer_load.assignments,
                    unique_experts,
                    top.c_str(),
                    layer_load.read_us / 1000.0,
                    pc_ff,
                    phone_ff,
                    total_ff > 0 ?
                        100.0 * (double) pc_ff /
                            (double) total_ff : 0.0,
                    total_ff > 0 ?
                        100.0 * (double) phone_ff /
                            (double) total_ff : 0.0);
            }

            GGML_ASSERT(drained_chunks == layer_chunk_count);

            restore_phone_prefill_direct_bindings(layer);
            for (auto it = phone_prefill_route_producer_seq.begin();
                 it != phone_prefill_route_producer_seq.end();) {
                if (it->first.first == layer) {
                    it = phone_prefill_route_producer_seq.erase(it);
                } else {
                    ++it;
                }
            }

            return GGML_STATUS_SUCCESS;
        };

    auto open_phone_prefill_lane1_return_gate =
        [&](int layer, int chunk) {
            if (!phone_prefill_chunk_join_active ||
                    !phone_prefill_producer_route ||
                    ggml_backend_meta_context::PREFILL_RETURN_LANES < 2) {
                return;
            }

            const bool has_next_chunk =
                find_down_chunk(0, true, layer, chunk + 1) != nullptr ||
                find_down_chunk(1, true, layer, chunk + 1) != nullptr;
            if (has_next_chunk) {
                return;
            }

            auto & gate =
                phone_prefill_lane1_return_gates[layer];
            if (gate == nullptr) {
                gate =
                    std::make_shared<
                        phone_prefill_lane1_return_gate>();
            }

            bool opened_now = false;
            {
                std::lock_guard<std::mutex> lock(gate->mutex);
                if (!gate->open) {
                    gate->open = true;
                    opened_now = true;
                }
            }
            gate->cv.notify_all();

            if (opened_now &&
                    (pipeline_debug ||
                     tensor_phone_stage_profile)) {
                printf(
                    "[PHONE_PREFILL_RETURN_LANE1_GATE_OPEN] "
                    "layer=%d chunk=%d reason=LAST_ROUTE_HANDOFF\n",
                    layer,
                    chunk);
            }
        };

    auto specialized_communication = [&](size_t i, bool & handled, bool & next_compute_complete,
                                         bool force_phone_block_exit,
                                         int force_phone_block_layer) -> ggml_status {
        std::vector<ggml_tensor *> nodes(n_backends, nullptr);
        size_t active_count = 0;
        size_t active_backend = 0;
        for (size_t j = 0; j < n_backends; ++j) {
            const auto & bcj = backend_ctx->backend_configs[j];
            nodes[j] = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (nodes[j]->flags & GGML_TENSOR_FLAG_COMPUTE) {
                ++active_count;
                active_backend = j;
            }
        }

        auto deferred_prefill_it =
            deferred_phone_prefill_return_sgs.find(i);
        if (deferred_prefill_it !=
                deferred_phone_prefill_return_sgs.end()) {
            int deferred_chunk_0 = -1;
            int deferred_layer_0 = -1;
            int deferred_chunk_1 = -1;
            int deferred_layer_1 = -1;
            const bool parsed_deferred =
                n_backends == 2 &&
                ggml_backend_meta_parse_prefill_down_chunk(
                    nodes[0]->name,
                    deferred_chunk_0,
                    deferred_layer_0) &&
                ggml_backend_meta_parse_prefill_down_chunk(
                    nodes[1]->name,
                    deferred_chunk_1,
                    deferred_layer_1) &&
                deferred_chunk_0 == deferred_chunk_1 &&
                deferred_layer_0 == deferred_layer_1;
            GGML_ASSERT(parsed_deferred);

            const bool has_next_chunk =
                find_down_chunk(
                    0,
                    true,
                    deferred_layer_0,
                    deferred_chunk_0 + 1) != nullptr ||
                find_down_chunk(
                    1,
                    true,
                    deferred_layer_0,
                    deferred_chunk_0 + 1) != nullptr;
            const bool last_chunk = !has_next_chunk;

            handled = true;

            if (pipeline_debug ||
                    tensor_phone_stage_profile) {
                printf(
                    "[PHONE_PREFILL_RETURN_DEFER] "
                    "layer=%d chunk=%d sg=%zu last=%d "
                    "active=%zu\n",
                    deferred_layer_0,
                    deferred_chunk_0,
                    i,
                    last_chunk ? 1 : 0,
                    active_count);
            }

            if (phone_prefill_chunk_join_active) {
                size_t pending_before = 0;
                for (const auto & branch :
                        pending_phone_prefill_pc_branches) {
                    if (branch.layer == deferred_layer_0) {
                        ++pending_before;
                    }
                }

                const bool wavefront_defer_layer =
                    return_wavefront_graph &&
                    phone_prefill_chunk_join_active;
                const int64_t reap_begin_us = ggml_time_us();
                const ggml_status join_status =
                    reap_phone_prefill_pc_layer(
                        deferred_layer_0,
                        last_chunk && !wavefront_defer_layer);
                const int64_t reap_us =
                    ggml_time_us() - reap_begin_us;
                if (join_status != GGML_STATUS_SUCCESS) {
                    return join_status;
                }

                size_t pending_after = 0;
                for (const auto & branch :
                        pending_phone_prefill_pc_branches) {
                    if (branch.layer == deferred_layer_0) {
                        ++pending_after;
                    }
                }

                if (pipeline_debug ||
                        tensor_phone_stage_profile) {
                    printf(
                        "[PHONE_PREFILL_CHUNK_REAP] "
                        "layer=%d boundary_chunk=%d last=%d "
                        "pending_before=%zu pending_after=%zu "
                        "wall_ms=%.3f\n",
                        deferred_layer_0,
                        deferred_chunk_0,
                        last_chunk ? 1 : 0,
                        pending_before,
                        pending_after,
                        reap_us / 1000.0);
                }

                if (last_chunk && !wavefront_defer_layer) {
                    GGML_ASSERT(pending_after == 0);
                } else if (last_chunk &&
                           wavefront_defer_layer &&
                           (pipeline_debug ||
                            tensor_phone_stage_profile)) {
                    printf(
                        "[PHONE_PREFILL_WAVE_LAYER_DEFER] "
                        "layer=%d last_chunk=%d pending=%zu\n",
                        deferred_layer_0,
                        deferred_chunk_0,
                        pending_after);
                }
            } else if (last_chunk) {
                const ggml_status drain_status =
                    drain_phone_prefill_pc_layer(
                        deferred_layer_0);
                if (drain_status != GGML_STATUS_SUCCESS) {
                    return drain_status;
                }
            }

            return GGML_STATUS_SUCCESS;
        }

        int prefill_norm_chunk = -1;
        int prefill_norm_layer = -1;
        const bool is_prefill_norm_input =
            n_backends == 2 &&
            active_count == 1 &&
            active_backend == 0 &&
            ggml_backend_meta_parse_prefill_norm_chunk(
                nodes[active_backend]->name, prefill_norm_chunk, prefill_norm_layer);

        if (is_prefill_norm_input) {
            // The optimized prefill input pipeline was originally written for
            // dense FFN graphs, where the norm boundary is followed directly
            // by one subgraph whose terminal node is prefill_ffn_down_chunk_*.
            //
            // MoE inserts Router/Top-K/MUL_MAT_ID/expert aggregation between
            // those points, so Meta may split the graph differently.  Do not
            // assert that the dense subgraph shape exists.  Use the optimized
            // path only when the next subgraph actually matches it; otherwise
            // fall through to the generic active_count==1 PC->Phone copy and
            // normal PARTIAL/AllReduce handling below.
            bool direct_prefill_down = false;
            ggml_tensor * pc_next_down = nullptr;
            ggml_tensor * phone_next_down = nullptr;
            int next_chunk = -1;
            int next_layer = -1;

            if (i + 1 < backend_ctx->n_subgraphs) {
                pc_next_down = get_prefill_down_boundary_node(0, i + 1);
                phone_next_down = get_prefill_down_boundary_node(1, i + 1);
                direct_prefill_down =
                    pc_next_down != nullptr &&
                    phone_next_down != nullptr &&
                    ggml_backend_meta_parse_prefill_down_chunk(
                        pc_next_down->name, next_chunk, next_layer) &&
                    next_chunk == prefill_norm_chunk &&
                    next_layer == prefill_norm_layer;
            }

            if (direct_prefill_down) {
                GGML_ASSERT(pending_prefill_input_task == 0);

                if (backend_ctx->prefill_input_worker == nullptr) {
                    backend_ctx->prefill_input_worker =
                        new ggml_backend_meta_transfer_worker();
                }

                handled = true;
                auto & bcj_src = backend_ctx->backend_configs[0];
                auto & bcj_dst = backend_ctx->backend_configs[1];
                ggml_tensor * src = nodes[0];
                ggml_tensor * dst = nodes[1];
                GGML_ASSERT(ggml_is_contiguous(src));
                GGML_ASSERT(ggml_is_contiguous(dst));

                pending_prefill_input_task =
                    backend_ctx->prefill_input_worker->enqueue(
                        [&, src, dst, i](uint64_t) -> ggml_status {
                            const int64_t copy_start_us = ggml_time_us();
                            ggml_backend_tensor_copy_async(
                                bcj_src.backend, bcj_dst.backend, src, dst);
                            const int64_t copy_us =
                                ggml_time_us() - copy_start_us;
                            record_copy_wait(copy_us);
                            record_meta_copy(i, 0, 1, src, copy_us);
                            return GGML_STATUS_SUCCESS;
                        });
                pending_prefill_input_layer = prefill_norm_layer;
                pending_prefill_input_chunk = prefill_norm_chunk;
                record_direct_copy();

                if (pipeline_debug) {
                    GGML_LOG_INFO(
                        "[PREFILL_INPUT] layer=%d chunk=%d tensor=%s "
                        "ne=[%" PRId64 ",%" PRId64 "]\n",
                        prefill_norm_layer, prefill_norm_chunk,
                        src->name, src->ne[0], src->ne[1]);
                }
                return GGML_STATUS_SUCCESS;
            }

            if (pipeline_debug) {
                GGML_LOG_INFO(
                    "[PREFILL_INPUT_FALLBACK] sg=%zu layer=%d chunk=%d "
                    "reason=non_dense_next_subgraph\n",
                    i, prefill_norm_layer, prefill_norm_chunk);
            }
            // Fall through.  The generic single-owner handoff below performs
            // a synchronous PC->Phone copy before the next phone computation.
        }

        int phone_route_chunk = -1;
        int phone_route_layer = -1;
        bool phone_route_boundary = false;
        if (n_backends == 2) {
            phone_route_boundary =
                ggml_backend_meta_parse_phone_route_weights(
                    nodes[1]->name, phone_route_chunk, phone_route_layer);
            if (!phone_route_boundary) {
                phone_route_boundary =
                    ggml_backend_meta_parse_phone_route_weights(
                        nodes[0]->name, phone_route_chunk, phone_route_layer);
            }
        }

        const bool phone_router_handoff =
            phone_route_boundary &&
            layer_is_tensor_phone_primary(phone_route_layer);

        if (phone_router_handoff) {
            handled = true;
            const int64_t tensor_phone_route_handoff_begin_us =
                tensor_phone_stage_profile ? ggml_time_us() : 0;

            char hidden_name[96];
            char topk_name[96];
            if (phone_route_chunk >= 0) {
                std::snprintf(
                    hidden_name, sizeof(hidden_name),
                    "prefill_ffn_norm_chunk_%d-%d",
                    phone_route_chunk, phone_route_layer);
                std::snprintf(
                    topk_name, sizeof(topk_name),
                    "phone_prefill_route_topk_chunk_%d-%d",
                    phone_route_chunk, phone_route_layer);
            } else {
                std::snprintf(
                    hidden_name, sizeof(hidden_name),
                    "ffn_norm-%d",
                    phone_route_layer);
                std::snprintf(
                    topk_name, sizeof(topk_name),
                    "phone_moe_route_topk-%d",
                    phone_route_layer);
            }

            ggml_tensor * src_hidden = find_exact_named_tensor(1, hidden_name);
            ggml_tensor * dst_hidden = find_exact_named_tensor(0, hidden_name);
            ggml_tensor * src_topk = find_exact_named_tensor(1, topk_name);
            ggml_tensor * dst_topk = find_exact_named_tensor(0, topk_name);
            ggml_tensor * src_weights = find_exact_named_tensor(
                1, nodes[1]->name);
            ggml_tensor * dst_weights = find_exact_named_tensor(
                0, nodes[0]->name);

            // The legacy mirrored-l_out fallback needs the Phone-owned
            // residual on PC so PC can evaluate the same l_out ADD.  In the
            // v2.3 single-owner path every Tensor-layer l_out is completed on
            // Phone, so PC consumes only normalized hidden/top-k/weights for
            // its expert FFN shard and must not receive ffn_inp at all.
            const bool phone_single_owner =
                std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER") != nullptr;
            const bool copy_residual =
                !phone_single_owner && phone_route_chunk <= 0;

            char residual_name[64] = {};
            if (copy_residual) {
                std::snprintf(
                    residual_name, sizeof(residual_name),
                    "ffn_inp-%d", phone_route_layer);
            }
            ggml_tensor * src_residual = copy_residual ?
                find_exact_named_tensor(1, residual_name) : nullptr;
            ggml_tensor * dst_residual = copy_residual ?
                find_exact_named_tensor(0, residual_name) : nullptr;

            GGML_ASSERT(src_hidden != nullptr);
            GGML_ASSERT(dst_hidden != nullptr);
            GGML_ASSERT(src_topk != nullptr);
            GGML_ASSERT(dst_topk != nullptr);
            GGML_ASSERT(src_weights != nullptr);
            GGML_ASSERT(dst_weights != nullptr);
            GGML_ASSERT(ggml_nbytes(src_hidden) == ggml_nbytes(dst_hidden));
            GGML_ASSERT(ggml_nbytes(src_topk) == ggml_nbytes(dst_topk));
            GGML_ASSERT(ggml_nbytes(src_weights) == ggml_nbytes(dst_weights));
            if (copy_residual) {
                GGML_ASSERT(src_residual != nullptr);
                GGML_ASSERT(dst_residual != nullptr);
                GGML_ASSERT(
                    ggml_nbytes(src_residual) ==
                    ggml_nbytes(dst_residual));
            }

            int parsed_topk_chunk = -1;
            int parsed_topk_layer = -1;
            GGML_ASSERT(ggml_backend_meta_parse_phone_route_topk(
                src_topk->name, parsed_topk_chunk, parsed_topk_layer));
            GGML_ASSERT(parsed_topk_chunk == phone_route_chunk);
            GGML_ASSERT(parsed_topk_layer == phone_route_layer);

            const auto producer_route_it =
                pending_phone_prefill_routes.find(
                    { phone_route_layer, phone_route_chunk });
            if (phone_prefill_producer_route &&
                    producer_route_it !=
                        pending_phone_prefill_routes.end() &&
                    producer_route_it->second.producer_seq != 0) {
                if (pipeline_debug ||
                        tensor_phone_stage_profile) {
                    printf(
                        "[PHONE_PREFILL_ROUTE_MAILBOX_HANDOFF] "
                        "layer=%d chunk=%d lane=%zu "
                        "seq=%" PRIu64 " task=%" PRIu64
                        " action=ALREADY_POSTED\n",
                        phone_route_layer,
                        phone_route_chunk,
                        producer_route_it->second.lane,
                        producer_route_it->second.producer_seq,
                        producer_route_it->second.task);
                }

                // Soft dual-return policy: lane 0 stays fully eager.  Lane 1
                // is released as soon as the final route for this layer has
                // been posted, not when the route transfer or FFN completes.
                open_phone_prefill_lane1_return_gate(
                    phone_route_layer,
                    phone_route_chunk);
                return GGML_STATUS_SUCCESS;
            }

            auto & bcj_src = backend_ctx->backend_configs[1];
            auto & bcj_dst = backend_ctx->backend_configs[0];
            // Router handoff runs on the graph thread.  RPC's staged transfer
            // callback is thread_local and is only installed by the transfer
            // workers, so these GET_TENSOR calls use the same control socket as
            // the preceding Phone graph submissions.  The server processes that
            // socket serially and OpenCL GET_TENSOR is a blocking CL_TRUE read,
            // which already establishes producer completion.  Keep the old
            // explicit fence only for legacy mirrored mode or strict A/B tests.
            const bool strict_phone_primary_fence =
                !phone_single_owner || phone_primary_strict_route;
            int64_t route_fence_us = 0;
            if (strict_phone_primary_fence) {
                const int64_t route_fence_begin = ggml_time_us();
                const ggml_backend_rpc_fence_t rpc_fence =
                    ggml_backend_meta_get_rpc_fence(bcj_src.backend);
                if (rpc_fence != nullptr) {
                    rpc_fence(bcj_src.backend);
                } else {
                    ggml_backend_synchronize(bcj_src.backend);
                }
                route_fence_us = ggml_time_us() - route_fence_begin;
            }

            bool live_route_trace = false;
            if (const char * value =
                    std::getenv("GGML_META_TP_CONTROL_TRACE_LAYER")) {
                char * end = nullptr;
                const long parsed = std::strtol(value, &end, 10);
                live_route_trace =
                    end != value && *end == '\0' &&
                    parsed == phone_route_layer &&
                    phone_route_chunk <= 0;
            }

            std::vector<float> live_src_hidden;
            std::vector<int32_t> live_src_topk;
            std::vector<float> live_src_weights;

            auto capture_live_f32 =
                [&](ggml_tensor * tensor,
                    std::vector<float> & out,
                    const char * logical,
                    const char * side) {
                    if (!live_route_trace ||
                            tensor == nullptr ||
                            tensor->type != GGML_TYPE_F32) {
                        return;
                    }
                    const size_t n = ggml_nelements(tensor);
                    out.resize(n);
                    ggml_backend_tensor_get(
                        tensor, out.data(), 0, n * sizeof(float));

                    double sum = 0.0;
                    double l2 = 0.0;
                    double max_abs = 0.0;
                    for (float x : out) {
                        const double xd = x;
                        sum += xd;
                        l2 += xd * xd;
                        max_abs = std::max(max_abs, std::abs(xd));
                    }
                    printf(
                        "[TP_ROUTE_LIVE_F32] layer=%d chunk=%d side=%s "
                        "logical=%s tensor=%s n=%zu sum=%.9f l2=%.9f "
                        "max=%.9f v0=%.9f v1=%.9f v2=%.9f v3=%.9f\n",
                        phone_route_layer, phone_route_chunk,
                        side, logical, tensor->name, n,
                        sum, std::sqrt(l2), max_abs,
                        n > 0 ? out[0] : 0.0f,
                        n > 1 ? out[1] : 0.0f,
                        n > 2 ? out[2] : 0.0f,
                        n > 3 ? out[3] : 0.0f);
                };

            auto capture_live_i32 =
                [&](ggml_tensor * tensor,
                    std::vector<int32_t> & out,
                    const char * logical,
                    const char * side) {
                    if (!live_route_trace ||
                            tensor == nullptr ||
                            tensor->type != GGML_TYPE_I32) {
                        return;
                    }
                    const size_t n = ggml_nelements(tensor);
                    out.resize(n);
                    ggml_backend_tensor_get(
                        tensor, out.data(), 0, n * sizeof(int32_t));

                    int64_t sum = 0;
                    int32_t min_v = n > 0 ? out[0] : 0;
                    int32_t max_v = n > 0 ? out[0] : 0;
                    uint64_t hash = UINT64_C(1469598103934665603);
                    for (int32_t x : out) {
                        sum += x;
                        min_v = std::min(min_v, x);
                        max_v = std::max(max_v, x);
                        const uint32_t bits = (uint32_t) x;
                        for (int b = 0; b < 4; ++b) {
                            hash ^= (bits >> (b * 8)) & UINT64_C(0xff);
                            hash *= UINT64_C(1099511628211);
                        }
                    }
                    printf(
                        "[TP_ROUTE_LIVE_I32] layer=%d chunk=%d side=%s "
                        "logical=%s tensor=%s n=%zu sum=%" PRId64
                        " min=%d max=%d hash=%" PRIu64
                        " v0=%d v1=%d v2=%d v3=%d "
                        "v4=%d v5=%d v6=%d v7=%d\n",
                        phone_route_layer, phone_route_chunk,
                        side, logical, tensor->name, n,
                        sum, min_v, max_v, hash,
                        n > 0 ? out[0] : 0,
                        n > 1 ? out[1] : 0,
                        n > 2 ? out[2] : 0,
                        n > 3 ? out[3] : 0,
                        n > 4 ? out[4] : 0,
                        n > 5 ? out[5] : 0,
                        n > 6 ? out[6] : 0,
                        n > 7 ? out[7] : 0);
                };

            capture_live_f32(
                src_hidden, live_src_hidden, "hidden", "PHONE_PRE_COPY");
            capture_live_i32(
                src_topk, live_src_topk, "topk", "PHONE_PRE_COPY");
            capture_live_f32(
                src_weights, live_src_weights, "weights", "PHONE_PRE_COPY");

            const ggml_backend_rpc_get_tensor_batch3_t route_get_batch3 =
                ggml_backend_meta_get_tensor_batch3(bcj_src.backend);
            const ggml_backend_rpc_set_stage_ready_t route_set_stage_ready =
                ggml_backend_meta_get_stage_ready_setter(bcj_src.backend);
            const ggml_backend_rpc_set_route_transfer_lane_t route_set_lane =
                ggml_backend_meta_get_route_transfer_lane_setter(bcj_src.backend);
            const ggml_backend_rpc_set_route_wait_seq_t route_set_wait_seq =
                ggml_backend_meta_get_route_wait_seq_setter(bcj_src.backend);
            const ggml_backend_rpc_route_mark_ready_t route_mark_ready =
                ggml_backend_meta_get_route_mark_ready(bcj_src.backend);
            const ggml_backend_rpc_fence_t route_rpc_fence =
                ggml_backend_meta_get_rpc_fence(bcj_src.backend);
            const bool phone_oneway_reduce =
                std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE") != nullptr;
            const bool has_multiple_prefill_chunks =
                phone_route_chunk >= 0 &&
                find_down_chunk(
                    0,
                    true,
                    phone_route_layer,
                    1) != nullptr &&
                find_down_chunk(
                    1,
                    true,
                    phone_route_layer,
                    1) != nullptr;

            const bool route_seq_supported =
                route_mark_ready != nullptr &&
                route_mark_ready(bcj_src.backend, 0);

            const bool allow_prefill_route_pipeline =
                phone_prefill_chunk_pipeline &&
                phone_route_chunk >= 0 &&
                has_multiple_prefill_chunks &&
                phone_single_owner &&
                phone_oneway_reduce &&
                !strict_phone_primary_fence &&
                !copy_residual &&
                !live_route_trace &&
                route_get_batch3 != nullptr &&
                route_set_stage_ready != nullptr &&
                route_set_lane != nullptr &&
                route_rpc_fence != nullptr &&
                (!phone_prefill_defer_phone_ffn ||
                 (route_set_wait_seq != nullptr &&
                  route_seq_supported));

            if (allow_prefill_route_pipeline) {
                const size_t lane =
                    phone_prefill_route_lane_for_chunk(
                        phone_route_chunk);

                auto & route_worker =
                    backend_ctx->prefill_route_workers[lane];
                if (route_worker == nullptr) {
                    route_worker = new ggml_backend_meta_transfer_worker();
                }

                if (pending_phone_prefill_route_lane_task[lane] != 0) {
                    const int64_t reuse_wait_begin_us = ggml_time_us();
                    const ggml_status old_status =
                        route_worker->wait(
                            pending_phone_prefill_route_lane_task[lane]);
                    const int64_t reuse_wait_us =
                        ggml_time_us() - reuse_wait_begin_us;
                    if (pipeline_debug) {
                        printf(
                            "[PHONE_PREFILL_ROUTE_LANE_REUSE] lane=%zu "
                            "layer=%d chunk=%d wait_ms=%.3f\n",
                            lane,
                            phone_route_layer,
                            phone_route_chunk,
                            reuse_wait_us / 1000.0);
                    }
                    if (old_status != GGML_STATUS_SUCCESS) {
                        return old_status;
                    }
                    pending_phone_prefill_route_lane_task[lane] = 0;
                }

                // The Phone compute RPC is fire-and-forget.  Before a
                // dedicated route socket reads Router outputs, explicitly
                // fence the compute socket so every producer queued before
                // this point (previous local FFN + current pre-route graph)
                // has completed on the Phone backend.
                const int64_t producer_fence_begin_us = ggml_time_us();
                if (!phone_prefill_defer_phone_ffn) {
                    route_rpc_fence(bcj_src.backend);
                }
                const int64_t producer_fence_us =
                    ggml_time_us() - producer_fence_begin_us;

                if (pipeline_debug || tensor_phone_stage_profile) {
                    printf(
                        "[PHONE_PREFILL_PRODUCER_FENCE] "
                        "layer=%d chunk=%d lane=%zu wait_ms=%.3f\n",
                        phone_route_layer,
                        phone_route_chunk,
                        lane,
                        producer_fence_us / 1000.0);
                }

                ggml_backend_t route_src_backend = bcj_src.backend;
                ggml_backend_t route_dst_backend = bcj_dst.backend;

                ggml_tensor * route_src_hidden = src_hidden;
                ggml_tensor * route_src_topk = src_topk;
                ggml_tensor * route_src_weights = src_weights;
                ggml_tensor * phone_stage_hidden = nullptr;
                ggml_tensor * phone_stage_topk = nullptr;
                ggml_tensor * phone_stage_weights = nullptr;

                uint64_t producer_seq = 0;
                if (phone_prefill_defer_phone_ffn) {
                    // Direct-producer mode: hidden/top-k/weights were rebound
                    // to private Phone buffers before their producer graphs
                    // were submitted.  No Phone->Phone copy occurs here.
                    phone_stage_hidden = src_hidden;
                    phone_stage_topk = src_topk;
                    phone_stage_weights = src_weights;

                    auto seq_it =
                        phone_prefill_route_producer_seq.find(
                            { phone_route_layer, phone_route_chunk });
                    GGML_ASSERT(
                        seq_it != phone_prefill_route_producer_seq.end());
                    producer_seq = seq_it->second;

                    if (pipeline_debug) {
                        printf(
                            "[PHONE_PREFILL_PHONE_DIRECT_STAGE] "
                            "layer=%d chunk=%d lane=%zu seq=%" PRIu64
                            " bytes=%zu copy_ms=0.000\n",
                            phone_route_layer,
                            phone_route_chunk,
                            lane,
                            producer_seq,
                            ggml_nbytes(src_hidden) +
                                ggml_nbytes(src_topk) +
                                ggml_nbytes(src_weights));
                    }
                }

                auto & stage_slots =
                    backend_ctx->backend_configs[0].prefill_route_stage_bufs;
                if (stage_slots.size() <=
                        static_cast<size_t>(phone_route_chunk)) {
                    stage_slots.resize(
                        static_cast<size_t>(phone_route_chunk) + 1);
                }

                ggml_tensor * stage_hidden = get_node_aux(dst_hidden);
                ggml_tensor * stage_topk = get_node_aux(dst_topk);
                ggml_tensor * stage_weights = get_node_aux(dst_weights);

                ggml_tensor * stage_tensors[3] = {
                    stage_hidden,
                    stage_topk,
                    stage_weights,
                };
                ggml_tensor * dst_tensors[3] = {
                    dst_hidden,
                    dst_topk,
                    dst_weights,
                };

                auto & stage_bufs =
                    stage_slots[static_cast<size_t>(phone_route_chunk)];
                for (size_t stage_index = 0; stage_index < 3; ++stage_index) {
                    const size_t need =
                        ggml_nbytes(dst_tensors[stage_index]);
                    auto & buf = stage_bufs[stage_index];
                    if (!buf ||
                            ggml_backend_buffer_get_size(buf.get()) < need) {
                        buf.reset(
                            ggml_backend_alloc_buffer(
                                route_dst_backend,
                                need));
                    }
                    GGML_ASSERT(buf != nullptr);
                    stage_tensors[stage_index]->buffer = buf.get();
                    stage_tensors[stage_index]->data =
                        ggml_backend_buffer_get_base(buf.get());
                }

                const uint64_t route_task =
                    route_worker->enqueue(
                        [&, route_worker, route_get_batch3,
                            route_set_stage_ready, route_set_lane,
                            route_set_wait_seq, producer_seq,
                            route_src_hidden, stage_hidden,
                            route_src_topk, stage_topk,
                            route_src_weights, stage_weights,
                            dst_hidden, dst_topk, dst_weights,
                            route_src_backend, route_dst_backend,
                            lane, phone_route_layer, phone_route_chunk, i]
                        (uint64_t task_id) -> ggml_status {
                            ggml_backend_meta_stage_ready_context stage_context {
                                route_worker,
                                task_id,
                            };

                            route_set_lane(static_cast<int>(lane));
                            if (route_set_wait_seq != nullptr) {
                                route_set_wait_seq(producer_seq);
                            }
                            route_set_stage_ready(
                                ggml_backend_meta_stage_ready,
                                &stage_context);

                            const int64_t route_begin_us = ggml_time_us();
                            const bool used = route_get_batch3(
                                route_src_backend,
                                route_dst_backend,
                                route_src_hidden, stage_hidden,
                                route_src_topk, stage_topk,
                                route_src_weights, stage_weights);
                            const int64_t route_us =
                                ggml_time_us() - route_begin_us;

                            route_set_stage_ready(nullptr, nullptr);
                            if (route_set_wait_seq != nullptr) {
                                route_set_wait_seq(0);
                            }
                            route_set_lane(-1);

                            if (!used) {
                                return GGML_STATUS_FAILED;
                            }

                            record_copy_wait(route_us);
                            record_meta_copy(
                                i, 1, 0, route_src_hidden, route_us);
                            record_meta_copy(
                                i, 1, 0, route_src_topk, 0);
                            record_meta_copy(
                                i, 1, 0, route_src_weights, 0);

                            if (pipeline_debug ||
                                    tensor_phone_stage_profile) {
                                printf(
                                    "[PHONE_PREFILL_ROUTE_READY] "
                                    "layer=%d chunk=%d lane=%zu "
                                    "task=%" PRIu64 " total_ms=%.3f\n",
                                    phone_route_layer,
                                    phone_route_chunk,
                                    lane,
                                    task_id,
                                    route_us / 1000.0);
                            }

                            return GGML_STATUS_SUCCESS;
                        });

                pending_phone_prefill_route_lane_task[lane] = route_task;
                pending_phone_prefill_routes[
                    { phone_route_layer, phone_route_chunk }] = {
                        lane,
                        route_task,
                        producer_seq,
                        route_worker,
                        src_hidden,
                        src_topk,
                        src_weights,
                        phone_stage_hidden,
                        phone_stage_topk,
                        phone_stage_weights,
                        stage_hidden,
                        stage_topk,
                        stage_weights,
                        dst_hidden,
                        dst_topk,
                        dst_weights,
                    };

                record_direct_copy();

                if (pipeline_debug ||
                        tensor_phone_stage_profile) {
                    printf(
                        "[PHONE_PREFILL_ROUTE_ENQUEUE] "
                        "layer=%d chunk=%d lane=%zu task=%" PRIu64
                        " bytes=%zu\n",
                        phone_route_layer,
                        phone_route_chunk,
                        lane,
                        route_task,
                        ggml_nbytes(src_hidden) +
                            ggml_nbytes(src_topk) +
                            ggml_nbytes(src_weights));
                }

                open_phone_prefill_lane1_return_gate(
                    phone_route_layer,
                    phone_route_chunk);
                return GGML_STATUS_SUCCESS;
            }

            int64_t residual_copy_us = 0;
            if (copy_residual) {
                const int64_t residual_copy_begin = ggml_time_us();
                ggml_backend_tensor_copy_async(
                    bcj_src.backend, bcj_dst.backend,
                    src_residual, dst_residual);
                residual_copy_us =
                    ggml_time_us() - residual_copy_begin;
                record_copy_wait(residual_copy_us);
                record_meta_copy(
                    i, 1, 0, src_residual, residual_copy_us);
            }

            int64_t hidden_copy_us = 0;
            int64_t topk_copy_us = 0;
            int64_t weights_copy_us = 0;
            int64_t route_batch_us = 0;
            bool route_batch_used = false;

            const ggml_backend_rpc_get_tensor_batch3_t get_tensor_batch3 =
                ggml_backend_meta_get_tensor_batch3(bcj_src.backend);
            const bool allow_route_batch =
                phone_single_owner &&
                !strict_phone_primary_fence &&
                !copy_residual &&
                get_tensor_batch3 != nullptr &&
                std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_DISABLE_ROUTE_BATCH") == nullptr;

            if (allow_route_batch) {
                const int64_t route_batch_begin = ggml_time_us();
                route_batch_used = get_tensor_batch3(
                    bcj_src.backend,
                    bcj_dst.backend,
                    src_hidden, dst_hidden,
                    src_topk, dst_topk,
                    src_weights, dst_weights);
                route_batch_us = ggml_time_us() - route_batch_begin;

                if (route_batch_used) {
                    // Account wall time once while still keeping the tensor byte
                    // counters intact for the three logical Router payloads.
                    hidden_copy_us = route_batch_us;
                    record_copy_wait(route_batch_us);
                    record_meta_copy(i, 1, 0, src_hidden, route_batch_us);
                    record_meta_copy(i, 1, 0, src_topk, 0);
                    record_meta_copy(i, 1, 0, src_weights, 0);
                }
            }

            if (!route_batch_used) {
                const int64_t hidden_copy_begin = ggml_time_us();
                ggml_backend_tensor_copy_async(
                    bcj_src.backend, bcj_dst.backend, src_hidden, dst_hidden);
                hidden_copy_us = ggml_time_us() - hidden_copy_begin;
                record_copy_wait(hidden_copy_us);
                record_meta_copy(i, 1, 0, src_hidden, hidden_copy_us);

                const int64_t topk_copy_begin = ggml_time_us();
                ggml_backend_tensor_copy_async(
                    bcj_src.backend, bcj_dst.backend, src_topk, dst_topk);
                topk_copy_us = ggml_time_us() - topk_copy_begin;
                record_copy_wait(topk_copy_us);
                record_meta_copy(i, 1, 0, src_topk, topk_copy_us);

                const int64_t weights_copy_begin = ggml_time_us();
                ggml_backend_tensor_copy_async(
                    bcj_src.backend, bcj_dst.backend, src_weights, dst_weights);
                weights_copy_us = ggml_time_us() - weights_copy_begin;
                record_copy_wait(weights_copy_us);
                record_meta_copy(i, 1, 0, src_weights, weights_copy_us);
            }

            // The following PC graph is submitted on the destination backend
            // after these copies.  Preserve the old explicit destination wait
            // only in strict/legacy mode; normal backend queue ordering carries
            // the dependency.
            int64_t route_dst_sync_us = 0;
            if (strict_phone_primary_fence) {
                const int64_t route_dst_sync_begin = ggml_time_us();
                ggml_backend_synchronize(bcj_dst.backend);
                route_dst_sync_us = ggml_time_us() - route_dst_sync_begin;
            }
            record_direct_copy();

            if (live_route_trace) {
                std::vector<float> live_dst_hidden;
                std::vector<int32_t> live_dst_topk;
                std::vector<float> live_dst_weights;

                capture_live_f32(
                    dst_hidden, live_dst_hidden, "hidden", "PC_POST_COPY");
                capture_live_i32(
                    dst_topk, live_dst_topk, "topk", "PC_POST_COPY");
                capture_live_f32(
                    dst_weights, live_dst_weights, "weights", "PC_POST_COPY");

                auto compare_live_f32 =
                    [&](const char * logical,
                        const std::vector<float> & src,
                        const std::vector<float> & dst) {
                        if (src.size() != dst.size() || src.empty()) {
                            printf(
                                "[TP_ROUTE_LIVE_DIFF] layer=%d chunk=%d "
                                "logical=%s status=SIZE src=%zu dst=%zu\n",
                                phone_route_layer, phone_route_chunk,
                                logical, src.size(), dst.size());
                            return;
                        }
                        double diff_l2 = 0.0;
                        double max_abs_diff = 0.0;
                        size_t max_index = 0;
                        for (size_t q = 0; q < src.size(); ++q) {
                            const double d =
                                (double) src[q] - (double) dst[q];
                            diff_l2 += d * d;
                            if (std::abs(d) > max_abs_diff) {
                                max_abs_diff = std::abs(d);
                                max_index = q;
                            }
                        }
                        printf(
                            "[TP_ROUTE_LIVE_DIFF] layer=%d chunk=%d "
                            "logical=%s max_abs_diff=%.9g "
                            "rms_diff=%.9g max_index=%zu\n",
                            phone_route_layer, phone_route_chunk,
                            logical, max_abs_diff,
                            std::sqrt(
                                diff_l2 / (double) src.size()),
                            max_index);
                    };

                compare_live_f32(
                    "hidden", live_src_hidden, live_dst_hidden);
                compare_live_f32(
                    "weights", live_src_weights, live_dst_weights);

                if (live_src_topk.size() == live_dst_topk.size() &&
                        !live_src_topk.empty()) {
                    size_t mismatch = 0;
                    size_t first_mismatch = 0;
                    for (size_t q = 0; q < live_src_topk.size(); ++q) {
                        if (live_src_topk[q] != live_dst_topk[q]) {
                            if (mismatch == 0) {
                                first_mismatch = q;
                            }
                            ++mismatch;
                        }
                    }
                    printf(
                        "[TP_ROUTE_LIVE_DIFF] layer=%d chunk=%d "
                        "logical=topk mismatches=%zu first_mismatch=%zu\n",
                        phone_route_layer, phone_route_chunk,
                        mismatch, first_mismatch);
                } else {
                    printf(
                        "[TP_ROUTE_LIVE_DIFF] layer=%d chunk=%d "
                        "logical=topk status=SIZE src=%zu dst=%zu\n",
                        phone_route_layer, phone_route_chunk,
                        live_src_topk.size(), live_dst_topk.size());
                }
            }

            if (tensor_phone_stage_profile) {
                const int64_t route_handoff_wall_us =
                    ggml_time_us() - tensor_phone_route_handoff_begin_us;
                auto & stage = tensor_phone_stage_entry(
                    phone_route_layer,
                    phone_route_chunk,
                    phone_route_chunk < 0);
                stage.route_handoff_us += route_handoff_wall_us;
                stage.route_handoff_count += 1;
                printf(
                    "[TENSOR_PHONE_STAGE_COMM] stage=route_handoff "
                    "mode=%s layer=%d chunk=%d sg=%zu wall_ms=%.3f "
                    "batch=%d batch_ms=%.3f copy_ms=%.3f\n",
                    phone_route_chunk < 0 ? "decode" : "prefill",
                    phone_route_layer,
                    phone_route_chunk < 0 ? 0 : phone_route_chunk,
                    i,
                    route_handoff_wall_us / 1000.0,
                    route_batch_used ? 1 : 0,
                    route_batch_us / 1000.0,
                    (residual_copy_us + hidden_copy_us +
                     topk_copy_us + weights_copy_us) / 1000.0);
            }

            if (pipeline_debug) {
                printf(
                    "%s layer=%d chunk=%d active=%zu "
                    "residual=%s residual_bytes=%zu hidden=%s hidden_bytes=%zu "
                    "topk=%s topk_bytes=%zu weights=%s weights_bytes=%zu "
                    "strict_fence=%d batch=%d route_fence_ms=%.3f dst_sync_ms=%.3f "
                    "copy_ms=%.3f batch_ms=%.3f action=%s\n",
                    phone_single_owner ?
                        "[TENSOR_PHONE_V23_ROUTER_HANDOFF]" :
                        "[TENSOR_PHONE_V2_ROUTER_HANDOFF]",
                    phone_route_layer, phone_route_chunk, active_count,
                    copy_residual ? src_residual->name : "(not-needed)",
                    copy_residual ? ggml_nbytes(src_residual) : 0,
                    src_hidden->name, ggml_nbytes(src_hidden),
                    src_topk->name, ggml_nbytes(src_topk),
                    src_weights->name, ggml_nbytes(src_weights),
                    strict_phone_primary_fence ? 1 : 0,
                    route_batch_used ? 1 : 0,
                    route_fence_us / 1000.0,
                    route_dst_sync_us / 1000.0,
                    (residual_copy_us + hidden_copy_us +
                     topk_copy_us + weights_copy_us) / 1000.0,
                    route_batch_us / 1000.0,
                    route_batch_used ?
                        "PHONE_TO_PC_BATCH3" :
                        "PHONE_TO_PC_ORDERED_GET");
            }

            open_phone_prefill_lane1_return_gate(
                phone_route_layer,
                phone_route_chunk);
            return GGML_STATUS_SUCCESS;
        }

        int phone_v2_l_out_layer_0 = -1;
        int phone_v2_l_out_layer_1 = -1;
        int phone_v2_l_out_parsed_0 = 0;
        int phone_v2_l_out_parsed_1 = 0;
        const bool phone_v2_l_out_boundary =
            n_backends == 2 &&
            active_count == 2 &&
            backend_ctx->tensor_phone_first_layer >= 0 &&
            std::sscanf(
                nodes[0]->name, "l_out-%d%n",
                &phone_v2_l_out_layer_0,
                &phone_v2_l_out_parsed_0) == 1 &&
            nodes[0]->name[phone_v2_l_out_parsed_0] == '\0' &&
            std::sscanf(
                nodes[1]->name, "l_out-%d%n",
                &phone_v2_l_out_layer_1,
                &phone_v2_l_out_parsed_1) == 1 &&
            nodes[1]->name[phone_v2_l_out_parsed_1] == '\0' &&
            phone_v2_l_out_layer_0 == phone_v2_l_out_layer_1 &&
            phone_v2_l_out_layer_0 >=
                backend_ctx->tensor_phone_first_layer &&
            phone_v2_l_out_layer_0 <
                backend_ctx->tensor_phone_last_layer;

        if (phone_v2_l_out_boundary) {
            // Both backends already hold the same complete FFN result after
            // the preceding all-reduce and the same residual ffn_inp after the
            // Router handoff.  Therefore both l_out ADDs are complete values,
            // not partials: do NOT all-reduce them again.
            handled = true;
            if (pipeline_debug) {
                printf(
                    "[TENSOR_PHONE_V2_L_OUT_SYNC] sg=%zu layer=%d "
                    "action=NO_REDUCE\n",
                    i, phone_v2_l_out_layer_0);
            }
            return GGML_STATUS_SUCCESS;
        }

        if (active_count == 1 && n_backends == 2 && active_backend == 0) {
            const bool phone_needed_next =
                i + 1 < backend_ctx->n_subgraphs && subgraph_will_execute_phone(i + 1);
            if (!phone_needed_next) {
                handled = true;
                if (pipeline_debug) {
                    GGML_LOG_INFO("[META_KEEP_PRIMARY] sg=%zu tensor=%s bytes=%zu next_phone_compute=0\n",
                           i, nodes[0]->name, ggml_nbytes(nodes[0]));
                }
                return GGML_STATUS_SUCCESS;
            }
        }

        int phone_primary_tensor_layer = -1;
        const bool phone_primary_tensor_sg =
            active_count == 1 &&
            active_backend == 1 &&
            subgraph_tensor_phone_primary_layer(
                i, phone_primary_tensor_layer);

        int inferred_phone_layer = -1;
        ggml_tensor * inferred_phone_residual = nullptr;
        bool inferred_phone_primary_tensor = false;
        if (active_count == 1 && active_backend == 1) {
            inferred_phone_residual =
                find_recent_ffn_inp_layer(
                    1, i, inferred_phone_layer);
            inferred_phone_primary_tensor =
                inferred_phone_residual != nullptr &&
                layer_is_tensor_phone_primary(
                    inferred_phone_layer);
        }

        const bool forced_phone_primary_tensor =
            force_phone_block_exit &&
            force_phone_block_layer >= 0 &&
            layer_is_tensor_phone_primary(
                force_phone_block_layer);

        const bool phone_owner_exit =
            n_backends == 2 && active_count == 1 && active_backend == 1 &&
            !phone_primary_tensor_sg &&
            !inferred_phone_primary_tensor &&
            !forced_phone_primary_tensor &&
            (force_phone_block_exit ||
             i + 1 >= backend_ctx->n_subgraphs ||
             !subgraph_will_execute_phone(i + 1));

        if (pipeline_debug &&
            (phone_primary_tensor_sg ||
             inferred_phone_primary_tensor ||
             forced_phone_primary_tensor)) {
            const int kept_layer =
                phone_primary_tensor_sg ?
                    phone_primary_tensor_layer :
                (inferred_phone_primary_tensor ?
                    inferred_phone_layer :
                    force_phone_block_layer);
            printf(
                "[TENSOR_PHONE_KEEP_OWNER] sg=%zu layer=%d node=%s "
                "source=%s next_pc=%d next_phone=%d\n",
                i, kept_layer,
                nodes[1] != nullptr ? nodes[1]->name : "(null)",
                phone_primary_tensor_sg ? "subgraph" :
                    (inferred_phone_primary_tensor ?
                        "recent_ffn_inp" : "forced_block"),
                i + 1 < backend_ctx->n_subgraphs ?
                    (int) subgraph_will_execute_pc(i + 1) : -1,
                i + 1 < backend_ctx->n_subgraphs ?
                    (int) subgraph_will_execute_phone(i + 1) : -1);
        }

        if (active_count == 1 && n_backends == 2 && active_backend == 1 &&
                std::getenv("GGML_META_TP_FFN_NUMERIC_TRACE") != nullptr &&
                backend_ctx->tensor_phone_first_layer >= 0) {
            int trace_lout_layer = -1;
            int trace_lout_parsed = 0;
            ggml_tensor * trace_lout = nodes[1];
            const bool is_first_tp_lout =
                trace_lout != nullptr &&
                std::sscanf(
                    trace_lout->name, "l_out-%d%n",
                    &trace_lout_layer, &trace_lout_parsed) == 1 &&
                trace_lout->name[trace_lout_parsed] == '\0' &&
                trace_lout_layer == backend_ctx->tensor_phone_first_layer;

            if (is_first_tp_lout &&
                    trace_lout->type == GGML_TYPE_F32 &&
                    trace_lout->src[0] != nullptr &&
                    trace_lout->src[1] != nullptr &&
                    trace_lout->src[0]->type == GGML_TYPE_F32 &&
                    trace_lout->src[1]->type == GGML_TYPE_F32 &&
                    ggml_nelements(trace_lout) ==
                        ggml_nelements(trace_lout->src[0]) &&
                    ggml_nelements(trace_lout) ==
                        ggml_nelements(trace_lout->src[1])) {
                auto & ph_cfg = backend_ctx->backend_configs[1];
                const ggml_backend_rpc_fence_t rpc_fence =
                    ggml_backend_meta_get_rpc_fence(ph_cfg.backend);
                if (rpc_fence != nullptr) {
                    rpc_fence(ph_cfg.backend);
                } else {
                    ggml_backend_synchronize(ph_cfg.backend);
                }

                const size_t n = ggml_nelements(trace_lout);
                std::vector<float> out(n);
                std::vector<float> src_a(n);
                std::vector<float> src_b(n);
                ggml_backend_tensor_get(
                    trace_lout, out.data(), 0, n * sizeof(float));
                ggml_backend_tensor_get(
                    trace_lout->src[0], src_a.data(), 0, n * sizeof(float));
                ggml_backend_tensor_get(
                    trace_lout->src[1], src_b.data(), 0, n * sizeof(float));

                double out_sum = 0.0;
                double out_l2 = 0.0;
                double diff_l2 = 0.0;
                double max_abs_diff = 0.0;
                size_t max_diff_index = 0;
                for (size_t q = 0; q < n; ++q) {
                    const double x = out[q];
                    const double expected =
                        (double) src_a[q] + (double) src_b[q];
                    const double d = x - expected;
                    out_sum += x;
                    out_l2 += x * x;
                    diff_l2 += d * d;
                    if (std::abs(d) > max_abs_diff) {
                        max_abs_diff = std::abs(d);
                        max_diff_index = q;
                    }
                }

                printf(
                    "[TP_LOUT_NUMERIC] layer=%d out=%s src0=%s src1=%s n=%zu "
                    "sum=%.9f l2=%.9f max_abs_diff=%.9g rms_diff=%.9g "
                    "max_diff_index=%zu out_v0=%.9f expected_v0=%.9f "
                    "out_v1=%.9f expected_v1=%.9f\n",
                    trace_lout_layer,
                    trace_lout->name,
                    trace_lout->src[0]->name,
                    trace_lout->src[1]->name,
                    n, out_sum, std::sqrt(out_l2),
                    max_abs_diff,
                    n > 0 ? std::sqrt(diff_l2 / (double) n) : 0.0,
                    max_diff_index,
                    n > 0 ? out[0] : 0.0f,
                    n > 0 ? src_a[0] + src_b[0] : 0.0f,
                    n > 1 ? out[1] : 0.0f,
                    n > 1 ? src_a[1] + src_b[1] : 0.0f);
            }
        }

        if (active_count == 1 && n_backends == 2 && active_backend == 1) {
            const bool pc_needed_next =
                i + 1 < backend_ctx->n_subgraphs && subgraph_will_execute_pc(i + 1);
            if (!pc_needed_next && !phone_owner_exit) {
                handled = true;
                if (pipeline_debug) {
                    printf("[META_KEEP_PHONE] sg=%zu tensor=%s bytes=%zu next_pc_compute=0\n",
                           i, nodes[1]->name, ggml_nbytes(nodes[1]));
                }
                return GGML_STATUS_SUCCESS;
            }
        }

        if (active_count == 1) {
            handled = true;
            auto & bcj_src = backend_ctx->backend_configs[active_backend];
            const bool defer_to_terminal_l_out =
                active_backend == 1 && force_phone_block_exit &&
                !forced_phone_primary_tensor &&
                !layer_is_tensor_phone_primary(force_phone_block_layer) &&
                subgraph_is_exact_l_out_bridge(
                    i + 1, force_phone_block_layer);
            if (defer_to_terminal_l_out) {
                ggml_tensor * disabled_l_out = disable_layer_output_producers(
                    0, i + 1, force_phone_block_layer);
                GGML_ASSERT(disabled_l_out != nullptr);

                deferred_phone_exit_layer = force_phone_block_layer;
                if (pipeline_debug) {
                    printf(
                        "[PHONE_EXIT_DEFER] layer=%d sg=%zu->%zu tensor=%s\n",
                        deferred_phone_exit_layer, i, i + 1, disabled_l_out->name);
                }
                return GGML_STATUS_SUCCESS;
            }

            if (active_backend == 1 && deferred_phone_exit_layer >= 0) {
                char expected[64];
                std::snprintf(
                    expected, sizeof(expected), "l_out-%d", deferred_phone_exit_layer);
                const bool already_final_l_out =
                    std::strcmp(nodes[1]->name, expected) == 0;
                if (already_final_l_out) {
                    ggml_tensor * dst_l_out = find_exact_named_tensor(0, expected);
                    GGML_ASSERT(dst_l_out != nullptr);
                    GGML_ASSERT(ggml_nbytes(dst_l_out) == ggml_nbytes(nodes[1]));

                    const ggml_backend_rpc_fence_t rpc_fence =
                        ggml_backend_meta_get_rpc_fence(bcj_src.backend);
                    const int64_t wait_begin_us = ggml_time_us();
                    if (pipeline_debug) {
                        printf("[PHONE_BLOCK_WAIT_BEGIN] layer=%d t=%" PRId64 "\n", deferred_phone_exit_layer,
                               wait_begin_us);
                    }
                    if (rpc_fence != nullptr) {
                        rpc_fence(bcj_src.backend);
                    } else {
                        ggml_backend_synchronize(bcj_src.backend);
                    }
                    const int64_t wait_end_us = ggml_time_us();
                    if (pipeline_debug) {
                        printf("[PHONE_BLOCK_WAIT_END] layer=%d t=%" PRId64 " dur=%.3f ms\n",
                               deferred_phone_exit_layer, wait_end_us, (wait_end_us - wait_begin_us) / 1000.0);
                    }
                    if (phone_exit_profile) {
                        printf(
                            "[PHONE_ASYNC_META_FENCE] sg=%zu layer=%d phase=pre_copy "
                            "wait_ms=%.3f rpc_fence=%d\n",
                            i,
                            deferred_phone_exit_layer,
                            (wait_end_us - wait_begin_us) / 1000.0,
                            rpc_fence != nullptr ? 1 : 0);
                    }

                    auto & bcj_dst = backend_ctx->backend_configs[0];
                    const int64_t copy_start_us = ggml_time_us();
                    if (pipeline_debug) {
                        printf("[PHONE_EXIT_COPY_BEGIN] layer=%d t=%" PRId64 " bytes=%zu\n",
                               deferred_phone_exit_layer, copy_start_us, ggml_nbytes(dst_l_out));
                    }
                    ggml_backend_tensor_copy_async(
                        bcj_src.backend, bcj_dst.backend, nodes[1], dst_l_out);
                    ggml_backend_synchronize(bcj_dst.backend);
                    const int64_t copy_us = ggml_time_us() - copy_start_us;
                    if (pipeline_debug) {
                        printf("[PHONE_EXIT_COPY_END] layer=%d t=%" PRId64 " dur=%.3f ms\n",
                               deferred_phone_exit_layer, copy_start_us + copy_us, copy_us / 1000.0);
                    }
                    if (phone_exit_profile) {
                        printf(
                            "[PHONE_ASYNC_META_COPY] sg=%zu layer=%d phase=final_l_out "
                            "bytes=%zu copy_ms=%.3f\n",
                            i,
                            deferred_phone_exit_layer,
                            ggml_nbytes(dst_l_out),
                            copy_us / 1000.0);
                    }
                    record_copy_wait(copy_us);
                    record_meta_copy(i, 1, 0, dst_l_out, copy_us);
                    record_direct_copy();

                    if (pipeline_debug) {
                        meta_debug_tensor(
                            bcj_src.backend, nodes[1], "PHONE FINAL_L_OUT");
                        meta_debug_tensor(
                            bcj_dst.backend, dst_l_out, "PC FINAL_L_OUT");
                        printf(
                            "[META_FINAL_L_OUT_HANDOFF] layer=%d 1->0 tensor=%s\n",
                            deferred_phone_exit_layer, dst_l_out->name);
                    }

                    debug_terminal_handoff_dst = dst_l_out;
                    deferred_phone_exit_layer = -1;
                    return GGML_STATUS_SUCCESS;
                }
            }

            const bool tp_layer_context =
                phone_primary_tensor_sg ||
                inferred_phone_primary_tensor ||
                forced_phone_primary_tensor;
            const bool internal_pc_handoff =
                n_backends == 2 && active_backend == 1 &&
                !tp_layer_context &&
                i + 1 < backend_ctx->n_subgraphs &&
                (subgraph_is_prefill_pc_only(i + 1) ||
                 subgraph_is_decode_pc_only_norm(i + 1));
            int layer = force_phone_block_exit ? force_phone_block_layer : -1;
            ggml_tensor * src_residual = nullptr;
            if (internal_pc_handoff || phone_owner_exit) {
                src_residual = force_phone_block_exit ?
                    find_recent_ffn_inp(1, i, layer) :
                    find_recent_ffn_inp_layer(1, i, layer);
            }
            ggml_tensor * dst_l_out = nullptr;
            if (internal_pc_handoff) {
                dst_l_out = disable_layer_output_producers(0, i + 1, layer);
            } else if (phone_owner_exit && src_residual != nullptr) {
                char expected[64];
                std::snprintf(expected, sizeof(expected), "l_out-%d", layer);
                dst_l_out = find_exact_named_tensor(0, expected);
                GGML_ASSERT(dst_l_out != nullptr);
                GGML_ASSERT(std::strcmp(dst_l_out->name, expected) == 0);
                GGML_ASSERT(ggml_nbytes(dst_l_out) == ggml_nbytes(nodes[1]));
            }
            const bool external_handoff_candidate =
                !tp_layer_context &&
                phone_owner_exit && !internal_pc_handoff &&
                src_residual != nullptr && dst_l_out != nullptr &&
                ggml_nbytes(nodes[1]) == ggml_nbytes(src_residual) &&
                ggml_nbytes(nodes[1]) == ggml_nbytes(dst_l_out);
            const bool external_backend_handoff =
                external_handoff_candidate &&
                (i + 1 >= backend_ctx->n_subgraphs ||
                 has_layer_output_producer(0, i + 1, layer));

            if (external_handoff_candidate && !external_backend_handoff && pipeline_debug) {
                printf("[META_HANDOFF_SKIP] sg=%zu layer=%d node=%s reason=no_pc_l_out_producer\n",
                    i, layer, nodes[1]->name);
            }

            if (external_backend_handoff && i + 1 < backend_ctx->n_subgraphs) {
                ggml_tensor * disabled_l_out =
                    disable_layer_output_producers(0, i + 1, layer);
                if (disabled_l_out == nullptr) {
                    fprintf(stderr,
                        "[META_HANDOFF_MISSING_PRODUCER] sg=%zu layer=%d node=%s dst=%s tp=%d forced=%d\n",
                        i, layer, nodes[1]->name, dst_l_out->name,
                        (int) layer_is_tensor_phone_primary(layer),
                        (int) force_phone_block_exit);
                }
                GGML_ASSERT(disabled_l_out != nullptr);
                GGML_ASSERT(disabled_l_out == dst_l_out);

                if (pipeline_debug) {
                    printf(
                        "[EXT_DISABLE_PRODUCER] sg=%zu layer=%d tensor=%s\n",
                        i + 1, layer, disabled_l_out->name);
                }
            }

            const bool layer_handoff_to_pc =
                internal_pc_handoff || external_backend_handoff;

            if (pipeline_debug && external_backend_handoff) {
                char expected[64];
                std::snprintf(expected, sizeof(expected), "l_out-%d", layer);

                ggml_tensor * outer_l_out = nullptr;
                for (int k = 0; k < cgraph->n_nodes; ++k) {
                    if (std::strcmp(cgraph->nodes[k]->name, expected) == 0) {
                        outer_l_out = cgraph->nodes[k];
                        break;
                    }
                }
                GGML_ASSERT(outer_l_out != nullptr);

                ggml_tensor * outer_simple0 =
                    ggml_backend_meta_buffer_simple_tensor(outer_l_out, 0);
                ggml_tensor * outer_simple1 = n_backends > 1 ?
                    ggml_backend_meta_buffer_simple_tensor(outer_l_out, 1) : nullptr;
                printf(
                    "[OUTER_MAP] layer=%d outer=%p dst=%p simple0=%p simple1=%p "
                    "outer_name=%s s0=%s s1=%s\n",
                    layer, (void *) outer_l_out, (void *) dst_l_out,
                    (void *) outer_simple0, (void *) outer_simple1,
                    outer_l_out->name,
                    outer_simple0 != nullptr ? outer_simple0->name : "(null)",
                    outer_simple1 != nullptr ? outer_simple1->name : "(null)");
            }

            if (pipeline_debug && dst_l_out != nullptr) {
                char expected[64];
                std::snprintf(expected, sizeof(expected), "l_out-%d", layer);
                printf(
                    "[HANDOFF_DST_CHECK] layer=%d dst=%s expected=%s ptr=%p\n",
                    layer, dst_l_out->name, expected, (void *) dst_l_out);
                GGML_ASSERT(std::strcmp(dst_l_out->name, expected) == 0);
            }

            if (pipeline_debug && phone_owner_exit) {
                printf(
                    "[PHONE_EXIT_DBG] sg=%zu layer=%d node=%s residual=%s "
                    "next_pc=%d next_phone=%d internal=%d external=%d forced=%d\n",
                    i, layer, nodes[1] != nullptr ? nodes[1]->name : "(null)",
                    src_residual != nullptr ? src_residual->name : "(null)",
                    i + 1 < backend_ctx->n_subgraphs ?
                        (int) subgraph_will_execute_pc(i + 1) : -1,
                    i + 1 < backend_ctx->n_subgraphs ?
                        (int) subgraph_will_execute_phone(i + 1) : -1,
                    (int) internal_pc_handoff, (int) external_backend_handoff,
                    (int) force_phone_block_exit);
            }

            GGML_ASSERT(!layer_handoff_to_pc ||
                (src_residual != nullptr && dst_l_out != nullptr &&
                 ggml_nbytes(nodes[1]) == ggml_nbytes(src_residual) &&
                 ggml_nbytes(nodes[1]) == ggml_nbytes(dst_l_out)));
            if (layer_handoff_to_pc) {
                const ggml_backend_rpc_fence_t rpc_fence =
                    ggml_backend_meta_get_rpc_fence(bcj_src.backend);
                auto fence_src = [&]() {
                    if (rpc_fence != nullptr) {
                        rpc_fence(bcj_src.backend);
                    } else {
                        ggml_backend_synchronize(bcj_src.backend);
                    }
                };

                const int64_t wait_begin_us = ggml_time_us();
                if (pipeline_debug && phone_owner_exit) {
                    printf("[PHONE_BLOCK_WAIT_BEGIN] layer=%d t=%" PRId64 "\n", layer, wait_begin_us);
                }
                fence_src();
                const int64_t wait_end_us = ggml_time_us();
                if (pipeline_debug && phone_owner_exit) {
                    printf("[PHONE_BLOCK_WAIT_END] layer=%d t=%" PRId64 " dur=%.3f ms\n", layer, wait_end_us,
                           (wait_end_us - wait_begin_us) / 1000.0);
                }
                if (phone_exit_profile) {
                    printf(
                        "[PHONE_ASYNC_META_FENCE] sg=%zu layer=%d phase=pre_add "
                        "wait_ms=%.3f rpc_fence=%d\n",
                        i,
                        layer,
                        (wait_end_us - wait_begin_us) / 1000.0,
                        rpc_fence != nullptr ? 1 : 0);
                }

                if (pipeline_debug) {
                    meta_debug_tensor(bcj_src.backend, nodes[1], "PHONE BEFORE_ADD");
                }

                ggml_tensor * node_layer = get_node_aux(nodes[1]);
                node_layer->view_src = nodes[1]->view_src == nullptr ? nodes[1] : nodes[1]->view_src;
                node_layer->view_offs = nodes[1]->view_offs;
                node_layer->op = GGML_OP_ADD;
                node_layer->src[0] = nodes[1];
                node_layer->src[1] = src_residual;
                node_layer->flags |= GGML_TENSOR_FLAG_COMPUTE;
                ggml_backend_view_init(node_layer);

                ggml_cgraph * layer_graph = get_cgraph_aux();
                layer_graph->nodes[0] = node_layer;
                layer_graph->n_nodes = 1;
                const int64_t add_submit_begin_us = ggml_time_us();
                const ggml_status status =
                    ggml_backend_graph_compute_async(bcj_src.backend, layer_graph);
                const int64_t add_submit_us =
                    ggml_time_us() - add_submit_begin_us;
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
                const int64_t add_fence_begin_us = ggml_time_us();
                fence_src();
                const int64_t add_fence_us =
                    ggml_time_us() - add_fence_begin_us;
                if (phone_exit_profile) {
                    printf(
                        "[PHONE_ASYNC_META_ADD] sg=%zu layer=%d "
                        "submit_ms=%.3f fence_ms=%.3f\n",
                        i,
                        layer,
                        add_submit_us / 1000.0,
                        add_fence_us / 1000.0);
                }

                if (pipeline_debug) {
                    meta_debug_tensor(bcj_src.backend, nodes[1], "PHONE POST_ADD");
                }

                auto & bcj_dst = backend_ctx->backend_configs[0];
                if (pipeline_debug && external_backend_handoff) {
                    printf(
                        "[EXT_HANDOFF_MAP] layer=%d phone_node=%s dst=%s "
                        "phone_ptr=%p dst_ptr=%p\n",
                        layer, nodes[1] != nullptr ? nodes[1]->name : "(null)",
                        dst_l_out != nullptr ? dst_l_out->name : "(null)",
                        (void *) nodes[1], (void *) dst_l_out);
                }
                const int64_t copy_start_us = ggml_time_us();
                if (pipeline_debug && phone_owner_exit) {
                    printf("[PHONE_EXIT_COPY_BEGIN] layer=%d t=%" PRId64 " bytes=%zu\n", layer, copy_start_us,
                           ggml_nbytes(dst_l_out));
                }
                ggml_backend_tensor_copy_async(
                    bcj_src.backend, bcj_dst.backend, nodes[1], dst_l_out);
                ggml_backend_synchronize(bcj_dst.backend);
                const int64_t copy_us = ggml_time_us() - copy_start_us;
                if (pipeline_debug && phone_owner_exit) {
                    printf("[PHONE_EXIT_COPY_END] layer=%d t=%" PRId64 " dur=%.3f ms\n", layer,
                           copy_start_us + copy_us, copy_us / 1000.0);
                }
                if (phone_exit_profile) {
                    printf(
                        "[PHONE_ASYNC_META_COPY] sg=%zu layer=%d phase=handoff "
                        "bytes=%zu copy_ms=%.3f external=%d internal=%d\n",
                        i,
                        layer,
                        ggml_nbytes(dst_l_out),
                        copy_us / 1000.0,
                        external_backend_handoff ? 1 : 0,
                        internal_pc_handoff ? 1 : 0);
                }
                record_copy_wait(copy_us);
                record_meta_copy(i, 1, 0, dst_l_out, copy_us);
                record_direct_copy();

                if (pipeline_debug) {
                    meta_debug_tensor(bcj_dst.backend, dst_l_out, "PC POST_COPY");
                }
                if (external_backend_handoff) {
                    debug_terminal_handoff_dst = dst_l_out;
                }

                if (pipeline_debug) {
                    printf(
                        "[META_LAYER_HANDOFF] layer=%d 1->0 tensor=%s bytes=%zu\n",
                        layer, dst_l_out->name, ggml_nbytes(dst_l_out));
                }
                return GGML_STATUS_SUCCESS;
            }
            GGML_ASSERT(ggml_is_contiguous(nodes[active_backend]));
            for (size_t j_dst = 0; j_dst < n_backends; ++j_dst) {
                if (j_dst == active_backend) {
                    continue;
                }
                auto & bcj_dst = backend_ctx->backend_configs[j_dst];
                GGML_ASSERT(ggml_is_contiguous(nodes[j_dst]));

                const bool decode_pc_to_phone_ffn_norm =
                    n_backends == 2 &&
                    active_backend == 0 &&
                    j_dst == 1 &&
                    nodes[active_backend]->ne[1] == 1 &&
                    std::strncmp(nodes[active_backend]->name, "ffn_norm-", 9) == 0 &&
                    get_ffn_down_boundary_node(0, i + 1) != nullptr &&
                    get_ffn_down_boundary_node(1, i + 1) != nullptr;

                if (decode_pc_to_phone_ffn_norm) {
                    const auto prepare_fused_input =
                        ggml_backend_meta_get_fused_ffn_input_preparer(bcj_dst.backend);
                    if (prepare_fused_input != nullptr) {
                        const bool prepared = prepare_fused_input(bcj_dst.backend, nodes[j_dst]);
                        GGML_ASSERT(prepared);
                    }
                }

                int phone_ffn_layer = -1;
                int phone_ffn_chunk = -1;
                const char * phone_ffn_mode = nullptr;
                bool pc_full_ffn_to_phone = false;
                if (n_backends == 2 &&
                        active_backend == 0 &&
                        j_dst == 1) {
                    int parsed = 0;
                    int decode_layer = -1;
                    const bool decode_single =
                        nodes[active_backend]->ne[1] == 1 &&
                        std::sscanf(
                            nodes[active_backend]->name,
                            "ffn_moe_out-%d%n",
                            &decode_layer, &parsed) == 1 &&
                        nodes[active_backend]->name[parsed] == '\0';

                    int decode_chunk = -1;
                    int decode_chunk_layer = -1;
                    const bool decode_chunked =
                        ggml_backend_meta_parse_decode_ffn_chunk(
                            nodes[active_backend]->name,
                            decode_chunk, decode_chunk_layer);

                    int prefill_chunk = -1;
                    int prefill_layer = -1;
                    const bool prefill_chunked =
                        ggml_backend_meta_parse_prefill_down_chunk(
                            nodes[active_backend]->name,
                            prefill_chunk, prefill_layer);

                    if (decode_single) {
                        phone_ffn_layer = decode_layer;
                        phone_ffn_chunk = 0;
                        phone_ffn_mode = "decode-single";
                    } else if (decode_chunked) {
                        phone_ffn_layer = decode_chunk_layer;
                        phone_ffn_chunk = decode_chunk;
                        phone_ffn_mode = "decode";
                    } else if (prefill_chunked) {
                        phone_ffn_layer = prefill_layer;
                        phone_ffn_chunk = prefill_chunk;
                        phone_ffn_mode = "prefill";
                    }

                    pc_full_ffn_to_phone =
                        phone_ffn_layer >= 0 &&
                        layer_is_tensor_phone_primary(phone_ffn_layer) &&
                        i + 1 < backend_ctx->n_subgraphs &&
                        subgraph_will_execute_phone(i + 1);
                }

                const bool strict_phone_primary_fence =
                    phone_primary_strict_ffn_handoff;

                // ggml_backend_tensor_copy_async() falls back to a blocking
                // copy for RPC because the RPC backend has no cpy_tensor_async
                // hook.  That fallback already synchronizes the PC producer,
                // and the RPC SET_TENSOR is ordered before the following Phone
                // graph on the same control socket.  Keep the old explicit
                // fences only as a correctness/debug fallback.
                int64_t source_fence_us = 0;
                if (pc_full_ffn_to_phone && strict_phone_primary_fence) {
                    const int64_t source_fence_start_us = ggml_time_us();
                    ggml_backend_synchronize(bcj_src.backend);
                    source_fence_us = ggml_time_us() - source_fence_start_us;
                }

                const int64_t copy_start_us = ggml_time_us();
                ggml_backend_tensor_copy_async(
                        bcj_src.backend, bcj_dst.backend, nodes[active_backend], nodes[j_dst]);

                if (pc_full_ffn_to_phone && strict_phone_primary_fence) {
                    const ggml_backend_rpc_fence_t rpc_fence =
                        ggml_backend_meta_get_rpc_fence(bcj_dst.backend);
                    if (rpc_fence != nullptr) {
                        rpc_fence(bcj_dst.backend);
                    } else {
                        ggml_backend_synchronize(bcj_dst.backend);
                    }
                }

                const int64_t copy_us = ggml_time_us() - copy_start_us;
                record_copy_wait(copy_us);
                record_meta_copy(i, active_backend, j_dst, nodes[active_backend], copy_us);

                if (pc_full_ffn_to_phone && pipeline_debug) {
                    printf(
                        "[TENSOR_PHONE_V24_FFN_DIRECT] sg=%zu layer=%d "
                        "chunk=%d mode=%s tensor=%s bytes=%zu "
                        "strict_fence=%d source_fence_ms=%.3f copy_ms=%.3f "
                        "action=PC_FULL_TO_PHONE_ORDERED_COPY\n",
                        i, phone_ffn_layer, phone_ffn_chunk,
                        phone_ffn_mode != nullptr ? phone_ffn_mode : "unknown",
                        nodes[active_backend]->name,
                        ggml_nbytes(nodes[active_backend]),
                        strict_phone_primary_fence ? 1 : 0,
                        source_fence_us / 1000.0,
                        copy_us / 1000.0);
                }
            }
            record_direct_copy();
            return GGML_STATUS_SUCCESS;
        }
        if (backend_ctx->compute_workers == nullptr) {
            backend_ctx->compute_workers =
                new ggml_backend_meta_compute_workers(backend_ctx, n_backends);
        }

        if (backend_ctx->transfer_worker == nullptr) {
            backend_ctx->transfer_worker =
                new ggml_backend_meta_transfer_worker();
        }
        int decode_chunk_0 = -1;
        int decode_layer_0 = -1;
        int decode_chunk_1 = -1;
        int decode_layer_1 = -1;
        const bool is_ffn_down_chunk =
            n_backends == 2 &&
            active_count == 2 &&
            ggml_backend_meta_parse_decode_ffn_chunk(
                nodes[0]->name, decode_chunk_0, decode_layer_0) &&
            ggml_backend_meta_parse_decode_ffn_chunk(
                nodes[1]->name, decode_chunk_1, decode_layer_1);

        if (std::getenv("GGML_META_TP_FFN_FLAG_TRACE") != nullptr &&
                n_backends == 2) {
            int trace_layer_0 = -1;
            int trace_layer_1 = -1;
            int trace_parsed_0 = 0;
            int trace_parsed_1 = 0;
            const bool trace_moe_0 =
                std::sscanf(nodes[0]->name, "ffn_moe_out-%d%n",
                            &trace_layer_0, &trace_parsed_0) == 1 &&
                nodes[0]->name[trace_parsed_0] == '\0';
            const bool trace_moe_1 =
                std::sscanf(nodes[1]->name, "ffn_moe_out-%d%n",
                            &trace_layer_1, &trace_parsed_1) == 1 &&
                nodes[1]->name[trace_parsed_1] == '\0';
            int trace_chunk_0 = -1;
            int trace_prefill_layer_0 = -1;
            int trace_chunk_1 = -1;
            int trace_prefill_layer_1 = -1;
            const bool trace_prefill_0 =
                ggml_backend_meta_parse_prefill_down_chunk(
                    nodes[0]->name, trace_chunk_0, trace_prefill_layer_0);
            const bool trace_prefill_1 =
                ggml_backend_meta_parse_prefill_down_chunk(
                    nodes[1]->name, trace_chunk_1, trace_prefill_layer_1);
            if (trace_moe_0 || trace_moe_1 || trace_prefill_0 || trace_prefill_1) {
                printf(
                    "[TP_FFN_BOUNDARY] sg=%zu active=%zu "
                    "pc={name=%s op=%s compute=%d ne=[%" PRId64 ",%" PRId64 "]} "
                    "phone={name=%s op=%s compute=%d ne=[%" PRId64 ",%" PRId64 "]}\n",
                    i,
                    active_count,
                    nodes[0]->name,
                    ggml_op_name(nodes[0]->op),
                    !!(nodes[0]->flags & GGML_TENSOR_FLAG_COMPUTE),
                    nodes[0]->ne[0], nodes[0]->ne[1],
                    nodes[1]->name,
                    ggml_op_name(nodes[1]->op),
                    !!(nodes[1]->flags & GGML_TENSOR_FLAG_COMPUTE),
                    nodes[1]->ne[0], nodes[1]->ne[1]);
            }
        }

        // Qwen3-MoE with LLAMA_CHUNKS=1 does not create
        // ffn_down_chunk_0-* boundaries. Its split expert result terminates at
        // ffn_moe_out-*. For Phone-primary Tensor layers that tensor is still a
        // PARTIAL result and must be reduced into the Phone owner exactly like
        // the chunked decode boundary. Falling through to the generic
        // all-reduce needlessly mirrors the result back to PC and bypasses the
        // Phone-primary ownership path.
        int decode_single_layer_0 = -1;
        int decode_single_layer_1 = -1;
        int decode_single_parsed_0 = 0;
        int decode_single_parsed_1 = 0;
        const bool is_single_decode_moe_out =
            n_backends == 2 &&
            active_count == 2 &&
            nodes[0]->ne[1] == 1 &&
            nodes[1]->ne[1] == 1 &&
            std::sscanf(
                nodes[0]->name, "ffn_moe_out-%d%n",
                &decode_single_layer_0,
                &decode_single_parsed_0) == 1 &&
            nodes[0]->name[decode_single_parsed_0] == '\0' &&
            std::sscanf(
                nodes[1]->name, "ffn_moe_out-%d%n",
                &decode_single_layer_1,
                &decode_single_parsed_1) == 1 &&
            nodes[1]->name[decode_single_parsed_1] == '\0' &&
            decode_single_layer_0 == decode_single_layer_1;

        int prefill_down_chunk_0 = -1;
        int prefill_down_layer_0 = -1;
        int prefill_down_chunk_1 = -1;
        int prefill_down_layer_1 = -1;
        const bool is_prefill_down_chunk =
            n_backends == 2 &&
            active_count == 2 &&
            ggml_backend_meta_parse_prefill_down_chunk(
                nodes[0]->name, prefill_down_chunk_0, prefill_down_layer_0) &&
            ggml_backend_meta_parse_prefill_down_chunk(
                nodes[1]->name, prefill_down_chunk_1, prefill_down_layer_1) &&
            prefill_down_chunk_0 == prefill_down_chunk_1 &&
            prefill_down_layer_0 == prefill_down_layer_1;

        const int phone_primary_decode_layer =
            is_ffn_down_chunk ? decode_layer_0 :
            (is_single_decode_moe_out ? decode_single_layer_0 : -1);
        const bool phone_primary_tensor_down =
            (is_prefill_down_chunk &&
             layer_attention_phone_owned(prefill_down_layer_0)) ||
            ((is_ffn_down_chunk || is_single_decode_moe_out) &&
             layer_attention_phone_owned(phone_primary_decode_layer));
        if (phone_primary_tensor_down) {
            const int layer = is_prefill_down_chunk ?
                prefill_down_layer_0 : phone_primary_decode_layer;
            const int chunk = is_prefill_down_chunk ?
                prefill_down_chunk_0 :
                (is_ffn_down_chunk ? decode_chunk_0 : 0);
            const char * mode = is_prefill_down_chunk ?
                "prefill" :
                (is_single_decode_moe_out ? "decode-single" : "decode");

            // Phone-primary v2.3:
            // - every Tensor-layer l_out is Phone-owned in single-owner mode;
            // - therefore every Tensor FFN can reduce only PC partial -> Phone;
            // - if the terminal Tensor activation is later needed on PC, the
            //   completed l_out crosses once at the ordinary backend boundary.
            const bool phone_l_out_owner =
                std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER") != nullptr;
            const bool one_way_reduce =
                phone_l_out_owner &&
                std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE") != nullptr;

            if (one_way_reduce) {
                handled = true;
                const int64_t tensor_phone_return_begin_us =
                    tensor_phone_stage_profile ? ggml_time_us() : 0;

                constexpr size_t j_src = 0; // PC FFN partial
                constexpr size_t j_dst = 1; // Phone FFN partial / owner
                auto & bcj_src = backend_ctx->backend_configs[j_src];
                auto & bcj_dst = backend_ctx->backend_configs[j_dst];
                ggml_tensor * node_src = nodes[j_src];
                ggml_tensor * node_dst = nodes[j_dst];

                GGML_ASSERT(ggml_is_contiguous(node_src));
                GGML_ASSERT(ggml_is_contiguous(node_dst));
                GGML_ASSERT(ggml_nbytes(node_src) == ggml_nbytes(node_dst));

                ggml_tensor * node_tmp = get_node_aux(node_dst);
                set_tmp_data(node_tmp, j_dst, 0);

                // Relaxed Phone-primary ordering:
                // - RPC has no cpy_tensor_async hook, so the generic async-copy
                //   API already falls back to source sync + blocking copy.
                // - PC->Phone SET_TENSOR and the following ADD graph are sent
                //   on the same RPC control socket, preserving command order.
                // - the next Phone->PC Router handoff also uses the graph
                //   thread's control socket; its blocking GET_TENSOR observes
                //   all prior Phone queue work in order.
                //
                // STRICT_REDUCE restores fence-before/fence-after behavior
                // for the one-way partial-return + ADD reduction path.
                const bool strict_phone_primary_fence =
                    phone_primary_strict_reduce;
                const auto fence_backend = [&](ggml_backend_t backend) {
                    const ggml_backend_rpc_fence_t rpc_fence =
                        ggml_backend_meta_get_rpc_fence(backend);
                    if (rpc_fence != nullptr) {
                        rpc_fence(backend);
                    } else {
                        ggml_backend_synchronize(backend);
                    }
                };

                int64_t src_fence_us = 0;
                if (strict_phone_primary_fence) {
                    const int64_t src_fence_start_us = ggml_time_us();
                    fence_backend(bcj_src.backend);
                    src_fence_us = ggml_time_us() - src_fence_start_us;
                }

                ggml_tensor * node_red = get_node_aux(node_dst);
                node_red->view_src =
                    node_dst->view_src == nullptr ?
                        node_dst : node_dst->view_src;
                node_red->view_offs = node_dst->view_offs;
                node_red->op = GGML_OP_ADD;
                node_red->src[0] = node_dst;
                node_red->src[1] = node_tmp;
                node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
                ggml_backend_view_init(node_red);

                ggml_cgraph * cgraph_aux = get_cgraph_aux();
                cgraph_aux->nodes[0] = node_red;
                cgraph_aux->n_nodes = 1;

                const ggml_backend_rpc_set_tensor_graph_t set_tensor_graph =
                    ggml_backend_meta_get_set_tensor_graph(bcj_dst.backend);
                const bool allow_fused_set_add =
                    !strict_phone_primary_fence &&
                    set_tensor_graph != nullptr &&
                    std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_DISABLE_FUSED_SET_ADD") == nullptr;

                bool fused_set_add_used = false;
                int64_t fused_set_add_us = 0;
                int64_t copy_us = 0;
                int64_t dst_fence_us = 0;
                int64_t add_submit_us = 0;

                if (allow_fused_set_add) {
                    const int64_t fused_begin_us = ggml_time_us();
                    fused_set_add_used = set_tensor_graph(
                        bcj_src.backend,
                        bcj_dst.backend,
                        node_src,
                        node_tmp,
                        cgraph_aux);
                    fused_set_add_us = ggml_time_us() - fused_begin_us;
                    if (fused_set_add_used) {
                        copy_us = fused_set_add_us;
                        record_copy_wait(fused_set_add_us);
                        record_meta_copy(
                            i, j_src, j_dst, node_src, fused_set_add_us);
                    }
                }

                if (!fused_set_add_used) {
                    const int64_t copy_start_us = ggml_time_us();
                    ggml_backend_tensor_copy_async(
                        bcj_src.backend, bcj_dst.backend,
                        node_src, node_tmp);
                    copy_us = ggml_time_us() - copy_start_us;
                    record_copy_wait(copy_us);
                    record_meta_copy(i, j_src, j_dst, node_src, copy_us);

                    if (strict_phone_primary_fence) {
                        const int64_t dst_fence_start_us = ggml_time_us();
                        fence_backend(bcj_dst.backend);
                        dst_fence_us = ggml_time_us() - dst_fence_start_us;
                    }

                    const int64_t add_start_us = ggml_time_us();
                    const ggml_status status =
                        ggml_backend_graph_compute_async(
                            bcj_dst.backend, cgraph_aux);
                    add_submit_us = ggml_time_us() - add_start_us;
                    reduce_add_us += add_submit_us;
                    if (status != GGML_STATUS_SUCCESS) {
                        return status;
                    }
                }

                // Fused mode sends the PC partial and the one-node Phone ADD
                // graph in one RPC command.  Fallback mode preserves the
                // already-validated ordered SET_TENSOR -> GRAPH_COMPUTE path.
                // The following Phone-owned graph remains ordered on the same
                // control socket / OpenCL queue.
                int64_t add_fence_us = 0;
                if (strict_phone_primary_fence) {
                    const int64_t add_fence_start_us = ggml_time_us();
                    fence_backend(bcj_dst.backend);
                    add_fence_us = ggml_time_us() - add_fence_start_us;
                }

                if (tensor_phone_stage_profile) {
                    const int64_t return_client_wall_us =
                        ggml_time_us() - tensor_phone_return_begin_us;
                    auto & stage = tensor_phone_stage_entry(
                        layer, chunk, !is_prefill_down_chunk);
                    stage.return_client_wall_us += return_client_wall_us;
                    stage.return_count += 1;
                    printf(
                        "[TENSOR_PHONE_STAGE_COMM] stage=return_set_add "
                        "mode=%s layer=%d chunk=%d sg=%zu wall_ms=%.3f "
                        "fused=%d fused_ms=%.3f copy_ms=%.3f "
                        "add_submit_ms=%.3f\n",
                        mode,
                        layer,
                        chunk,
                        i,
                        return_client_wall_us / 1000.0,
                        fused_set_add_used ? 1 : 0,
                        fused_set_add_us / 1000.0,
                        copy_us / 1000.0,
                        add_submit_us / 1000.0);
                }

                if (pipeline_debug) {
                    printf(
                        "[TENSOR_PHONE_V22_FFN_ONEWAY] sg=%zu layer=%d "
                        "chunk=%d mode=%s bytes=%zu strict_fence=%d fused=%d "
                        "src_fence_ms=%.3f copy_ms=%.3f fused_ms=%.3f "
                        "add_submit_ms=%.3f dst_fence_ms=%.3f "
                        "add_fence_ms=%.3f action=%s\n",
                        i, layer, chunk, mode,
                        ggml_nbytes(node_src),
                        strict_phone_primary_fence ? 1 : 0,
                        fused_set_add_used ? 1 : 0,
                        src_fence_us / 1000.0,
                        copy_us / 1000.0,
                        fused_set_add_us / 1000.0,
                        add_submit_us / 1000.0,
                        dst_fence_us / 1000.0,
                        add_fence_us / 1000.0,
                        fused_set_add_used ?
                            "PC_TO_PHONE_FUSED_SET_ADD" :
                            "PC_TO_PHONE_ORDERED");
                }
                return GGML_STATUS_SUCCESS;
            }

            // Proven generic fallback when single-owner / one-way mode is
            // disabled.
            if (pipeline_debug) {
                printf(
                    "[TENSOR_PHONE_V2_FFN_ALLREDUCE] sg=%zu layer=%d "
                    "chunk=%d mode=%s action=GENERIC_ALLREDUCE reason=%s\n",
                    i, layer, chunk, mode,
                    phone_l_out_owner ?
                        "oneway-disabled" : "single-owner-disabled");
            }
            return GGML_STATUS_SUCCESS;
        }

        if (is_prefill_down_chunk) {
            GGML_ASSERT(pending_prefill_input_task == 0);
            handled = true;

            constexpr size_t j_src = 1;
            constexpr size_t j_dst = 0;
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];
            ggml_tensor * node_src = nodes[j_src];
            ggml_tensor * node_dst = nodes[j_dst];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;

            const ggml_backend_rpc_set_stage_ready_t set_stage_ready =
                ggml_backend_meta_get_stage_ready_setter(bcj_src.backend);
            const ggml_backend_rpc_fence_t rpc_fence =
                ggml_backend_meta_get_rpc_fence(bcj_src.backend);
            const ggml_backend_rpc_snapshot_arm_t snapshot_arm =
                ggml_backend_meta_get_snapshot_arm(bcj_src.backend);
            const ggml_backend_rpc_set_snapshot_read_t set_snapshot_read =
                ggml_backend_meta_get_snapshot_read_setter(bcj_src.backend);
            const bool use_snapshot_pipeline =
                snapshot_prepares[i].prepared &&
                snapshot_arm != nullptr &&
                set_snapshot_read != nullptr;
            const uint64_t snapshot_seq =
                use_snapshot_pipeline ? snapshot_prepares[i].seq : 0;
            const uint32_t snapshot_slot =
                use_snapshot_pipeline ? snapshot_prepares[i].slot : 0;
            const bool dual_prefill_return =
                use_snapshot_pipeline &&
                (std::getenv("GGML_META_PREFILL_DUAL_RETURN") != nullptr || return_wavefront_graph);
            const size_t return_lane = dual_prefill_return ? (snapshot_slot & 1u) : 0;
            ggml_backend_meta_transfer_worker * reduce_worker = backend_ctx->transfer_worker;
            if (dual_prefill_return) {
                auto & lane_worker = backend_ctx->prefill_reduce_workers[return_lane];
                if (lane_worker == nullptr) {
                    lane_worker = new ggml_backend_meta_transfer_worker();
                }
                reduce_worker = lane_worker;
                if (pending_prefill_reduce_task[return_lane] != 0) {
                    const int old_layer = pending_prefill_reduce_layer[return_lane];
                    const int64_t wait_start_us = ggml_time_us();
                    const ggml_status status = wait_prefill_reduce_lane(return_lane);
                    const int64_t wait_us = ggml_time_us() - wait_start_us;
                    ++lane_reuse_wait_count;
                    lane_reuse_wait_us += wait_us;
                    lane_reuse_wait_max_us = std::max(lane_reuse_wait_max_us, wait_us);
                    ++lane_reuse_wait_count_by_lane[return_lane];
                    lane_reuse_wait_us_by_lane[return_lane] += wait_us;
                    if (pipeline_debug) {
                        printf(
                            "[PREFILL_LANE_REUSE_WAIT] lane=%zu old_layer=%d new_layer=%d new_chunk=%d "
                            "wait_ms=%.3f\n",
                            return_lane, old_layer, prefill_down_layer_0, prefill_down_chunk_0,
                            wait_us / 1000.0);
                    }
                    if (status != GGML_STATUS_SUCCESS) {
                        return status;
                    }
                }
                set_prefill_tmp_data(node_tmp, j_dst, return_lane);
            } else {
                set_tmp_data(node_tmp, j_dst, 0);
            }
            const bool handoff_to_phone =
                prefill_layer_hands_off_to_phone(i, prefill_down_layer_0);
            const bool layer_handoff =
                handoff_to_phone && prefill_is_last_down_chunk(i, prefill_down_layer_0);
            ggml_tensor * pc_l_out = handoff_to_phone ?
                find_layer_output(0, prefill_down_layer_0) : nullptr;
            ggml_tensor * phone_l_out = layer_handoff ?
                disable_layer_output_producers(1, i + 1, prefill_down_layer_0) : nullptr;
            ggml_tensor * pc_residual = handoff_to_phone ?
                find_recent_ffn_inp(0, i, prefill_down_layer_0) : nullptr;
            GGML_ASSERT(!handoff_to_phone ||
                (pc_l_out != nullptr && pc_residual != nullptr &&
                 ggml_nbytes(pc_l_out) == ggml_nbytes(pc_residual)));
            GGML_ASSERT(!layer_handoff ||
                (phone_l_out != nullptr &&
                 ggml_nbytes(pc_l_out) == ggml_nbytes(phone_l_out)));
            GGML_ASSERT(!return_wavefront_graph || !handoff_to_phone);

            ggml_tensor * pc_output_chunk = nullptr;
            ggml_tensor * residual_chunk = nullptr;
            ggml_cgraph * chunk_layer_graph = nullptr;
            if (handoff_to_phone) {
                const size_t chunk_offset = down_chunk_offset(
                    0, true, prefill_down_layer_0, prefill_down_chunk_0);

                residual_chunk = get_node_aux(node_dst);
                residual_chunk->view_src = pc_residual;
                residual_chunk->view_offs = chunk_offset;
                ggml_backend_view_init(residual_chunk);
                GGML_ASSERT(chunk_offset + ggml_nbytes(residual_chunk) <= ggml_nbytes(pc_residual));

                ggml_tensor * node_layer = get_node_aux(node_dst);
                node_layer->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
                node_layer->view_offs = node_dst->view_offs;
                node_layer->op = GGML_OP_ADD;
                node_layer->src[0] = node_dst;
                node_layer->src[1] = residual_chunk;
                node_layer->flags |= GGML_TENSOR_FLAG_COMPUTE;
                ggml_backend_view_init(node_layer);

                chunk_layer_graph = get_cgraph_aux();
                chunk_layer_graph->nodes[0] = node_layer;
                chunk_layer_graph->n_nodes = 1;

                pc_output_chunk = get_node_aux(node_dst);
                pc_output_chunk->view_src = pc_l_out;
                pc_output_chunk->view_offs = chunk_offset;
                ggml_backend_view_init(pc_output_chunk);
                GGML_ASSERT(ggml_nbytes(pc_output_chunk) == ggml_nbytes(node_dst));
            }

            const int64_t return_enqueue_us = ggml_time_us();
            const uint64_t reduce_task = reduce_worker->enqueue(
                [&, node_src, node_dst, node_tmp, cgraph_aux, i, set_stage_ready, rpc_fence,
                    set_snapshot_read, use_snapshot_pipeline, snapshot_slot,
                    snapshot_seq, return_enqueue_us, handoff_to_phone, prefill_down_layer_0,
                    prefill_down_chunk_0, layer_handoff, pc_l_out, phone_l_out,
                    pc_output_chunk, chunk_layer_graph, reduce_worker, return_lane](uint64_t task_id) -> ggml_status {
                    ggml_backend_meta_stage_ready_context stage_context {
                        reduce_worker,
                        task_id,
                    };
                    const int64_t return_begin_us = ggml_time_us();
                    const int64_t worker_queue_us = return_begin_us - return_enqueue_us;
                    int64_t rpc_src_sync_us = 0;
                    int64_t pc_dst_sync_us = 0;
                    int64_t snapshot_copy_us = 0;
                    int64_t reduce_add_submit_us = 0;
                    int64_t reduce_sync_us = 0;
                    if (pipeline_debug) {
                            GGML_LOG_INFO("[PREFILL_RETURN_BEGIN] layer=%d chunk=%d lane=%zu t=%" PRId64 "\n",
                                   prefill_down_layer_0, prefill_down_chunk_0, return_lane, return_begin_us);
                    }
                    auto finish_return = [&](ggml_status result) {
                        const int64_t return_end_us = ggml_time_us();
                        if (return_path_debug && use_snapshot_pipeline) {
                            printf(
                                "[RETURN_PATH] layer=%d chunk=%d lane=%zu slot=%u seq=%" PRIu64
                                " worker_queue=%.3f rpc_src_sync=%.3f pc_dst_sync=%.3f "
                                "snapshot_copy=%.3f reduce_add_submit=%.3f reduce_sync=%.3f "
                                "total=%.3f status=%d\n",
                                prefill_down_layer_0, prefill_down_chunk_0, return_lane,
                                snapshot_slot, snapshot_seq,
                                worker_queue_us / 1000.0,
                                rpc_src_sync_us / 1000.0,
                                pc_dst_sync_us / 1000.0,
                                snapshot_copy_us / 1000.0,
                                reduce_add_submit_us / 1000.0,
                                reduce_sync_us / 1000.0,
                                (return_end_us - return_begin_us) / 1000.0,
                                (int) result);
                            fflush(stdout);
                        }
                        if (pipeline_debug) {
                            GGML_LOG_INFO("[PREFILL_RETURN_END] layer=%d chunk=%d lane=%zu t=%" PRId64 " dur=%.3f ms\n",
                                   prefill_down_layer_0, prefill_down_chunk_0, return_lane, return_end_us,
                                   (return_end_us - return_begin_us) / 1000.0);
                        }
                        return result;
                    };

                    if (use_snapshot_pipeline) {
                        set_snapshot_read(true, snapshot_slot, snapshot_seq);
                    } else {
                        const int64_t fence_start_us = ggml_time_us();
                        if (rpc_fence != nullptr) {
                            rpc_fence(bcj_src.backend);
                        } else {
                            ggml_backend_synchronize(bcj_src.backend);
                        }
                        const int64_t fence_us = ggml_time_us() - fence_start_us;
                        if (pipeline_debug && has_copy_detail_budget()) {
                            printf("[META_FENCE] sg=%zu 1->0 time=%.3f ms\n",
                                   i, fence_us / 1000.0);
                        }
                    }

                    const int64_t copy_start_us = ggml_time_us();
                    if (!use_snapshot_pipeline && set_stage_ready != nullptr) {
                        set_stage_ready(ggml_backend_meta_stage_ready, &stage_context);
                    }

                    if (use_snapshot_pipeline && bcj_dst.backend->iface.cpy_tensor_async == nullptr) {
                        // This is exactly the generic ggml_backend_tensor_copy_async()
                        // fallback, expanded here so the two hidden synchronizations can
                        // be measured separately. Keep the semantics unchanged while
                        // diagnosing return-wavefront slowdown.
                        const int64_t src_sync_start_us = ggml_time_us();
                        ggml_backend_synchronize(bcj_src.backend);
                        rpc_src_sync_us = ggml_time_us() - src_sync_start_us;

                        const int64_t dst_sync_start_us = ggml_time_us();
                        ggml_backend_synchronize(bcj_dst.backend);
                        pc_dst_sync_us = ggml_time_us() - dst_sync_start_us;

                        const int64_t snapshot_copy_start_us = ggml_time_us();
                        ggml_backend_tensor_copy(node_src, node_tmp);
                        snapshot_copy_us = ggml_time_us() - snapshot_copy_start_us;
                    } else {
                        const int64_t snapshot_copy_start_us = ggml_time_us();
                        ggml_backend_tensor_copy_async(
                            bcj_src.backend, bcj_dst.backend, node_src, node_tmp);
                        snapshot_copy_us = ggml_time_us() - snapshot_copy_start_us;
                    }

                    if (use_snapshot_pipeline) {
                        set_snapshot_read(false, 0, 0);
                    } else if (set_stage_ready != nullptr) {
                        set_stage_ready(nullptr, nullptr);
                    }
                    const int64_t copy_us = ggml_time_us() - copy_start_us;
                    record_copy_wait(copy_us);
                    record_meta_copy(i, 1, 0, node_src, copy_us);

                    const int64_t add_start_us = ggml_time_us();
                    const ggml_status status = ggml_backend_graph_compute_async(
                        bcj_dst.backend, cgraph_aux);
                    const int64_t add_us = ggml_time_us() - add_start_us;
                    reduce_add_submit_us = add_us;
                    record_reduce_add(add_us, status == GGML_STATUS_SUCCESS);

                    if (status != GGML_STATUS_SUCCESS) {
                        return finish_return(status);
                    }

                    if (return_wavefront_graph && !handoff_to_phone) {
                        // The pending task is the exact DOWN_READY(L,C) fence.
                        // Do not publish it until the PC reduce ADD is complete.
                        const int64_t reduce_sync_start_us = ggml_time_us();
                        ggml_backend_synchronize(bcj_dst.backend);
                        reduce_sync_us = ggml_time_us() - reduce_sync_start_us;
                        if (pipeline_debug) {
                            GGML_LOG_INFO(
                                "[RETURN_WAVEFRONT_DOWN_READY] layer=%d chunk=%d lane=%zu\n",
                                prefill_down_layer_0, prefill_down_chunk_0, return_lane);
                        }
                    }

                    if (!handoff_to_phone) {
                        return finish_return(status);
                    }

                    ggml_backend_synchronize(bcj_dst.backend);
                    const ggml_status layer_status =
                        ggml_backend_graph_compute_async(bcj_dst.backend, chunk_layer_graph);
                    if (layer_status != GGML_STATUS_SUCCESS) {
                        return finish_return(layer_status);
                    }
                    ggml_backend_synchronize(bcj_dst.backend);
                    ggml_backend_tensor_copy_async(
                        bcj_dst.backend, bcj_dst.backend, node_dst, pc_output_chunk);
                    if (!layer_handoff) {
                        return finish_return(status);
                    }
                    ggml_backend_synchronize(bcj_dst.backend);

                    const int64_t handoff_start_us = ggml_time_us();
                    ggml_backend_tensor_copy_async(
                        bcj_dst.backend, bcj_src.backend, pc_l_out, phone_l_out);
                    const int64_t handoff_us = ggml_time_us() - handoff_start_us;
                    record_copy_wait(handoff_us);
                    record_meta_copy(i, 0, 1, pc_l_out, handoff_us);
                    record_direct_copy();

                    if (pipeline_debug) {
                        printf(
                            "[META_LAYER_HANDOFF] layer=%d 0->1 tensor=%s bytes=%zu\n",
                            prefill_down_layer_0, pc_l_out->name, ggml_nbytes(pc_l_out));
                    }
                    return finish_return(status);
                });

            pending_prefill_reduce_task[return_lane]   = reduce_task;
            pending_prefill_reduce_layer[return_lane]  = prefill_down_layer_0;
            pending_prefill_reduce_chunk[return_lane]  = prefill_down_chunk_0;
            pending_prefill_reduce_worker[return_lane] = reduce_worker;

            if (use_snapshot_pipeline) {
                const int64_t snapshot_arm_start_us = ggml_time_us();
                const int64_t enqueue_to_arm_us = snapshot_arm_start_us - return_enqueue_us;
                const bool armed = snapshot_arm(
                    bcj_src.backend,
                    node_src,
                    0,
                    ggml_nbytes(node_src),
                    snapshot_slot,
                    snapshot_seq);
                const int64_t snapshot_arm_call_us = ggml_time_us() - snapshot_arm_start_us;
                GGML_ASSERT(armed);

                if (return_path_debug) {
                    printf(
                        "[SNAPSHOT_ARM_META] layer=%d chunk=%d lane=%zu slot=%u seq=%" PRIu64
                        " enqueue_to_arm=%.3f arm_call=%.3f bytes=%zu\n",
                        prefill_down_layer_0, prefill_down_chunk_0, return_lane,
                        snapshot_slot, snapshot_seq,
                        enqueue_to_arm_us / 1000.0,
                        snapshot_arm_call_us / 1000.0,
                        ggml_nbytes(node_src));
                    fflush(stdout);
                }

                if (pipeline_debug) {
                    GGML_LOG_INFO("[PREFILL_SNAPSHOT_ARM] sg=%zu layer=%d chunk=%d "
                           "slot=%u seq=%" PRIu64 " bytes=%zu\n",
                           i, prefill_down_layer_0, prefill_down_chunk_0,
                           snapshot_slot, snapshot_seq, ggml_nbytes(node_src));
                }
            }
            if (pipeline_debug) {
                GGML_LOG_INFO("[PREFILL_REDUCE] layer=%d chunk=%d tensor=%s "
                       "ne=[%" PRId64 ",%" PRId64 "]\n",
                       prefill_down_layer_0, prefill_down_chunk_0, node_src->name,
                       node_src->ne[0], node_src->ne[1]);
            }
            return GGML_STATUS_SUCCESS;
        }

        if (!is_ffn_down_chunk) {
            return GGML_STATUS_SUCCESS;
        }

        handled = true;

        constexpr size_t j_src = 1; // Phone
        constexpr size_t j_dst = 0; // PC

        auto & bcj_src = backend_ctx->backend_configs[j_src];
        auto & bcj_dst = backend_ctx->backend_configs[j_dst];

        const bool handoff_to_phone =
            decode_layer_hands_off_to_phone(i, decode_layer_0);
        const bool layer_handoff =
            handoff_to_phone && decode_is_last_down_chunk(i, decode_layer_0);

        ggml_tensor * phone_next_wdown = get_ffn_down_boundary_node(j_src, i + 1);
        ggml_tensor * pc_next_wdown    = get_ffn_down_boundary_node(j_dst, i + 1);
        const bool next_is_ffn_down_chunk =
            phone_next_wdown != nullptr && pc_next_wdown != nullptr;

        // The legacy decode overlap executes only the terminal node of the
        // next Phone chunk as a one-node early graph. This is valid for dense
        // FFN chunking because ffn_down_chunk_* is the down-projection
        // MUL_MAT itself and all of its inputs are already ready.
        //
        // Qwen3-MoE chunking names the boundary after expert weighting and
        // aggregation. Its terminal node is typically ADD and depends on the
        // chunk-local MUL_MAT_ID/weight/aggregation chain. Running only that
        // ADD early consumes inputs that have not been computed yet and
        // corrupts the hidden state. Keep MoE on the normal ordered path until
        // an early graph containing the complete dependency chain is built.
        const bool next_chunk_supports_single_node_early =
            next_is_ffn_down_chunk &&
            phone_next_wdown->op == GGML_OP_MUL_MAT &&
            pc_next_wdown->op == GGML_OP_MUL_MAT;

        if (pipeline_debug &&
            next_is_ffn_down_chunk &&
            !next_chunk_supports_single_node_early) {
            printf(
                "[DECODE_EARLY_SKIP] layer=%d chunk=%d next_phone_op=%s next_pc_op=%s\n",
                decode_layer_0,
                decode_chunk_0 + 1,
                ggml_op_name(phone_next_wdown->op),
                ggml_op_name(pc_next_wdown->op));
        }

        ggml_cgraph * phone_early_graph = nullptr;
        if (next_chunk_supports_single_node_early) {
            phone_early_graph = get_early_graph(j_src, i + 1);
            phone_early_graph->nodes[0] = phone_next_wdown;
            phone_early_graph->n_nodes = 1;
        }

        const ggml_backend_rpc_set_stage_ready_t set_stage_ready =
            ggml_backend_meta_get_stage_ready_setter(bcj_src.backend);
        const ggml_backend_rpc_fence_t rpc_fence =
            ggml_backend_meta_get_rpc_fence(bcj_src.backend);
        const ggml_backend_rpc_snapshot_arm_t snapshot_arm =
            ggml_backend_meta_get_snapshot_arm(bcj_src.backend);
        const ggml_backend_rpc_set_snapshot_read_t set_snapshot_read =
            ggml_backend_meta_get_snapshot_read_setter(bcj_src.backend);

        ggml_tensor * node_src = nodes[j_src];
        ggml_tensor * node_dst = nodes[j_dst];

        // Dense decode chunking has a single down-projection MUL_MAT as
        // the reduction boundary. The RPC snapshot fast path was designed for
        // that shape. A MoE chunk boundary is the terminal expert-aggregation
        // node after a delayed all-reduce chain; snapshotting that boundary can
        // race/fuse against a graph shape it was never designed to represent.
        // Keep MoE decode chunks on the conservative fence -> copy -> reduce
        // path until a dependency-complete MoE snapshot path is implemented.
        const bool snapshot_compatible_decode_chunk =
            node_src->op == GGML_OP_MUL_MAT &&
            node_dst->op == GGML_OP_MUL_MAT;
        const bool use_snapshot_pipeline =
            !handoff_to_phone &&
            snapshot_compatible_decode_chunk &&
            snapshot_arm != nullptr &&
            set_snapshot_read != nullptr;

        if (pipeline_debug &&
            !handoff_to_phone &&
            !snapshot_compatible_decode_chunk) {
            printf(
                "[DECODE_SNAPSHOT_SKIP] layer=%d chunk=%d phone_op=%s pc_op=%s\n",
                decode_layer_0,
                decode_chunk_0,
                ggml_op_name(node_src->op),
                ggml_op_name(node_dst->op));
        }

        const uint64_t snapshot_seq =
            snapshot_prepares[i].prepared ?
                snapshot_prepares[i].seq :
                (use_snapshot_pipeline ? backend_ctx->next_snapshot_seq++ : 0);
        const uint32_t snapshot_slot =
            snapshot_prepares[i].prepared ?
                snapshot_prepares[i].slot :
                snapshot_seq & 1;

        GGML_ASSERT(ggml_is_contiguous(nodes[j_src]));
        GGML_ASSERT(ggml_is_contiguous(nodes[j_dst]));

        // -----------------------------------------------------
        // 这些对象仍然由主线程准备，不要放进 transfer worker
        // -----------------------------------------------------

        ggml_tensor * pc_residual = handoff_to_phone ?
            find_recent_ffn_inp(0, i, decode_layer_0) : nullptr;
        ggml_tensor * phone_l_out = handoff_to_phone ?
            find_layer_output(1, decode_layer_0) : nullptr;
        if (layer_handoff) {
            ggml_tensor * disabled_l_out =
                disable_layer_output_producers(1, i + 1, decode_layer_0);
            GGML_ASSERT(disabled_l_out == phone_l_out);
        }
        GGML_ASSERT(!handoff_to_phone ||
            (pc_residual != nullptr && phone_l_out != nullptr &&
             ggml_nbytes(pc_residual) == ggml_nbytes(phone_l_out)));

        const size_t chunk_offset = handoff_to_phone ?
            down_chunk_offset(0, false, decode_layer_0, decode_chunk_0) : 0;

        ggml_tensor * node_tmp = get_node_aux(handoff_to_phone ? node_src : node_dst);
        set_tmp_data(node_tmp, handoff_to_phone ? j_src : j_dst, 0);

        ggml_tensor * node_red = get_node_aux(handoff_to_phone ? node_src : node_dst);

        node_red->view_src = handoff_to_phone ?
            phone_l_out :
            (node_dst->view_src == nullptr ? node_dst : node_dst->view_src);
        node_red->view_offs = handoff_to_phone ? chunk_offset : node_dst->view_offs;

        node_red->op     = GGML_OP_ADD;
        node_red->src[0] = handoff_to_phone ? node_src : node_dst;
        node_red->src[1] = node_tmp;
        node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

        ggml_backend_view_init(node_red);

        ggml_cgraph * cgraph_aux = get_cgraph_aux();
        cgraph_aux->nodes[0] = node_red;
        cgraph_aux->n_nodes  = 1;

        ggml_cgraph * chunk_layer_graph = nullptr;
        if (handoff_to_phone) {
            ggml_tensor * residual_chunk = get_node_aux(node_dst);
            residual_chunk->view_src = pc_residual;
            residual_chunk->view_offs = chunk_offset;
            ggml_backend_view_init(residual_chunk);
            GGML_ASSERT(chunk_offset + ggml_nbytes(residual_chunk) <= ggml_nbytes(pc_residual));

            ggml_tensor * node_layer = get_node_aux(node_dst);
            node_layer->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_layer->view_offs = node_dst->view_offs;
            node_layer->op = GGML_OP_ADD;
            node_layer->src[0] = node_dst;
            node_layer->src[1] = residual_chunk;
            node_layer->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_layer);

            chunk_layer_graph = get_cgraph_aux();
            chunk_layer_graph->nodes[0] = node_layer;
            chunk_layer_graph->n_nodes = 1;

            GGML_ASSERT(chunk_offset + ggml_nbytes(node_red) <= ggml_nbytes(phone_l_out));
        }

        // -----------------------------------------------------
        // 真正的传输 + reduce 放到 transfer worker
        // -----------------------------------------------------

        const uint64_t task_id =
            backend_ctx->transfer_worker->enqueue(
                [&, node_src, node_dst, node_tmp, cgraph_aux, j_src, j_dst, i,
                    set_stage_ready, rpc_fence, set_snapshot_read,
                    use_snapshot_pipeline, snapshot_slot, snapshot_seq,
                    handoff_to_phone, decode_layer_0, decode_chunk_0,
                    phone_l_out, chunk_layer_graph](uint64_t task_id) -> ggml_status {

                    if (handoff_to_phone) {
                        ggml_backend_synchronize(bcj_dst.backend);
                        const ggml_status layer_status =
                            ggml_backend_graph_compute_async(bcj_dst.backend, chunk_layer_graph);
                        if (layer_status != GGML_STATUS_SUCCESS) {
                            return layer_status;
                        }
                        ggml_backend_synchronize(bcj_dst.backend);

                        const int64_t copy_start_us = ggml_time_us();
                        ggml_backend_tensor_copy_async(
                            bcj_dst.backend, bcj_src.backend, node_dst, node_tmp);
                        const int64_t copy_us = ggml_time_us() - copy_start_us;
                        record_copy_wait(copy_us);
                        record_meta_copy(i, j_dst, j_src, node_dst, copy_us);
                        record_direct_copy();

                        ggml_backend_synchronize(bcj_src.backend);
                        const int64_t add_start_us = ggml_time_us();
                        const ggml_status status =
                            ggml_backend_graph_compute_async(bcj_src.backend, cgraph_aux);
                        reduce_add_us += ggml_time_us() - add_start_us;
                        if (status != GGML_STATUS_SUCCESS) {
                            return status;
                        }
                        ggml_backend_synchronize(bcj_src.backend);

                        if (pipeline_debug) {
                            printf(
                                "[META_REDUCE_TO_PHONE] layer=%d chunk=%d "
                                "tensor=%s bytes=%zu\n",
                                decode_layer_0, decode_chunk_0,
                                phone_l_out->name, ggml_nbytes(node_dst));
                        }
                        return GGML_STATUS_SUCCESS;
                    }

                    // Phone -> PC
                    ggml_backend_meta_stage_ready_context stage_context {
                        backend_ctx->transfer_worker,
                        task_id,
                    };

                    if (use_snapshot_pipeline) {
                        set_snapshot_read(true, snapshot_slot, snapshot_seq);
                    } else {
                        const int64_t fence_start_us = ggml_time_us();
                        if (rpc_fence != nullptr) {
                            rpc_fence(bcj_src.backend);
                        } else {
                            ggml_backend_synchronize(bcj_src.backend);
                        }
                        const int64_t fence_us = ggml_time_us() - fence_start_us;

                        if (pipeline_debug && has_copy_detail_budget()) {
                            printf("[META_FENCE] sg=%zu %zu->%zu time=%.3f ms\n",
                                   i, j_src, j_dst, fence_us / 1000.0);
                        }
                    }

                    const int64_t copy_start_us = ggml_time_us();

                    if (!use_snapshot_pipeline && set_stage_ready != nullptr) {
                        set_stage_ready(ggml_backend_meta_stage_ready, &stage_context);
                    }

                    ggml_backend_tensor_copy_async(
                            bcj_src.backend,
                            bcj_dst.backend,
                            node_src,
                            node_tmp);

                    if (use_snapshot_pipeline) {
                        set_snapshot_read(false, 0, 0);
                    } else if (set_stage_ready != nullptr) {
                        set_stage_ready(nullptr, nullptr);
                    }

                    const int64_t copy_us =
                        ggml_time_us() - copy_start_us;

                    record_copy_wait(copy_us);

                    record_meta_copy(
                            i,
                            j_src,
                            j_dst,
                            node_src,
                            copy_us);

                    // PC local result + Phone result
                    const int64_t add_start_us = ggml_time_us();

                    const ggml_status status =
                        ggml_backend_graph_compute_async(
                                bcj_dst.backend,
                                cgraph_aux);

                    reduce_add_us +=
                        ggml_time_us() - add_start_us;

                    if (status == GGML_STATUS_SUCCESS) {
                        ++reduce_to_primary_count;
                    }

                    return status;
                });

        const bool primary_mirror_back = std::getenv("GGML_META_PRIMARY_MIRROR_BACK") != nullptr;
        if (primary_mirror_back && !handoff_to_phone) {
            if (use_snapshot_pipeline) {
                const bool armed = snapshot_arm(
                        bcj_src.backend,
                        node_src,
                        0,
                        ggml_nbytes(node_src),
                        snapshot_slot,
                        snapshot_seq);
                GGML_ASSERT(armed);
            }

            const ggml_status status = backend_ctx->transfer_worker->wait(task_id);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }

            ggml_backend_synchronize(bcj_dst.backend);

            const size_t nbytes = ggml_nbytes(node_dst);
            GGML_ASSERT(ggml_nbytes(node_src) == nbytes);
            std::vector<uint8_t> full_data(nbytes);

            ggml_backend_tensor_get(node_dst, full_data.data(), 0, nbytes);
            ggml_backend_tensor_set(node_src, full_data.data(), 0, nbytes);

            return GGML_STATUS_SUCCESS;
        }
        bool phone_early_started = false;
        if (use_snapshot_pipeline) {
            const int64_t snapshot_arm_us = ggml_time_us();
            if (pipeline_gap.valid && pipeline_gap.subgraph == i) {
                if (pipeline_debug) {
                    const int64_t submit_us = pipeline_gap.pc_submit_us - pipeline_gap.pc_start_us;
                    const int64_t gap_us = snapshot_arm_us - pipeline_gap.pc_submit_us;
                    pipeline_submit_sum_us += submit_us;
                    pipeline_gap_sum_us += gap_us;
                    pipeline_gap_max_us = std::max(pipeline_gap_max_us, gap_us);
                    ++pipeline_gap_count;
                }
                pipeline_gap.valid = false;
            }
            const bool armed = snapshot_arm(
                bcj_src.backend,
                node_src,
                0,
                ggml_nbytes(node_src),
                snapshot_slot,
                snapshot_seq);
            GGML_ASSERT(armed);
            if (phone_early_graph != nullptr) {
            prepare_graph_snapshot(i + 1, phone_next_wdown);
            backend_ctx->compute_workers->start_graph(
            j_src,
            phone_early_graph);

        phone_early_started = true;
    }
        }

        const ggml_status transfer_status =
    backend_ctx->transfer_worker->wait(task_id);

if (transfer_status != GGML_STATUS_SUCCESS) {
    if (phone_early_started) {
        backend_ctx->compute_workers->wait(j_src);
    }

    return transfer_status;
}

if (phone_early_graph == nullptr) {
    return GGML_STATUS_SUCCESS;
}

// 非 snapshot 路径下，Phone 还没有启动。
// 此时 transfer 已经完成，可以安全启动。
if (!phone_early_started) {
    backend_ctx->compute_workers->start_graph(
        j_src,
        phone_early_graph);

    phone_early_started = true;
}

// transfer + PC reduce 已完成，
// PC 现在可以消费完整 chunk k。
const int64_t pc_start_us = ggml_time_us();

backend_ctx->compute_workers->start(
    j_dst,
    i + 1);

// 注意：这里不再先 wait Phone。
// PC / Phone 同时运行。
const ggml_status pc_status =
    backend_ctx->compute_workers->wait(j_dst);

const ggml_status phone_status =
    backend_ctx->compute_workers->wait(j_src);

if (pc_status != GGML_STATUS_SUCCESS) {
    return pc_status;
}

if (phone_status != GGML_STATUS_SUCCESS) {
    return phone_status;
}

        if (use_snapshot_pipeline) {
            pipeline_gap.valid        = true;
            pipeline_gap.subgraph     = i + 1;
            pipeline_gap.pc_start_us  = pc_start_us;
            pipeline_gap.pc_submit_us = backend_ctx->compute_workers->completed_at(j_dst);
        }

        next_compute_complete = true;
        return GGML_STATUS_SUCCESS;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);
        std::vector<uint8_t> has_data(n_backends, 0);
        for (size_t j = 0; j < n_backends; ++j) {
            const auto & bcj = backend_ctx->backend_configs[j];
            const ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            has_data[j] = (node->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
        }
        if (pipeline_debug && i == 1 && n_backends == 2) {
            for (size_t j = 0; j < n_backends; ++j) {
                auto & bcj = backend_ctx->backend_configs[j];
                ggml_cgraph * graph = bcj.cgraphs[i].cgraph_main;
                ggml_tensor * node = graph->nodes[graph->n_nodes - 1];
                meta_debug_tensor(bcj.backend, node,
                        j == 0 ? "B pre-reduce backend0" : "C pre-reduce backend1");
            }
        }
        if (i < 6) {
    auto * n0 = backend_ctx->backend_configs[0]
                    .cgraphs[i].cgraph_main->nodes[
                        backend_ctx->backend_configs[0]
                            .cgraphs[i].cgraph_main->n_nodes - 1];

    auto * n1 = backend_ctx->backend_configs[1]
                    .cgraphs[i].cgraph_main->nodes[
                        backend_ctx->backend_configs[1]
                            .cgraphs[i].cgraph_main->n_nodes - 1];

    GGML_LOG_INFO(
        "[REDUCE_ACTIVE] sg=%zu "
        "name0=%s compute0=%d bytes0=%zu "
        "name1=%s compute1=%d bytes1=%zu\n",
        i,
        n0->name,
        !!(n0->flags & GGML_TENSOR_FLAG_COMPUTE),
        ggml_nbytes(n0),
        n1->name,
        !!(n1->flags & GGML_TENSOR_FLAG_COMPUTE),
        ggml_nbytes(n1));
}
        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
            node_zero->src[0] = node;
            ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->buffer = node->buffer;
            node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[j] = get_cgraph_aux();
            step_cgraphs[j]->nodes[0] = node_zero;
            step_cgraphs[j]->n_nodes = 1;
            const int64_t zero_start_us = ggml_time_us();
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
            reduce_zero_us += ggml_time_us() - zero_start_us;
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto reduce_fence_backend = [&](ggml_backend_t backend) {
            const ggml_backend_rpc_fence_t rpc_fence =
                ggml_backend_meta_get_rpc_fence(backend);
            if (rpc_fence != nullptr) {
                rpc_fence(backend);
            } else {
                ggml_backend_synchronize(backend);
            }
        };

        // Cross-primary control-path trace.  This is deliberately read-only and
        // runs at the first FFN reduction for the requested layer, after
        // Attention / residual / FFN norm / Router have already executed but
        // before the FFN partials are reduced.  It therefore works for both
        // legacy PC-primary Tensor and TENSOR_PHONE_PRIMARY without executing a
        // shadow graph or changing ownership.
        if (n_backends == 2) {
            const char * trace_layer_env =
                std::getenv("GGML_META_TP_CONTROL_TRACE_LAYER");
            if (trace_layer_env != nullptr) {
                char * trace_end = nullptr;
                const long trace_layer_long =
                    std::strtol(trace_layer_env, &trace_end, 10);

                if (trace_end != trace_layer_env &&
                        *trace_end == '\0' &&
                        trace_layer_long >= 0) {
                    const int trace_layer = (int) trace_layer_long;

                    ggml_tensor * reduce_node =
                        backend_ctx->backend_configs[0]
                            .cgraphs[i].cgraph_main->nodes[
                                backend_ctx->backend_configs[0]
                                    .cgraphs[i].cgraph_main->n_nodes - 1];

                    int current_layer = -1;
                    int current_chunk = -1;
                    int parsed_chars = 0;
                    const char * current_mode = nullptr;

                    if (reduce_node->ne[1] == 1 &&
                            std::sscanf(
                                reduce_node->name,
                                "ffn_moe_out-%d%n",
                                &current_layer,
                                &parsed_chars) == 1 &&
                            reduce_node->name[parsed_chars] == '\0') {
                        current_chunk = 0;
                        current_mode = "decode-single";
                    } else {
                        int parsed_chunk = -1;
                        int parsed_layer = -1;
                        if (ggml_backend_meta_parse_decode_ffn_chunk(
                                reduce_node->name,
                                parsed_chunk,
                                parsed_layer)) {
                            current_layer = parsed_layer;
                            current_chunk = parsed_chunk;
                            current_mode = "decode";
                        } else if (ggml_backend_meta_parse_prefill_down_chunk(
                                       reduce_node->name,
                                       parsed_chunk,
                                       parsed_layer)) {
                            current_layer = parsed_layer;
                            current_chunk = parsed_chunk;
                            current_mode = "prefill";
                        }
                    }

                    // Chunk zero is sufficient to compare both paths while
                    // keeping the trace small.  Decode-single also maps here.
                    if (current_layer == trace_layer &&
                            current_chunk == 0) {
                        const bool phone_primary_owner =
                            backend_ctx->tensor_phone_first_layer >= 0 &&
                            trace_layer >=
                                backend_ctx->tensor_phone_first_layer &&
                            trace_layer <
                                backend_ctx->tensor_phone_last_layer;
                        const size_t owner =
                            phone_primary_owner ? 1 : 0;
                        auto & owner_cfg =
                            backend_ctx->backend_configs[owner];

                        reduce_fence_backend(owner_cfg.backend);

                        auto trace_f32 =
                            [&](const char * logical,
                                const char * tensor_name) {
                                ggml_tensor * tensor =
                                    find_exact_named_tensor(
                                        owner, tensor_name);
                                if (tensor == nullptr) {
                                    printf(
                                        "[TP_CONTROL_TRACE] layer=%d "
                                        "chunk=%d mode=%s primary=%s "
                                        "owner=%zu logical=%s "
                                        "tensor=%s status=MISSING\n",
                                        trace_layer,
                                        current_chunk,
                                        current_mode != nullptr ?
                                            current_mode : "unknown",
                                        phone_primary_owner ?
                                            "PHONE" : "PC",
                                        owner,
                                        logical,
                                        tensor_name);
                                    return;
                                }
                                if (tensor->type != GGML_TYPE_F32) {
                                    printf(
                                        "[TP_CONTROL_TRACE] layer=%d "
                                        "chunk=%d mode=%s primary=%s "
                                        "owner=%zu logical=%s "
                                        "tensor=%s status=TYPE type=%d\n",
                                        trace_layer,
                                        current_chunk,
                                        current_mode != nullptr ?
                                            current_mode : "unknown",
                                        phone_primary_owner ?
                                            "PHONE" : "PC",
                                        owner,
                                        logical,
                                        tensor->name,
                                        (int) tensor->type);
                                    return;
                                }

                                const size_t n =
                                    ggml_nelements(tensor);
                                std::vector<float> data(n);
                                ggml_backend_tensor_get(
                                    tensor,
                                    data.data(),
                                    0,
                                    n * sizeof(float));

                                double sum = 0.0;
                                double l2 = 0.0;
                                double max_abs = 0.0;
                                for (float x : data) {
                                    const double xd = x;
                                    sum += xd;
                                    l2 += xd * xd;
                                    max_abs =
                                        std::max(
                                            max_abs,
                                            std::abs(xd));
                                }

                                printf(
                                    "[TP_CONTROL_F32] layer=%d chunk=%d "
                                    "mode=%s primary=%s owner=%zu "
                                    "logical=%s tensor=%s "
                                    "ne=[%" PRId64 ",%" PRId64
                                    ",%" PRId64 ",%" PRId64 "] "
                                    "n=%zu sum=%.9f l2=%.9f "
                                    "max=%.9f "
                                    "v0=%.9f v1=%.9f v2=%.9f "
                                    "v3=%.9f v4=%.9f v5=%.9f "
                                    "v6=%.9f v7=%.9f\n",
                                    trace_layer,
                                    current_chunk,
                                    current_mode != nullptr ?
                                        current_mode : "unknown",
                                    phone_primary_owner ?
                                        "PHONE" : "PC",
                                    owner,
                                    logical,
                                    tensor->name,
                                    tensor->ne[0],
                                    tensor->ne[1],
                                    tensor->ne[2],
                                    tensor->ne[3],
                                    n,
                                    sum,
                                    std::sqrt(l2),
                                    max_abs,
                                    n > 0 ? data[0] : 0.0f,
                                    n > 1 ? data[1] : 0.0f,
                                    n > 2 ? data[2] : 0.0f,
                                    n > 3 ? data[3] : 0.0f,
                                    n > 4 ? data[4] : 0.0f,
                                    n > 5 ? data[5] : 0.0f,
                                    n > 6 ? data[6] : 0.0f,
                                    n > 7 ? data[7] : 0.0f);
                            };

                        auto trace_i32 =
                            [&](const char * logical,
                                const char * tensor_name) {
                                ggml_tensor * tensor =
                                    find_exact_named_tensor(
                                        owner, tensor_name);
                                if (tensor == nullptr) {
                                    return;
                                }
                                if (tensor->type != GGML_TYPE_I32) {
                                    printf(
                                        "[TP_CONTROL_TRACE] layer=%d "
                                        "chunk=%d mode=%s primary=%s "
                                        "owner=%zu logical=%s tensor=%s "
                                        "status=TYPE type=%d\n",
                                        trace_layer,
                                        current_chunk,
                                        current_mode != nullptr ?
                                            current_mode : "unknown",
                                        phone_primary_owner ?
                                            "PHONE" : "PC",
                                        owner,
                                        logical,
                                        tensor->name,
                                        (int) tensor->type);
                                    return;
                                }

                                const size_t n =
                                    ggml_nelements(tensor);
                                std::vector<int32_t> data(n);
                                ggml_backend_tensor_get(
                                    tensor,
                                    data.data(),
                                    0,
                                    n * sizeof(int32_t));

                                int64_t sum = 0;
                                int32_t min_v =
                                    n > 0 ? data[0] : 0;
                                int32_t max_v =
                                    n > 0 ? data[0] : 0;
                                uint64_t hash =
                                    UINT64_C(1469598103934665603);
                                for (int32_t x : data) {
                                    sum += x;
                                    min_v = std::min(min_v, x);
                                    max_v = std::max(max_v, x);
                                    const uint32_t bits =
                                        (uint32_t) x;
                                    for (int b = 0; b < 4; ++b) {
                                        hash ^=
                                            (bits >> (b * 8)) &
                                            UINT64_C(0xff);
                                        hash *=
                                            UINT64_C(1099511628211);
                                    }
                                }

                                printf(
                                    "[TP_CONTROL_I32] layer=%d chunk=%d "
                                    "mode=%s primary=%s owner=%zu "
                                    "logical=%s tensor=%s "
                                    "ne=[%" PRId64 ",%" PRId64
                                    ",%" PRId64 ",%" PRId64 "] "
                                    "n=%zu sum=%" PRId64
                                    " min=%d max=%d hash=%" PRIu64
                                    " v0=%d v1=%d v2=%d v3=%d "
                                    "v4=%d v5=%d v6=%d v7=%d\n",
                                    trace_layer,
                                    current_chunk,
                                    current_mode != nullptr ?
                                        current_mode : "unknown",
                                    phone_primary_owner ?
                                        "PHONE" : "PC",
                                    owner,
                                    logical,
                                    tensor->name,
                                    tensor->ne[0],
                                    tensor->ne[1],
                                    tensor->ne[2],
                                    tensor->ne[3],
                                    n,
                                    sum,
                                    min_v,
                                    max_v,
                                    hash,
                                    n > 0 ? data[0] : 0,
                                    n > 1 ? data[1] : 0,
                                    n > 2 ? data[2] : 0,
                                    n > 3 ? data[3] : 0,
                                    n > 4 ? data[4] : 0,
                                    n > 5 ? data[5] : 0,
                                    n > 6 ? data[6] : 0,
                                    n > 7 ? data[7] : 0);
                            };

                        char residual_name[64];
                        char hidden_name[96];
                        char logits_name[64];
                        char topk_name[96];
                        char weights_name[96];

                        std::snprintf(
                            residual_name,
                            sizeof(residual_name),
                            "ffn_inp-%d",
                            trace_layer);
                        std::snprintf(
                            logits_name,
                            sizeof(logits_name),
                            "ffn_moe_logits-%d",
                            trace_layer);

                        if (current_mode != nullptr &&
                                std::strcmp(
                                    current_mode,
                                    "prefill") == 0) {
                            std::snprintf(
                                hidden_name,
                                sizeof(hidden_name),
                                "prefill_ffn_norm_chunk_%d-%d",
                                current_chunk,
                                trace_layer);
                            if (phone_primary_owner) {
                                std::snprintf(
                                    topk_name,
                                    sizeof(topk_name),
                                    "phone_prefill_route_topk_chunk_%d-%d",
                                    current_chunk,
                                    trace_layer);
                                std::snprintf(
                                    weights_name,
                                    sizeof(weights_name),
                                    "phone_prefill_route_weights_chunk_%d-%d",
                                    current_chunk,
                                    trace_layer);
                            } else {
                                std::snprintf(
                                    topk_name,
                                    sizeof(topk_name),
                                    "ffn_moe_topk-%d",
                                    trace_layer);
                                std::snprintf(
                                    weights_name,
                                    sizeof(weights_name),
                                    "ffn_moe_weights-%d",
                                    trace_layer);
                            }
                        } else {
                            std::snprintf(
                                hidden_name,
                                sizeof(hidden_name),
                                "ffn_norm-%d",
                                trace_layer);
                            if (phone_primary_owner) {
                                std::snprintf(
                                    topk_name,
                                    sizeof(topk_name),
                                    "phone_moe_route_topk-%d",
                                    trace_layer);
                                std::snprintf(
                                    weights_name,
                                    sizeof(weights_name),
                                    "phone_moe_route_weights-%d",
                                    trace_layer);
                            } else {
                                std::snprintf(
                                    topk_name,
                                    sizeof(topk_name),
                                    "ffn_moe_topk-%d",
                                    trace_layer);
                                std::snprintf(
                                    weights_name,
                                    sizeof(weights_name),
                                    "ffn_moe_weights-%d",
                                    trace_layer);
                            }
                        }

                        trace_f32("ffn_inp", residual_name);
                        trace_f32("ffn_norm", hidden_name);
                        trace_f32("router_logits", logits_name);
                        trace_i32("router_topk", topk_name);
                        trace_f32("router_weights", weights_name);

                        if (!phone_primary_owner) {
                            // Different model variants may rename the final
                            // post-softmax router weights.  Print these if
                            // present so the A/B trace still captures the
                            // actual control tensor used by expert dispatch.
                            const char * suffixes[] = {
                                "ffn_moe_weights_softmax",
                                "ffn_moe_weights_norm",
                                "ffn_moe_weights_scaled",
                            };
                            for (const char * suffix : suffixes) {
                                char alt_name[96];
                                std::snprintf(
                                    alt_name,
                                    sizeof(alt_name),
                                    "%s-%d",
                                    suffix,
                                    trace_layer);
                                ggml_tensor * alt =
                                    find_exact_named_tensor(
                                        owner, alt_name);
                                if (alt != nullptr) {
                                    trace_f32(
                                        suffix,
                                        alt_name);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Numeric proof for the first Phone-primary Tensor layer.  This is
        // deliberately diagnostic-only: it does not change graph ownership or
        // reduction semantics.  Capture the two FFN partials before reduction
        // so we can prove whether each backend later contains their exact sum.
        bool tp_numeric_trace = false;
        int tp_numeric_layer = -1;
        int tp_numeric_chunk = -1;
        const char * tp_numeric_mode = nullptr;
        std::vector<float> tp_numeric_pre_pc;
        std::vector<float> tp_numeric_pre_phone;

        if (std::getenv("GGML_META_TP_FFN_NUMERIC_TRACE") != nullptr &&
                n_backends == 2 &&
                backend_ctx->tensor_phone_first_layer >= 0) {
            ggml_tensor * trace_node =
                backend_ctx->backend_configs[0].cgraphs[i].cgraph_main->nodes[
                    backend_ctx->backend_configs[0].cgraphs[i].cgraph_main->n_nodes - 1];

            int parsed = 0;
            int layer = -1;
            int chunk = -1;
            int chunk_layer = -1;
            if (trace_node->ne[1] == 1 &&
                    std::sscanf(
                        trace_node->name, "ffn_moe_out-%d%n",
                        &layer, &parsed) == 1 &&
                    trace_node->name[parsed] == '\0') {
                tp_numeric_layer = layer;
                tp_numeric_chunk = 0;
                tp_numeric_mode = "decode-single";
            } else if (ggml_backend_meta_parse_decode_ffn_chunk(
                           trace_node->name, chunk, chunk_layer)) {
                tp_numeric_layer = chunk_layer;
                tp_numeric_chunk = chunk;
                tp_numeric_mode = "decode";
            } else if (ggml_backend_meta_parse_prefill_down_chunk(
                           trace_node->name, chunk, chunk_layer)) {
                tp_numeric_layer = chunk_layer;
                tp_numeric_chunk = chunk;
                tp_numeric_mode = "prefill";
            }

            tp_numeric_trace =
                tp_numeric_layer == backend_ctx->tensor_phone_first_layer &&
                backend_ctx->backend_configs[0].cgraphs[i].cgraph_main
                    ->nodes[backend_ctx->backend_configs[0].cgraphs[i].cgraph_main->n_nodes - 1]->type == GGML_TYPE_F32 &&
                backend_ctx->backend_configs[1].cgraphs[i].cgraph_main
                    ->nodes[backend_ctx->backend_configs[1].cgraphs[i].cgraph_main->n_nodes - 1]->type == GGML_TYPE_F32;

            if (tp_numeric_trace) {
                auto & pc_cfg = backend_ctx->backend_configs[0];
                auto & ph_cfg = backend_ctx->backend_configs[1];
                ggml_tensor * pc_node =
                    pc_cfg.cgraphs[i].cgraph_main->nodes[
                        pc_cfg.cgraphs[i].cgraph_main->n_nodes - 1];
                ggml_tensor * ph_node =
                    ph_cfg.cgraphs[i].cgraph_main->nodes[
                        ph_cfg.cgraphs[i].cgraph_main->n_nodes - 1];

                if (ggml_nelements(pc_node) != ggml_nelements(ph_node)) {
                    tp_numeric_trace = false;
                } else {
                    reduce_fence_backend(pc_cfg.backend);
                    reduce_fence_backend(ph_cfg.backend);

                    const size_t n = ggml_nelements(pc_node);
                    tp_numeric_pre_pc.resize(n);
                    tp_numeric_pre_phone.resize(n);
                    ggml_backend_tensor_get(
                        pc_node, tp_numeric_pre_pc.data(), 0,
                        n * sizeof(float));
                    ggml_backend_tensor_get(
                        ph_node, tp_numeric_pre_phone.data(), 0,
                        n * sizeof(float));

                    auto print_pre = [&](const char * side,
                                         const std::vector<float> & values) {
                        double sum = 0.0;
                        double l2 = 0.0;
                        double max_abs = 0.0;
                        for (float x : values) {
                            sum += x;
                            l2 += (double) x * x;
                            max_abs = std::max(max_abs, std::abs((double) x));
                        }
                        printf(
                            "[TP_ALLREDUCE_NUMERIC] phase=PRE layer=%d chunk=%d "
                            "mode=%s side=%s n=%zu sum=%.9f l2=%.9f max=%.9f "
                            "v0=%.9f v1=%.9f v2=%.9f v3=%.9f\n",
                            tp_numeric_layer, tp_numeric_chunk,
                            tp_numeric_mode != nullptr ? tp_numeric_mode : "unknown",
                            side, values.size(), sum, std::sqrt(l2), max_abs,
                            values.size() > 0 ? values[0] : 0.0f,
                            values.size() > 1 ? values[1] : 0.0f,
                            values.size() > 2 ? values[2] : 0.0f,
                            values.size() > 3 ? values[3] : 0.0f);
                    };
                    print_pre("PC", tp_numeric_pre_pc);
                    print_pre("PHONE", tp_numeric_pre_phone);
                }
            }
        }

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            // Correctness-first ordering for heterogeneous all-reduce.
            //
            // ggml_backend_tensor_copy_async() falls back to
            // ggml_backend_synchronize() when the destination has no async
            // copy hook.  RPC's ordinary synchronize hook is intentionally a
            // no-op in this fork, so that fallback does NOT guarantee that a
            // remote producer has finished before another transfer socket
            // reads its tensor.  Make both dependencies explicit here:
            //   producer complete -> copy complete/visible on destination -> ADD.
            const int64_t src_fence_start_us = ggml_time_us();
            reduce_fence_backend(bcj_src.backend);
            const int64_t src_fence_us = ggml_time_us() - src_fence_start_us;

            const int64_t copy_start_us = ggml_time_us();
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);
            const int64_t copy_us = ggml_time_us() - copy_start_us;
            record_copy_wait(copy_us);
            record_meta_copy(i, j_src, j_dst, node_src, copy_us);

            const int64_t dst_fence_start_us = ggml_time_us();
            reduce_fence_backend(bcj_dst.backend);
            const int64_t dst_fence_us = ggml_time_us() - dst_fence_start_us;

            if (pipeline_debug) {
                printf(
                    "[META_ALLREDUCE_FENCE] sg=%zu %zu->%zu tensor=%s "
                    "src_fence_ms=%.3f copy_ms=%.3f dst_fence_ms=%.3f\n",
                    i, j_src, j_dst, node_src->name,
                    src_fence_us / 1000.0,
                    copy_us / 1000.0,
                    dst_fence_us / 1000.0);
            }

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            if (!has_data[j_src]) {
                ++reduce_zero_copy_skips;
                continue;
            }
            push_data(j_src, j_dst, i_buf);
            const int64_t add_start_us = ggml_time_us();
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            reduce_add_us += ggml_time_us() - add_start_us;
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            has_data[j_dst] = 1;
        }
        if (2*offset_j_max < n_backends) {
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);
            const std::vector<uint8_t> step_has_data = has_data;

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                if (!step_has_data[j]) {
                    ++reduce_zero_copy_skips;
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const int64_t add_start_us = ggml_time_us();
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                reduce_add_us += ggml_time_us() - add_start_us;
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            for (size_t j = 0; j < 2*offset_j_max; ++j) {
                const size_t j_other = j ^ offset_j;
                if (j_other < n_backends) {
                    has_data[j] = step_has_data[j] || step_has_data[j_other];
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        if (tp_numeric_trace) {
            auto & pc_cfg = backend_ctx->backend_configs[0];
            auto & ph_cfg = backend_ctx->backend_configs[1];
            ggml_tensor * pc_node =
                pc_cfg.cgraphs[i].cgraph_main->nodes[
                    pc_cfg.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * ph_node =
                ph_cfg.cgraphs[i].cgraph_main->nodes[
                    ph_cfg.cgraphs[i].cgraph_main->n_nodes - 1];

            reduce_fence_backend(pc_cfg.backend);
            reduce_fence_backend(ph_cfg.backend);

            const size_t n = tp_numeric_pre_pc.size();
            std::vector<float> post_pc(n);
            std::vector<float> post_phone(n);
            ggml_backend_tensor_get(
                pc_node, post_pc.data(), 0, n * sizeof(float));
            ggml_backend_tensor_get(
                ph_node, post_phone.data(), 0, n * sizeof(float));

            auto print_post = [&](const char * side,
                                  const std::vector<float> & values) {
                double sum = 0.0;
                double l2 = 0.0;
                double max_abs = 0.0;
                double diff_l2 = 0.0;
                double max_abs_diff = 0.0;
                size_t max_diff_index = 0;
                for (size_t k = 0; k < n; ++k) {
                    const double x = values[k];
                    const double expected =
                        (double) tp_numeric_pre_pc[k] +
                        (double) tp_numeric_pre_phone[k];
                    const double diff = x - expected;
                    sum += x;
                    l2 += x * x;
                    max_abs = std::max(max_abs, std::abs(x));
                    diff_l2 += diff * diff;
                    if (std::abs(diff) > max_abs_diff) {
                        max_abs_diff = std::abs(diff);
                        max_diff_index = k;
                    }
                }
                const double rms_diff =
                    n > 0 ? std::sqrt(diff_l2 / (double) n) : 0.0;
                printf(
                    "[TP_ALLREDUCE_NUMERIC] phase=POST layer=%d chunk=%d "
                    "mode=%s side=%s n=%zu sum=%.9f l2=%.9f max=%.9f "
                    "max_abs_diff=%.9g rms_diff=%.9g max_diff_index=%zu "
                    "v0=%.9f v1=%.9f v2=%.9f v3=%.9f\n",
                    tp_numeric_layer, tp_numeric_chunk,
                    tp_numeric_mode != nullptr ? tp_numeric_mode : "unknown",
                    side, values.size(), sum, std::sqrt(l2), max_abs,
                    max_abs_diff, rms_diff, max_diff_index,
                    values.size() > 0 ? values[0] : 0.0f,
                    values.size() > 1 ? values[1] : 0.0f,
                    values.size() > 2 ? values[2] : 0.0f,
                    values.size() > 3 ? values[3] : 0.0f);
            };

            print_post("PC", post_pc);
            print_post("PHONE", post_phone);

            double cross_l2 = 0.0;
            double cross_max = 0.0;
            for (size_t k = 0; k < n; ++k) {
                const double diff =
                    (double) post_pc[k] - (double) post_phone[k];
                cross_l2 += diff * diff;
                cross_max = std::max(cross_max, std::abs(diff));
            }
            printf(
                "[TP_ALLREDUCE_NUMERIC] phase=CROSS layer=%d chunk=%d "
                "mode=%s pc_phone_max_abs_diff=%.9g "
                "pc_phone_rms_diff=%.9g\n",
                tp_numeric_layer, tp_numeric_chunk,
                tp_numeric_mode != nullptr ? tp_numeric_mode : "unknown",
                cross_max,
                n > 0 ? std::sqrt(cross_l2 / (double) n) : 0.0);
        }

        if (pipeline_debug && i == 1 && n_backends == 2) {
            auto & bcj = backend_ctx->backend_configs[0];
            ggml_cgraph * graph = bcj.cgraphs[i].cgraph_main;
            ggml_tensor * node = graph->nodes[graph->n_nodes - 1];
            meta_debug_tensor(bcj.backend, node, "D post-reduce backend0");
        }

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            const size_t j_src = j - 2*offset_j_max;
            if (!has_data[j_src]) {
                ++reduce_zero_copy_skips;
                continue;
            }
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            const int64_t copy_start_us = ggml_time_us();
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
            const int64_t copy_us = ggml_time_us() - copy_start_us;
            record_copy_wait(copy_us);
            record_meta_copy(i, j_src, j, node_src, copy_us);
            has_data[j] = 1;
        }

        return GGML_STATUS_SUCCESS;
    };


    if (backend_ctx->compute_workers == nullptr) {
        backend_ctx->compute_workers = new ggml_backend_meta_compute_workers(backend_ctx, n_backends);
    }
    ggml_backend_meta_compute_workers & compute_workers = *backend_ctx->compute_workers;
    compute_workers.reset_timings();
    int64_t compute_wall_us = 0;
    int64_t reduce_wall_us  = 0;
    bool compute_complete = false;
    auto is_decode_pc_only_norm_sg = [&](size_t i) -> bool {
        return subgraph_is_decode_pc_only_norm(i);
    };
    auto parse_prefill_sg = [&](size_t i, bool norm, int & chunk, int & layer) -> bool {
        if (n_backends != 2) {
            return false;
        }

        ggml_cgraph * pc_graph = backend_ctx->backend_configs[0].cgraphs[i].cgraph_main;
        ggml_cgraph * phone_graph = backend_ctx->backend_configs[1].cgraphs[i].cgraph_main;
        if (pc_graph == nullptr || phone_graph == nullptr ||
                pc_graph->n_nodes == 0 || phone_graph->n_nodes == 0) {
            return false;
        }

        ggml_tensor * pc_last = pc_graph->nodes[pc_graph->n_nodes - 1];
        ggml_tensor * phone_last = phone_graph->nodes[phone_graph->n_nodes - 1];
        int phone_chunk = -1;
        int phone_layer = -1;
        const bool parsed = norm ?
            ggml_backend_meta_parse_prefill_norm_chunk(pc_last->name, chunk, layer) &&
                ggml_backend_meta_parse_prefill_norm_chunk(
                    phone_last->name, phone_chunk, phone_layer) :
            ggml_backend_meta_parse_prefill_down_chunk(pc_last->name, chunk, layer) &&
                ggml_backend_meta_parse_prefill_down_chunk(
                    phone_last->name, phone_chunk, phone_layer);
        return parsed && chunk == phone_chunk && layer == phone_layer;
    };
    auto parse_prefill_wave_attn_sg =
        [&](size_t i, int & chunk_begin, int & chunk_count, int & layer) -> bool {
            if (n_backends != 2 || i >= backend_ctx->n_subgraphs) {
                return false;
            }

            auto find_wave_attn =
                [&](size_t backend, int & out_begin, int & out_count, int & out_layer) -> bool {
                    ggml_cgraph * graph =
                        backend_ctx->backend_configs[backend].cgraphs[i].cgraph_main;
                    if (graph == nullptr || graph->n_nodes == 0) {
                        return false;
                    }

                    bool found = false;
                    for (int k = 0; k < graph->n_nodes; ++k) {
                        int parsed_begin = -1;
                        int parsed_count = 1;
                        int parsed_layer = -1;
                        bool parsed = ggml_backend_meta_parse_prefill_wave_attn_out_group(
                            graph->nodes[k]->name, parsed_begin, parsed_count, parsed_layer);
                        if (!parsed) {
                            parsed = ggml_backend_meta_parse_prefill_wave_attn_out_chunk(
                                graph->nodes[k]->name, parsed_begin, parsed_layer);
                            parsed_count = 1;
                        }
                        if (!parsed) {
                            continue;
                        }
                        if (parsed_count <= 0) {
                            return false;
                        }
                        if (found &&
                                (parsed_begin != out_begin ||
                                 parsed_count != out_count ||
                                 parsed_layer != out_layer)) {
                            return false;
                        }
                        out_begin = parsed_begin;
                        out_count = parsed_count;
                        out_layer = parsed_layer;
                        found = true;
                    }
                    return found;
                };

            int phone_begin = -1;
            int phone_count = -1;
            int phone_layer = -1;
            if (!find_wave_attn(0, chunk_begin, chunk_count, layer) ||
                    !find_wave_attn(1, phone_begin, phone_count, phone_layer)) {
                return false;
            }
            return chunk_begin == phone_begin &&
                   chunk_count == phone_count &&
                   layer == phone_layer;
        };

    auto phone_sg_pair_same_layer = [&](size_t i, int & layer) -> bool {
        if (n_backends != 2 || i + 1 >= backend_ctx->n_subgraphs ||
                !subgraph_is_phone_only(i) || !subgraph_is_phone_only(i + 1)) {
            return false;
        }

        ggml_cgraph * g0 = backend_ctx->backend_configs[1].cgraphs[i].cgraph_main;
        ggml_cgraph * g1 = backend_ctx->backend_configs[1].cgraphs[i + 1].cgraph_main;
        if (g0 == nullptr || g1 == nullptr || g0->n_nodes == 0 || g1->n_nodes == 0) {
            return false;
        }

        int attn_layer = -1;
        int ffn_layer = -1;
        const bool is_attn =
            std::sscanf(g0->nodes[g0->n_nodes - 1]->name, "attn_out-%d", &attn_layer) == 1;
        const bool is_ffn =
            std::sscanf(g1->nodes[0]->name, "ffn_inp-%d", &ffn_layer) == 1;
        if (!is_attn || !is_ffn || attn_layer != ffn_layer) {
            return false;
        }

        // A Phone-primary Tensor layer also has Phone-owned Attention and
        // ffn_inp, but its FFN is split across PC+Phone. It must not enter
        // the pure PHONE_ONLY block-fusion / block-exit path.
        if (layer_is_tensor_phone_primary(attn_layer)) {
            return false;
        }

        layer = attn_layer;
        return true;
    };
    auto find_phone_block = [&](size_t start, size_t & end, int & first_layer, int & last_layer) -> bool {
        int layer = -1;
        if (!phone_sg_pair_same_layer(start, layer)) {
            return false;
        }

        first_layer = layer;
        last_layer = layer;
        end = start + 1;
        for (size_t next = start + 2; next + 1 < backend_ctx->n_subgraphs; next += 2) {
            int next_layer = -1;
            if (!phone_sg_pair_same_layer(next, next_layer) || next_layer != last_layer + 1) {
                break;
            }
            last_layer = next_layer;
            end = next + 1;
        }
        return true;
    };
    auto build_phone_block_graph = [&](size_t start, size_t end) -> ggml_cgraph * {
        GGML_ASSERT(start <= end && end < backend_ctx->n_subgraphs);
        ggml_cgraph * fused = backend_ctx->cgraphs_phone_fused[start];
        GGML_ASSERT(fused != nullptr);

        fused->n_nodes = 0;
        fused->n_leafs = 0;
        ggml_hash_set_reset(&fused->visited_hash_set);
        uint64_t uid = 0;
        for (size_t sg = start; sg <= end; ++sg) {
            ggml_cgraph * source =
                backend_ctx->backend_configs[1].cgraphs[sg].cgraph_main;
            GGML_ASSERT(source != nullptr);
            GGML_ASSERT(fused->n_nodes + source->n_nodes <= fused->size);
            for (int k = 0; k < source->n_nodes; ++k) {
                ggml_tensor * node = source->nodes[k];
                fused->nodes[fused->n_nodes++] = node;
                const size_t source_pos = ggml_hash_find(&source->visited_hash_set, node);
                const size_t fused_pos = ggml_hash_insert(&fused->visited_hash_set, node);
                fused->use_counts[fused_pos] = source->use_counts[source_pos];
            }
            if (sg == start) {
                uid = source->uid;
            } else {
                uid ^= source->uid + 0x9e3779b97f4a7c15ULL + (uid << 6) + (uid >> 2);
            }
        }
        fused->uid = uid != 0 ? uid : 1;
        return fused;
    };

    struct meta_layer_timing {
        int64_t compute_pc_us    = 0;
        int64_t compute_phone_us = 0;
        int64_t copy_0to1_us     = 0;
        int64_t copy_1to0_us     = 0;
        int64_t total_us         = 0;
    };
    std::map<std::pair<int, int>, meta_layer_timing> layer_timings;

    auto subgraph_layer = [&](size_t sg) -> int {
        for (size_t backend = 0; backend < n_backends; ++backend) {
            ggml_cgraph * graph = backend_ctx->backend_configs[backend].cgraphs[sg].cgraph_main;
            if (graph == nullptr) {
                continue;
            }
            for (int node_id = 0; node_id < graph->n_nodes; ++node_id) {
                const char * name = graph->nodes[node_id]->name;
                int chunk = -1;
                int layer = -1;
                if (ggml_backend_meta_parse_prefill_norm_chunk(name, chunk, layer) ||
                    ggml_backend_meta_parse_prefill_down_chunk(name, chunk, layer) ||
                    std::sscanf(name, "attn_out-%d", &layer) == 1 ||
                    std::sscanf(name, "ffn_inp-%d", &layer) == 1 ||
                    std::sscanf(name, "l_out-%d", &layer) == 1) {
                    return layer;
                }
            }
        }
        return -1;
    };

    auto tensor_phone_stage_route_identity =
        [&](size_t sg, int & layer, int & chunk) -> bool {
            layer = -1;
            chunk = -1;
            if (sg >= backend_ctx->n_subgraphs) {
                return false;
            }
            for (size_t backend = 0; backend < n_backends; ++backend) {
                ggml_cgraph * graph =
                    backend_ctx->backend_configs[backend]
                        .cgraphs[sg].cgraph_main;
                if (graph == nullptr || graph->n_nodes <= 0) {
                    continue;
                }
                const char * name =
                    graph->nodes[graph->n_nodes - 1]->name;
                if (ggml_backend_meta_parse_phone_route_weights(
                        name, chunk, layer) &&
                        layer_is_tensor_phone_primary(layer)) {
                    return true;
                }
            }
            return false;
        };

    auto tensor_phone_stage_ffn_identity =
        [&](size_t sg, int & layer, int & chunk, bool & decode)
            -> bool {
            layer = -1;
            chunk = -1;
            decode = false;
            if (sg >= backend_ctx->n_subgraphs) {
                return false;
            }

            for (size_t backend = 0; backend < n_backends; ++backend) {
                ggml_cgraph * graph =
                    backend_ctx->backend_configs[backend]
                        .cgraphs[sg].cgraph_main;
                if (graph == nullptr || graph->n_nodes <= 0) {
                    continue;
                }

                const char * name =
                    graph->nodes[graph->n_nodes - 1]->name;
                int parsed_layer = -1;
                int parsed_chunk = -1;
                if (ggml_backend_meta_parse_prefill_down_chunk(
                        name, parsed_chunk, parsed_layer) &&
                        layer_is_tensor_phone_primary(parsed_layer)) {
                    layer = parsed_layer;
                    chunk = parsed_chunk;
                    decode = false;
                    return true;
                }
                if (ggml_backend_meta_parse_decode_ffn_chunk(
                        name, parsed_chunk, parsed_layer) &&
                        layer_is_tensor_phone_primary(parsed_layer)) {
                    layer = parsed_layer;
                    chunk = parsed_chunk;
                    decode = true;
                    return true;
                }

                int parsed = 0;
                if (std::sscanf(
                        name, "ffn_moe_out-%d%n",
                        &parsed_layer, &parsed) == 1 &&
                        name[parsed] == '\0' &&
                        layer_is_tensor_phone_primary(parsed_layer)) {
                    layer = parsed_layer;
                    chunk = 0;
                    decode = true;
                    return true;
                }
            }
            return false;
        };

    auto backend_times_snapshot = [&]() {
        std::lock_guard<std::mutex> lock(compute_workers.mutex);
        return compute_workers.backend_time_us;
    };
    auto copy_time_snapshot = [&](size_t src, size_t dst) {
        std::lock_guard<std::mutex> lock(meta_copy_stats_mutex);
        return reduce_copy_by_direction[src*n_backends + dst].total_us;
    };

    // Async Phone submit may release the hybrid scheduler only after graph
    // rebuild/preparation is complete. This avoids racing the Meta STC/simple
    // tensor cache with the next CPU-stage graph construction.
    {
        std::lock_guard<std::mutex> lock(backend_ctx->async_graph_mutex);
        if (backend_ctx->async_graph_active) {
            backend_ctx->async_graph_prepared = true;
            backend_ctx->async_graph_cv.notify_all();
        }
    }

    const int64_t meta_execute_begin_us = ggml_time_us();
    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        const size_t timing_sg = i;
        const int64_t layer_wall_start_us = ggml_time_us();
        const auto backend_times_before = backend_times_snapshot();
        const int64_t copy_0to1_before = n_backends > 1 ? copy_time_snapshot(0, 1) : 0;
        const int64_t copy_1to0_before = n_backends > 1 ? copy_time_snapshot(1, 0) : 0;
        int timing_first_layer = subgraph_layer(i);
        int timing_last_layer  = timing_first_layer;

        size_t communication_sg = i;
        bool phone_block_fused = false;
        int phone_block_last_layer = -1;
        int prefill_norm_chunk = -1;
        int prefill_norm_layer = -1;
        int prefill_down_chunk = -1;
        int prefill_down_layer = -1;
        int prefill_wave_attn_chunk = -1;
        int prefill_wave_attn_chunk_count = 1;
        int prefill_wave_attn_layer = -1;
        const bool is_prefill_norm_sg = parse_prefill_sg(
            i, true, prefill_norm_chunk, prefill_norm_layer);
        const bool is_prefill_down_sg = parse_prefill_sg(
            i, false, prefill_down_chunk, prefill_down_layer);
        const bool is_prefill_wave_attn_sg = parse_prefill_wave_attn_sg(
            i, prefill_wave_attn_chunk, prefill_wave_attn_chunk_count,
            prefill_wave_attn_layer);
        const bool is_phone_only_sg = subgraph_is_phone_only(i);
if (pipeline_debug && is_prefill_norm_sg) {
    auto * g_pc =
        backend_ctx->backend_configs[0]
            .cgraphs[i].cgraph_main;

    auto * g_phone =
        backend_ctx->backend_configs[1]
            .cgraphs[i].cgraph_main;

    int pc_active = 0;
    int phone_active = 0;

     for (int k = 0; k < g_phone->n_nodes; ++k) {
        ggml_tensor * pn = g_phone->nodes[k];
        ggml_tensor * cn = g_pc->nodes[k];

        const bool pc_compute =
            (cn->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
        const bool phone_compute =
            (pn->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;

        if (pc_compute) {
            ++pc_active;
        }
        if (phone_compute) {
            ++phone_active;
        }

        if (!phone_compute) {
            continue;
        }

        GGML_LOG_INFO(
            "[PREFILL_PHONE_NODE] "
            "sg=%zu layer=%d chunk=%d k=%d "
            "name=%s op=%s "
            "ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] "
            "pc_compute=%d phone_compute=%d\n",
            i,
            prefill_norm_layer,
            prefill_norm_chunk,
            k,
            pn->name,
            ggml_op_name(pn->op),
            pn->ne[0],
            pn->ne[1],
            pn->ne[2],
            pn->ne[3],
            !!(cn->flags & GGML_TENSOR_FLAG_COMPUTE),
            !!(pn->flags & GGML_TENSOR_FLAG_COMPUTE));
    }

    GGML_LOG_INFO(
        "[PREFILL_NORM_SG] sg=%zu "
        "layer=%d chunk=%d "
        "first=%s last=%s nodes=%d "
        "pc_active=%d phone_active=%d\n",
        i,
        prefill_norm_layer,
        prefill_norm_chunk,
        g_pc->nodes[0]->name,
        g_pc->nodes[g_pc->n_nodes - 1]->name,
        g_pc->n_nodes,
        pc_active,
        phone_active);
}
auto prefill_norm_sg_has_prework =
    [&](size_t sg) -> bool {
        if (sg >= backend_ctx->n_subgraphs) {
            return false;
        }

        ggml_cgraph * graph =
            backend_ctx->backend_configs[0]
                .cgraphs[sg].cgraph_main;

        if (graph == nullptr || graph->n_nodes <= 1) {
            return false;
        }

        // 最后一个节点是 prefill_ffn_norm_chunk。
        // 如果它前面还有 PC COMPUTE 节点，
        // 那么这个 SG 不能假设与之前的 reduce 无关。
        for (int k = 0; k < graph->n_nodes - 1; ++k) {
            if (graph->nodes[k]->flags &
                    GGML_TENSOR_FLAG_COMPUTE) {
                return true;
            }
        }

        return false;
    };
        const bool norm_can_overlap =
            is_prefill_norm_sg &&
            has_pending_prefill_reduce_for_layer(prefill_norm_layer) &&
            !prefill_norm_sg_has_prework(i);

        const bool down_can_overlap =
            is_prefill_down_sg &&
            has_pending_prefill_reduce_for_layer(prefill_down_layer);

        const bool continues_prefill_layer =
            norm_can_overlap || down_can_overlap || is_prefill_wave_attn_sg;
        if (return_wavefront_graph &&
                (is_prefill_wave_attn_sg || is_prefill_norm_sg || is_prefill_down_sg)) {
            if (is_prefill_wave_attn_sg) {
                // Attention(L,C) consumes OUT(L-1,C), not the most recently
                // submitted down tensor in graph order.  Let same-layer chunk
                // prework run while current-layer returns are in flight, and
                // when crossing a layer boundary wait only for the exact
                // predecessor chunk.  Older than L-1 is outside the V1
                // one-layer lookahead window and must be drained first.
                bool older_waited = false;
                int64_t older_wait_us = 0;
                const ggml_status older_status = wait_prefill_reduces_before(
                    prefill_wave_attn_layer - 1, older_waited, older_wait_us);
                if (older_status != GGML_STATUS_SUCCESS) {
                    return older_status;
                }

                bool older_phone_waited = false;
                int64_t older_phone_wait_us = 0;
                if (phone_prefill_chunk_join_active) {
                    const ggml_status phone_status =
                        reap_phone_prefill_pc_layers_before(
                            prefill_wave_attn_layer - 1,
                            older_phone_waited,
                            older_phone_wait_us);
                    if (phone_status != GGML_STATUS_SUCCESS) {
                        return phone_status;
                    }
                }

                if (older_waited || older_phone_waited) {
                    ++return_wave_old_layer_wait_count;
                    return_wave_old_layer_wait_us +=
                        older_wait_us + older_phone_wait_us;
                }

                bool waited = false;
                int64_t dependency_wait_us = 0;
                if (prefill_wave_attn_layer > return_wavefront_first_layer) {
                    for (int dep_chunk = prefill_wave_attn_chunk;
                         dep_chunk < prefill_wave_attn_chunk + prefill_wave_attn_chunk_count;
                         ++dep_chunk) {
                        bool reduce_dep_waited = false;
                        const int64_t reduce_wait_start_us =
                            ggml_time_us();
                        const ggml_status reduce_status =
                            wait_prefill_reduce_dependency(
                                prefill_wave_attn_layer - 1,
                                dep_chunk,
                                reduce_dep_waited);
                        const int64_t reduce_dep_wait_us =
                            ggml_time_us() - reduce_wait_start_us;
                        if (reduce_status != GGML_STATUS_SUCCESS) {
                            return reduce_status;
                        }

                        bool phone_dep_waited = false;
                        int64_t phone_dep_wait_us = 0;
                        if (phone_prefill_chunk_join_active) {
                            const ggml_status phone_status =
                                wait_phone_prefill_pc_dependency(
                                    prefill_wave_attn_layer - 1,
                                    dep_chunk,
                                    phone_dep_waited,
                                    phone_dep_wait_us);
                            if (phone_status != GGML_STATUS_SUCCESS) {
                                return phone_status;
                            }
                        }

                        const bool dep_waited =
                            reduce_dep_waited ||
                            phone_dep_waited;
                        const int64_t dep_wait_us =
                            reduce_dep_wait_us +
                            phone_dep_wait_us;
                        dependency_wait_us += dep_wait_us;
                        waited = waited || dep_waited;
                        if (dep_waited) {
                            ++return_wave_dependency_wait_count;
                            return_wave_dependency_wait_us += dep_wait_us;
                            return_wave_dependency_wait_max_us =
                                std::max(
                                    return_wave_dependency_wait_max_us,
                                    dep_wait_us);
                            if (pipeline_debug) {
                                printf(
                                    "[RETURN_WAVEFRONT_DEP_WAIT] "
                                    "layer=%d chunk=%d "
                                    "group_begin=%d group_count=%d "
                                    "predecessor=%d wait_ms=%.3f "
                                    "generic=%d phone=%d\n",
                                    prefill_wave_attn_layer,
                                    dep_chunk,
                                    prefill_wave_attn_chunk,
                                    prefill_wave_attn_chunk_count,
                                    prefill_wave_attn_layer - 1,
                                    dep_wait_us / 1000.0,
                                    reduce_dep_waited ? 1 : 0,
                                    phone_dep_waited ? 1 : 0);
                            }
                        }
                    }
                }

                const bool predecessor_layer_still_in_flight =
                    prefill_wave_attn_layer > return_wavefront_first_layer &&
                    (has_pending_prefill_reduce_for_layer(
                         prefill_wave_attn_layer - 1) ||
                     has_pending_phone_prefill_pc_for_layer(
                         prefill_wave_attn_layer - 1));
                if (has_pending_prefill_reduce() ||
                        !pending_phone_prefill_pc_branches.empty()) {
                    ++return_wave_overlap_boundaries;
                }
                if (predecessor_layer_still_in_flight) {
                    return_wave_ahead_attn_chunks += prefill_wave_attn_chunk_count;
                }

                if (pipeline_debug) {
                    printf(
                        "[RETURN_WAVEFRONT_PREWORK] layer=%d group_begin=%d group_count=%d waited=%d "
                        "wait_ms=%.3f predecessor_pending=%d any_pending=%d\n",
                        prefill_wave_attn_layer, prefill_wave_attn_chunk,
                        prefill_wave_attn_chunk_count, waited ? 1 : 0,
                        dependency_wait_us / 1000.0,
                        predecessor_layer_still_in_flight ? 1 : 0,
                        (has_pending_prefill_reduce() ||
                         !pending_phone_prefill_pc_branches.empty()) ? 1 : 0);
                }
            } else if (is_prefill_down_sg &&
                       prefill_down_layer > return_wavefront_first_layer &&
                       (has_pending_prefill_reduce_for_layer(
                            prefill_down_layer - 1) ||
                        has_pending_phone_prefill_pc_for_layer(
                            prefill_down_layer - 1))) {
                // The exact predecessor dependency was already enforced at
                // this chunk's Attention prework.  Reaching down while another
                // L-1 chunk still returns means Phone FFN(L,C) is submitted
                // ahead of the old layer's complete return frontier.
                ++return_wave_ahead_phone_submits;
            }
            // Norm subgraphs need no additional return wait: their Attention
            // prework has already enforced the exact OUT dependency.
        } else if ((has_pending_prefill_reduce() ||
                    (return_wavefront_graph &&
                     phone_prefill_chunk_join_active &&
                     !pending_phone_prefill_pc_branches.empty())) &&
                   !continues_prefill_layer) {
            size_t pending_lanes = 0;
            int barrier_layer = -1;
            for (size_t lane = 0; lane < ggml_backend_meta_context::PREFILL_RETURN_LANES; ++lane) {
                if (pending_prefill_reduce_task[lane] != 0) {
                    ++pending_lanes;
                    barrier_layer = std::max(barrier_layer, pending_prefill_reduce_layer[lane]);
                }
            }
            for (const auto & branch :
                    pending_phone_prefill_pc_branches) {
                ++pending_lanes;
                barrier_layer =
                    std::max(barrier_layer, branch.layer);
            }

            int last_lane = -1;
            const int64_t wait_start_us = ggml_time_us();
            ggml_status status = GGML_STATUS_SUCCESS;
            if (has_pending_prefill_reduce()) {
                status = wait_all_prefill_reduces(&last_lane);
            }
            while (status == GGML_STATUS_SUCCESS &&
                   return_wavefront_graph &&
                   phone_prefill_chunk_join_active &&
                   !pending_phone_prefill_pc_branches.empty()) {
                const int pending_layer =
                    pending_phone_prefill_pc_branches.front().layer;
                status = reap_phone_prefill_pc_layer(
                    pending_layer, true);
            }
            const int64_t wait_us = ggml_time_us() - wait_start_us;
            ++layer_barrier_wait_count;
            layer_barrier_wait_us += wait_us;
            if (return_wavefront_graph && barrier_layer >= return_wavefront_first_layer) {
                return_wave_layer_barrier_us[barrier_layer] += wait_us;
            }
            if (wait_us >= layer_barrier_wait_max_us) {
                layer_barrier_wait_max_us = wait_us;
                layer_barrier_wait_max_layer = barrier_layer;
                layer_barrier_wait_max_pending = (int64_t) pending_lanes;
                layer_barrier_wait_max_last_lane = last_lane;
            }
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        int64_t subgraph_compute_wall_us = 0;
        if (!compute_complete) {
    if (n_backends == 2) {
    ggml_tensor * phone_wdown =
        get_ffn_down_boundary_node(1, i);

    ggml_tensor * pc_wdown =
        get_ffn_down_boundary_node(0, i);

    bool decode_handoff_to_phone = false;

    if (phone_wdown != nullptr) {
        int chunk = -1;
        int layer = -1;

        if (ggml_backend_meta_parse_decode_ffn_chunk(
                phone_wdown->name,
                chunk,
                layer)) {

            decode_handoff_to_phone =
                decode_layer_hands_off_to_phone(i, layer);
        }
    }

    // Snapshot preparation must use exactly the same compatibility rule as
    // the communication path below.  Otherwise RPC graph_compute() sees an
    // active prepared snapshot and emits GRAPH_RECOMPUTE_SNAPSHOT even when
    // specialized_communication() later chooses the conservative fence/copy
    // path.  The produced snapshot is then never consumed, leaving the remote
    // slot READY; after both slots fill, a later decode blocks forever waiting
    // for a free slot.
    //
    // Dense decode chunks end directly in MUL_MAT and are compatible with the
    // existing snapshot path.  Qwen3-MoE chunks end after expert aggregation
    // (typically ADD), so do not prepare a snapshot for them yet.
    const bool snapshot_compatible_decode_chunk =
        phone_wdown != nullptr &&
        pc_wdown != nullptr &&
        phone_wdown->op == GGML_OP_MUL_MAT &&
        pc_wdown->op == GGML_OP_MUL_MAT;

    if (phone_wdown != nullptr &&
            pc_wdown != nullptr &&
            phone_wdown->ne[1] == 1 &&
            (phone_wdown->flags &
                GGML_TENSOR_FLAG_COMPUTE) &&
            !decode_handoff_to_phone &&
            snapshot_compatible_decode_chunk) {

        prepare_graph_snapshot(i, phone_wdown);
    } else if (pipeline_debug &&
            phone_wdown != nullptr &&
            pc_wdown != nullptr &&
            !decode_handoff_to_phone &&
            !snapshot_compatible_decode_chunk) {
        int chunk = -1;
        int layer = -1;
        ggml_backend_meta_parse_decode_ffn_chunk(
            phone_wdown->name, chunk, layer);
        printf(
            "[DECODE_SNAPSHOT_PREPARE_SKIP] layer=%d chunk=%d phone_op=%s pc_op=%s\n",
            layer,
            chunk,
            ggml_op_name(phone_wdown->op),
            ggml_op_name(pc_wdown->op));
    }
}

    if (n_backends == 2 && is_prefill_down_sg &&
            !layer_attention_phone_owned(prefill_down_layer)) {
        ggml_tensor * phone_prefill_down = get_prefill_down_boundary_node(1, i);
        ggml_tensor * pc_prefill_down = get_prefill_down_boundary_node(0, i);
        if (phone_prefill_down != nullptr && pc_prefill_down != nullptr &&
                (phone_prefill_down->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            prepare_graph_snapshot(i, phone_prefill_down);

            if (pipeline_debug) {
                GGML_LOG_INFO("[PREFILL_SNAPSHOT_PREPARE] sg=%zu layer=%d chunk=%d "
                       "tensor=%s bytes=%zu\n",
                       i, prefill_down_layer, prefill_down_chunk,
                       phone_prefill_down->name, ggml_nbytes(phone_prefill_down));
            }
        }
    }

    const int next_layer = is_prefill_down_sg ? prefill_down_layer : prefill_norm_layer;
    const int next_chunk = is_prefill_down_sg ? prefill_down_chunk : prefill_norm_chunk;
    const bool trace_prefill_chunk = pipeline_debug && (is_prefill_down_sg || is_prefill_norm_sg);
    const int64_t next_begin_us = ggml_time_us();
    if (trace_prefill_chunk) {
        GGML_LOG_INFO("[PREFILL_NEXT_BEGIN] layer=%d chunk=%d t=%" PRId64 "\n", next_layer, next_chunk, next_begin_us);
    }

        const int64_t compute_start_us = ggml_time_us();

    if (is_prefill_wave_attn_sg && prefill_wave_attn_chunk == 0) {
        auto it = return_wave_layer_start_us.find(prefill_wave_attn_layer);
        if (it == return_wave_layer_start_us.end() || compute_start_us < it->second) {
            return_wave_layer_start_us[prefill_wave_attn_layer] = compute_start_us;
        }
    }

    for (size_t backend = 0; backend < n_backends; ++backend) {
        if ((std::getenv("GGML_META_TP_INPUT_TRACE") != nullptr ||
             std::getenv("GGML_META_TP_FFN_NUMERIC_TRACE") != nullptr) &&
            backend_ctx->tensor_phone_first_layer >= 0) {
            for (int layer = backend_ctx->tensor_phone_first_layer;
                 layer < backend_ctx->tensor_phone_last_layer; ++layer) {
                debug_layer_input(backend, i, layer, false);
            }
        } else {
            debug_layer_input(backend, i, 24, false);
        }
    }

    ggml_status compute_status = GGML_STATUS_SUCCESS;
    if (is_prefill_down_sg) {
        const bool has_async_prefill_input =
            pending_prefill_input_task != 0;

        if (has_async_prefill_input) {
            GGML_ASSERT(prefill_down_layer == pending_prefill_input_layer);
            GGML_ASSERT(prefill_down_chunk == pending_prefill_input_chunk);
        }

        if (return_wavefront_graph) {
            // Dense graphs normally arrive here with an async PC->Phone input
            // task. MoE can insert Router/Top-K/expert subgraphs between the
            // norm and down boundaries, so Meta may legitimately fall back to
            // the generic handoff path. That path is dependency-correct; it
            // only gives up some H2D overlap, so allow it in the return
            // wavefront instead of rejecting the whole MoE experiment.
            if (!has_async_prefill_input && pipeline_debug) {
                GGML_LOG_INFO(
                    "[RETURN_WAVEFRONT_INPUT_FALLBACK] layer=%d chunk=%d "
                    "mode=generic_handoff\n",
                    prefill_down_layer, prefill_down_chunk);
            }
            // Bound Phone producer pressure by producer completion, not by
            // full snapshot return completion. The snapshot client publishes
            // seq readiness as soon as the first payload byte arrives, which
            // can only happen after the server slot reached READY.
            constexpr size_t phone_inflight_limit = 2;
            GGML_ASSERT(return_wave_snapshot_ready_waiter != nullptr);

            if (return_wave_phone_credit_seqs.size() >= phone_inflight_limit) {
                const uint64_t oldest_seq = return_wave_phone_credit_seqs.front();
                const size_t pending_before = return_wave_phone_credit_seqs.size();
                const int64_t wait_start_us = ggml_time_us();
                const bool ready = return_wave_snapshot_ready_waiter(
                    backend_ctx->backend_configs[1].backend,
                    oldest_seq);
                const int64_t wait_us = ggml_time_us() - wait_start_us;

                ++return_wave_phone_credit_wait_count;
                return_wave_phone_credit_wait_us += wait_us;
                return_wave_phone_credit_wait_max_us =
                    std::max(return_wave_phone_credit_wait_max_us, wait_us);

                if (return_path_debug || pipeline_debug) {
                    printf(
                        "[PHONE_CREDIT_WAIT] limit=%zu pending_before=%zu old_seq=%" PRIu64
                        " new_layer=%d new_chunk=%d wait_ms=%.3f ready=%d\n",
                        phone_inflight_limit, pending_before, oldest_seq,
                        prefill_down_layer, prefill_down_chunk,
                        wait_us / 1000.0, ready ? 1 : 0);
                    fflush(stdout);
                }

                if (!ready) {
                    return GGML_STATUS_FAILED;
                }
                return_wave_phone_credit_seqs.pop_front();
            }
        }

        const auto route_it = pending_phone_prefill_routes.find(
            { prefill_down_layer, prefill_down_chunk });
        const bool pipeline_phone_prefill_down =
            phone_prefill_chunk_pipeline &&
            !return_wavefront_graph &&
            !has_async_prefill_input &&
            layer_attention_phone_owned(prefill_down_layer) &&
            route_it != pending_phone_prefill_routes.end() &&
            std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER") != nullptr &&
            std::getenv("LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE") != nullptr;

        if (pipeline_phone_prefill_down) {
            const phone_prefill_route_task route = route_it->second;
            GGML_ASSERT(route.worker != nullptr);
            GGML_ASSERT(route.task != 0);

            const int64_t stage_wait_begin_us = ggml_time_us();
            const ggml_status route_stage_status =
                route.worker->wait_stage_ready(route.task);
            const int64_t stage_wait_us =
                ggml_time_us() - stage_wait_begin_us;
            if (route_stage_status != GGML_STATUS_SUCCESS) {
                return route_stage_status;
            }

            if (backend_ctx->prefill_pc_worker == nullptr) {
                backend_ctx->prefill_pc_worker =
                    new ggml_backend_meta_transfer_worker();
            }

            ggml_cgraph * pc_graph =
                backend_ctx->backend_configs[0].cgraphs[i].cgraph_main;
            ggml_cgraph * phone_graph =
                backend_ctx->backend_configs[1].cgraphs[i].cgraph_main;
            ggml_backend_t pc_backend =
                backend_ctx->backend_configs[0].backend;
            ggml_backend_t phone_backend =
                backend_ctx->backend_configs[1].backend;
            GGML_ASSERT(pc_graph != nullptr);
            GGML_ASSERT(phone_graph != nullptr);

            const uint64_t phone_ffn_seq =
                phone_prefill_chunk_join_active ?
                    backend_ctx->next_phone_prefill_ffn_seq++ : 0;

            ggml_tensor * pc_return_src = nullptr;
            ggml_tensor * phone_return_dst = nullptr;
            ggml_tensor * phone_return_stage = nullptr;
            std::shared_ptr<std::vector<uint8_t>> return_payload;
            if (phone_prefill_async_return_active) {
                GGML_ASSERT(pc_graph->n_nodes > 0);
                GGML_ASSERT(phone_graph->n_nodes > 0);
                pc_return_src =
                    pc_graph->nodes[pc_graph->n_nodes - 1];
                phone_return_dst =
                    phone_graph->nodes[phone_graph->n_nodes - 1];

                int pc_return_chunk = -1;
                int pc_return_layer = -1;
                int phone_return_chunk = -1;
                int phone_return_layer = -1;
                GGML_ASSERT(
                    ggml_backend_meta_parse_prefill_down_chunk(
                        pc_return_src->name,
                        pc_return_chunk,
                        pc_return_layer));
                GGML_ASSERT(
                    ggml_backend_meta_parse_prefill_down_chunk(
                        phone_return_dst->name,
                        phone_return_chunk,
                        phone_return_layer));
                GGML_ASSERT(pc_return_chunk == prefill_down_chunk);
                GGML_ASSERT(phone_return_chunk == prefill_down_chunk);
                GGML_ASSERT(pc_return_layer == prefill_down_layer);
                GGML_ASSERT(phone_return_layer == prefill_down_layer);
                GGML_ASSERT(
                    ggml_are_same_layout(
                        pc_return_src,
                        phone_return_dst));

                const size_t return_bytes =
                    ggml_nbytes(phone_return_dst);
                auto & return_bufs =
                    backend_ctx->backend_configs[1].
                        prefill_phone_return_stage_bufs;
                const size_t return_slot =
                    static_cast<size_t>(prefill_down_chunk);
                if (return_bufs.size() <= return_slot) {
                    return_bufs.resize(return_slot + 1);
                }
                auto & return_buf = return_bufs[return_slot];
                if (!return_buf ||
                        ggml_backend_buffer_get_size(return_buf.get()) <
                            return_bytes) {
                    return_buf.reset(
                        ggml_backend_alloc_buffer(
                            phone_backend,
                            return_bytes));
                    GGML_ASSERT(return_buf != nullptr);
                }

                phone_return_stage =
                    get_node_aux(phone_return_dst);
                phone_return_stage->buffer =
                    return_buf.get();
                phone_return_stage->data =
                    ggml_backend_buffer_get_base(return_buf.get());

                auto & host_payloads =
                    backend_ctx->prefill_pc_return_host_payloads;
                if (host_payloads.size() <= return_slot) {
                    host_payloads.resize(return_slot + 1);
                }
                auto & host_payload =
                    host_payloads[return_slot];
                if (host_payload == nullptr ||
                        host_payload->size() != return_bytes) {
                    host_payload =
                        std::make_shared<std::vector<uint8_t>>(
                            return_bytes);
                }
                return_payload = host_payload;

                if (pipeline_debug) {
                    printf(
                        "[PHONE_PREFILL_RETURN_STAGE_PREP] "
                        "layer=%d chunk=%d sg=%zu bytes=%zu stage=%p\n",
                        prefill_down_layer,
                        prefill_down_chunk,
                        i,
                        return_bytes,
                        (void *) phone_return_stage);
                }
            }

            if (tensor_expert_load_profile &&
                    prefill_down_chunk == 0) {
                const int64_t pc_ff =
                    tensor_moe_ffn_shard_width(pc_graph);
                const int64_t phone_ff =
                    tensor_moe_ffn_shard_width(phone_graph);
                tensor_expert_shard_by_layer[prefill_down_layer] = {
                    pc_ff,
                    phone_ff,
                };

                const int64_t total_ff =
                    pc_ff >= 0 && phone_ff >= 0 ?
                        pc_ff + phone_ff : -1;
                printf(
                    "[TENSOR_EXPERT_SHARD] "
                    "layer=%d parity=%s pc_ff=%" PRId64
                    " phone_ff=%" PRId64 " total_ff=%" PRId64
                    " pc_pct=%.3f phone_pct=%.3f\n",
                    prefill_down_layer,
                    (prefill_down_layer & 1) != 0 ? "odd" : "even",
                    pc_ff,
                    phone_ff,
                    total_ff,
                    total_ff > 0 ?
                        100.0 * (double) pc_ff / (double) total_ff : 0.0,
                    total_ff > 0 ?
                        100.0 * (double) phone_ff / (double) total_ff : 0.0);
            }

            const uint64_t pc_task =
                backend_ctx->prefill_pc_worker->enqueue(
                    [&, route, pc_graph, pc_backend,
                        pc_return_src, return_payload,
                        prefill_down_layer, prefill_down_chunk, i]
                    (uint64_t task_id) -> ggml_status {
                        const int64_t route_wait_begin_us = ggml_time_us();
                        const ggml_status route_status =
                            route.worker->wait(route.task);
                        const int64_t route_wait_us =
                            ggml_time_us() - route_wait_begin_us;
                        if (route_status != GGML_STATUS_SUCCESS) {
                            return route_status;
                        }

                        const int64_t stage_copy_begin_us =
                            ggml_time_us();
                        ggml_backend_tensor_copy_async(
                            pc_backend,
                            pc_backend,
                            route.stage_hidden,
                            route.dst_hidden);
                        ggml_backend_tensor_copy_async(
                            pc_backend,
                            pc_backend,
                            route.stage_topk,
                            route.dst_topk);
                        ggml_backend_tensor_copy_async(
                            pc_backend,
                            pc_backend,
                            route.stage_weights,
                            route.dst_weights);
                        ggml_backend_synchronize(pc_backend);
                        const int64_t stage_copy_us =
                            ggml_time_us() - stage_copy_begin_us;

                        if (tensor_expert_load_profile &&
                                route.dst_topk != nullptr) {
                            const int64_t expert_read_begin_us =
                                ggml_time_us();

                            const size_t n_topk =
                                ggml_nelements(route.dst_topk);
                            std::vector<int32_t> selected;
                            if (route.dst_topk->type == GGML_TYPE_I32) {
                                selected.resize(n_topk);
                                ggml_backend_tensor_get(
                                    route.dst_topk,
                                    selected.data(),
                                    0,
                                    n_topk * sizeof(int32_t));
                            }

                            const int64_t expert_read_us =
                                ggml_time_us() -
                                expert_read_begin_us;

                            std::vector<uint64_t> chunk_counts;
                            uint64_t valid_assignments = 0;
                            for (int32_t expert : selected) {
                                if (expert < 0) {
                                    continue;
                                }
                                const size_t expert_index =
                                    static_cast<size_t>(expert);
                                if (chunk_counts.size() <= expert_index) {
                                    chunk_counts.resize(
                                        expert_index + 1, 0);
                                }
                                ++chunk_counts[expert_index];
                                ++valid_assignments;
                            }

                            size_t unique_experts = 0;
                            for (uint64_t count : chunk_counts) {
                                unique_experts += count != 0 ? 1 : 0;
                            }
                            const std::string top =
                                tensor_expert_top_string(
                                    chunk_counts, 8);

                            {
                                std::lock_guard<std::mutex> lock(
                                    tensor_expert_load_mutex);
                                auto & layer_load =
                                    tensor_expert_load_by_layer[
                                        prefill_down_layer];
                                layer_load.assignments +=
                                    valid_assignments;
                                layer_load.read_us +=
                                    expert_read_us;
                                if (layer_load.counts.size() <
                                        chunk_counts.size()) {
                                    layer_load.counts.resize(
                                        chunk_counts.size(), 0);
                                }
                                for (size_t expert = 0;
                                     expert < chunk_counts.size();
                                     ++expert) {
                                    layer_load.counts[expert] +=
                                        chunk_counts[expert];
                                }
                            }

                            printf(
                                "[TENSOR_EXPERT_LOAD_CHUNK] "
                                "layer=%d chunk=%d tokens=%" PRId64
                                " topk=%" PRId64
                                " assignments=%" PRIu64
                                " unique=%zu top=%s read_ms=%.3f\n",
                                prefill_down_layer,
                                prefill_down_chunk,
                                route.dst_topk->ne[1],
                                route.dst_topk->ne[0],
                                valid_assignments,
                                unique_experts,
                                top.c_str(),
                                expert_read_us / 1000.0);
                        }

                        const int64_t pc_begin_us = ggml_time_us();
                        const ggml_status status =
                            ggml_backend_graph_compute(
                                pc_backend,
                                pc_graph);
                        const int64_t pc_compute_us =
                            ggml_time_us() - pc_begin_us;

                        int64_t return_stage_us = 0;
                        if (status == GGML_STATUS_SUCCESS &&
                                return_payload != nullptr) {
                            GGML_ASSERT(pc_return_src != nullptr);
                            GGML_ASSERT(
                                return_payload->size() ==
                                ggml_nbytes(pc_return_src));
                            const int64_t return_stage_begin_us =
                                ggml_time_us();
                            ggml_backend_tensor_get(
                                pc_return_src,
                                return_payload->data(),
                                0,
                                return_payload->size());
                            return_stage_us =
                                ggml_time_us() -
                                return_stage_begin_us;
                        }

                        if (pipeline_debug ||
                                tensor_phone_stage_profile) {
                            printf(
                                "[PHONE_PREFILL_PC_BRANCH] "
                                "layer=%d chunk=%d sg=%zu task=%" PRIu64
                                " route_wait_ms=%.3f stage_copy_ms=%.3f "
                                "pc_compute_ms=%.3f return_stage_ms=%.3f "
                                "status=%d\n",
                                prefill_down_layer,
                                prefill_down_chunk,
                                i,
                                task_id,
                                route_wait_us / 1000.0,
                                stage_copy_us / 1000.0,
                                pc_compute_us / 1000.0,
                                return_stage_us / 1000.0,
                                (int) status);
                        }

                        return status;
                    });

            uint64_t return_task = 0;
            size_t return_lane = 0;
            if (phone_prefill_async_return_active) {
                GGML_ASSERT(return_payload != nullptr);
                GGML_ASSERT(phone_return_stage != nullptr);
                const bool use_chunk_join_return =
                    phone_prefill_chunk_join_active;
                return_lane =
                    use_chunk_join_return ?
                        static_cast<size_t>(prefill_down_chunk) %
                            ggml_backend_meta_context::PREFILL_RETURN_LANES :
                        0;
                auto & return_worker_slot =
                    backend_ctx->prefill_return_workers[return_lane];
                if (return_worker_slot == nullptr) {
                    return_worker_slot =
                        new ggml_backend_meta_transfer_worker();
                }

                std::shared_ptr<phone_prefill_lane1_return_gate>
                    lane1_return_gate;
                if (use_chunk_join_return &&
                        return_lane == 1) {
                    auto & gate =
                        phone_prefill_lane1_return_gates[
                            prefill_down_layer];
                    if (gate == nullptr) {
                        gate =
                            std::make_shared<
                                phone_prefill_lane1_return_gate>();
                    }
                    lane1_return_gate = gate;
                }

                ggml_backend_meta_transfer_worker * pc_worker =
                    backend_ctx->prefill_pc_worker;
                ggml_backend_meta_transfer_worker * return_worker =
                    return_worker_slot;
                const ggml_backend_rpc_set_tensor_async_return_t
                    async_return_set =
                        phone_prefill_async_return_set;
                const ggml_backend_rpc_set_tensor_async_return_wait_t
                    async_return_wait =
                        phone_prefill_async_return_wait;
                return_task = return_worker->enqueue(
                    [pc_worker,
                     pc_task,
                     async_return_set,
                     async_return_wait,
                     use_chunk_join_return,
                     return_lane,
                     lane1_return_gate,
                     phone_ffn_seq,
                     phone_backend,
                     phone_return_stage,
                     return_payload,
                     prefill_down_layer,
                     prefill_down_chunk,
                     i,
                     pipeline_debug,
                     tensor_phone_stage_profile]
                    (uint64_t task_id) -> ggml_status {
                        const int64_t wait_begin_us =
                            ggml_time_us();
                        const ggml_status pc_status =
                            pc_worker->wait(pc_task);
                        const int64_t pc_wait_us =
                            ggml_time_us() - wait_begin_us;
                        if (pc_status != GGML_STATUS_SUCCESS) {
                            return pc_status;
                        }

                        int64_t lane_gate_wait_us = 0;
                        if (lane1_return_gate != nullptr) {
                            const int64_t gate_wait_begin_us =
                                ggml_time_us();
                            std::unique_lock<std::mutex> gate_lock(
                                lane1_return_gate->mutex);
                            lane1_return_gate->cv.wait(
                                gate_lock,
                                [&]() {
                                    return lane1_return_gate->open ||
                                        lane1_return_gate->cancelled;
                                });
                            lane_gate_wait_us =
                                ggml_time_us() -
                                gate_wait_begin_us;
                            if (lane1_return_gate->cancelled) {
                                return GGML_STATUS_FAILED;
                            }
                        }

                        const int64_t send_begin_us =
                            ggml_time_us();
                        const bool sent =
                            use_chunk_join_return ?
                                async_return_wait(
                                    phone_backend,
                                    phone_return_stage,
                                    return_payload->data(),
                                    return_payload->size(),
                                    phone_ffn_seq,
                                    return_lane) :
                                async_return_set(
                                    phone_backend,
                                    phone_return_stage,
                                    return_payload->data(),
                                    return_payload->size());
                        const int64_t send_us =
                            ggml_time_us() - send_begin_us;

                        if (pipeline_debug ||
                                tensor_phone_stage_profile) {
                            printf(
                                "[PHONE_PREFILL_RETURN_ASYNC] "
                                "layer=%d chunk=%d sg=%zu task=%" PRIu64
                                " pc_wait_ms=%.3f lane_gate_wait_ms=%.3f "
                                "send_ms=%.3f return_lane=%zu bytes=%zu "
                                "ffn_seq=%" PRIu64
                                " chunk_join=%d status=%d\n",
                                prefill_down_layer,
                                prefill_down_chunk,
                                i,
                                task_id,
                                pc_wait_us / 1000.0,
                                lane_gate_wait_us / 1000.0,
                                send_us / 1000.0,
                                return_lane,
                                return_payload->size(),
                                phone_ffn_seq,
                                use_chunk_join_return ? 1 : 0,
                                sent ? 1 : 0);
                        }

                        return sent ?
                            GGML_STATUS_SUCCESS :
                            GGML_STATUS_FAILED;
                    });
            }

            const bool defer_local_phone_ffn =
                !phone_prefill_producer_route &&
                phone_prefill_defer_phone_ffn &&
                route.phone_stage_hidden != nullptr &&
                route.phone_stage_topk != nullptr &&
                route.phone_stage_weights != nullptr;

            ggml_status phone_status = GGML_STATUS_SUCCESS;
            int64_t phone_wall_us = 0;
            if (defer_local_phone_ffn) {
                pending_phone_prefill_phone_branches.push_back({
                    prefill_down_layer,
                    prefill_down_chunk,
                    i,
                    route.src_hidden,
                    route.src_topk,
                    route.src_weights,
                    route.phone_stage_hidden,
                    route.phone_stage_topk,
                    route.phone_stage_weights,
                });

                if (pipeline_debug) {
                    printf(
                        "[PHONE_PREFILL_PHONE_DEFER] "
                        "layer=%d chunk=%d sg=%zu pending=%zu\n",
                        prefill_down_layer,
                        prefill_down_chunk,
                        i,
                        pending_phone_prefill_phone_branches.size());
                }
            } else {
                const int64_t phone_begin_us = ggml_time_us();
                compute_workers.start(1, i);
                phone_status = compute_workers.wait(1);
                phone_wall_us = ggml_time_us() - phone_begin_us;
                if (phone_status != GGML_STATUS_SUCCESS) {
                    return phone_status;
                }

                if (phone_prefill_chunk_join_active) {
                    GGML_ASSERT(phone_ffn_seq != 0);
                    GGML_ASSERT(phone_prefill_ffn_mark_ready != nullptr);
                    const bool marked =
                        phone_prefill_ffn_mark_ready(
                            phone_backend,
                            phone_ffn_seq);
                    GGML_ASSERT(marked);

                    if (pipeline_debug ||
                            tensor_phone_stage_profile) {
                        printf(
                            "[PHONE_PREFILL_FFN_JOIN_MARK] "
                            "layer=%d chunk=%d sg=%zu seq=%" PRIu64 "\n",
                            prefill_down_layer,
                            prefill_down_chunk,
                            i,
                            phone_ffn_seq);
                    }
                }
            }

            pending_phone_prefill_pc_branches.push_back({
                prefill_down_layer,
                prefill_down_chunk,
                i,
                pc_task,
                return_task,
                return_lane,
                phone_ffn_seq,
                phone_return_stage,
            });
            deferred_phone_prefill_return_sgs[i] =
                prefill_down_layer;

            if (pipeline_debug ||
                    tensor_phone_stage_profile) {
                printf(
                    "[PHONE_PREFILL_PHONE_BRANCH] "
                    "layer=%d chunk=%d sg=%zu route_stage_wait_ms=%.3f "
                    "phone_wall_ms=%.3f pc_task=%" PRIu64
                    " ffn_seq=%" PRIu64 " deferred=%d\n",
                    prefill_down_layer,
                    prefill_down_chunk,
                    i,
                    stage_wait_us / 1000.0,
                    phone_wall_us / 1000.0,
                    pc_task,
                    phone_ffn_seq,
                    defer_local_phone_ffn ? 1 : 0);
            }

            compute_status = phone_status;
        } else {
            compute_workers.start(0, i);

            if (has_async_prefill_input) {
                GGML_ASSERT(backend_ctx->prefill_input_worker != nullptr);
                const int64_t input_wait_start_us = ggml_time_us();
                const ggml_status input_status =
                    backend_ctx->prefill_input_worker->wait(
                        pending_prefill_input_task);
                tensor_wait_us += ggml_time_us() - input_wait_start_us;
                pending_prefill_input_task = 0;
                pending_prefill_input_layer = -1;
                pending_prefill_input_chunk = -1;
                if (input_status != GGML_STATUS_SUCCESS) {
                    compute_workers.wait(0);
                    if (has_pending_prefill_reduce()) {
                        wait_all_prefill_reduces(nullptr);
                    }
                    return input_status;
                }
            } else if (pipeline_debug) {
                GGML_LOG_INFO(
                    "[PREFILL_DOWN_FALLBACK] sg=%zu layer=%d chunk=%d "
                    "input=generic_handoff\n",
                    i, prefill_down_layer, prefill_down_chunk);
            }

            const int64_t chunk_submit_begin_us = ggml_time_us();
            if (pipeline_debug) {
                GGML_LOG_INFO("[PREFILL_CHUNK_SUBMIT_BEGIN] layer=%d chunk=%d t=%" PRId64 "\n", prefill_down_layer,
                       prefill_down_chunk, chunk_submit_begin_us);
            }
            compute_workers.start(1, i);
            const ggml_status pc_status = compute_workers.wait(0);
            const ggml_status phone_status = compute_workers.wait(1);

            if (return_wavefront_graph && phone_status == GGML_STATUS_SUCCESS) {
                if (snapshot_prepares[i].prepared) {
                    return_wave_phone_credit_seqs.push_back(snapshot_prepares[i].seq);
                } else if (pipeline_debug) {
                    // MoE can end a logical prefill-down subgraph with a mirrored
                    // boundary node that is not COMPUTE on Phone. In that case no
                    // RPC snapshot was prepared and communication falls back to
                    // the synchronous/generic reduce path below. There is no
                    // snapshot producer credit to track.
                    GGML_LOG_INFO(
                        "[RETURN_WAVEFRONT_SNAPSHOT_FALLBACK] sg=%zu "
                        "layer=%d chunk=%d reason=no_prepared_snapshot\n",
                        i, prefill_down_layer, prefill_down_chunk);
                }
            }

            const int64_t chunk_submit_end_us = ggml_time_us();
            if (pipeline_debug) {
                GGML_LOG_INFO("[PREFILL_CHUNK_SUBMIT_END] layer=%d chunk=%d t=%" PRId64 " dur=%.3f ms\n",
                       prefill_down_layer, prefill_down_chunk, chunk_submit_end_us,
                       (chunk_submit_end_us - chunk_submit_begin_us) / 1000.0);
            }
            compute_status = pc_status != GGML_STATUS_SUCCESS ? pc_status : phone_status;
        }
    } else if (is_prefill_norm_sg) {
        // Phone-primary prefill owns the residual -> FFN norm -> Router
        // control path.  Do not reuse the legacy PC-only prefill-norm
        // execution rule here: the terminal prefill_ffn_norm_chunk_* is a
        // VIEW of ffn_norm, so executing only backend 0 leaves the Phone view
        // pointing at stale allocator contents even though its logical owner
        // is backend 1.
        const bool phone_primary_prefill_norm =
            n_backends == 2 &&
            layer_is_tensor_phone_primary(prefill_norm_layer);

        const size_t norm_backend = phone_primary_prefill_norm ? 1 : 0;

        ggml_cgraph * norm_graph =
            backend_ctx->backend_configs[norm_backend].
                cgraphs[i].cgraph_main;
        const bool direct_hidden_bound =
            phone_primary_prefill_norm &&
            prefill_norm_chunk == 0 &&
            bind_phone_prefill_hidden_direct(
                prefill_norm_layer);
        uint64_t saved_norm_uid = 0;
        if (direct_hidden_bound && norm_graph != nullptr) {
            saved_norm_uid = norm_graph->uid;
            norm_graph->uid = 0;
            if (pipeline_debug) {
                printf(
                    "[PHONE_PREFILL_DIRECT_HIDDEN] "
                    "layer=%d sg=%zu bytes=%zu\n",
                    prefill_norm_layer,
                    i,
                    ggml_nbytes(
                        find_exact_named_tensor(
                            1,
                            norm_graph->nodes[
                                norm_graph->n_nodes - 1]->name)));
            }
        }

        compute_workers.start(norm_backend, i);
        compute_status = compute_workers.wait(norm_backend);

        if (direct_hidden_bound && norm_graph != nullptr) {
            norm_graph->uid = saved_norm_uid;
        }

        if (pipeline_debug) {
            ggml_cgraph * graph =
                backend_ctx->backend_configs[norm_backend]
                    .cgraphs[i].cgraph_main;
            if (graph != nullptr && graph->n_nodes > 0) {
                printf(
                    "[PREFILL_NORM_EXEC] sg=%zu layer=%d chunk=%d "
                    "backend=%zu primary=%s first=%s last=%s nodes=%d\n",
                    i, prefill_norm_layer, prefill_norm_chunk,
                    norm_backend,
                    phone_primary_prefill_norm ? "PHONE" : "PC",
                    graph->nodes[0]->name,
                    graph->nodes[graph->n_nodes - 1]->name,
                    graph->n_nodes);
            }
        }

    } else if (subgraph_is_prefill_pc_only(i)) {

        compute_workers.start(0, i);
        compute_status = compute_workers.wait(0);

        if (pipeline_debug) {
            ggml_cgraph * graph =
                backend_ctx->backend_configs[0].cgraphs[i].cgraph_main;
            if (graph != nullptr && graph->n_nodes > 0 &&
                    std::strcmp(graph->nodes[0]->name, "l_out-24") == 0) {
                meta_debug_tensor(
                    backend_ctx->backend_configs[0].backend,
                    graph->nodes[0], "CPU L24 IMMEDIATE");
            }

            auto * g =
                backend_ctx->backend_configs[0]
                    .cgraphs[i].cgraph_main;
            printf(
                "[PREFILL_PC_ONLY_SG] sg=%zu "
                "first=%s last=%s nodes=%d\n",
                i,
                g->nodes[0]->name,
                g->nodes[g->n_nodes - 1]->name,
                g->n_nodes);
        }

    } else if (is_decode_pc_only_norm_sg(i) ||
               subgraph_is_decode_pc_only_ffn(i)) {
        // 整个 Attention/tail -> ffn_norm 区域只让 PC 执行。
        compute_workers.start(0, i);
        compute_status = compute_workers.wait(0);

        if (pipeline_debug) {
            auto * g =
                backend_ctx->backend_configs[0]
                    .cgraphs[i].cgraph_main;

            printf(
                "[DECODE_PC_ONLY_SG] sg=%zu "
                "first=%s last=%s nodes=%d\n",
                i,
                g->nodes[0]->name,
                g->nodes[g->n_nodes - 1]->name,
                g->n_nodes);
        }
    } else if (is_phone_only_sg) {
        size_t phone_block_end = i;
        int first_fused_layer = -1;
        phone_block_fused = find_phone_block(
            i, phone_block_end, first_fused_layer, phone_block_last_layer);
        if (phone_block_fused) {
            timing_first_layer = first_fused_layer;
            timing_last_layer  = phone_block_last_layer;
        }
        ggml_cgraph * phone_graph = phone_block_fused ?
            build_phone_block_graph(i, phone_block_end) :
            backend_ctx->backend_configs[1].cgraphs[i].cgraph_main;
        communication_sg = phone_block_fused ? phone_block_end : i;

        int direct_route_chunk = -1;
        int direct_route_layer = -1;
        bool direct_route_bound = false;
        bool direct_route_rebound = false;
        uint64_t saved_phone_graph_uid = 0;
        if (!phone_block_fused &&
                phone_graph != nullptr &&
                phone_graph->n_nodes > 0 &&
                ggml_backend_meta_parse_phone_route_weights(
                    phone_graph->nodes[
                        phone_graph->n_nodes - 1]->name,
                    direct_route_chunk,
                    direct_route_layer)) {
            // Producer-route snapshots the ordinary producer tensors
            // immediately after this graph completes.  It must therefore
            // detect the boundary without rebinding those tensors into the
            // legacy deferred-Phone-FFN staging storage.
            if (phone_prefill_producer_route) {
                direct_route_bound = true;
            } else if (bind_phone_prefill_route_outputs_direct(
                    direct_route_layer,
                    direct_route_chunk)) {
                direct_route_bound = true;
                direct_route_rebound = true;
                saved_phone_graph_uid = phone_graph->uid;
                phone_graph->uid = 0;
            }
        }

        bool producer_route_prearmed = false;
        if (direct_route_bound) {
            producer_route_prearmed =
                prearm_phone_prefill_route(
                    i,
                    direct_route_layer,
                    direct_route_chunk);
        }

        const int64_t phone_submit_begin_us = ggml_time_us();
        if (pipeline_debug && phone_block_fused) {
            printf("[PHONE_BLOCK_SUBMIT_BEGIN] layers=%d..%d t=%" PRId64 "\n", first_fused_layer,
                   phone_block_last_layer, phone_submit_begin_us);
        }
        const int64_t phone_worker_submit_begin_us = ggml_time_us();
        compute_workers.start_graph(1, phone_graph);
        const int64_t phone_worker_submit_us =
            ggml_time_us() - phone_worker_submit_begin_us;
        const int64_t phone_worker_wait_begin_us = ggml_time_us();
        compute_status = compute_workers.wait(1);
        const int64_t phone_worker_wait_us =
            ggml_time_us() - phone_worker_wait_begin_us;
        if (phone_exit_profile) {
            printf(
                "[PHONE_ASYNC_META_WORKER] sg=%zu fused=%d layers=%d..%d "
                "nodes=%d submit_ms=%.3f wait_ms=%.3f status=%d\n",
                i,
                phone_block_fused ? 1 : 0,
                first_fused_layer,
                phone_block_last_layer,
                phone_graph != nullptr ? phone_graph->n_nodes : 0,
                phone_worker_submit_us / 1000.0,
                phone_worker_wait_us / 1000.0,
                (int) compute_status);
        }

        if (direct_route_bound) {
            if (direct_route_rebound) {
                phone_graph->uid = saved_phone_graph_uid;
            }
            ggml_backend_t phone_backend =
                backend_ctx->backend_configs[1].backend;

            uint64_t seq = 0;
            bool published = false;
            int64_t snapshot_us = 0;
            if (producer_route_prearmed) {
                const auto route_it =
                    pending_phone_prefill_routes.find(
                        { direct_route_layer,
                          direct_route_chunk });
                GGML_ASSERT(
                    route_it !=
                    pending_phone_prefill_routes.end());
                const phone_prefill_route_task & route =
                    route_it->second;
                seq = route.producer_seq;
                GGML_ASSERT(seq != 0);

                const ggml_backend_rpc_route_snapshot_ready_t
                    snapshot_ready =
                        ggml_backend_meta_get_route_snapshot_ready(
                            phone_backend);
                GGML_ASSERT(snapshot_ready != nullptr);

                const int64_t snapshot_begin_us =
                    ggml_time_us();
                published = snapshot_ready(
                    phone_backend,
                    seq,
                    static_cast<uint32_t>(route.lane),
                    route.src_hidden,
                    route.src_topk,
                    route.src_weights);
                snapshot_us =
                    ggml_time_us() - snapshot_begin_us;
            } else {
                const ggml_backend_rpc_route_mark_ready_t mark_ready =
                    ggml_backend_meta_get_route_mark_ready(
                        phone_backend);
                GGML_ASSERT(mark_ready != nullptr);

                seq =
                    backend_ctx->next_phone_prefill_route_seq++;
                phone_prefill_route_producer_seq[
                    { direct_route_layer,
                      direct_route_chunk }] = seq;
                published = mark_ready(phone_backend, seq);
            }
            GGML_ASSERT(published);

            if (pipeline_debug ||
                    tensor_phone_stage_profile) {
                printf(
                    "[PHONE_PREFILL_ROUTE_PRODUCER_SEQ] "
                    "layer=%d chunk=%d seq=%" PRIu64
                    " mailbox=%d snapshot_ms=%.3f\n",
                    direct_route_layer,
                    direct_route_chunk,
                    seq,
                    producer_route_prearmed ? 1 : 0,
                    snapshot_us / 1000.0);
            }
        }

        // Diagnostic-only shadow execution for the first Phone-primary Tensor
        // layer's Attention subgraph.  The real path still consumes the Phone
        // result.  We run the PC counterpart after Phone completes, compare
        // the final Attention tensor, then restore the original COMPUTE flags
        // before communication/ownership logic continues.
        if (compute_status == GGML_STATUS_SUCCESS &&
                std::getenv("GGML_META_TP_ATTN_SHADOW_PC") != nullptr &&
                !phone_block_fused &&
                n_backends == 2 &&
                backend_ctx->tensor_phone_first_layer >= 0 &&
                phone_graph != nullptr &&
                phone_graph->n_nodes > 0) {
            int shadow_layer = -1;
            int shadow_parsed = 0;
            const char * first_name = phone_graph->nodes[0]->name;
            const bool is_first_tp_attn =
                std::sscanf(
                    first_name, "norm-%d%n",
                    &shadow_layer, &shadow_parsed) == 1 &&
                first_name[shadow_parsed] == '\0' &&
                shadow_layer == backend_ctx->tensor_phone_first_layer;

            if (is_first_tp_attn) {
                ggml_cgraph * pc_graph =
                    backend_ctx->backend_configs[0].cgraphs[i].cgraph_main;

                if (pc_graph != nullptr &&
                        pc_graph->n_nodes == phone_graph->n_nodes) {
                    std::vector<int32_t> saved_flags;
                    saved_flags.reserve((size_t) pc_graph->n_nodes);
                    for (int k = 0; k < pc_graph->n_nodes; ++k) {
                        saved_flags.push_back(pc_graph->nodes[k]->flags);
                        pc_graph->nodes[k]->flags |= GGML_TENSOR_FLAG_COMPUTE;
                    }

                    compute_workers.start_graph(0, pc_graph);
                    const ggml_status shadow_status = compute_workers.wait(0);

                    for (int k = 0; k < pc_graph->n_nodes; ++k) {
                        pc_graph->nodes[k]->flags = saved_flags[(size_t) k];
                    }

                    ggml_tensor * pc_last =
                        pc_graph->nodes[pc_graph->n_nodes - 1];
                    ggml_tensor * phone_last =
                        phone_graph->nodes[phone_graph->n_nodes - 1];

                    auto shadow_fence = [&](ggml_backend_t backend) {
                        const ggml_backend_rpc_fence_t rpc_fence =
                            ggml_backend_meta_get_rpc_fence(backend);
                        if (rpc_fence != nullptr) {
                            rpc_fence(backend);
                        } else {
                            ggml_backend_synchronize(backend);
                        }
                    };

                    if (shadow_status == GGML_STATUS_SUCCESS &&
                            pc_last != nullptr &&
                            phone_last != nullptr &&
                            pc_last->type == GGML_TYPE_F32 &&
                            phone_last->type == GGML_TYPE_F32 &&
                            ggml_nelements(pc_last) ==
                                ggml_nelements(phone_last)) {
                        shadow_fence(
                            backend_ctx->backend_configs[0].backend);
                        shadow_fence(
                            backend_ctx->backend_configs[1].backend);

                        const size_t n = ggml_nelements(pc_last);
                        std::vector<float> pc_values(n);
                        std::vector<float> phone_values(n);
                        ggml_backend_tensor_get(
                            pc_last, pc_values.data(), 0,
                            n * sizeof(float));
                        ggml_backend_tensor_get(
                            phone_last, phone_values.data(), 0,
                            n * sizeof(float));

                        double pc_sum = 0.0;
                        double phone_sum = 0.0;
                        double pc_l2 = 0.0;
                        double phone_l2 = 0.0;
                        double diff_l2 = 0.0;
                        double max_abs_diff = 0.0;
                        size_t max_diff_index = 0;
                        for (size_t q = 0; q < n; ++q) {
                            const double a = pc_values[q];
                            const double b = phone_values[q];
                            const double d = a - b;
                            pc_sum += a;
                            phone_sum += b;
                            pc_l2 += a * a;
                            phone_l2 += b * b;
                            diff_l2 += d * d;
                            if (std::abs(d) > max_abs_diff) {
                                max_abs_diff = std::abs(d);
                                max_diff_index = q;
                            }
                        }

                        printf(
                            "[TP_ATTN_SHADOW] layer=%d sg=%zu "
                            "pc_last=%s phone_last=%s n=%zu "
                            "pc_sum=%.9f phone_sum=%.9f "
                            "pc_l2=%.9f phone_l2=%.9f "
                            "max_abs_diff=%.9g rms_diff=%.9g "
                            "max_diff_index=%zu "
                            "pc_v0=%.9f phone_v0=%.9f "
                            "pc_v1=%.9f phone_v1=%.9f\n",
                            shadow_layer, i,
                            pc_last->name, phone_last->name, n,
                            pc_sum, phone_sum,
                            std::sqrt(pc_l2), std::sqrt(phone_l2),
                            max_abs_diff,
                            n > 0 ?
                                std::sqrt(diff_l2 / (double) n) :
                                0.0,
                            max_diff_index,
                            n > 0 ? pc_values[0] : 0.0f,
                            n > 0 ? phone_values[0] : 0.0f,
                            n > 1 ? pc_values[1] : 0.0f,
                            n > 1 ? phone_values[1] : 0.0f);
                    } else {
                        printf(
                            "[TP_ATTN_SHADOW] layer=%d sg=%zu status=SKIP "
                            "shadow_status=%d pc_last=%s phone_last=%s "
                            "pc_type=%d phone_type=%d pc_n=%" PRId64
                            " phone_n=%" PRId64 "\n",
                            shadow_layer, i, (int) shadow_status,
                            pc_last != nullptr ? pc_last->name : "(null)",
                            phone_last != nullptr ? phone_last->name : "(null)",
                            pc_last != nullptr ? (int) pc_last->type : -1,
                            phone_last != nullptr ? (int) phone_last->type : -1,
                            pc_last != nullptr ? ggml_nelements(pc_last) : 0,
                            phone_last != nullptr ? ggml_nelements(phone_last) : 0);
                    }
                } else {
                    printf(
                        "[TP_ATTN_SHADOW] layer=%d sg=%zu status=SKIP "
                        "reason=graph-shape pc_nodes=%d phone_nodes=%d\n",
                        shadow_layer, i,
                        pc_graph != nullptr ? pc_graph->n_nodes : -1,
                        phone_graph->n_nodes);
                }
            }
        }

        const int64_t phone_submit_end_us = ggml_time_us();
        if (pipeline_debug && phone_block_fused) {
            printf("[PHONE_BLOCK_SUBMIT_END] layers=%d..%d t=%" PRId64 " dur=%.3f ms\n", first_fused_layer,
                   phone_block_last_layer, phone_submit_end_us,
                   (phone_submit_end_us - phone_submit_begin_us) / 1000.0);
        }

        if (pipeline_debug) {
            if (phone_block_fused) {
                printf(
                    "[PHONE_BLOCK_FUSED] sg=%zu..%zu layers=%d..%d nodes=%d\n",
                    i, phone_block_end, first_fused_layer,
                    phone_block_last_layer, phone_graph->n_nodes);
            } else {
                printf(
                    "[PHONE_ONLY_SG] sg=%zu first=%s last=%s nodes=%d\n",
                    i, phone_graph->nodes[0]->name,
                    phone_graph->nodes[phone_graph->n_nodes - 1]->name,
                    phone_graph->n_nodes);
            }
        }
    } else {
        compute_status = compute_workers.compute(i);
    }

    subgraph_compute_wall_us = ggml_time_us() - compute_start_us;
    compute_wall_us += subgraph_compute_wall_us;

    if (return_wavefront_graph) {
        int wave_profile_layer = -1;
        if (is_prefill_wave_attn_sg) {
            wave_profile_layer = prefill_wave_attn_layer;
        } else if (is_prefill_down_sg) {
            wave_profile_layer = prefill_down_layer;
        } else if (is_prefill_norm_sg) {
            wave_profile_layer = prefill_norm_layer;
        } else {
            wave_profile_layer = subgraph_layer(i);
        }
        if (wave_profile_layer >= return_wavefront_first_layer) {
            return_wave_layer_compute_wall_us[wave_profile_layer] += subgraph_compute_wall_us;
        }
    }

    if (trace_prefill_chunk) {
        const int64_t next_end_us = ggml_time_us();
        GGML_LOG_INFO("[PREFILL_NEXT_END] layer=%d chunk=%d t=%" PRId64 " dur=%.3f ms\n", next_layer, next_chunk,
               next_end_us, (next_end_us - next_begin_us) / 1000.0);
    }

    if (compute_status != GGML_STATUS_SUCCESS) {
        if (has_pending_prefill_reduce()) {
            wait_all_prefill_reduces(nullptr);
        }
        return compute_status;
    }

    // Complete deferred TP input traces only for sources that are produced by
    // this subgraph itself. This distinguishes a real zero from a PRE trace
    // that simply inspected the destination before its ADD/other producer ran.
    if ((std::getenv("GGML_META_TP_INPUT_TRACE") != nullptr ||
         std::getenv("GGML_META_TP_FFN_NUMERIC_TRACE") != nullptr) &&
            backend_ctx->tensor_phone_first_layer >= 0) {
        for (size_t backend = 0; backend < n_backends; ++backend) {
            for (int layer = backend_ctx->tensor_phone_first_layer;
                 layer < backend_ctx->tensor_phone_last_layer; ++layer) {
                debug_layer_input(backend, communication_sg, layer, true);
            }
        }
    }

    // Separate the synchronization effect from the tensor reads in the
    // attention trace. This gate is diagnostic and leaves normal execution
    // unchanged until the source of the decode corruption is confirmed.
    if (std::getenv("GGML_META_TP_ATTN_FENCE") != nullptr &&
            n_backends == 2 && is_phone_only_sg &&
            !phone_block_fused) {
        ggml_cgraph * phone_graph =
            backend_ctx->backend_configs[1].cgraphs[i].cgraph_main;
        if (phone_graph != nullptr && phone_graph->n_nodes > 0 &&
                phone_graph->nodes[0]->ne[1] == 1 &&
                std::strcmp(phone_graph->nodes[0]->name, "l_out-30") == 0) {
            ggml_backend_t phone_backend = backend_ctx->backend_configs[1].backend;
            const ggml_backend_rpc_fence_t rpc_fence =
                ggml_backend_meta_get_rpc_fence(phone_backend);
            if (rpc_fence != nullptr) {
                rpc_fence(phone_backend);
            } else {
                ggml_backend_synchronize(phone_backend);
            }
            printf("[TP_ATTN_FENCE] sg=%zu rpc_fence=%d\n", i,
                   (int) (rpc_fence != nullptr));
        }
    }

    // Inspect the first failing decode attention block before its boundary
    // tensor is copied to the other backend.  The regular TP input trace
    // observes ffn_inp-31 only after that copy, when both copies are NaN.
    if (std::getenv("GGML_META_TP_ATTN_TRACE") != nullptr && n_backends == 2) {
        for (size_t backend = 0; backend < n_backends; ++backend) {
            auto & bcj = backend_ctx->backend_configs[backend];
            ggml_cgraph * graph = bcj.cgraphs[communication_sg].cgraph_main;
            if (graph == nullptr || graph->n_nodes == 0 ||
                    graph->nodes[0]->ne[1] != 1 ||
                    std::strcmp(graph->nodes[0]->name, "l_out-30") != 0) {
                continue;
            }
            printf("[TP_ATTN_TRACE] sg=%zu backend=%zu nodes=%d last=%s\n",
                   communication_sg, backend, graph->n_nodes,
                   graph->nodes[graph->n_nodes - 1]->name);
            for (int k = 0; k < graph->n_nodes; ++k) {
                ggml_tensor * node = graph->nodes[k];
                if (!(node->flags & GGML_TENSOR_FLAG_COMPUTE) ||
                        node->type != GGML_TYPE_F32 ||
                        (std::strstr(node->name, "-31") == nullptr &&
                         k != graph->n_nodes - 1)) {
                    continue;
                }
                char tag[96];
                std::snprintf(tag, sizeof(tag), "TP attn31 sg%zu backend%zu", communication_sg, backend);
                meta_debug_tensor(bcj.backend, node, tag);
            }
        }
    }
} else {
    compute_complete = false;
}

        if (pipeline_debug && communication_sg == 0) {
            auto & bcj = backend_ctx->backend_configs[0];
            ggml_cgraph * graph = bcj.cgraphs[communication_sg].cgraph_main;
            ggml_tensor * node = graph->nodes[graph->n_nodes - 1];
            if (std::strcmp(node->name, "attn_out-0") == 0) {
                meta_debug_tensor(bcj.backend, node, "A attn_out-0 backend0");
            }
        }

        int terminal_tp_layer = -1;
        const bool terminal_phone_exit =
            communication_sg + 1 >= backend_ctx->n_subgraphs &&
            subgraph_is_phone_only(communication_sg) &&
            !subgraph_tensor_phone_primary_layer(
                communication_sg, terminal_tp_layer);
        const bool force_phone_block_exit =
            phone_block_fused &&
            phone_block_last_layer >= 0 &&
            !layer_is_tensor_phone_primary(
                phone_block_last_layer);
        int64_t subgraph_comm_wall_us = 0;
        int64_t subgraph_specialized_us = 0;
        int64_t subgraph_comm_allreduce_us = 0;
        int64_t subgraph_fallback_us = 0;
        bool subgraph_specialized_handled = false;
        bool subgraph_comm_attempted = false;
        bool subgraph_comm_handled = false;
        bool subgraph_fallback_used = false;
        if (n_backends > 1 &&
                (communication_sg < backend_ctx->n_subgraphs - 1 || terminal_phone_exit)) {
            const int64_t reduce_start_us = ggml_time_us();
            bool communication_complete = false;

            const int64_t specialized_start_us = ggml_time_us();
            const ggml_status specialized_status =
                specialized_communication(
                    communication_sg, communication_complete, compute_complete,
                    force_phone_block_exit, phone_block_last_layer);
            subgraph_specialized_us =
                ggml_time_us() - specialized_start_us;
            subgraph_specialized_handled = communication_complete;
            if (specialized_status != GGML_STATUS_SUCCESS) {
                return specialized_status;
            }

            if (!communication_complete && backend_ctx->comm_ctx) {
                subgraph_comm_attempted = true;
                ++reduce_comm_count;
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[communication_sg].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                const int64_t comm_start_us = ggml_time_us();
                communication_complete = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
                subgraph_comm_allreduce_us =
                    ggml_time_us() - comm_start_us;
                subgraph_comm_handled = communication_complete;
                reduce_comm_us += subgraph_comm_allreduce_us;
            }

            if (!communication_complete) {
                subgraph_fallback_used = true;
                ++reduce_fallback_count;
                const int64_t fallback_start_us = ggml_time_us();
                const ggml_status status = allreduce_fallback(communication_sg);
                subgraph_fallback_us =
                    ggml_time_us() - fallback_start_us;
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            ++reduce_count;
            const int64_t reduce_us = ggml_time_us() - reduce_start_us;
            subgraph_comm_wall_us = reduce_us;
            reduce_wall_us += reduce_us;
            reduce_max_us = std::max(reduce_max_us, reduce_us);

            if (meta_sg_timing) {
                const int64_t subgraph_comm_unaccounted_us =
                    std::max<int64_t>(
                        0,
                        subgraph_comm_wall_us -
                            subgraph_specialized_us -
                            subgraph_comm_allreduce_us -
                            subgraph_fallback_us);
                printf(
                    "[META_SG_COMM_TIMING] sg=%zu "
                    "total_ms=%.3f specialized_ms=%.3f "
                    "specialized_handled=%d "
                    "comm_allreduce_ms=%.3f comm_attempted=%d "
                    "comm_handled=%d fallback_ms=%.3f "
                    "fallback_used=%d unaccounted_ms=%.3f\n",
                    communication_sg,
                    subgraph_comm_wall_us / 1000.0,
                    subgraph_specialized_us / 1000.0,
                    subgraph_specialized_handled ? 1 : 0,
                    subgraph_comm_allreduce_us / 1000.0,
                    subgraph_comm_attempted ? 1 : 0,
                    subgraph_comm_handled ? 1 : 0,
                    subgraph_fallback_us / 1000.0,
                    subgraph_fallback_used ? 1 : 0,
                    subgraph_comm_unaccounted_us / 1000.0);
            }
        }

        if (meta_sg_timing) {
            const int64_t subgraph_total_wall_us =
                ggml_time_us() - layer_wall_start_us;
            const int64_t subgraph_other_wall_us =
                std::max<int64_t>(
                    0,
                    subgraph_total_wall_us -
                        subgraph_compute_wall_us -
                        subgraph_comm_wall_us);
            printf(
                "[META_SG_TIMING] sg=%zu comm_sg=%zu "
                "layer=%d..%d fused=%d "
                "total_ms=%.3f compute_ms=%.3f comm_ms=%.3f "
                "other_ms=%.3f\n",
                timing_sg,
                communication_sg,
                timing_first_layer,
                timing_last_layer,
                phone_block_fused ? 1 : 0,
                subgraph_total_wall_us / 1000.0,
                subgraph_compute_wall_us / 1000.0,
                subgraph_comm_wall_us / 1000.0,
                subgraph_other_wall_us / 1000.0);
        }

        if (phone_block_fused) {
            i = communication_sg;
        }

        if (timing_first_layer >= 0) {
            const auto backend_times_after = backend_times_snapshot();
            auto & timing = layer_timings[{ timing_first_layer, timing_last_layer }];
            if (!backend_times_after.empty()) {
                timing.compute_pc_us += backend_times_after[0] - backend_times_before[0];
            }
            if (backend_times_after.size() > 1) {
                timing.compute_phone_us += backend_times_after[1] - backend_times_before[1];
            }
            if (n_backends > 1) {
                timing.copy_0to1_us += copy_time_snapshot(0, 1) - copy_0to1_before;
                timing.copy_1to0_us += copy_time_snapshot(1, 0) - copy_1to0_before;
            }
            timing.total_us += ggml_time_us() - layer_wall_start_us;
        }

        const auto tensor_backend_times_after = backend_times_snapshot();
        const int64_t pc_compute_us = !tensor_backend_times_after.empty() ?
            tensor_backend_times_after[0] - backend_times_before[0] : 0;
        const int64_t phone_compute_us = tensor_backend_times_after.size() > 1 ?
            tensor_backend_times_after[1] - backend_times_before[1] : 0;

        if (tensor_phone_stage_profile) {
            int stage_layer = -1;
            int stage_chunk = -1;
            bool stage_decode = false;
            const bool stage_is_route =
                tensor_phone_stage_route_identity(
                    communication_sg, stage_layer, stage_chunk);
            const bool stage_is_ffn =
                !stage_is_route &&
                tensor_phone_stage_ffn_identity(
                    communication_sg,
                    stage_layer,
                    stage_chunk,
                    stage_decode);

            if (stage_is_route) {
                stage_decode = stage_chunk < 0;
                auto & stage = tensor_phone_stage_entry(
                    stage_layer, stage_chunk, stage_decode);
                stage.pre_route_wall_us += subgraph_compute_wall_us;
                stage.pre_route_pc_worker_us += pc_compute_us;
                stage.pre_route_phone_worker_us += phone_compute_us;
                stage.route_compute_count += 1;

                printf(
                    "[TENSOR_PHONE_STAGE_SG] stage=pre_route_compute "
                    "mode=%s layer=%d chunk=%d sg=%zu wall_ms=%.3f "
                    "pc_worker_ms=%.3f phone_worker_ms=%.3f\n",
                    stage_decode ? "decode" : "prefill",
                    stage_layer,
                    stage_chunk < 0 ? 0 : stage_chunk,
                    communication_sg,
                    subgraph_compute_wall_us / 1000.0,
                    pc_compute_us / 1000.0,
                    phone_compute_us / 1000.0);
            } else if (stage_is_ffn) {
                auto & stage = tensor_phone_stage_entry(
                    stage_layer, stage_chunk, stage_decode);
                stage.ffn_wall_us += subgraph_compute_wall_us;
                stage.pc_ffn_worker_us += pc_compute_us;
                stage.phone_ffn_worker_us += phone_compute_us;
                stage.ffn_compute_count += 1;

                printf(
                    "[TENSOR_PHONE_STAGE_SG] stage=ffn_parallel_compute "
                    "mode=%s layer=%d chunk=%d sg=%zu wall_ms=%.3f "
                    "pc_worker_ms=%.3f phone_worker_ms=%.3f\n",
                    stage_decode ? "decode" : "prefill",
                    stage_layer,
                    stage_chunk < 0 ? 0 : stage_chunk,
                    communication_sg,
                    subgraph_compute_wall_us / 1000.0,
                    pc_compute_us / 1000.0,
                    phone_compute_us / 1000.0);
            }
        }

        if (is_prefill_down_sg) {
            tensor_pc_ffn_us += pc_compute_us;
            tensor_phone_us  += phone_compute_us;
        } else if (is_prefill_wave_attn_sg || is_prefill_norm_sg || subgraph_is_prefill_pc_only(i)) {
            // Return-wavefront Attention is its own subgraph.  It must be
            // included here or attn_ms only sees the tiny norm/PC-only pieces.
            tensor_attn_us += pc_compute_us;
        }
    }

    while (!pending_phone_prefill_pc_branches.empty()) {
        const int pending_layer =
            pending_phone_prefill_pc_branches.front().layer;
        const ggml_status drain_status =
            drain_phone_prefill_pc_layer(pending_layer);
        if (drain_status != GGML_STATUS_SUCCESS) {
            return drain_status;
        }
    }

    for (size_t lane = 0;
         lane < ggml_backend_meta_context::PREFILL_ROUTE_LANES;
         ++lane) {
        if (pending_phone_prefill_route_lane_task[lane] == 0) {
            continue;
        }
        auto * worker = backend_ctx->prefill_route_workers[lane];
        GGML_ASSERT(worker != nullptr);
        const ggml_status route_status =
            worker->wait(
                pending_phone_prefill_route_lane_task[lane]);
        if (route_status != GGML_STATUS_SUCCESS) {
            return route_status;
        }
        pending_phone_prefill_route_lane_task[lane] = 0;
    }

    debug_handoff_watch("AFTER_META_LOOP");

    if (pipeline_debug && debug_terminal_handoff_dst != nullptr) {
        printf(
            "[HANDOFF_PENDING] prefill_reduce={%" PRIu64 ",%" PRIu64 "}"
            " prefill_input=%" PRIu64 "\n",
            pending_prefill_reduce_task[0], pending_prefill_reduce_task[1], pending_prefill_input_task);
    }

    if (has_pending_prefill_reduce()) {
        debug_handoff_watch("BEFORE_PENDING_REDUCE_WAIT");

        const ggml_status status = wait_all_prefill_reduces(nullptr);

        debug_handoff_watch("AFTER_PENDING_REDUCE_WAIT");

        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }

    if (return_wavefront_graph && return_wave_snapshot_ready_waiter != nullptr) {
        // Full returns are drained now, so any remaining producer credits are
        // already READY. Consume their notifications to avoid carrying stale
        // seq entries into a later graph execution.
        while (!return_wave_phone_credit_seqs.empty()) {
            const uint64_t seq = return_wave_phone_credit_seqs.front();
            const bool ready = return_wave_snapshot_ready_waiter(
                backend_ctx->backend_configs[1].backend,
                seq);
            GGML_ASSERT(ready);
            return_wave_phone_credit_seqs.pop_front();
        }
    }

    const int64_t meta_execute_end_us = ggml_time_us();
    const int64_t meta_graph_timing_end_us = meta_execute_end_us;
    const int64_t meta_graph_total_all_us =
        meta_graph_timing_end_us - meta_graph_start_us;
    const int64_t meta_execute_us =
        meta_execute_end_us - meta_execute_begin_us;
    const int64_t meta_other_all_us = std::max<int64_t>(
        0, meta_graph_total_all_us - meta_rebuild_us - meta_execute_us);

    if (meta_timing_debug) {
        printf(
            "[META_GRAPH_TIMING] uid=%" PRIu64
            " nodes=%d subgraphs=%zu needs_rebuild=%d "
            "total_ms=%.3f rebuild_ms=%.3f execute_ms=%.3f other_ms=%.3f\n",
            cgraph->uid,
            cgraph->n_nodes,
            backend_ctx->n_subgraphs,
            needs_rebuild ? 1 : 0,
            meta_graph_total_all_us / 1000.0,
            meta_rebuild_us / 1000.0,
            meta_execute_us / 1000.0,
            meta_other_all_us / 1000.0);
    }

    if (return_wavefront_graph) {
        printf(
            "[RETURN_WAVEFRONT_SUM] enabled=1 dependency_wait_count=%" PRId64
            " dependency_wait_ms=%.3f dependency_wait_max_ms=%.3f"
            " overlap_boundaries=%" PRId64
            " ahead_attn=%" PRId64
            " ahead_phone_submit=%" PRId64
            " phone_credit_wait_count=%" PRId64
            " phone_credit_wait_ms=%.3f phone_credit_wait_max_ms=%.3f"
            " old_layer_wait_count=%" PRId64
            " old_layer_wait_ms=%.3f"
            " lane_reuse_wait_count=%" PRId64
            " lane_reuse_wait_ms=%.3f lane_reuse_wait_max_ms=%.3f"
            " legacy_layer_barrier_count=%" PRId64
            " legacy_layer_barrier_wait_ms=%.3f legacy_layer_barrier_wait_max_ms=%.3f\n",
            return_wave_dependency_wait_count,
            return_wave_dependency_wait_us / 1000.0,
            return_wave_dependency_wait_max_us / 1000.0,
            return_wave_overlap_boundaries,
            return_wave_ahead_attn_chunks,
            return_wave_ahead_phone_submits,
            return_wave_phone_credit_wait_count,
            return_wave_phone_credit_wait_us / 1000.0,
            return_wave_phone_credit_wait_max_us / 1000.0,
            return_wave_old_layer_wait_count,
            return_wave_old_layer_wait_us / 1000.0,
            lane_reuse_wait_count,
            lane_reuse_wait_us / 1000.0,
            lane_reuse_wait_max_us / 1000.0,
            layer_barrier_wait_count,
            layer_barrier_wait_us / 1000.0,
            layer_barrier_wait_max_us / 1000.0);
    }

    debug_handoff_watch("BEFORE_META_RETURN");

    if (pipeline_debug) {
        std::string backend_times;
        for (size_t j = 0; j < n_backends; ++j) {
            backend_times += (j == 0 ? "" : ",") + std::to_string(j) + ":" +
                std::to_string(compute_workers.backend_time_us[j] / 1000.0);
        }
        const double reduce_avg_us = reduce_count > 0 ? double(reduce_wall_us) / reduce_count : 0.0;
        const int64_t reduce_profiled_us = reduce_copy_wait_us + reduce_add_us + reduce_zero_us + reduce_comm_us;
        const int64_t reduce_other_us = std::max<int64_t>(0, reduce_wall_us - reduce_profiled_us);
        printf("[META_PIPELINE] uid=%" PRIu64 " subgraphs=%zu compute=%.3f ms backends_ms={%s} "
                "reduce_total=%.3f ms count=%zu avg=%.3f ms max=%.3f ms comm=%.3f ms(%zu) "
                "direct=%zu reduce_to_primary=%zu fallback=%zu copy_wait=%.3f ms zero_copy_skips=%zu "
                "add=%.3f ms zero=%.3f ms other=%.3f ms\n",
                cgraph->uid, backend_ctx->n_subgraphs, compute_wall_us / 1000.0,
                backend_times.c_str(), reduce_wall_us / 1000.0, reduce_count, reduce_avg_us / 1000.0,
                reduce_max_us / 1000.0, reduce_comm_us / 1000.0, reduce_comm_count,
                direct_copy_count, reduce_to_primary_count, reduce_fallback_count,
                reduce_copy_wait_us / 1000.0, reduce_zero_copy_skips, reduce_add_us / 1000.0,
                reduce_zero_us / 1000.0, reduce_other_us / 1000.0);
        for (size_t j_src = 0; j_src < n_backends; ++j_src) {
            for (size_t j_dst = 0; j_dst < n_backends; ++j_dst) {
                const auto & stats = reduce_copy_by_direction[j_src*n_backends + j_dst];
                if (stats.count == 0) {
                    continue;
                }
                printf("[META_COPY_SUM] %zu->%zu count=%zu total=%.3f ms avg=%.3f ms max=%.3f ms bytes=%.3f MiB\n",
                       j_src, j_dst, stats.count, stats.total_us / 1000.0,
                       (double(stats.total_us) / stats.count) / 1000.0, stats.max_us / 1000.0,
                       stats.bytes / (1024.0 * 1024.0));
            }
        }
        for (const auto & entry : layer_timings) {
            const int first_layer = entry.first.first;
            const int last_layer  = entry.first.second;
            const auto & timing   = entry.second;
            printf("[META_LAYER_SUM] layer=%d", first_layer);
            if (last_layer != first_layer) {
                printf("..%d", last_layer);
            }
            printf(" compute_pc=%.3f compute_phone=%.3f copy_0to1=%.3f copy_1to0=%.3f total=%.3f ms\n",
                   timing.compute_pc_us / 1000.0, timing.compute_phone_us / 1000.0,
                   timing.copy_0to1_us / 1000.0, timing.copy_1to0_us / 1000.0, timing.total_us / 1000.0);
        }
        if (pipeline_gap_count > 0) {
            printf("[META_PIPELINE_GAP_SUM] count=%zu submit_avg=%.3f ms gap_avg=%.3f ms max=%.3f ms\n",
                   pipeline_gap_count,
                   pipeline_submit_sum_us / 1000.0 / pipeline_gap_count,
                   pipeline_gap_sum_us / 1000.0 / pipeline_gap_count,
                   pipeline_gap_max_us / 1000.0);
        }
    }


    const int64_t h2d_us = n_backends > 1 ? reduce_copy_by_direction[1].total_us : 0;
    const int64_t d2h_us = n_backends > 1 ? reduce_copy_by_direction[n_backends].total_us : 0;
    const int64_t tensor_reduce_us = reduce_add_us + reduce_zero_us + reduce_comm_us;

    ggml_backend_rpc_snapshot_stats rpc_snapshot_stats_end {};
    int64_t return_transfer_count = 0;
    int64_t return_payload_bytes = 0;
    int64_t return_request_us = 0;
    int64_t return_ready_wait_us = 0;
    int64_t return_recv_payload_us = 0;
    int64_t return_rpc_total_us = 0;
    if (have_rpc_snapshot_stats &&
        rpc_snapshot_stats_getter(
            backend_ctx->backend_configs[1].backend,
            &rpc_snapshot_stats_end)) {
        return_transfer_count = (int64_t) (
            rpc_snapshot_stats_end.transfer_count -
            rpc_snapshot_stats_begin.transfer_count);
        return_payload_bytes = (int64_t) (
            rpc_snapshot_stats_end.payload_bytes -
            rpc_snapshot_stats_begin.payload_bytes);
        return_request_us =
            rpc_snapshot_stats_end.request_us -
            rpc_snapshot_stats_begin.request_us;
        return_ready_wait_us =
            rpc_snapshot_stats_end.ready_first_byte_us -
            rpc_snapshot_stats_begin.ready_first_byte_us;
        return_recv_payload_us =
            rpc_snapshot_stats_end.recv_payload_us -
            rpc_snapshot_stats_begin.recv_payload_us;
        return_rpc_total_us =
            rpc_snapshot_stats_end.total_us -
            rpc_snapshot_stats_begin.total_us;
    }

    const int64_t explicit_return_wait_us =
        lane_reuse_wait_us + layer_barrier_wait_us;
    const int64_t return_overlap_est_us =
        std::max<int64_t>(
            0, return_rpc_total_us - explicit_return_wait_us);
    const double return_overlap_ratio =
        return_rpc_total_us > 0 ?
            (double) return_overlap_est_us /
                (double) return_rpc_total_us :
            0.0;

    if (return_transfer_count > 0) {
        printf(
            "[RETURN_CRITICAL_PATH_GRAPH] returns=%" PRId64
            " payload_mib=%.3f request_ms=%.3f ready_wait_ms=%.3f "
            "recv_payload_ms=%.3f rpc_total_ms=%.3f "
            "d2h_accounted_ms=%.3f lane_wait_ms=%.3f "
            "barrier_wait_ms=%.3f exposed_wait_ms=%.3f "
            "overlap_est_ms=%.3f overlap_ratio=%.3f\n",
            return_transfer_count,
            return_payload_bytes / 1048576.0,
            return_request_us / 1000.0,
            return_ready_wait_us / 1000.0,
            return_recv_payload_us / 1000.0,
            return_rpc_total_us / 1000.0,
            d2h_us / 1000.0,
            lane_reuse_wait_us / 1000.0,
            layer_barrier_wait_us / 1000.0,
            explicit_return_wait_us / 1000.0,
            return_overlap_est_us / 1000.0,
            return_overlap_ratio);
    }

    const int64_t meta_graph_end_us = return_wavefront_graph ? ggml_time_us() : 0;
    const int64_t meta_total_us =
        return_wavefront_graph ? meta_graph_end_us - meta_graph_start_us : 0;
    const int64_t main_accounted_us =
        return_wavefront_graph ? compute_wall_us + reduce_wall_us + layer_barrier_wait_us : 0;
    const int64_t other_main_us =
        return_wavefront_graph ? std::max<int64_t>(0, meta_total_us - main_accounted_us) : 0;

    if (return_wavefront_graph) {
        printf(
            "[TENSOR_RUNTIME_SUM] "
            "attn_ms=%.3f pc_ffn_ms=%.3f h2d_ms=%.3f phone_ms=%.3f "
            "d2h_ms=%.3f reduce_ms=%.3f wait_ms=%.3f "
            "compute_wall_ms=%.3f reduce_wall_ms=%.3f "
            "meta_total_ms=%.3f other_main_ms=%.3f\n",
            tensor_attn_us / 1000.0,
            tensor_pc_ffn_us / 1000.0,
            h2d_us / 1000.0,
            tensor_phone_us / 1000.0,
            d2h_us / 1000.0,
            tensor_reduce_us / 1000.0,
            tensor_wait_us / 1000.0,
            compute_wall_us / 1000.0,
            reduce_wall_us / 1000.0,
            meta_total_us / 1000.0,
            other_main_us / 1000.0);
    }
    if (tensor_phone_stage_profile) {
        for (const auto & item : tensor_phone_stage_timings) {
            const int layer = item.first.first;
            const int chunk = item.first.second;
            const tensor_phone_stage_timing & stage = item.second;

            const int64_t worker_parallel_est_us =
                std::max(
                    stage.pc_ffn_worker_us,
                    stage.phone_ffn_worker_us);
            const int64_t ffn_overhead_est_us =
                std::max<int64_t>(
                    0,
                    stage.ffn_wall_us - worker_parallel_est_us);
            const int64_t client_path_est_us =
                stage.pre_route_wall_us +
                stage.route_handoff_us +
                stage.ffn_wall_us +
                stage.return_client_wall_us;

            printf(
                "[TENSOR_PHONE_STAGE] mode=%s layer=%d chunk=%d "
                "pre_route_wall_ms=%.3f pre_route_pc_worker_ms=%.3f "
                "pre_route_phone_worker_ms=%.3f route_handoff_ms=%.3f "
                "ffn_wall_ms=%.3f pc_ffn_worker_ms=%.3f "
                "phone_ffn_worker_ms=%.3f worker_parallel_est_ms=%.3f "
                "ffn_overhead_est_ms=%.3f return_client_wall_ms=%.3f "
                "client_path_est_ms=%.3f counts=%d/%d/%d/%d\n",
                stage.decode ? "decode" : "prefill",
                layer,
                chunk,
                stage.pre_route_wall_us / 1000.0,
                stage.pre_route_pc_worker_us / 1000.0,
                stage.pre_route_phone_worker_us / 1000.0,
                stage.route_handoff_us / 1000.0,
                stage.ffn_wall_us / 1000.0,
                stage.pc_ffn_worker_us / 1000.0,
                stage.phone_ffn_worker_us / 1000.0,
                worker_parallel_est_us / 1000.0,
                ffn_overhead_est_us / 1000.0,
                stage.return_client_wall_us / 1000.0,
                client_path_est_us / 1000.0,
                stage.route_compute_count,
                stage.route_handoff_count,
                stage.ffn_compute_count,
                stage.return_count);
        }
    }

    {
        std::lock_guard<std::mutex> lock(backend_ctx->tensor_profile_mutex);
        backend_ctx->tensor_profile.attn_us   += tensor_attn_us;
        backend_ctx->tensor_profile.pc_ffn_us += tensor_pc_ffn_us;
        backend_ctx->tensor_profile.h2d_us    += h2d_us;
        backend_ctx->tensor_profile.phone_us  += tensor_phone_us;
        backend_ctx->tensor_profile.d2h_us    += d2h_us;
        backend_ctx->tensor_profile.reduce_us += tensor_reduce_us;
        backend_ctx->tensor_profile.wait_us   += tensor_wait_us;
        backend_ctx->tensor_profile.return_transfer_count +=
            return_transfer_count;
        backend_ctx->tensor_profile.return_payload_bytes +=
            return_payload_bytes;
        backend_ctx->tensor_profile.return_request_us +=
            return_request_us;
        backend_ctx->tensor_profile.return_ready_wait_us +=
            return_ready_wait_us;
        backend_ctx->tensor_profile.return_recv_payload_us +=
            return_recv_payload_us;
        backend_ctx->tensor_profile.return_rpc_total_us +=
            return_rpc_total_us;
        backend_ctx->tensor_profile.graph_compute_count += 1;
        backend_ctx->tensor_profile.graph_total_us += meta_graph_total_all_us;
        backend_ctx->tensor_profile.graph_rebuild_us += meta_rebuild_us;
        backend_ctx->tensor_profile.graph_execute_us += meta_execute_us;
        backend_ctx->tensor_profile.graph_other_us += meta_other_all_us;
        backend_ctx->tensor_profile.simple_backend_count =
            std::max<int64_t>(
                backend_ctx->tensor_profile.simple_backend_count,
                (int64_t) n_backends);
        for (size_t j = 0;
             j < n_backends &&
             j < GGML_BACKEND_META_MAX_DEVICES;
             ++j) {
            backend_ctx->tensor_profile.simple_backend_compute_us[j] +=
                compute_workers.backend_time_us[j];
            backend_ctx->tensor_profile.simple_backend_compute_calls[j] +=
                compute_workers.backend_call_count[j];
        }

        for (const auto & entry : layer_timings) {
            const int first_layer = entry.first.first;
            const int last_layer  = entry.first.second;
            const meta_layer_timing & timing = entry.second;
            if (first_layer < 0 || last_layer < first_layer) {
                continue;
            }

            const int64_t dst =
                backend_ctx->tensor_profile.layer_timing_entries;
            if (dst >= GGML_BACKEND_META_MAX_LAYER_TIMINGS) {
                break;
            }

            backend_ctx->tensor_profile.layer_timing_first[dst] =
                first_layer;
            backend_ctx->tensor_profile.layer_timing_last[dst] =
                last_layer;
            backend_ctx->tensor_profile.layer_timing_pc_compute_us[dst] =
                timing.compute_pc_us;
            backend_ctx->tensor_profile.layer_timing_phone_compute_us[dst] =
                timing.compute_phone_us;
            backend_ctx->tensor_profile.layer_timing_copy_0to1_us[dst] =
                timing.copy_0to1_us;
            backend_ctx->tensor_profile.layer_timing_copy_1to0_us[dst] =
                timing.copy_1to0_us;
            backend_ctx->tensor_profile.layer_timing_wall_us[dst] =
                timing.total_us;
            backend_ctx->tensor_profile.layer_timing_entries =
                dst + 1;
        }

        if (needs_rebuild) {
            backend_ctx->tensor_profile.graph_rebuild_count += 1;
        }
        if (return_wavefront_graph) {
            backend_ctx->tensor_profile.compute_wall_us += compute_wall_us;
            backend_ctx->tensor_profile.reduce_wall_us  += reduce_wall_us;
            backend_ctx->tensor_profile.meta_total_us   += meta_total_us;
            backend_ctx->tensor_profile.other_main_us   += other_main_us;

            if (backend_ctx->tensor_profile_wave_span_begin_us == 0 ||
                meta_graph_start_us < backend_ctx->tensor_profile_wave_span_begin_us) {
                backend_ctx->tensor_profile_wave_span_begin_us = meta_graph_start_us;
            }
            backend_ctx->tensor_profile_wave_span_end_us =
                std::max(backend_ctx->tensor_profile_wave_span_end_us, meta_graph_end_us);

            for (const auto & [layer, start_us] : return_wave_layer_start_us) {
                auto it = backend_ctx->tensor_profile_wave_layer_start_us.find(layer);
                if (it == backend_ctx->tensor_profile_wave_layer_start_us.end() ||
                    start_us < it->second) {
                    backend_ctx->tensor_profile_wave_layer_start_us[layer] = start_us;
                }
            }
            for (const auto & [layer, value_us] : return_wave_layer_compute_wall_us) {
                backend_ctx->tensor_profile_wave_layer_compute_wall_us[layer] += value_us;
            }
            for (const auto & [layer, value_us] : return_wave_layer_barrier_us) {
                backend_ctx->tensor_profile_wave_layer_barrier_us[layer] += value_us;
            }
        }
        backend_ctx->tensor_profile.lane_reuse_wait_count += lane_reuse_wait_count;
        backend_ctx->tensor_profile.lane_reuse_wait_us += lane_reuse_wait_us;
        backend_ctx->tensor_profile.lane_reuse_wait_max_us = std::max(
            backend_ctx->tensor_profile.lane_reuse_wait_max_us, lane_reuse_wait_max_us);
        for (size_t lane = 0; lane < ggml_backend_meta_context::PREFILL_RETURN_LANES; ++lane) {
            backend_ctx->tensor_profile.lane_reuse_wait_count_by_lane[lane] +=
                lane_reuse_wait_count_by_lane[lane];
            backend_ctx->tensor_profile.lane_reuse_wait_us_by_lane[lane] += lane_reuse_wait_us_by_lane[lane];
        }
        backend_ctx->tensor_profile.layer_barrier_wait_count += layer_barrier_wait_count;
        backend_ctx->tensor_profile.layer_barrier_wait_us += layer_barrier_wait_us;
        if (layer_barrier_wait_count > 0 &&
            layer_barrier_wait_max_us >= backend_ctx->tensor_profile.layer_barrier_wait_max_us) {
            backend_ctx->tensor_profile.layer_barrier_wait_max_us = layer_barrier_wait_max_us;
            backend_ctx->tensor_profile.layer_barrier_wait_max_layer = layer_barrier_wait_max_layer;
            backend_ctx->tensor_profile.layer_barrier_wait_max_pending = layer_barrier_wait_max_pending;
            backend_ctx->tensor_profile.layer_barrier_wait_max_last_lane = layer_barrier_wait_max_last_lane;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_graph_compute(
        ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_meta_context * backend_ctx =
        (ggml_backend_meta_context *) backend->context;

    bool async_enabled = false;
    {
        std::lock_guard<std::mutex> lock(backend_ctx->async_graph_mutex);
        async_enabled = backend_ctx->async_graph_compute_enabled;
    }
    if (!async_enabled) {
        return ggml_backend_meta_graph_compute_impl(backend, cgraph);
    }

    // There must be at most one in-flight graph for the dedicated Phone
    // scheduler. Join a stale completed worker defensively before reuse.
    const ggml_status previous_status =
        ggml_backend_meta_wait_async_graph_impl(backend_ctx);
    if (previous_status != GGML_STATUS_SUCCESS) {
        return previous_status;
    }

    {
        std::lock_guard<std::mutex> lock(backend_ctx->async_graph_mutex);
        backend_ctx->async_graph_status = GGML_STATUS_SUCCESS;
        backend_ctx->async_graph_active = true;
        backend_ctx->async_graph_prepared = false;
    }

    backend_ctx->async_graph_thread =
        std::thread([backend, cgraph, backend_ctx]() {
            const ggml_status status =
                ggml_backend_meta_graph_compute_impl(backend, cgraph);
            {
                std::lock_guard<std::mutex> lock(
                    backend_ctx->async_graph_mutex);
                backend_ctx->async_graph_status = status;
                backend_ctx->async_graph_active = false;
            }
            backend_ctx->async_graph_cv.notify_all();
        });

    // Wait only until rebuild/preparation has finished. The expensive Meta
    // execution/Phone handoff then continues on async_graph_thread.
    {
        std::unique_lock<std::mutex> lock(backend_ctx->async_graph_mutex);
        backend_ctx->async_graph_cv.wait(
            lock,
            [&]() {
                return backend_ctx->async_graph_prepared ||
                       !backend_ctx->async_graph_active;
            });

        if (!backend_ctx->async_graph_active) {
            const ggml_status status = backend_ctx->async_graph_status;
            lock.unlock();
            if (backend_ctx->async_graph_thread.joinable()) {
                backend_ctx->async_graph_thread.join();
            }
            return status;
        }
    }

    if (std::getenv("GGML_META_ASYNC_GRAPH_DEBUG") != nullptr) {
        printf(
            "[META_ASYNC_GRAPH_SUBMIT] uid=%" PRIu64
            " nodes=%d prepared=1\n",
            cgraph->uid,
            cgraph->n_nodes);
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}

bool ggml_backend_meta_set_async_graph_compute(
        ggml_backend_t backend, bool enabled) {
    if (!ggml_backend_is_meta(backend)) {
        return false;
    }

    auto * backend_ctx =
        (ggml_backend_meta_context *) backend->context;
    std::lock_guard<std::mutex> lock(backend_ctx->async_graph_mutex);
    backend_ctx->async_graph_compute_enabled = enabled;
    return true;
}

enum ggml_status ggml_backend_meta_wait_async_graph(
        ggml_backend_t backend) {
    if (!ggml_backend_is_meta(backend)) {
        return GGML_STATUS_SUCCESS;
    }
    auto * backend_ctx =
        (ggml_backend_meta_context *) backend->context;
    return ggml_backend_meta_wait_async_graph_impl(backend_ctx);
}

bool ggml_backend_meta_set_tensor_phone_primary_layers(
        ggml_backend_t backend, int first_layer, int last_layer) {
    if (!ggml_backend_is_meta(backend)) {
        return false;
    }
    GGML_ASSERT((first_layer == -1 && last_layer == -1) ||
                (first_layer >= 0 && last_layer > first_layer));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    if (std::getenv("GGML_META_PIPELINE_DEBUG") != nullptr &&
        (backend_ctx->tensor_phone_first_layer != first_layer ||
         backend_ctx->tensor_phone_last_layer != last_layer)) {
        printf("[META_TP_POLICY] layers=[%d,%d)\n", first_layer, last_layer);
    }
    backend_ctx->tensor_phone_first_layer = first_layer;
    backend_ctx->tensor_phone_last_layer  = last_layer;
    return true;
}

bool ggml_backend_meta_tensor_profile_reset(ggml_backend_t backend) {
    if (!ggml_backend_is_meta(backend)) {
        return false;
    }
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    std::lock_guard<std::mutex> lock(backend_ctx->tensor_profile_mutex);
    backend_ctx->tensor_profile = {};
    backend_ctx->tensor_profile_wave_layer_start_us.clear();
    backend_ctx->tensor_profile_wave_layer_compute_wall_us.clear();
    backend_ctx->tensor_profile_wave_layer_barrier_us.clear();
    backend_ctx->tensor_profile_wave_span_begin_us = 0;
    backend_ctx->tensor_profile_wave_span_end_us   = 0;
    return true;
}

bool ggml_backend_meta_tensor_profile_get(
        ggml_backend_t backend, ggml_backend_meta_tensor_profile * profile) {
    if (!ggml_backend_is_meta(backend) || profile == nullptr) {
        return false;
    }
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    std::lock_guard<std::mutex> lock(backend_ctx->tensor_profile_mutex);
    *profile = backend_ctx->tensor_profile;

    const auto median_us = [](std::vector<int64_t> values) -> int64_t {
        if (values.empty()) {
            return 0;
        }
        std::sort(values.begin(), values.end());
        const size_t n = values.size();
        return n % 2 != 0 ?
            values[n / 2] :
            (values[n / 2 - 1] + values[n / 2]) / 2;
    };

    const auto & starts = backend_ctx->tensor_profile_wave_layer_start_us;
    profile->wave_layer_start_count = (int64_t) starts.size();
    if (!starts.empty()) {
        const int64_t first_start_us = starts.begin()->second;
        const int64_t last_start_us  = starts.rbegin()->second;

        if (backend_ctx->tensor_profile_wave_span_begin_us > 0) {
            profile->wave_fill_us = std::max<int64_t>(
                0, first_start_us - backend_ctx->tensor_profile_wave_span_begin_us);
        }
        if (backend_ctx->tensor_profile_wave_span_end_us > 0) {
            profile->wave_drain_us = std::max<int64_t>(
                0, backend_ctx->tensor_profile_wave_span_end_us - last_start_us);
        }
        if (backend_ctx->tensor_profile_wave_span_begin_us > 0 &&
            backend_ctx->tensor_profile_wave_span_end_us >=
                backend_ctx->tensor_profile_wave_span_begin_us) {
            profile->wave_span_us =
                backend_ctx->tensor_profile_wave_span_end_us -
                backend_ctx->tensor_profile_wave_span_begin_us;
        }

        std::vector<int64_t> intervals;
        intervals.reserve(starts.size() > 1 ? starts.size() - 1 : 0);
        auto prev = starts.begin();
        for (auto it = std::next(prev); it != starts.end(); ++it) {
            if (it->first == prev->first + 1 && it->second >= prev->second) {
                intervals.push_back(it->second - prev->second);
            }
            prev = it;
        }

        if (!intervals.empty()) {
            profile->wave_ii_count = (int64_t) intervals.size();
            profile->wave_ii_sum_us =
                std::accumulate(intervals.begin(), intervals.end(), int64_t(0));
            profile->wave_ii_min_us =
                *std::min_element(intervals.begin(), intervals.end());
            profile->wave_ii_max_us =
                *std::max_element(intervals.begin(), intervals.end());
            profile->wave_ii_median_us = median_us(intervals);

            // Use symmetric first/last windows. For an odd interval count,
            // leave the middle interval out of both windows.
            const size_t half = intervals.size() / 2;
            if (half > 0) {
                std::vector<int64_t> early(intervals.begin(), intervals.begin() + half);
                std::vector<int64_t> late(intervals.end() - half, intervals.end());

                profile->wave_early_ii_count = (int64_t) early.size();
                profile->wave_early_ii_sum_us =
                    std::accumulate(early.begin(), early.end(), int64_t(0));
                profile->wave_early_ii_median_us = median_us(early);

                profile->wave_late_ii_count = (int64_t) late.size();
                profile->wave_late_ii_sum_us =
                    std::accumulate(late.begin(), late.end(), int64_t(0));
                profile->wave_late_ii_median_us = median_us(late);
            }
        }

        const size_t layer_half = starts.size() / 2;
        if (layer_half > 0) {
            size_t index = 0;
            for (const auto & [layer, start_us] : starts) {
                GGML_UNUSED(start_us);
                const bool early = index < layer_half;
                const bool late  = index >= starts.size() - layer_half;

                const auto compute_it =
                    backend_ctx->tensor_profile_wave_layer_compute_wall_us.find(layer);
                const auto barrier_it =
                    backend_ctx->tensor_profile_wave_layer_barrier_us.find(layer);
                const int64_t compute_us =
                    compute_it != backend_ctx->tensor_profile_wave_layer_compute_wall_us.end() ?
                        compute_it->second : 0;
                const int64_t barrier_us =
                    barrier_it != backend_ctx->tensor_profile_wave_layer_barrier_us.end() ?
                        barrier_it->second : 0;

                if (early) {
                    ++profile->wave_early_layer_count;
                    profile->wave_early_compute_wall_us += compute_us;
                    profile->wave_early_barrier_us += barrier_us;
                }
                if (late) {
                    ++profile->wave_late_layer_count;
                    profile->wave_late_compute_wall_us += compute_us;
                    profile->wave_late_barrier_us += barrier_us;
                }
                ++index;
            }
        }
    }
    return true;
}
